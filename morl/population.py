import numpy as np
from copy import deepcopy
from utils import generate_weights_batch_dfs
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from gaussian_process import GPPairwise
from dataset import DatasetPairwise
import math


class decisionmaker:
    def __init__(self,obj_num):
        self.random_state = np.random.RandomState(0)
        self.terms = [[(0, 1), (1, 1)]]
        self.coeffs = [6.25]
        self.obj_num = obj_num
        self.dataset = DatasetPairwise(self.obj_num)
        self.gp = GPPairwise(self.obj_num, kernel_width=0.45, std_noise=0.1)
        self.min_point = None
        self.max_point = None
        self.thresh_dist = 0.001
        self.temp_linear_prior = False
        self.keep_set_small = False
        self.cool = 0
        self.flame = 1

    def sample(self, sample_points):
        """ returns a sample of the GP utility at sample_points """
        return self.gp.sample(sample_points)

    def set_prior(self):
        if self.temp_linear_prior:
            if self.dataset.comparisons.shape[0] < 10:
                self.gp.prior_mean_type = 'linear'
            else:
                self.gp.prior_mean_type = 'zero'

    def update_gp(self, dataset):
        self.set_prior()
        self.gp.update(dataset)

'''
Population class maintains the population of the policies by performance buffer strategy.
'''
class Population:
    def __init__(self, args):
        self.sample_batch = []  # all samples in population
        self.obj_num = args.obj_num
        self.z_min = np.zeros(args.obj_num)  # reference point
        self.pbuffer_size = args.pbuffer_size
        self.pbuffers = None
        self.pbuffer_dist = None
        self.pbuffer_vec = []

        if self.obj_num == 2:
            self.pbuffer_num = args.pbuffer_num
            self.dtheta = np.pi / 2.0 / self.pbuffer_num
        else:
            generate_weights_batch_dfs(0, args.obj_num, 0.0, 1.0, 1.0 / (args.pbuffer_num - 1), [], self.pbuffer_vec)
            for i in range(len(self.pbuffer_vec)):
                self.pbuffer_vec[i] = self.pbuffer_vec[i] / np.linalg.norm(self.pbuffer_vec[i])
            self.pbuffer_num = len(self.pbuffer_vec)

        self.ask = []
        self.time_ask = 0
        self.sigmean = 1.0
        self.sigvar=0.1

    '''
    insert the sample to the performance buffers (storing the index).
    '''

    def find_buffer_id(self, f):
        max_dot, buffer_id = -np.inf, -1
        for i in range(self.pbuffer_num):
            dot = np.dot(self.pbuffer_vec[i], f)
            if dot > max_dot:
                max_dot, buffer_id = dot, i
        return buffer_id

    def insert_pbuffer_3D(self, index, objs):
        f = objs - self.z_min
        if np.min(f) < 1e-7:
            return False

        dist = np.linalg.norm(f)
        buffer_id = self.find_buffer_id(f)

        inserted = False
        for i in range(len(self.pbuffers[buffer_id])):
            if self.pbuffer_dist[buffer_id][i] < dist:
                self.pbuffers[buffer_id].insert(i, index)
                self.pbuffer_dist[buffer_id].insert(i, dist)
                inserted = True
                break
        if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
            self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
            self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
        elif (not inserted) and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
            self.pbuffers[buffer_id].append(index)
            self.pbuffer_dist[buffer_id].append(dist)
            inserted = True

        return inserted

    def insert_pbuffer_2D(self, index, objs):
        f = objs - self.z_min
        if np.min(f) < 1e-7:
            return False

        dist = np.linalg.norm(f)
        theta = np.arccos(np.clip(f[1] / dist, -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)
        if buffer_id < 0 or buffer_id >= self.pbuffer_num:
            return False

        inserted = False

        for i in range(len(self.pbuffers[buffer_id])):
            if self.pbuffer_dist[buffer_id][i] < dist:
                self.pbuffers[buffer_id].insert(i, index)
                self.pbuffer_dist[buffer_id].insert(i, dist)
                inserted = True
                break
        if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
            self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
            self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
        elif (not inserted) and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
            self.pbuffers[buffer_id].append(index)
            self.pbuffer_dist[buffer_id].append(dist)
            inserted = True

        return inserted

    '''
    update the population by a new offspring sample_batch.
    '''
    def update(self, sample_batch,obj_num):
        ### population = Union(population, offspring) ###
        all_sample_batch = self.sample_batch + sample_batch
        self.sample_batch = []
        self.pbuffers = [[] for _ in range(self.pbuffer_num)]  # store the sample indices in each pbuffer
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)]  # store the sample distance in each pbuffer

        ### select the population by performance buffer ###
        for i, sample in enumerate(all_sample_batch):
            if obj_num == 2:
                self.insert_pbuffer_2D(i, sample.objs)
            else:
                self.insert_pbuffer_3D(i,sample.objs)

        for pbuffer in self.pbuffers:
            for index in pbuffer:
                self.sample_batch.append(all_sample_batch[index])


    '''
    The prediction-guided task selection.
    '''
    #check if a new policy for interaction is too close to an old one.
    def CheckIfBeenAsked(self, args,newObj, oldObj):
        newObj = np.array(newObj).squeeze()
        oldObj = np.array(oldObj).squeeze()
        if np.linalg.norm(newObj - oldObj) > 1e-3:
            return True
        else:
            return False

    def SelectPolicyForInteraction(self,args, policyBatch,premean,prevar, banId=-1):
        maxx1 = -100000000
        maxx2 = -100000000
        counter = np.zeros((1, len(policyBatch)))
        counter = counter.squeeze()
        for i in range(len(counter)):
            counter[i] = 1
        for i in range(len(policyBatch)):
            if i == banId:
                counter[i] = 10000000
                continue
            for j in range(self.time_ask):
                if args.obj_num == 2:
                    if ~(self.CheckIfBeenAsked(args, policyBatch[i], self.ask[j][0:2])) or ~(self.CheckIfBeenAsked(args, policyBatch[i], self.ask[j][2:4])):
                        counter[i] += 1.0
                else:
                    if ~(self.CheckIfBeenAsked(args, policyBatch[i], self.ask[j][0:3])) or ~(self.CheckIfBeenAsked(args, policyBatch[i], self.ask[j][3:6])):
                        counter[i] += 1.0
            eita = self.sigmean * premean[i] + self.sigvar * (prevar[i] *
                                                          (float((self.time_ask)) / float(counter[i])) ** 0.5)
            if eita >= maxx1:
                maxx1 = eita
            if eita < maxx1 and eita >= maxx2:
                maxx2 = eita
        chse = []
        for i in range(len(policyBatch)):
            if self.sigmean * premean[i] + self.sigvar * (prevar[i] *
                                                          (float((self.time_ask)) / float(counter[i])) ** 0.5) == maxx1:
                chse.append(i)
        selected = np.zeros(2,)
        if len(chse) > 1:
            selected = np.random.randint(0,len(chse),size=2)
        else:
            selected[0] = chse[0]
            chse=[]
            for i in range(len(policyBatch)):
                if self.sigmean * premean[i] + self.sigvar * (prevar[i] *
                                                              (float((self.time_ask)) / float(
                                                                  counter[i])) ** 0.5) == maxx2:
                    chse.append(i)
            selected[1] = np.random.randint(0,len(chse))
        return selected

    def Interaction(self,args,preference,data1,data2):
        data1 = np.array(data1).squeeze()
        data2 = np.array(data2).squeeze()
        preference_setting = np.array(args.preference_setting).squeeze()
        getp=0
        if np.linalg.norm(data1-data2) < 1e-3:
            pass
        else:
            if np.dot(preference_setting,data1) > np.dot(preference_setting,data2) :
                getp=1
            else :
                getp=2
        if getp == 1:
            preference.dataset.add_single_comparison(data1, data2)
        if getp == 2:
            preference.dataset.add_single_comparison(data2, data1)

        if getp != 0:
            preference.update_gp(preference.dataset)

    def InsertNewAsk(self,data1,data2):
        self.time_ask += 1
        newAsk = []
        for i in range(len(data1)):
            newAsk.append(data1[i])
        for i in range(len(data2)):
            newAsk.append(data2[i])
        self.ask.append(newAsk)


    def preference_learning(self, args, iteration, ep, opt_graph, scalarization_template, preference):
        N = args.num_tasks
        candidates = []
        for sample in self.sample_batch:
            weight_center = opt_graph.weights[sample.optgraph_id]
            weight_center = weight_center / np.sum(weight_center)
            candidates.append({'sample': sample, 'weight': weight_center})
        elite_batch, scalarization_batch = [], []
        best_scalar = []
        temp=[]
        if iteration>=4:
            temp1 = ep.obj_batch
            for i in range(len(candidates)):
                temp1 = np.vstack([temp1, candidates[i]['sample'].objs]) if len(temp1) > 0 else np.array(
                    [candidates[i]['sample'].objs])
            temp = temp1
            if iteration ==4 or (iteration >4 and iteration%1==0):
                premean, prevar = preference.gp.get_predictive_params(temp, pointwise=True)
                selected = self.SelectPolicyForInteraction(args,temp,premean,prevar)
                data1 = temp[int(selected[0])]
                data2 = temp[int(selected[1])]
                self.InsertNewAsk(data1,data2)
                self.Interaction(args,preference,data1,data2)
            temp1 = []
            for i in range(len(temp)):
                if i < len(ep.obj_batch):
                    continue
                temp1.append(temp[i])
            temp = temp1
        else:
            selected = np.random.randint(0,len(candidates),size=2)
            data1 = candidates[selected[0]]['sample'].objs
            data2 = candidates[selected[1]]['sample'].objs
            self.InsertNewAsk(data1, data2)
            self.Interaction(args,preference,data1,data2)
            temp=[]
            for i in range(len(candidates)):
                temp.append(candidates[i]['sample'].objs)
            temp = np.array(temp)
        prem, prev = preference.gp.get_predictive_params(temp, pointwise=True)
        ff = np.zeros((1, 10000))
        for i in range(N):
            maxx = -100000
            best_id = -1
            for i in range(len(temp)):
                if prem[i] + 0.01 * math.sqrt(prev[i]) > maxx and ff[0][i] == 0:
                    best_id = i
                    maxx = prem[i] + 0.01 * math.sqrt(prev[i])
            if best_id == -1:
                break
            ff[0][best_id] = 1
            elite_batch.append(candidates[best_id]['sample'])
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(candidates[best_id]['sample'].weights)
            scalarization_batch.append(scalarization)
            best_scalar.append(scalarization)
        return elite_batch, scalarization_batch, best_scalar

