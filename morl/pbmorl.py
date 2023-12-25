import os, sys
import time
from copy import deepcopy
import numpy as np
import torch
from multiprocessing import Process, Queue, Event
import pickle
# import our packages
from scalarization_methods import WeightedSumScalarization
from sample import Sample
from sample import Task
from ep import EP
from population import Population
from opt_graph import OptGraph
from utils import generate_preferred_task
from initialization import initialize_warm_up_batch
from moppo import MOPPO_worker
from population import decisionmaker
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir,'externals/baselines/'))
sys.path.append(os.path.join(base_dir,'externals/pytorch-a2c-ppo-acktr-gail/'))

def run(args):
    # --------------------> Preparation <-------------------- #
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)
    start_time = time.time()
    ep = EP()
    population = Population(args)
    opt_graph = OptGraph()
    preference = decisionmaker(obj_num=args.obj_num)
    elite_batch, scalarization_batch = initialize_warm_up_batch(args, device)
    rl_num_updates = args.warmup_iter
    num_search = len(scalarization_batch)
    for sample, scalarization in zip(elite_batch, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)
    iteration = 0
    best_scalar = []
    args.save_dir = "./"+args.env_name+"/"
    args.num_tasks = int(num_search* (1.0-args.population_ratio))
    total_iterations = (args.num_env_steps - args.warmup_iter*args.num_steps)/args.update_iter*args.num_steps
    while iteration < total_iterations:
        task_batch = []
        if iteration >0:
            generate_preferred_task(args,num_search,best_scalar,elite_batch,scalarization_batch)
        for elite, scalarization in \
                zip(elite_batch, scalarization_batch):
            task_batch.append(Task(elite, scalarization))
        processes = []
        results_queue = Queue()
        done_event = Event()
        for task_id, task in enumerate(task_batch):
            p = Process(target = MOPPO_worker, \
                args = (args,task_id, task, device, iteration, rl_num_updates, start_time, results_queue, done_event))
            p.start()
            processes.append(p)
        all_offspring_batch = [[] for _ in range(len(processes))]
        cnt_done_workers = 0
        llen = len(processes)
        while cnt_done_workers < llen:
            rl_results = results_queue.get()
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            wei = rl_results['weights']
            for sample in offsprings:
                sample.weights = wei
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            if rl_results['done']:
                cnt_done_workers += 1
        all_sample_batch = []
        last_offspring_batch = [None] * len(processes)
        offspring_batch = []
        for task_id in range(len(processes)):
            offsprings = all_offspring_batch[task_id]
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % 1 == 0:
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]
        done_event.set()

        ep.update(all_sample_batch)
        if iteration > 0 and iteration % 10 == 0:
            epans = EP()
            tempep = np.array([])
            for sample in ep.sample_batch:
                tempep = np.vstack([tempep, sample.objs]) if len(tempep) > 0 else np.array([sample.objs])
            u, _ = preference.gp.get_predictive_params(tempep, pointwise=False, data_from=False)
            l = len(tempep)
            ff = np.zeros((1, l))
            while len(epans.sample_batch) < len(ep.sample_batch)*0.85:
                maxx = -100000.0
                sig = 0
                for i in range(len(u)):
                    if u[i] > maxx and ff[0][i] == 0:
                        maxx = u[i]
                        sig = i
                ff[0][sig] = 1
                if u[sig] < 0:
                    break
                epans.update([ep.sample_batch[sig]])
            ep = epans
        ave_ep = np.zeros(args.obj_num, )
        if len(ep.sample_batch):
            for i in range(len(ep.sample_batch)):
                for j in range(args.obj_num):
                    ave_ep[j] += ep.sample_batch[i].objs[j]
            ave_ep /= float(len(ep.sample_batch))
        population.update(offspring_batch,args.obj_num)
        elite_batch, scalarization_batch, best_scalar = population.preference_learning(args, iteration, ep, opt_graph,
                                                                                       scalarization_template,
                                                                                       preference)
        iteration += 1
        rl_num_updates = args.update_iter

        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for sample in ep.sample_batch:
                obj = sample.objs
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))
        with open(os.path.join(ep_dir,'weights.txt'),'w') as fp:
            for sample in ep.sample_batch:
                w = sample.weights
                filestr=""
                for j in range(args.obj_num):
                    filestr += (str(w[j])+',')
                filestr += '\n'
                fp.write(filestr)
        if len(ep.sample_batch) > 0:
            with open(os.path.join(ep_dir,'average.txt'),'w') as fp:
                fp.write(('{:5f}'+(args.obj_num -1)*',{:5f}'+'\n').format(*ave_ep))


    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)
    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
        with open(os.path.join(args.save_dir, 'final', 'EP_env_params_{}.pkl'.format(i)), 'wb') as fp:
            pickle.dump(sample.env_params, fp)

    with open(os.path.join(args.save_dir, 'final', 'objs.txt'), 'w') as fp:
        for i, obj in enumerate(ep.obj_batch):
            objs = obj
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))

    if args.obj_rms:
        with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
            for sample in ep.sample_batch:
                fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))

