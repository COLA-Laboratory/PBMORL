import os, sys
import json

import gym
import mo_gymnasium as mo_gym

import includes
from scalarization_methods import WeightedSumScalarization

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from copy import deepcopy

    
def check_dominated(obj_batch, obj):
    return (np.logical_and(
                (obj_batch >= obj).all(axis=1), 
                (obj_batch > obj).any(axis=1))
            ).any()
            
# return sorted indices of nondominated objs
def get_ep_indices(obj_batch_input):
    if len(obj_batch_input) == 0: return np.array([])
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    ep_indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx]):
            ep_indices.append(idx)
    return ep_indices

currentEnv = None

def CheckMOGYM(env_name):
    global currentEnv
    if currentEnv == None:
        with open('../configs/environment.json','r') as f:
            envs = json.load(f)
            if env_name in envs["mogym"][0]:
                currentEnv = True
            else:
                currentEnv = False
    return currentEnv

def FormulateReward(packedReturn,env_name=""):
    global currentEnv
    ob = None
    done = None
    info = None
    if CheckMOGYM(env_name):
        ind=0
        for thing in packedReturn:
            if ind == 0:
                ob = thing
            if ind == 1:
                info = {'obj':thing}
            if ind == 2:
                done = thing
            ind +=1
    else:
        ind=0
        for thing in packedReturn:
            if ind == 0:
                ob = thing
            if ind == 2:
                done = thing
            if ind == 3:
                info = thing
            ind+=1
    return ob,done,info

def pbmorl_make_env(env_name):
    with open('../configs/environment.json','r') as f:
        envs = json.load(f)
    if env_name in envs["mujoco"][0]:
        with open('../configs/location/mujoco.json','r') as fmujoco:
            return gym.make(env_name)
    elif env_name in envs["mogym"][0]:
        mogymEnv = mo_gym.make(env_name)
        return mogymEnv
    elif env_name in envs["mmsd"][0]:
        mmsdEnv = includes.get_env(envs["mmsd"][0][env_name])
        envInstance = mmsdEnv()
        return envInstance
    elif env_name in envs["evogym"][0]:
        if env_name.find("-design") >0:
            pass
        else:
            body = None
            with open("../configs/evorobot.json","r") as fp:
                robots = json.load(fp)
                body = robots[env_name]
                body = np.array(body)
            evoEnv = gym.make(env_name,body = body)
            return evoEnv
    else:
        pass
    return 0



# update ep with a new point
def update_ep(ep_objs_batch, new_objs):
    if (new_objs < 0).any():
        return deepcopy(ep_objs_batch)
    new_ep_objs_batch = []
    on_ep = True
    for i in range(len(ep_objs_batch)):
        dominated = False
        if (new_objs >= ep_objs_batch[i]).all():
            dominated = True
        if (ep_objs_batch[i] >= new_objs - 1e-5).all() and (ep_objs_batch[i] > new_objs + 1e-5).any():
            on_ep = False
        if not dominated:
            new_ep_objs_batch.append(deepcopy(ep_objs_batch[i]))
    if on_ep:
        inserted = False
        for i in range(len(new_ep_objs_batch)): # gaurantee the new ep objs is still in order with first objective
            if new_objs[0] < new_ep_objs_batch[i][0]:
                new_ep_objs_batch.insert(i, deepcopy(new_objs))
                inserted = True
                break
        if not inserted:
            new_ep_objs_batch.append(deepcopy(new_objs))
        
    return new_ep_objs_batch

def generate_weights_batch_dfs(i, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch):
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(deepcopy(weight))
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight


#generate new tasks based on learnt preference information
def generate_preferred_task(args,num_search,best_scalar,elite_batch,scalarization_batch):
    best_weight = []
    weight_candidates = []
    generate_weights_batch_dfs(0, args.obj_num, 0.0, 1.0, args.delta_weight / 4.0, [], weight_candidates)
    if best_scalar != []:
        for scalar in best_scalar:
            best_weight.append(scalar.weights)
    for j in range(len(weight_candidates)):
        minn = 10000
        sig = 0
        for i in range(len(best_weight)):
            ans = 0
            for k in range(args.obj_num):
                ans += abs(best_weight[i][k] - weight_candidates[j][k]) ** 2
            if ans < minn:
                minn = ans
                sig = i
        for k in range(args.obj_num):
            weight_candidates[j][k] = weight_candidates[j][k] + 0.2 * (best_weight[sig][k] - weight_candidates[j][k])
    now_number = args.num_tasks
    while now_number < num_search:
        ind1 = np.random.randint(len(elite_batch))
        ind2 = np.random.randint(len(weight_candidates))
        elite_batch.append(elite_batch[ind1])
        scalar = WeightedSumScalarization(num_objs=args.obj_num, weights=weight_candidates[ind2])
        scalarization_batch.append(scalar)
        now_number += 1

#sample points in objective sapce and evaluate the preference model
def preference_evaluation(preference,obj_num):
    testbatch=[]
    preferenceAns = []
    for i in range(0,6000,200):
        for j in range(0,6000,200):
            obj1 = float(i) / 6000.0
            obj2 = float(j) / 6000.0
            if obj_num == 3:
                for k in range(0,6000,200):
                    obj3=float(k)/6000.0
                    testbatch.append([obj1,obj2,obj3])
            else:
                testbatch.append([obj1, obj2])
    testbatch=np.array(testbatch)
    testbatch=testbatch.squeeze()
    u,_=preference.gp.get_predictive_params(testbatch, pointwise=False, data_from=False)
    for i in range(len(u)):
        tempAns=[]
        for j in range(obj_num):
            tempAns.append(testbatch[i][j])
        tempAns.append(u[i])
        preferenceAns.append(tempAns)
    return preferenceAns


