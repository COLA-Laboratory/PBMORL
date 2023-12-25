import os, sys

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from collections import deque
from copy import deepcopy
import time
import torch
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.storage import RolloutStorage
from sample import Sample
from utils import pbmorl_make_env,CheckMOGYM,FormulateReward

'''
Evaluate a policy sample.
'''


def evaluation(args, sample):
    eval_env = pbmorl_make_env(args.env_name)
    objs = np.zeros(args.obj_num)
    policy = sample.actor_critic
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            if not CheckMOGYM(args.env_name):
                eval_env.seed(args.seed + eval_id)
            if CheckMOGYM(args.env_name):
                ob,_ = eval_env.reset()
            else:
                ob = eval_env.reset()
            done = False
            gamma = 1.0
            while not done:
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                packedReturn = eval_env.step(action)
                ob, done, info = FormulateReward(packedReturn,args.env_name)
                objs += gamma * info['obj']
                if not args.raw:
                    gamma *= args.gamma
    eval_env.close()
    objs /= args.eval_num
    return objs


def MOPPO_worker(args, task_id, task, device, iteration, num_updates, start_time, results_queue, done_event):
    torch.set_num_threads(1)
    scalarization = task.scalarization
    env_params, actor_critic, agent = task.sample.env_params, task.sample.actor_critic, task.sample.agent
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                         gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                         obj_rms=args.obj_rms, ob_rms=args.ob_rms)
    if env_params['ob_rms'] is not None:
        envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
    if env_params['ret_rms'] is not None:
        envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
    if env_params['obj_rms'] is not None:
        envs.venv.obj_rms = deepcopy(env_params['obj_rms'])
    rollouts = RolloutStorage(num_steps=args.num_steps, num_processes=args.num_processes,
                              obs_shape=envs.observation_space.shape, action_space=envs.action_space,
                              recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
                              obj_num=args.obj_num)
    if not CheckMOGYM(args.env_name):
        obs = envs.reset()
    else:
        obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    episode_lens = deque(maxlen=10)
    episode_objs = deque(maxlen=10)  # for each cost component we care
    episode_obj = np.array([None] * args.num_processes)

    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    offspring_batch = []

    start_iter, final_iter = iteration, min(iteration + num_updates, total_num_updates)
    #start_iter=0
    if iteration == 0:
        final_iter=start_iter+args.warmup_iter
    else:
        final_iter=start_iter+args.update_iter
        
    for j in range(start_iter, final_iter):
        torch.manual_seed(j)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule( \
                agent.optimizer, j * args.lr_decay_ratio, \
                total_num_updates, args.lr)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            packedReturn = envs.step(action)
            ob, done, infos = FormulateReward(packedReturn,args.env_name)
            obj_tensor = torch.zeros([args.num_processes, args.obj_num])
            if not CheckMOGYM(args.env_name):
                for idx, info in enumerate(infos):
                    obj_tensor[idx] = torch.from_numpy(info['obj'])
                    episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_lens.append(info['episode']['l'])
                        if episode_obj[idx] is not None:
                            episode_objs.append(episode_obj[idx])
                            episode_obj[idx] = None
            else:
                obj_tensor[0] = infos['obj']

            if not CheckMOGYM(args.env_name):
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
            else:
                masks = torch.FloatTensor([0.0] if done else [1.0])
                bad_masks = torch.FloatTensor([1.0])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

        value_loss, action_loss, dist_entropy = agent.update(rollouts, scalarization, obj_rms_var)

        rollouts.after_update()

        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None

        # evaluate new sample
        sample = Sample(env_params, deepcopy(actor_critic), deepcopy(agent))
        objs = evaluation(args, sample)
        sample.objs = objs
        offspring_batch.append(sample)

        # put results back every update_iter iterations, to avoid the multi-processing crash
        if (j + 1) % args.update_iter == 0 or j == final_iter - 1:
            offspring_batch = np.array(offspring_batch)
            results = {}
            results['task_id'] = task_id
            results['weights'] = task.scalarization.weights
            results['offspring_batch'] = offspring_batch
            if j == final_iter - 1:
                results['done'] = True
            else:
                results['done'] = False
            results_queue.put(results)
            offspring_batch = []

    envs.close()

    done_event.wait()
