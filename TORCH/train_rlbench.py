# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from RLBenchEnv import SimulationEnvironment
from rlbench.tasks import ReachTarget

#Agents
from agents.pg import initialize_pg
from agents.ddpg import initialize_ddpg
from agents.dqn import initialize_dqn
from agents.td3 import initialize_td3
from agents.sac import initialize_sac
from agents.a2c import initialize_a2c
from agents.ppo import initialize_ppo


import logz


MAX_EPISODES = 20000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e3
BATCH_SIZE = 256
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
EPISODE_LENGTH = 200  # max steps in each episode
TEST_EVERY_STEPS = 200  # e2  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
REWARD_SCALE = 1

#TODO : revoir ça
SAVE_INTERVAL = 20 if agent.on_policy else 20000
EVAL_INTERVAL = 20 if agent.on_policy else 20000
LOG_INTERVAL = 1 if agent.on_policy else 1000


def run_train_episode(env, agent, itr):
    # collect rollouts
    agent.collect_rollouts(itr, False)
    # update agent
    agent.update()

# evaluate agent, calculate reward mean of 5 episodes
#TODO : passer sous Torch ici
""" 
def evaluate_episode(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        target_pose = np.expand_dims(obs[-3:], axis=0)

        for i in range(EPISODE_LENGTH):
            batch_obs = np.expand_dims(obs[8:15], axis=0)
            batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)
            action = agent.predict(batch_obs_full.astype('float32'))

            # Add gripper action again
            action = np.append(action, 0)

            action = np.squeeze(action)
            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)
"""

#TODO passer sous Torch ici
""" 
def test_episode(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    target_pose = np.expand_dims(obs[-3:], axis=0)

    for i in range(EPISODE_LENGTH):
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)
        action = agent.predict(batch_obs_full.astype('float32'))

        # Add gripper action again
        action = np.append(action, 0)

        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if render:
            env.render()

        if done:
            break

    return total_reward
"""


def main(args):

    alg_name = 'td3' #TODO pass alg_name to params
    cpu = True #TODO same here


    state_types = [ 'gripper_open',
                    'joint_velocities',
                    'joint_positions',
                    'joint_forces',
                    'gripper_pose',
                    'gripper_joint_positions',
                    'gripper_touch_forces',
                    'task_low_dim_state']

    # Create rlbench gym env
    env = SimulationEnvironment(task_name=ReachTarget, 
                                state_type_list=state_types, 
                                headless=args.headless)
     
    env.reset()
    obs_dim = 7 + 3  # 7 joint positions plus 3 target poses

    # drop gripper action to speed up training
    act_dim = env.action_space.shape[0]-1

    max_action = float(env.action_space.high[0])


    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    print(device)

    if alg_name == 'td3': #TODO : ajouter d'autres algos facilement ?
            agent, params = initialize_td3( env = env, 
                                            device = device, 
                                            obs_dim = obs_dim, 
                                            act_dim = act_dim,
                                            max_action = max_action,
                                            min_action = -max_action)
    

    test_flag = 0
    itr = 0
    if args.train:
        logger = CustomLogger('train_log/train_gym.txt')
        while itr < MAX_EPISODES:
            run_train_episode(env, agent, itr)

            # log diagnostics
            if itr % LOG_INTERVAL == 0:
                print("********** Iteration %i ************" % itr)
                stats = agent.log_progress()
                logz.log_tabular("Iteration", itr)
                for x in stats.keys():
                    logz.log_tabular(x, stats[x])
                logz.dump_tabular()


            #TODO eval agent
            """ 
            if itr % EVAL_INTERVAL == 0:
                agent.eval_env.setup_recording(os.path.join(logdir, 'video_' + str(itr)))
                episode_returns, episode_lengths = agent.eval(
                    num_eval_episodes, render)
                print("---------------------------------------")
                print("Evaluation over %d episodes: %f" %
                    (num_eval_episodes, np.mean(episode_returns)))
                print("---------------------------------------")
            """


            #TODO save models
            """
            if itr % SAVE_INTERVAL == 0:
                agent.save(logdir, itr)
            """


            itr += 1
    print('Done')
    env.close()
    

    if args.test: #TODO
        for i in range(20):
            test_reward = test_episode(env, agent)
            print('Steps {}, Test reward: {}'.format(
                i, test_reward))

    print('Done')
    env.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test mode.')
    parser.add_argument('--train', dest='train',
                        action='store_true', default=False)
    parser.add_argument('--test', dest='test',
                        action='store_true', default=False)
    parser.add_argument('--headless', dest='headless',
                        action='store_true', default=False)
    parser.add_argument('--mode', help='render mode name', default='None')
    
    args = parser.parse_args()

    main(args) #TODO maybe gerer GPU ici ?
