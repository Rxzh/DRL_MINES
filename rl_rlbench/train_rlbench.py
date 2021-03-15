# -*- coding: utf-8 -*-

import os
import numpy as np
import parl
from parl import layers
import paddle
import tensorflow as tf
import argparse

from parl.utils import ReplayMemory  # 经验回放

from td3_model import RLBenchModel
from td3_agent import RLBenchAgent
from parl.algorithms import TD3


from td3 import td3Agent

from utils import CustomLogger, distance_cal
from RLBenchEnv import SimulationEnvironment

from rlbench.tasks import ReachTarget

paddle.enable_static()

MAX_EPISODES = 20000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 20 #1e3
BATCH_SIZE = 256
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
EPISODE_LENGTH = 50  # max steps in each episode
TEST_EVERY_STEPS = 200  # e2  
REWARD_SCALE = 1

def action_mapping(model_output_act, low_bound, high_bound):
    """ mapping action space [-1, 1] of model output 
        to new action space [low_bound, high_bound].

    Args:
        model_output_act: np.array, which value is in [-1, 1]
        low_bound: float, low bound of env action space
        high_bound: float, high bound of env action space

    Returns:
        action: np.array, which value is in [low_bound, high_bound]
    """
    assert np.all(((model_output_act<=1.0), (model_output_act>=-1.0))), \
        'the action should be in range [-1.0, 1.0]'
    assert high_bound > low_bound
    action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    action = np.clip(action, low_bound, high_bound)
    return action


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    # Use joint positions only
    target_pose = np.expand_dims(obs[-3:], axis=0)
    max_action = float(env.action_space.high[0])

    for steps in range(EPISODE_LENGTH):

        batch_obs = np.expand_dims(obs[8:14], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
            action[-1] = 0 # set gripper state as close

        else:
            """
            action = agent.predict(batch_obs_full.astype('float32'))
             # Add gripper action here, 0: close, 1: open
            action = np.append(action, 0)
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-max_action, max_action]
            action = np.clip(
                np.random.normal(action, EXPL_NOISE * max_action), -max_action,
                max_action)
            """

            action = agent.make_action(batch_obs_full, epsilon = EXPL_NOISE)


        next_obs, reward, done, info = env.step(action)

        # Use joint positions and target position only
        obs_full = np.concatenate((obs[8:14], obs[-3:]))
        next_obs_full = np.concatenate((next_obs[8:14], next_obs[-3:]))
        # Feed first 7 action into rpm here, gripper state is dropped here

        agent.memorize( obs = obs_full,
                        act = action[0:7],
                        reward = reward,
                        done = done,
                        new_obs = next_obs_full)
        
        """
        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward,  batch_terminal, batch_next_obs,idx = rpm.sample_batch(
                BATCH_SIZE)

            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
            
        """

    
        if rpm.size() > WARMUP_SIZE:
            agent.train()



        obs = next_obs
        total_reward += reward

        distance, _ = distance_cal(obs)

        if done:
            break

    return total_reward, distance


# evaluate agent, calculate reward mean of 5 episodes
def evaluate_episode(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        target_pose = np.expand_dims(obs[-3:], axis=0)

        for i in range(EPISODE_LENGTH):
            
            batch_obs = np.expand_dims(obs[8:14], axis=0)
            batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)

            """
            action = agent.predict(batch_obs_full.astype('float32'))

            # Add gripper action again
            action = np.append(action, 0)

            action = np.squeeze(action)

            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])
            """


            action = agent.make_action(batch_obs_full.astype('float32'),epsilon=EXPL_NOISE)

            # Add gripper action again
            #action = np.append(action, 0)

            #action = np.squeeze(action)

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


def test_episode(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    target_pose = np.expand_dims(obs[-3:], axis=0)

    for i in range(EPISODE_LENGTH):
        batch_obs = np.expand_dims(obs[8:14], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)

        action = agent.make_action(batch_obs_full.astype('float32'),epsilon=EXPL_NOISE)
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
    print("total reward = ", total_reward)
    return total_reward


def main(args):

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
                                #headless=True)
                                headless=args.headless)

    env.reset()
    obs_dim = 7 + 3  # 7 joint positions plus 3 target poses

    # drop gripper action to speed up training
    act_dim = env.action_space.shape[0]-1

    max_action = float(env.action_space.high[0])

    """
    model = RLBenchModel(act_dim, max_action)
    algorithm = TD3(model, max_action=max_action,
                    gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR,
                    critic_lr=CRITIC_LR)

    agent = RLBenchAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)
    """
        
    agent = td3Agent(env, is_discrete=False,batch_size=BATCH_SIZE)
    rpm = agent.buffer


    # load model
    if os.path.exists('model_dir/gym_actor_steps_20000.ckpt') and False:
        agent.restore_actor('model_dir/gym_actor_steps_20000.ckpt')
        agent.restore_critic('model_dir/gym_critic_steps_20000.ckpt')
        print('model loaded')

    test_flag = 0
    total_steps = 0
    if args.train:
        logger = CustomLogger('train_log/train_gym.txt')
        while total_steps < MAX_EPISODES:
            train_reward, distance = run_train_episode(env, agent, rpm)
            total_steps += 1
            logger.info('Steps: {}, Distance: {:.4f}, Reward: {}'.format(
                total_steps, distance, train_reward))

            if total_steps // TEST_EVERY_STEPS >= test_flag:
                while total_steps // TEST_EVERY_STEPS >= test_flag:
                    test_flag += 1

                evaluate_reward = evaluate_episode(env, agent)
                logger.info('Steps {}, Evaluate reward: {}'.format(
                    total_steps, evaluate_reward))

                actor_ckpt = 'model_dir/gym_actor_steps_{}.ckpt'.format(
                    total_steps)
                critic_ckpt = 'model_dir/gym_critic_steps_{}.ckpt'.format(
                    total_steps)
                """
                agent.save_actor(actor_ckpt)
                agent.save_critic(critic_ckpt)
                """
                agent.critic.save_network('networks/')

                agent.actor.save_network('networks/')

    if args.test:
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
                        action='store_true', default=True)
    parser.add_argument('--mode', help='render mode name', default='None')
    
    args = parser.parse_args()
    with tf.device('/gpu:0'):
        tf.debugging.set_log_device_placement(True)
        main(args)

