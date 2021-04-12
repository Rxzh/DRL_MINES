from networks.policy import Policy
from networks.q_fun import QFunction

def initialize_td3(env, device, obs_dim, act_dim,max_action, min_action):
    net_params = {
        'q_lr': 0.005,
        'policy_lr': 0.005,
        'q_hidden': [64, 64],
        'policy_hidden': [64, 64], #[400,300] ? comme le chinois
    }
    alg_params = {
        'batch_size': 32,
        'learning_starts': 1000,
        'learning_freq': 1,
        'discount_factor': 1.00,
        'sync_every': 3000,
        'tau': None,
        'exploration_noise': 0.02,
        'grad_norm_clip_val': None,
        'target_noise': 0.2,
        'target_noise_clip': 0.5,
        'policy_delay': 2,
        'max_path_frames': env.max_episode_steps - 1,
        'replay_buffer_size': 50000,
    }
    params = {**net_params, **alg_params}



    policy = Policy(obs_dim, act_dim, 'deterministic', max_action, min_action,
        lr=net_params['policy_lr'], hidden=net_params['policy_hidden'], target=True)

    q_fun = QFunction(obs_dim, act_dim, discrete=False, double=True,
        lr=net_params['q_lr'], hidden=net_params['q_hidden'], target=True)
    agent = TD3(env, device, policy, q_fun, max_action, min_action,act_dim **alg_params)
    return agent, params


import numpy as np
import os
import torch
import torch.nn.functional as F
from agents.common_off_policy import OffPolicyAgent

class TD3(OffPolicyAgent):
    def __init__(
            self,
            env,
            device,
            policy,
            q_fun,
            max_action,
            min_action,
            act_dim,
            batch_size=64,
            learning_starts=1000,
            learning_freq=4,
            replay_buffer_size=50000,
            max_path_frames=np.inf,
            discount_factor=0.99,
            exploration_noise=0.1,
            sync_every=5,
            tau=None,
            grad_norm_clip_val=None,
            target_noise=0.2,
            target_noise_clip=0.5,
            policy_delay=2,
    ):
        super(TD3, self).__init__(
            env,
            device,
            batch_size,
            learning_starts,
            learning_freq,
            replay_buffer_size,
            max_path_frames,
        )
        self.gamma = discount_factor
        self.grad_norm_clip_val = grad_norm_clip_val
        self.exploration_noise = exploration_noise

        # policy network
        self.policy = policy
        self.policy.to_(self.device)
        self.policy.sync_target()

        # q function network
        self.q_fun = q_fun
        self.q_fun.to_(self.device)
        self.q_fun.sync_target()

        # utilities for policy and target updates (sync every x iterations)
        self._num_updates = 0
        self.tau = tau
        self.policy_delay = policy_delay
        self.sync_every = sync_every if tau is None else policy_delay

        # utilities for action clipping
        self.max_action = max_action
        self.min_action = min_action
        self.act_dim = act_dim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self._max_ac = torch.tensor(self.max_action).to(device)
        self._min_ac = torch.tensor(self.min_action).to(device)



    #AJOUTE DEPUIS GymENV Que l'on utilisera pas   
    def process_state(self, ob):
        # to tensor of shape [1 , observation_dim]
        return torch.FloatTensor(ob).view(1, -1).to(self.device)

    def process_action(self, ac):
        # from tensor of shape [1]              if env is discrete
        # from tensor of shape [1, action_dim]  if env is continuous
        if self.is_discrete:
            return ac.item()
        else:
            return np.squeeze(ac.cpu().detach().numpy(), 0)
    ##################################

    @torch.no_grad()
    def act(self, ob, eval=False):
        if not eval and self._frame < self.learning_starts:
            ac = self.env.action_space.sample()
        else:
            ac = self._exploit(ob, eval)
            if not eval and self.exploration_noise != 0:
                ac = ac + np.random.normal(0, self.exploration_noise, size=self.act_dim)
                ac = ac.clip(self.min_action, self.max_action)
        return ac

    def _exploit(self, ob, eval):
        ob = self.process_state(ob)
        ac = self.policy.forward(ob, eval=eval)
        return self.process_action(ac)

    def update(self):
        self._num_updates += 1

        # sample transitions from buffer
        data = self.replay_buffer.sample(self.batch_size)
        states = data['states']					# shape (N, ob_dim)
        next_states = data['next_states']		# shape (N, ob_dim)
        actions = data['actions']	            # shape (N, ac_dim)
        rewards = data['rewards']				# shape (N, 1)
        done_mask = data['done_mask']  			# shape (N, 1)

        with torch.no_grad():
            # compute next actions
            next_actions_tmp = self.policy.target_net(next_states)
            noise = torch.randn_like(next_actions_tmp) * self.target_noise
            noise = noise.clamp(- self.target_noise_clip, self.target_noise_clip)
            # next_actions = (next_actions_tmp + noise).clamp(self.env.action_space.low[0], self.env.action_space.high[0])
            next_actions  = torch.max(torch.min(next_actions_tmp + noise, self._max_ac), self._min_ac)

            # compute q targets
            q1_targets_next, q2_targets_next = self.q_fun.target_net(next_states, next_actions)
            q_targets_next = torch.min(q1_targets_next, q2_targets_next)
            q_targets = rewards + (1 - done_mask) * self.gamma * q_targets_next

        # compute q values
        q1_values, q2_values = self.q_fun.net(states, actions)    # shape (N, 1)

        # update the critic
        critic_loss = F.mse_loss(q1_values, q_targets) + F.mse_loss(q2_values, q_targets)
        self.q_fun.optimize(critic_loss, self.grad_norm_clip_val)
  
        # periodically update the actor
        if self._num_updates % self.policy_delay == 0:
            self._update_actor(states)

        # periodically update the target networks
        if self._num_updates % self.sync_every == 0:
            self.q_fun.sync_target(self.tau)
            self.policy.sync_target(self.tau)

    def _update_actor(self, states):
        # temporally freeze q-networks 
        for p in self.q_fun.net.parameters():
            p.requires_grad = False

        # compute loss and update network
        greedy_actions = self.policy.net(states)
        actor_loss = - self.q_fun.net(states, greedy_actions, q1=True, q2=False).mean()  
        self.policy.optimize(actor_loss, self.grad_norm_clip_val)

        # unfreeze q-networks
        for p in self.q_fun.net.parameters():
            p.requires_grad = True

    def save(self, dir, itr):
        self.policy.save(os.path.join(dir, str(itr) + '_policy'))
        self.q_fun.save(os.path.join(dir, str(itr) + '_qfun'))

