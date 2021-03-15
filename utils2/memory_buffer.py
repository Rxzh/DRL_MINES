import random
import numpy as np

from collections import deque
from .sumtree import SumTree

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size,act_dim = 0,obs_dim = 0, with_per = False):
        """ Initialization
        """
        self.buffer_size = int(buffer_size)

        self.buffer = list()

        self.obs = np.zeros((buffer_size, obs_dim), dtype='float32')
        self.action = np.zeros((buffer_size, act_dim), dtype='float32')
        self.reward = np.zeros((buffer_size, ), dtype='float32')
        self.terminal = np.zeros((buffer_size, ), dtype='bool')
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype='float32')
        
        self._curr_size = 0
        self._curr_pos = 0



        """
        if(with_per):
            # Prioritized Experience Replay(propositional)
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self._curr_size = 0
        self.with_per = with_per
        self.buffer_size = buffer_size
        """

    def memorize(self, state, action, reward, done, new_state, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, action, reward, done, new_state)
                    # Check if buffer is already full

        if self._curr_size < self.buffer_size:
            self.buffer.append(experience)
            self._curr_size += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

        

        self.with_per = False
        """
        if self._curr_size < self.buffer_size:
            self._curr_size += 1

        self.obs[self._curr_pos] = state
        self.action[self._curr_pos] = action
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = new_state
        self.terminal[self._curr_pos] = done
        self._curr_pos = (self._curr_pos + 1) % self.buffer_size
        """

        """
        if(self.with_per):
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self._curr_size += 1
        else:
            # Check if buffer is already full
            if self._curr_size < self.buffer_size:
                self.buffer.append(experience)
                self._curr_size += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)
        """

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self._curr_size

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() / batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch])
        # Sample randomly from Buffer
        
        elif self._curr_size < batch_size:
            idx = None
            batch = random.sample(self.buffer, self._curr_size)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self._curr_size = 0
