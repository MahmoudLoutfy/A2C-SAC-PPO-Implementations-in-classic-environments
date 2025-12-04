import argparse
import math
import random
import time
import os
import itertools
from abc import ABC, abstractmethod
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ==========================================
# Prioritized Replay Buffer (NEW)
# ==========================================

class SumTree:
    """Efficient sum tree for prioritized sampling"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, obs_dim, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full priority)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to prevent zero priority
        
    def _get_beta(self):
        """Beta anneals from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, obs, action, reward, next_obs, done, n_steps):
        # New experiences get maximum priority to ensure they're sampled
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        self.tree.add(max_priority, (obs, action, reward, next_obs, done, n_steps))
    
    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.frame += 1
        beta = self._get_beta()
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()  # Normalize
        
        # Unpack batch
        obs_batch = np.array([x[0] for x in batch])
        act_batch = np.array([x[1] for x in batch]).reshape(-1, 1)
        rew_batch = np.array([x[2] for x in batch]).reshape(-1, 1)
        next_obs_batch = np.array([x[3] for x in batch])
        done_batch = np.array([x[4] for x in batch]).reshape(-1, 1)
        n_steps_batch = np.array([x[5] for x in batch]).reshape(-1, 1)
        
        return (
            torch.FloatTensor(obs_batch).to(self.device),
            torch.LongTensor(act_batch).to(self.device),
            torch.FloatTensor(rew_batch).to(self.device),
            torch.FloatTensor(next_obs_batch).to(self.device),
            torch.FloatTensor(done_batch).to(self.device),
            torch.FloatTensor(n_steps_batch).to(self.device),
            torch.FloatTensor(weights).unsqueeze(1).to(self.device),
            idxs
        )
    
    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(idxs, td_errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# ==========================================
# Discretization Helper
# ==========================================

class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=5):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Wrapper only for Box spaces"
        
        self.bins = bins
        self.orig_shape = env.action_space.shape
        self.dims = self.orig_shape[0]
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.grids = [np.linspace(l, h, bins) for l, h in zip(self.low, self.high)]
        self.n_actions = bins ** self.dims
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_map = list(itertools.product(*self.grids))

    def action(self, action):
        return np.array(self.action_map[action], dtype=np.float32)


# ==========================================
# N-Step Buffer
# ==========================================

class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        if len(self.buffer) == self.n_step:
            return self._get_n_step_info()
        return None

    def finish_episode(self):
        transitions = []
        while len(self.buffer) > 0:
            transitions.append(self._get_n_step_info())
            self.buffer.popleft()
        return transitions

    def _get_n_step_info(self):
        curr_obs, curr_act, _, _, _ = self.buffer[0]
        n_step_reward = 0
        n_step_gamma = 1
        
        for i in range(len(self.buffer)):
            r = self.buffer[i][2]
            n_step_reward += r * n_step_gamma
            n_step_gamma *= self.gamma
        
        final_next_obs = self.buffer[-1][3]
        final_done = self.buffer[-1][4]
        actual_steps = len(self.buffer)
        
        return (curr_obs, curr_act, n_step_reward, final_next_obs, final_done, actual_steps)
    
# ==========================================
# 2. Abstract Base Class for RL Algorithms
# ==========================================

class RLAlgorithm(ABC):
    def _init_(self, observation_space, action_space, device, args):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.args = args

    @abstractmethod
    def select_action(self, observation, evaluate=False):
        pass

    @abstractmethod
    def push_transition(self, obs, action, reward, next_obs, done, n_steps):
        pass

    @abstractmethod
    def train(self, batch_size):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


# ==========================================
# Improved Discrete SAC
# ==========================================

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DiscreteSACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample(self, obs):
        probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)
        return action, probs, log_probs


class DiscreteSACCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        q1 = self.q1_net(obs)
        q2 = self.q2_net(obs)
        return q1, q2


class DiscreteSAC(RLAlgorithm):
    def __init__(self, observation_space, action_space, device, args):
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        self.device = device
        self.args = args
        
        self.n_step = args.n_step
        self.gamma = args.gamma
        self.tau = args.tau
        
        # CHANGE 1: Reduced target entropy for less exploration once successful
        self.target_entropy = -0.5 * np.log(1 / self.act_dim)  # Was -0.98
        
        # CHANGE 2: Lower initial alpha
        init_alpha = args.init_alpha  # Will set to 0.5 in args
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
        self.alpha = init_alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr * 0.1)
        
        hidden_dim = 256
        
        self.actor = DiscreteSACActor(self.obs_dim, self.act_dim, hidden_dim).to(device)
        self.critic = DiscreteSACCritic(self.obs_dim, self.act_dim, hidden_dim).to(device)
        self.critic_target = DiscreteSACCritic(self.obs_dim, self.act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.critic_target.apply(init_weights)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # CHANGE 3: Use Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(args.buffer_size, observation_space.shape, device)
        
        self.train_steps = 0
        self.actor_update_freq = 1
        
        print(f"Improved Discrete SAC initialized:")
        print(f"  Using Prioritized Experience Replay")
        print(f"  Target entropy: {self.target_entropy:.3f} (reduced for exploitation)")
        print(f"  Initial alpha: {init_alpha}")

    def select_action(self, observation, evaluate=False):
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                probs = self.actor(observation)
                action = torch.argmax(probs, dim=1)
            else:
                action, _, _ = self.actor.sample(observation)
        return action.item()

    def push_transition(self, obs, action, reward, next_obs, done, n_steps):
        self.buffer.add(obs, action, reward, next_obs, done, n_steps)

    def train(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        self.train_steps += 1
        state, action, reward, next_state, done, n_steps, weights, idxs = self.buffer.sample(batch_size)

        # No reward normalization - keep raw rewards
        
        # Critic Update with TD error tracking for PER
        with torch.no_grad():
            next_probs = self.actor(next_state)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            target_q1, target_q2 = self.critic_target(next_state)
            target_q = torch.min(target_q1, target_q2)
            
            soft_target_q = (next_probs * (target_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            discount = torch.pow(self.gamma, n_steps)
            q_target = reward + (1 - done) * discount * soft_target_q

        current_q1, current_q2 = self.critic(state)
        current_q1_a = current_q1.gather(1, action)
        current_q2_a = current_q2.gather(1, action)

        # CHANGE 4: Weighted loss for PER
        td_error1 = (current_q1_a - q_target).abs()
        td_error2 = (current_q2_a - q_target).abs()
        
        q_loss = (weights * F.smooth_l1_loss(current_q1_a, q_target, reduction='none')).mean() + \
                 (weights * F.smooth_l1_loss(current_q2_a, q_target, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # CHANGE 5: Update priorities in PER buffer
        td_errors = ((td_error1 + td_error2) / 2).detach().cpu().numpy().flatten()
        self.buffer.update_priorities(idxs, td_errors)

        # Actor Update
        actor_loss = torch.tensor(0.0)
        if self.train_steps % self.actor_update_freq == 0:
            for param in self.critic.parameters():
                param.requires_grad = False
            
            probs = self.actor(state)
            log_probs = torch.log(probs + 1e-8)
            
            with torch.no_grad():
                q1, q2 = self.critic(state)
                q = torch.min(q1, q2)

            actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            for param in self.critic.parameters():
                param.requires_grad = True

        # Alpha Update
        alpha_loss = torch.tensor(0.0)
        if self.train_steps % self.actor_update_freq == 0:
            with torch.no_grad():
                probs = self.actor(state)
                log_probs = torch.log(probs + 1e-8)
                expected_log_pi = (probs * log_probs).sum(dim=1, keepdim=True)
            
            alpha_loss = -(self.log_alpha * (expected_log_pi + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            # CHANGE 6: Much lower alpha maximum (was 2.0, now 0.2)
            self.alpha = np.clip(self.alpha, 0.01, 0.2)
            self.log_alpha.data = torch.tensor(np.log(self.alpha), device=self.device)

        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": q_loss.item(),
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "alpha": self.alpha,
            "q_mean": current_q1_a.mean().item(),
            "avg_td_error": td_errors.mean(),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()

# ==========================================
# PPO Implementation - IMPROVED
# ==========================================

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.device = device
        
        # Deeper network for discrete actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
            
        # Deeper critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO(RLAlgorithm):
    def __init__(self, observation_space, action_space, device, args):
        super()._init_(observation_space, action_space, device, args)
        
        
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n

        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.gamma = args.gamma
        self.K_epochs = args.k_epochs
        self.eps_clip = args.eps_clip
        
        self.ent_coef = args.ent_coef
        
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.obs_dim, self.act_dim, device).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.policy_old = ActorCritic(self.obs_dim, self.act_dim, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.temp_transition = None

    def select_action(self, observation, evaluate=False):
        state = torch.FloatTensor(observation).to(self.device)
        if evaluate:
            with torch.no_grad():
                action_probs = self.policy_old.actor(state)
                action = torch.argmax(action_probs).item()
                return action
        else:
            action, action_logprob, state_val = self.policy_old.act(state)
            
            self.temp_transition = {
                'logprob': action_logprob,
                'state_val': state_val
            }

            return action.item()
            
    def push_transition(self, obs, action, reward, next_obs, done, n_steps): 
        if self.temp_transition is None:
            return

        self.buffer.states.append(torch.FloatTensor(obs).to(self.device))
        self.buffer.actions.append(torch.tensor(action).to(self.device))
        self.buffer.logprobs.append(self.temp_transition['logprob'])
        self.buffer.state_values.append(self.temp_transition['state_val'])
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        
        self.temp_transition = None

    def compute_gae(self, rewards, values, dones):  
        advantages = []
        gae = 0
        
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def train(self, batch_size):
        if len(self.buffer.states) < batch_size:
            return None

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        old_state_values = torch.stack(self.buffer.state_values, dim=0).squeeze().detach()
        
        rewards = self.buffer.rewards
        dones = self.buffer.is_terminals
        
        advantages = self.compute_gae(rewards, old_state_values.cpu().numpy(), dones)
        returns = advantages + old_state_values
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_actor_loss = 0
        total_value_loss = 0
        batch_size_actual = len(old_states)
        mini_batch_size = 64
        
        for _ in range(self.K_epochs):
            indices = np.arange(batch_size_actual)
            np.random.shuffle(indices)
            
            for start_idx in range(0, batch_size_actual, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size_actual)
                mb_indices = indices[start_idx:end_idx]
                
                mb_states = old_states[mb_indices]
                mb_actions = old_actions[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_state_values[mb_indices]
                
                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                state_values = state_values.squeeze()
                
                ratios = torch.exp(logprobs - mb_old_logprobs)
                
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_pred_clipped = mb_old_values + torch.clamp(
                    state_values - mb_old_values,
                    -self.eps_clip,
                    self.eps_clip
                )
                value_loss_unclipped = self.MseLoss(state_values, mb_returns)
                value_loss_clipped = self.MseLoss(value_pred_clipped, mb_returns)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
                
                loss = actor_loss + 0.5 * value_loss - self.ent_coef * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
        n_updates = self.K_epochs * (batch_size_actual // mini_batch_size + 1)
        return {
            "ppo_loss": total_loss / n_updates,
            "actor_loss": total_actor_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "value_mean": old_state_values.mean().item(),
            "advantage_mean": advantages.mean().item()
        }

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy_old.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


# ==========================================
# Main Training Loop
# ==========================================

def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"): return True
    if s in ("no", "false", "f", "n", "0"): return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="ppo", help="Algorithm: sac, ppo")
    p.add_argument("--env-id", type=str, default="MountainCar-v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", nargs="?", const=True, type=str2bool, default=True)
    p.add_argument("--total-timesteps", type=int, default=1500000)  # Can solve faster now
    p.add_argument("--track", nargs="?", const=True, type=str2bool, default=False)
    p.add_argument("--wandb-project", type=str, default="rl-mountaincar")
    
    p.add_argument("--discretize", nargs="?", const=True, type=str2bool, default=True)
    p.add_argument("--bins", type=int, default=5)
    
    # TUNED HYPERPARAMETERS FOR MOUNTAINCAR
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=2048)  # Smaller for sparse rewards
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--tau", type=float, default=0.005)

    # PPO Specific
    p.add_argument("--k-epochs", type=int, default=10, help="PPO Update Epochs")
    p.add_argument("--eps-clip", type=float, default=0.2, help="PPO Clip ratio")
    p.add_argument("--lr-actor", type=float, default=0.0003, help="PPO Actor LR")
    p.add_argument("--lr-critic", type=float, default=0.0003, help="PPO Critic LR")
    p.add_argument("--ent-coef", type=float, default=0, help="Entropy coefficient for PPO")

    # SAC Specific
    p.add_argument("--learning-starts", type=int, default=0)  # More exploration
    p.add_argument("--init-alpha", type=float, default=0.2)  # Lower starting alpha
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--grad-steps", type=int, default=1)  # Multiple updates per step
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if isinstance(env.action_space, gym.spaces.Box) and args.discretize:
        print(f"Discretizing with {args.bins} bins per dimension.")
        env = DiscretizedActionWrapper(env, bins=args.bins)

    if args.algo == "sac":
        agent = DiscreteSAC(env.observation_space, env.action_space, device, args)
    elif args.algo == "ppo":
        agent = PPO(env.observation_space, env.action_space, device, args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    if args.track and _WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.algo}_{args.env_id}")

    print(f"Starting training on {args.env_id}...")

    obs, info = env.reset(seed=args.seed)
    start_time = time.time()
    n_step_collector = NStepBuffer(n_step=args.n_step, gamma=args.gamma)

    for global_step in range(1, args.total_timesteps + 1):
        
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        transition = n_step_collector.add(obs, action, reward, next_obs, done)
        if transition:
            n_obs, n_act, n_rew, n_next, n_done, n_steps = transition
            agent.push_transition(n_obs, n_act, n_rew, n_next, n_done, n_steps)
            
        if done:
            remaining_transitions = n_step_collector.finish_episode()
            for t in remaining_transitions:
                n_obs, n_act, n_rew, n_next, n_done, n_steps = t
                agent.push_transition(n_obs, n_act, n_rew, n_next, n_done, n_steps)
                
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                if isinstance(ep_r, np.ndarray): ep_r = ep_r.item()
                if isinstance(ep_l, np.ndarray): ep_l = ep_l.item()
                
                #print(f"Step: {global_step}, Reward: {ep_r:.2f}, Length: {ep_l}, Alpha: {agent.alpha:.3f}")
                if args.track and _WANDB_AVAILABLE:
                    wandb.log({"episodic_return": ep_r, "episodic_length": ep_l}, step=global_step)
            
            obs, info = env.reset()
        else:
            obs = next_obs

        # CHANGE 7: Train multiple times per step after learning starts
        if global_step >= args.learning_starts:
            for _ in range(args.grad_steps):
                metrics = agent.train(args.batch_size)
                if metrics and global_step % 1000 == 0:
                    if args.track and _WANDB_AVAILABLE:
                        wandb.log(metrics, step=global_step)

    agent.save(f"PPO_MountainCar.pth")
    env.close()
    if args.track and _WANDB_AVAILABLE: 
        wandb.finish()