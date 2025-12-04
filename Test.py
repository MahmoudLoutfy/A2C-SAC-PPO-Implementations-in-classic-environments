import argparse
import time
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Import necessary components from your training script
import itertools


# ==========================================
# Discretization Helper (from training code)
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
# Network Architectures (from training code)
# ==========================================

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


class ActorCritic(nn.Module):
    """PPO Actor-Critic"""
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.device = device
        
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
        return action.detach()


# ==========================================
# Agent Wrappers for Testing
# ==========================================

class SACAgent:
    """Wrapper for SAC model during testing"""
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
        self.actor.eval()
    
    def select_action(self, observation, deterministic=True):
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(observation)
            if deterministic:
                action = torch.argmax(probs, dim=1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
        return action.item()


class PPOAgent:
    """Wrapper for PPO model during testing"""
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        self.policy.eval()
    
    def select_action(self, observation, deterministic=True):
        observation = torch.FloatTensor(observation).to(self.device)
        with torch.no_grad():
            if deterministic:
                action_probs = self.policy.actor(observation)
                action = torch.argmax(action_probs)
            else:
                action = self.policy.act(observation)
        return action.item()


# ==========================================
# Utility Functions
# ==========================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ('yes', 'true', 't', 'y', '1'):
        return True
    if s in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env-id', type=str, default='CartPole-v1', help='Environment ID')
    p.add_argument('--algo', type=str, default='sac', choices=['sac', 'ppo'], help='Algorithm type')
    p.add_argument('--model-path', type=str, default=None, 
                   help='Path to model checkpoint; if omitted uses Trained/<algo>_<env-name>.pth')
    p.add_argument('--episodes', type=int, default=100, help='Number of test episodes')
    p.add_argument('--cuda', nargs='?', const=True, type=str2bool, default=False, help='Use CUDA')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--discretize', nargs='?', const=True, type=str2bool, default=True,
                   help='Discretize continuous action spaces')
    p.add_argument('--bins', type=int, default=5, help='Bins per dimension for discretization')
    p.add_argument('--sleep', type=float, default=0.01, help='Sleep between frames for rendering')
    p.add_argument('--save-dir', type=str, default='eval_results', help='Directory to save results')
    p.add_argument('--render', nargs='?', const=True, type=str2bool, default=False, 
                   help='Render environment during testing')
    p.add_argument('--deterministic', nargs='?', const=True, type=str2bool, default=True,
                   help='Use deterministic actions (argmax for discrete)')
    p.add_argument('--show-plots', nargs='?', const=True, type=str2bool, default=False,
                   help='Show plots interactively')
    return p.parse_args()


def load_checkpoint(path: str):
    """Load a torch checkpoint"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ckpt = torch.load(path, map_location='cpu')
    return ckpt


def build_agent_from_checkpoint(env, ckpt, algo, device):
    """Build agent from checkpoint based on algorithm type"""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Building {algo.upper()} agent with obs_dim={obs_dim}, act_dim={act_dim}")
    
    if algo == 'sac':
        # SAC checkpoint structure: {'actor': ..., 'critic': ..., 'critic_target': ..., 'log_alpha': ...}
        actor = DiscreteSACActor(obs_dim, act_dim, hidden_dim=256).to(device)
        actor.load_state_dict(ckpt['actor'])
        agent = SACAgent(actor, device)
        
    elif algo == 'ppo':
        # PPO checkpoint is the policy state dict directly
        policy = ActorCritic(obs_dim, act_dim, device).to(device)
        policy.load_state_dict(ckpt)
        agent = PPOAgent(policy, device)
        
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return agent


def get_env_name_without_version(env_id):
    """Convert 'CartPole-v1' to 'CartPole'"""
    return env_id.rsplit('-', 1)[0]


# ==========================================
# Main Testing Loop
# ==========================================

if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    try:
        if args.render:
            env = gym.make(args.env_id, render_mode='human')
            human_render = True
        else:
            env = gym.make(args.env_id)
            human_render = False
    except TypeError:
        env = gym.make(args.env_id)
        human_render = args.render

    # Discretize continuous action spaces if needed
    if args.discretize and isinstance(env.action_space, gym.spaces.Box):
        print(f"Discretizing continuous action space with {args.bins} bins per dimension")
        env = DiscretizedActionWrapper(env, bins=args.bins)

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        env_name = get_env_name_without_version(args.env_id)
        model_path = os.path.join('Trained', f'{args.algo.upper()}_{env_name}.pth')

    print(f"Loading model from: {model_path}")

    # Load checkpoint
    try:
        ckpt = load_checkpoint(model_path)
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        sys.exit(1)

    # Build agent
    try:
        agent = build_agent_from_checkpoint(env, ckpt, args.algo, device)
    except Exception as e:
        print(f"Error building agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Agent loaded successfully!")

    # Prepare save directory
    os.makedirs(args.save_dir, exist_ok=True)
    safe_env = args.env_id.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_prefix = f"{args.algo}_{safe_env}_{timestamp}"
    rewards_file = os.path.join(args.save_dir, f"rewards_{results_prefix}.csv")
    lengths_file = os.path.join(args.save_dir, f"lengths_{results_prefix}.csv")
    combined_file = os.path.join(args.save_dir, f"results_{results_prefix}.csv")
    plot_file = os.path.join(args.save_dir, f"rewards_plot_{results_prefix}.png")
    lengths_plot_file = os.path.join(args.save_dir, f"lengths_plot_{results_prefix}.png")

    episode_rewards = []
    episode_lengths = []

    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print(f"Deterministic actions: {args.deterministic}")

    for ep in range(args.episodes):
        # Reset environment
        try:
            reset_res = env.reset(seed=(args.seed + ep))
        except TypeError:
            # some envs may not accept seed kwarg in reset; fallback
            reset_res = env.reset()

        # env.reset may return obs or (obs, info). Handle both robustly.
        try:
            obs, info = reset_res  # try unpacking (modern gym returns (obs, info))
        except Exception:
            # reset_res was a single value (obs) — use empty info dict
            obs = reset_res
            info = {}

        # If obs itself is a tuple (rare nested case), make sure obs is the array
        if isinstance(obs, tuple) and len(obs) > 0:
            # prefer the first element as observation; keep original info if present
            obs = obs[0]


        ep_ret = 0.0
        ep_len = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs, deterministic=args.deterministic)

            # Step environment
            step_res = env.step(action)
            if len(step_res) == 5:
                next_obs, reward, terminated, truncated, info = step_res
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step_res

            # Accumulate
            try:
                reward_scalar = float(np.asarray(reward).item())
            except Exception:
                reward_scalar = float(reward)
            
            ep_ret += reward_scalar
            ep_len += 1
            obs = next_obs

            # Sleep for rendering
            if human_render:
                time.sleep(float(args.sleep))

        episode_rewards.append(ep_ret)
        episode_lengths.append(ep_len)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep + 1}/{args.episodes} - Return: {ep_ret:.2f}, Length: {ep_len}")

    env.close()

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"{'='*60}")
    print(f"Mean Return: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Min Return:  {np.min(episode_rewards):.2f}")
    print(f"Max Return:  {np.max(episode_rewards):.2f}")
    print(f"{'='*60}")

    # Save results to CSV
    try:
        # Save individual files
        np.savetxt(rewards_file, np.array(episode_rewards), delimiter=',', 
                   header='return', comments='')
        np.savetxt(lengths_file, np.array(episode_lengths), delimiter=',',
                   header='length', comments='')
        
        # Save combined file
        data = np.vstack([
            np.arange(1, len(episode_rewards) + 1),
            np.array(episode_rewards),
            np.array(episode_lengths)
        ]).T
        np.savetxt(combined_file, data, delimiter=',',
                   header='episode,return,length', comments='')
        
        print(f"\nResults saved to {args.save_dir}/")
        print(f"  - {os.path.basename(rewards_file)}")
        print(f"  - {os.path.basename(lengths_file)}")
        print(f"  - {os.path.basename(combined_file)}")
        
    except Exception as e:
        print(f"Failed to save results to CSV: {e}")

    # Create plots
    try:
        if not human_render:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Rewards plot
        mean_reward_f = float(mean_reward)
        std_reward_f = float(std_reward)

        fig, ax = plt.subplots(figsize=(10, 5))
        episodes = np.arange(1, len(episode_rewards) + 1)
        ax.plot(episodes, episode_rewards, marker='o', linestyle='-', markersize=3, alpha=0.7)
        ax.axhline(y=mean_reward_f, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
        ax.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, 
                        alpha=0.2, color='r', label=f'±1 std: {std_reward:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return')
        ax.set_title(f'{args.algo.upper()} on {args.env_id} - Episode Returns ({args.episodes} episodes)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        print(f"  - {os.path.basename(plot_file)}")
        if args.show_plots and human_render:
            plt.show()
        plt.close()

        # Lengths plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, episode_lengths, marker='o', linestyle='-', markersize=3, alpha=0.7)
        ax.axhline(y=mean_reward_f, color='r', linestyle='--', label=f'Mean: {mean_length:.2f}')
        ax.fill_between(episodes, mean_length - std_length, mean_length + std_length,
                        alpha=0.2, color='r', label=f'±1 std: {std_length:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length (steps)')
        ax.set_title(f'{args.algo.upper()} on {args.env_id} - Episode Lengths ({args.episodes} episodes)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(lengths_plot_file, dpi=150)
        print(f"  - {os.path.basename(lengths_plot_file)}")
        if args.show_plots and human_render:
            plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Failed to create/save plots: {e}")

    print("\nTesting finished successfully!")