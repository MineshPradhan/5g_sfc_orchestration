import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SFCOrchestrationEnv(gym.Env):
    """
    Realistic 5G SFC Orchestration Environment.
    Simulates traffic fluctuations across eMBB, URLLC, and mMTC slices.
    """
    def __init__(self, num_servers=20, max_sfcs=30, num_slices=3):
        super().__init__()
        self.num_servers = num_servers
        self.max_sfcs = max_sfcs
        self.num_slices = num_slices
        
        # Actions: 0=NOP, 1=PROACTIVE_DEPLOY, 2=SCALE_UP, 3=SCALE_DOWN, 4=LOAD_BALANCE_MIGRATE
        self.action_space = spaces.MultiDiscrete([5] * num_slices)
        obs_dim = num_servers * 3 + num_slices * 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.step_cnt = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        # Initialize normal server loads (30% - 50% baseline utilization)
        self.server_cpu = np.random.uniform(0.30, 0.50, self.num_servers)
        self.server_mem = np.random.uniform(0.30, 0.50, self.num_servers)
        self.server_bw  = np.random.uniform(0.30, 0.50, self.num_servers)
        return self._get_obs(), {}
        
    def _get_obs(self):
        srv = np.stack([self.server_cpu, self.server_mem, self.server_bw], axis=-1).flatten()
        # Simulated predicted traffic demands (mean, std) for 3 slices
        dem = np.random.uniform(0.2, 0.9, self.num_slices * 2).astype(np.float32)
        return np.concatenate([srv, dem]).astype(np.float32)
        
    def step(self, action):
        self.step_cnt += 1
        
        # 1. Background dynamic traffic arrival (pushes random servers towards overload)
        traffic_shock = np.random.exponential(scale=0.08, size=self.num_servers)
        # Random surge on 2-3 hot servers
        hot_nodes = np.random.choice(self.num_servers, size=3, replace=False)
        traffic_shock[hot_nodes] += np.random.uniform(0.15, 0.35, size=3)
        self.server_cpu += traffic_shock
        
        # 2. Apply Agent Actions
        for sl, act in enumerate(action):
            if act == 1: # PROACTIVE DEPLOY / SPREAD: spreads demand to least loaded servers
                min_idx = np.argsort(self.server_cpu)[:3]
                self.server_cpu[min_idx] += 0.05
            elif act == 2: # SCALE UP (Add VNF replicas to absorb high loads, reducing per-server stress)
                max_idx = np.argsort(self.server_cpu)[-3:]
                self.server_cpu[max_idx] -= 0.12 # Offloads bottleneck servers
            elif act == 3: # SCALE DOWN (Consolidate / energy save)
                min_idx = np.argsort(self.server_cpu)[:2]
                self.server_cpu[min_idx] -= 0.05
            elif act == 4: # SMART MIGRATION: balance busiest server with emptiest server
                h_idx = np.argmax(self.server_cpu)
                l_idx = np.argmin(self.server_cpu)
                shift = (self.server_cpu[h_idx] - self.server_cpu[l_idx]) * 0.4
                self.server_cpu[h_idx] -= shift
                self.server_cpu[l_idx] += shift

        # Natural traffic decay / service completions
        self.server_cpu = np.clip(self.server_cpu * 0.90, 0.05, 1.0)
        self.server_mem = np.clip(self.server_mem * 0.92, 0.05, 1.0)
        self.server_bw  = np.clip(self.server_bw * 0.92, 0.05, 1.0)
        
        # 3. Multi-Objective Reward Calculation
        # - SLA violation: CPU > 0.80 causes queuing delays and packet loss
        overload = np.maximum(0.0, self.server_cpu - 0.80)
        sla_penalty = np.sum(overload) * 15.0  # Heavy penalty for SLA breach
        
        # - Resource balance reward (lower variance is better)
        variance_penalty = np.var(self.server_cpu) * 5.0
        
        # - Energy cost: idle power + active power
        energy_penalty = np.mean(self.server_cpu) * 0.5
        
        reward = 2.0 - sla_penalty - variance_penalty - energy_penalty
        
        done = self.step_cnt >= 200
        return self._get_obs(), float(reward), done, False, {}