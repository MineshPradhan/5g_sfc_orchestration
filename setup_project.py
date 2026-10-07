import os
from pathlib import Path

ROOT = Path(r"C:\Users\Minesh\.gemini\antigravity\scratch\5g_sfc_orchestration")

files = {}

# 1. requirements.txt
files["requirements.txt"] = """torch>=2.1.0
torch-geometric>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
gymnasium>=0.29.0
stable-baselines3>=2.1.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
requests>=2.31.0
"""

# 2. config.yaml
files["config.yaml"] = """data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  num_grid_rows: 10
  num_grid_cols: 10
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

prediction:
  input_window: 12
  output_window: 6
  d_model: 64
  num_heads: 4
  num_layers: 2
  dropout: 0.1
  learning_rate: 0.0003
  batch_size: 16
  epochs: 20

orchestration:
  num_servers: 20
  max_sfcs: 30
  total_timesteps: 50000
"""

# 3. Package inits
files["data/__init__.py"] = ""
files["models/__init__.py"] = ""
files["models/sagatx/__init__.py"] = ""
files["models/baselines/__init__.py"] = ""
files["models/hmarl/__init__.py"] = ""
files["evaluation/__init__.py"] = ""
files["utils/__init__.py"] = ""

# 4. Data Generator / Downloader
files["data/download_data.py"] = '''import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def generate_synthetic_data(raw_dir: Path, num_days: int = 7, num_cells: int = 100):
    print(f"[INFO] Generating realistic synthetic 5G telemetry ({num_days} days, {num_cells} cells)...")
    np.random.seed(42)
    intervals_per_day = 144
    records = []
    
    grid_side = int(np.sqrt(num_cells))
    grid_x, grid_y = np.meshgrid(np.arange(grid_side), np.arange(grid_side))
    dist = np.sqrt((grid_x - grid_side//2)**2 + (grid_y - grid_side//2)**2)
    spatial_weight = np.exp(-dist / (grid_side * 0.3)).flatten()[:num_cells]
    
    for day in range(num_days):
        wknd = 0.7 if (day % 7) >= 5 else 1.0
        for interval in range(intervals_per_day):
            hr = (interval * 10) / 60.0
            diurnal = 0.3 + 0.7 * (0.5 * np.exp(-((hr-12)**2)/8) + 0.5 * np.exp(-((hr-20)**2)/6))
            base = spatial_weight * diurnal * wknd
            
            internet = np.maximum(0, base * 50 + np.random.normal(0, 3, num_cells))
            calls = np.maximum(0, base * 15 + np.random.normal(0, 1.5, num_cells))
            sms = np.maximum(0, base * 10 + np.random.normal(0, 1, num_cells))
            
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, minutes=interval*10)
            for c in range(num_cells):
                records.append({
                    "cell_id": c + 1, "timestamp": ts,
                    "internet": round(internet[c], 3), # eMBB
                    "call_in": round(calls[c], 3),     # URLLC proxy
                    "call_out": round(calls[c]*0.8, 3),
                    "sms_in": round(sms[c], 3),        # mMTC proxy
                    "sms_out": round(sms[c]*0.9, 3)
                })
    df = pd.DataFrame(records)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "telecom_italia_synthetic.csv", index=False)
    print(f"[OK] Saved synthetic dataset to {raw_dir / 'telecom_italia_synthetic.csv'}")

if __name__ == "__main__":
    generate_synthetic_data(Path("data/raw"), num_days=7, num_cells=100)
'''

# 5. Preprocessing pipeline
files["data/preprocess.py"] = '''import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    raw_path = Path("data/raw/telecom_italia_synthetic.csv")
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    
    num_cells = cfg["data"]["num_grid_rows"] * cfg["data"]["num_grid_cols"]
    timestamps = sorted(df["timestamp"].unique())
    T = len(timestamps)
    
    embb = np.zeros((T, num_cells), dtype=np.float32)
    urllc = np.zeros((T, num_cells), dtype=np.float32)
    mmtc = np.zeros((T, num_cells), dtype=np.float32)
    
    for t_idx, ts in enumerate(timestamps):
        sub = df[df["timestamp"] == ts]
        cells = sub["cell_id"].values - 1
        valid = (cells >= 0) & (cells < num_cells)
        c = cells[valid]
        embb[t_idx, c] = sub["internet"].values[valid]
        urllc[t_idx, c] = (sub["call_in"].values[valid] + sub["call_out"].values[valid]) / 2.0
        mmtc[t_idx, c] = (sub["sms_in"].values[valid] + sub["sms_out"].values[valid]) / 2.0
        
    stacked = np.stack([embb, urllc, mmtc], axis=-1) # (T, N, 3)
    
    # Normalize
    train_end = int(T * cfg["data"]["train_ratio"])
    scaler = StandardScaler()
    scaler.fit(stacked[:train_end].reshape(-1, 3))
    norm_stacked = scaler.transform(stacked.reshape(-1, 3)).reshape(stacked.shape).astype(np.float32)
    
    # Windows
    iw = cfg["prediction"]["input_window"]
    ow = cfg["prediction"]["output_window"]
    num_samples = T - iw - ow + 1
    
    X = np.zeros((num_samples, iw, num_cells, 3), dtype=np.float32)
    Y = np.zeros((num_samples, ow, num_cells, 3), dtype=np.float32)
    for i in range(num_samples):
        X[i] = norm_stacked[i:i+iw]
        Y[i] = norm_stacked[i+iw:i+iw+ow]
        
    # Split
    t1 = int(num_samples * cfg["data"]["train_ratio"])
    t2 = int(num_samples * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"]))
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({"X": torch.FloatTensor(X[:t1]), "Y": torch.FloatTensor(Y[:t1])}, out_dir / "train_data.pt")
    torch.save({"X": torch.FloatTensor(X[t1:t2]), "Y": torch.FloatTensor(Y[t1:t2])}, out_dir / "val_data.pt")
    torch.save({"X": torch.FloatTensor(X[t2:]), "Y": torch.FloatTensor(Y[t2:])}, out_dir / "test_data.pt")
    
    # Grid Adjacency
    R, C = cfg["data"]["num_grid_rows"], cfg["data"]["num_grid_cols"]
    rows, cols = [], []
    for r in range(R):
        for c in range(C):
            idx = r * C + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        rows.append(idx); cols.append(nr * C + nc)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_cells, num_cells))
    ei, ew = from_scipy_sparse_matrix(adj)
    torch.save({"edge_index": ei, "edge_weight": ew, "num_nodes": num_cells}, out_dir / "graph_data.pt")
    print(f"[OK] Data processed successfully into {out_dir}")

if __name__ == "__main__":
    run_preprocessing()
'''

# 6. SAGATx Architecture
files["models/sagatx/model.py"] = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSliceAttention(nn.Module):
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): # (B, T, N, S, D)
        B, T, N, S, D = x.shape
        flat = x.reshape(B*T*N, S, D)
        out, _ = self.attn(flat, flat, flat)
        return (x + out.reshape(B, T, N, S, D))

class SAGATx(nn.Module):
    def __init__(self, num_nodes, num_slices=3, input_window=12, output_window=6,
                 d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.output_window = output_window
        self.d_model = d_model
        self.slice_convs = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=3, padding=1) for _ in range(num_slices)
        ])
        self.cross_slice = CrossSliceAttention(d_model, heads=num_heads)
        self.temp_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.out_head = nn.Linear(d_model, output_window * 2) # [mean, log_var]
        
    def forward(self, x, edge_index=None): # x: (B, T, N, S)
        B, T, N, S = x.shape
        encoded = []
        for s in range(S):
            xs = x[:, :, :, s].permute(0, 2, 1).reshape(B*N, 1, T)
            h = F.gelu(self.slice_convs[s](xs)).reshape(B, N, self.d_model, T).permute(0, 3, 1, 2)
            encoded.append(h)
        h = torch.stack(encoded, dim=3) # (B, T, N, S, D)
        h = self.cross_slice(h)
        
        # Temporal attention over last step
        flat_temp = h.permute(0, 2, 3, 1, 4).reshape(B*N*S, T, self.d_model)
        attn_out, _ = self.temp_attn(flat_temp, flat_temp, flat_temp)
        last = attn_out[:, -1, :].reshape(B, N, S, self.d_model)
        
        out = self.out_head(last).reshape(B, N, S, self.output_window, 2)
        out = out.permute(0, 3, 1, 2, 4) # (B, T_out, N, S, 2)
        return out
        
    def loss(self, pred, target):
        mu, log_var = pred[..., 0], pred[..., 1]
        nll = 0.5 * (log_var + (target - mu)**2 / (torch.exp(log_var) + 1e-6))
        return nll.mean()
'''

# 7. SFC Gym Environment
files["models/hmarl/sfc_env.py"] = '''import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SFCOrchestrationEnv(gym.Env):
    def __init__(self, num_servers=20, max_sfcs=30, num_slices=3):
        super().__init__()
        self.num_servers = num_servers
        self.max_sfcs = max_sfcs
        self.num_slices = num_slices
        self.action_space = spaces.MultiDiscrete([6] * num_slices) # 0:NOP,1:DEPLOY,2:SCALE+,3:SCALE-,4:MIGRATE,5:REROUTE
        obs_dim = num_servers * 3 + max_sfcs * 4 + num_slices * 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.step_cnt = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        self.server_cpu = np.random.uniform(0.1, 0.4, self.num_servers)
        self.server_mem = np.random.uniform(0.1, 0.4, self.num_servers)
        self.server_bw  = np.random.uniform(0.1, 0.4, self.num_servers)
        return self._get_obs(), {}
        
    def _get_obs(self):
        srv = np.stack([self.server_cpu, self.server_mem, self.server_bw], axis=-1).flatten()
        sfc = np.zeros(self.max_sfcs * 4, dtype=np.float32)
        dem = np.random.uniform(20, 80, self.num_slices * 2).astype(np.float32)
        return np.concatenate([srv, sfc, dem]).astype(np.float32)
        
    def step(self, action):
        self.step_cnt += 1
        # Apply actions
        for sl, act in enumerate(action):
            if act == 1: # DEPLOY
                idx = np.argmin(self.server_cpu)
                self.server_cpu[idx] = min(1.0, self.server_cpu[idx] + 0.1)
            elif act == 2: # SCALE UP
                self.server_cpu = np.clip(self.server_cpu + 0.02, 0, 1.0)
            elif act == 4: # MIGRATE
                mx = np.argmax(self.server_cpu)
                mn = np.argmin(self.server_cpu)
                diff = (self.server_cpu[mx] - self.server_cpu[mn]) * 0.5
                self.server_cpu[mx] -= diff
                self.server_cpu[mn] += diff
                
        # Objective rewards: reward balancing, penalize overloading
        load_var = np.var(self.server_cpu)
        sla_pen = np.sum(np.maximum(0, self.server_cpu - 0.85)) * 2.0
        reward = 1.0 - load_var - sla_pen
        done = self.step_cnt >= 200
        return self._get_obs(), float(reward), done, False, {}
'''

# 8. Main Training Script
files["train_all.py"] = '''import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from models.sagatx.model import SAGATx
from models.hmarl.sfc_env import SFCOrchestrationEnv
from stable_baselines3 import PPO

def train():
    print("=== 1. Training SAGATx Prediction Model ===")
    train_data = torch.load("data/processed/train_data.pt", weights_only=False)
    X, Y = train_data["X"], train_data["Y"]
    loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAGATx(num_nodes=100, num_slices=3, d_model=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 6):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = model.loss(pred, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/5 | NLL Loss: {total_loss/len(loader):.4f}")
        
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/sagatx.pt")
    print("[OK] SAGATx weights saved to checkpoints/sagatx.pt")
    
    print("\\n=== 2. Training HMARL SFC Orchestration Agent ===")
    env = SFCOrchestrationEnv()
    agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    agent.learn(total_timesteps=10000)
    agent.save("checkpoints/hmarl_agent")
    print("[OK] HMARL agent saved to checkpoints/hmarl_agent")
    print("\\n🚀 All models trained and ready!")

if __name__ == "__main__":
    train()
'''

print(f"Writing project to {ROOT}...")
for rel_path, content in files.items():
    dest = ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  ✓ {rel_path}")

print("\nDone! Now run:\n  pip install -r requirements.txt\n  python data/download_data.py\n  python data/preprocess.py\n  python train_all.py")