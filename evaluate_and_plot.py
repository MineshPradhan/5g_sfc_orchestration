import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from models.sagatx.model import SAGATx
from models.hmarl.sfc_env import SFCOrchestrationEnv
from stable_baselines3 import PPO

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
Path("results").mkdir(exist_ok=True)

print("="*60)
print(" 1. EVALUATING SAGATx PREDICTION PERFORMANCE")
print("="*60)

test_data = torch.load("data/processed/test_data.pt", weights_only=False)
X_test, Y_test = test_data["X"], test_data["Y"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAGATx(num_nodes=100, num_slices=3, d_model=64).to(device)
model.load_state_dict(torch.load("checkpoints/sagatx.pt", map_location=device))
model.eval()

with torch.no_grad():
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)
    preds, targets = [], []
    for bx, by in test_loader:
        bx = bx.to(device)
        out = model(bx)
        # out has shape (B, T_out, N, S, 2) where index 0 is mu (mean)
        mu = out[..., 0].cpu().numpy()
        preds.append(mu)
        targets.append(by.numpy())

y_pred = np.concatenate(preds, axis=0)
y_true = np.concatenate(targets, axis=0)

# Baseline: Historical Naive / Persistence baseline
y_baseline = np.repeat(X_test[:, -1:, :, :].numpy(), y_true.shape[1], axis=1)

slice_names = ["eMBB (High Throughput)", "URLLC (Low Latency)", "mMTC (Massive IoT)"]
pred_results = []

for s, sname in enumerate(slice_names):
    mae_model = np.mean(np.abs(y_pred[..., s] - y_true[..., s]))
    rmse_model = np.sqrt(np.mean((y_pred[..., s] - y_true[..., s])**2))
    
    mae_base = np.mean(np.abs(y_baseline[..., s] - y_true[..., s]))
    rmse_base = np.sqrt(np.mean((y_baseline[..., s] - y_true[..., s])**2))
    
    improvement = ((mae_base - mae_model) / mae_base) * 100
    pred_results.append({
        "Slice": sname,
        "Baseline MAE": mae_base,
        "SAGATx MAE": mae_model,
        "SAGATx RMSE": rmse_model,
        "MAE Improvement (%)": improvement
    })

df_pred = pd.DataFrame(pred_results)
print("\n--- Prediction Accuracy Summary ---")
print(df_pred.to_string(index=False))

# Plot 1: Prediction Trajectory for Node 0
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
sample_steps = min(60, y_true.shape[0])
for s in range(3):
    axes[s].plot(y_true[:sample_steps, 0, 0, s], label="Ground Truth", color="black", lw=2)
    axes[s].plot(y_pred[:sample_steps, 0, 0, s], label="SAGATx (Proposed)", color="#007acc", lw=1.5, ls="--")
    axes[s].plot(y_baseline[:sample_steps, 0, 0, s], label="Reactive Baseline", color="#e74c3c", lw=1, alpha=0.6)
    axes[s].set_title(slice_names[s], fontsize=11, fontweight="bold")
    axes[s].set_ylabel("Normalized Load")
    if s == 0:
        axes[s].legend(loc="upper right")
axes[-1].set_xlabel("Time Intervals (10-min bins)")
plt.tight_layout()
plt.savefig("results/traffic_prediction_comparison.png", dpi=300)
print("\n[Saved] Figure -> results/traffic_prediction_comparison.png")

print("\n" + "="*60)
print(" 2. EVALUATING CLOSED-LOOP SFC ORCHESTRATION")
print("="*60)

env = SFCOrchestrationEnv()
agent = PPO.load("checkpoints/hmarl_agent")

def run_simulation(policy_type="hmarl", episodes=20):
    total_rewards = []
    load_variances = []
    sla_violations = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0
        while not done:
            if policy_type == "hmarl":
                action, _ = agent.predict(obs, deterministic=True)
            elif policy_type == "reactive":
                # Reactive heuristic: Scale up if overloaded (>0.8), migrate if high disparity
                action = np.zeros(env.num_slices, dtype=int)
                if np.max(env.server_cpu) > 0.8:
                    action[:] = 2 # SCALE UP
                elif (np.max(env.server_cpu) - np.min(env.server_cpu)) > 0.4:
                    action[:] = 4 # MIGRATE
            elif policy_type == "static":
                action = np.zeros(env.num_slices, dtype=int) # NOP
                
            obs, r, done, _, _ = env.step(action)
            ep_rew += r
            load_variances.append(np.var(env.server_cpu))
            sla_violations.append(np.sum(env.server_cpu > 0.85))
            
        total_rewards.append(ep_rew)
        
    return {
        "Mean Reward": np.mean(total_rewards),
        "Avg Server Load Variance": np.mean(load_variances),
        "Total Overload Incidents": np.sum(sla_violations)
    }

metrics_hmarl = run_simulation("hmarl")
metrics_reactive = run_simulation("reactive")
metrics_static = run_simulation("static")

df_orch = pd.DataFrame([
    {"Strategy": "Static SFC Allocation", **metrics_static},
    {"Strategy": "Conventional Reactive Scaling", **metrics_reactive},
    {"Strategy": "Proposed SAGATx + HMARL", **metrics_hmarl},
])

print("\n--- Orchestration Policy Comparison ---")
print(df_orch.to_string(index=False))

# Plot 2: Orchestration KPI Bar Chart
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

strategies = ["Static", "Reactive", "SAGATx-HMARL (Ours)"]
colors = ["#95a5a6", "#e67e22", "#2ecc71"]

variances = [metrics_static["Avg Server Load Variance"], 
             metrics_reactive["Avg Server Load Variance"], 
             metrics_hmarl["Avg Server Load Variance"]]
ax[0].bar(strategies, variances, color=colors)
ax[0].set_title("Resource Fragmentation (Lower is Better)", fontweight="bold")
ax[0].set_ylabel("Server Load Variance")

overloads = [metrics_static["Total Overload Incidents"], 
             metrics_reactive["Total Overload Incidents"], 
             metrics_hmarl["Total Overload Incidents"]]
ax[1].bar(strategies, overloads, color=colors)
ax[1].set_title("SLA Violations / Congestion Events (Lower is Better)", fontweight="bold")
ax[1].set_ylabel("Total Severe Overload Steps")

plt.tight_layout()
plt.savefig("results/orchestration_benchmarks.png", dpi=300)
print("[Saved] Figure -> results/orchestration_benchmarks.png")

print("\n" + "="*60)
print(" All benchmarks complete! Figures and tables are in /results.")
print("="*60)