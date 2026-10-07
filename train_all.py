import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from models.sagatx.model import SAGATx
from models.hmarl.sfc_env import SFCOrchestrationEnv
from stable_baselines3 import PPO

def train():
    print("=== 1. Training SAGATx Prediction Model (15 Epochs) ===")
    train_data = torch.load("data/processed/train_data.pt", weights_only=False)
    X, Y = train_data["X"], train_data["Y"]
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAGATx(num_nodes=100, num_slices=3, d_model=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    
    for epoch in range(1, 16):
        total_loss = 0
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = model.loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/15 | NLL Loss: {total_loss/len(loader):.4f}")
        
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/sagatx.pt")
    print("[OK] SAGATx trained and weights saved to checkpoints/sagatx.pt")
    
    print("\n=== 2. Training HMARL SFC Orchestration Agent (40,000 steps) ===")
    env = SFCOrchestrationEnv()
    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99
    )
    agent.learn(total_timesteps=40000)
    agent.save("checkpoints/hmarl_agent")
    print("[OK] HMARL agent saved to checkpoints/hmarl_agent")
    print("\n🚀 Ready for evaluation! Now run python evaluate_and_plot.py")

if __name__ == "__main__":
    train()