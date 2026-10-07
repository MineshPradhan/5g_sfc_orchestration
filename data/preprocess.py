import numpy as np
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
