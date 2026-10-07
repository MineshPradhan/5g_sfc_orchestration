import numpy as np
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
