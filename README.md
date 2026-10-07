# 5G-PredictSFC: Slice-Aware Graph Attention Transformer & Hierarchical Multi-Agent RL for Proactive SFC Orchestration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://github.com/DLR-RM/stable-baselines3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework for **joint multi-slice 5G traffic prediction** and **closed-loop proactive Service Function Chain (SFC) orchestration**. This framework eliminates reactive adaptation delay in 5G NFV infrastructures, reducing SLA violations by up to **98.2%**.

---

## 📌 Problem & Overview

In 5G and beyond (6G) networks, heterogeneous traffic classes (**eMBB**, **URLLC**, and **mMTC**) generate highly dynamic, bursty workloads. Conventional Network Function Virtualization (NFV) and Service Function Chaining (SFC) systems are **reactive**—they only instantiate, scale, or migrate Virtual Network Functions (VNFs) *after* traffic surges arrive.

This causes:
- **Severe SLA violations** during adaptation delay
- **Resource fragmentation and bottleneck hotspots**
- **Excessive reconfiguration and migration overhead**

### 💡 The Solution: SAGATx-HMARL
1. **SAGATx (Predictor)**: A Slice-Aware Graph Attention Transformer that jointly forecasts spatial network topology and heterogeneous per-slice demands (eMBB, URLLC, mMTC) using cross-slice attention.
2. **HMARL (Orchestrator)**: A Hierarchical Multi-Agent Reinforcement Learning policy (using PPO) that takes proactive placement, scaling, and migration decisions before congestion occurs.

┌────────────────────────────────────────────────────────────────────────┐ │ SAGATx-HMARL CLOSED-LOOP │ │ │ │ [Telecom Ingestion] ──► [SAGATx Predictor] ──► [Predicted Demand] │ │ (eMBB, URLLC, mMTC) • Cross-Slice Attn (μ, σ² intervals) │ │ • Spatial Graph Conv │ │ │ ▼ │ │ [NFV Infrastructure] ◄── [HMARL Orchestrator] ◄─ [Resource State] │ │ • Server CPU/Mem/BW • Meta-PPO Agent (Server telemetry) │ │ • VNF Placements • Multi-Objective QoS │ └────────────────────────────────────────────────────────────────────────┘


---

## 📊 Benchmark Results

Evaluated over 20 simulation episodes with bursty traffic shocks against **Static Allocation** and **Conventional Reactive Heuristics**:

### 1. SFC Orchestration Performance

| Strategy | Cumulative Reward | Server Load Variance (Lower is better) | Severe SLA Overload Steps (Lower is better) | SLA Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Static Allocation** | -4,710.97 | 0.01534 | 44,497 | — |
| **Conventional Reactive Scaling** | -188.28 | 0.00897 | 4,647 | Baseline |
| **Proposed SAGATx + HMARL** | **+331.27** | **0.00371** | **84** | **-98.19%** |

- **98.2% reduction in SLA breaches** compared to reactive scaling.
- **58.6% lower resource fragmentation**, demonstrating balanced cluster utilization.

### 2. Traffic Prediction Performance (MAE on Test Set)

| Network Slice | Baseline MAE | SAGATx MAE | Improvement |
| :--- | :---: | :---: | :---: |
| **eMBB** (High Throughput) | 0.6063 | **0.4798** | **+20.85%** |
| **URLLC** (Low Latency) | 0.7763 | **0.5980** | **+22.97%** |
| **mMTC** (Massive IoT) | 0.7673 | **0.5911** | **+22.96%** |

---

## 📁 Repository Structure

5g_sfc_orchestration/ ├── config.yaml # Model parameters, slices, training configs ├── requirements.txt # Python dependencies ├── data/ │ ├── download_data.py # Telecom Italia dataset generator / downloader │ └── preprocess.py # Multi-slice tensor mapping & graph adjacency builder ├── models/ │ ├── sagatx/ │ │ └── model.py # Slice-Aware Graph Attention Transformer │ └── hmarl/ │ └── sfc_env.py # Gymnasium-based 5G NFV cluster environment ├── checkpoints/ # Saved model weights (.pt, .zip) ├── results/ # Generated benchmark evaluation plots ├── train_all.py # End-to-end training pipeline └── evaluate_and_plot.py # Automated benchmark & plotting script


---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- PyTorch 2.0+ (CUDA supported)

### 2. Installation

```bash
git clone https://github.com/<your-username>/5g-predictsfc.git
cd 5g-predictsfc

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## Data Ingestion & Preprocessing

# Generate/fetch Telecom Italia 5G traffic data
python data/download_data.py

# Preprocess into normalized sliding-window tensors & spatial adjacency graph
python data/preprocess.py

## Training

# Train both the SAGATx forecasting model and the HMARL agent in one command:

bash
python train_all.py

# Trained checkpoints are saved to checkpoints/sagatx.pt and checkpoints/hmarl_agent.zip.

## Evaluation and Plotting

python evaluate_and_plot.py

This prints the metrics table in your terminal and generates two publication-ready figures in results/:

results/traffic_prediction_comparison.png: Time-series forecasting tracking across slices.
results/orchestration_benchmarks.png: Comparative bar charts for SLA breaches and load variance.