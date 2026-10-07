import torch
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
