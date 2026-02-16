import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Fixed-point int32 fake-quant
# -----------------------------

INT32_MIN = -(2**31)
INT32_MAX =  (2**31 - 1)

def fake_quant_int32_fixed(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """
    Fake-quantize to signed int32 fixed-point with STE.

    Forward:
      x_hat = clamp(round(x * S), int32_min, int32_max) / S
    Backward:
      dL/dx approx dL/dx_hat (straight-through).
    """
    if frac_bits < 0 or frac_bits > 30:
        raise ValueError("frac_bits should be in [0, 30] for practical fixed-point.")
    scale = float(2**frac_bits)

    # Quantize to int32 domain
    q = torch.round(x * scale)
    q = torch.clamp(q, INT32_MIN, INT32_MAX)

    # Dequantize
    x_hat = q / scale

    # STE: forward uses x_hat, gradients flow as if identity
    return x + (x_hat - x).detach()


# -----------------------------
# Synthetic dataset
# -----------------------------

class SumSignDataset(Dataset):
    """
    Generate sequences x[t] ~ N(0, 1).
    Label y = 1 if sum_t x[t] > 0 else 0.

    This is intentionally simple and suitable for a single LSTM cell + classifier.
    """
    def __init__(self, n_samples: int, seq_len: int, input_size: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, seq_len, input_size)).astype(np.float32)
        s = self.x.sum(axis=1).sum(axis=1)  # sum across time and features
        self.y = (s > 0).astype(np.float32)  # shape [n_samples]
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


# -----------------------------
# Quantized LSTM Cell module
# -----------------------------

class QuantLSTMCell(nn.Module):
    """
    LSTMCell with fake-quantization applied to:
    - inputs x
    - hidden state h and cell state c
    - weights and biases
    - internal gate pre-activations (optional, but helps match RTL behavior)
    """
    def __init__(self, input_size: int, hidden_size: int, frac_bits: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.frac_bits = frac_bits

        # Use PyTorch parameters with float storage; we fake-quantize at forward.
        # We'll implement the LSTM equations ourselves to control quant points.
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard LSTM init (similar to nn.LSTMCell)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        x_t: [B, input_size]
        state: (h, c), each [B, hidden_size]
        """
        h, c = state

        # Fake-quantize inputs and states
        x_q = fake_quant_int32_fixed(x_t, self.frac_bits)
        h_q = fake_quant_int32_fixed(h,   self.frac_bits)
        c_q = fake_quant_int32_fixed(c,   self.frac_bits)

        # Fake-quantize parameters
        W_ih_q = fake_quant_int32_fixed(self.W_ih, self.frac_bits)
        W_hh_q = fake_quant_int32_fixed(self.W_hh, self.frac_bits)
        b_ih_q = fake_quant_int32_fixed(self.b_ih, self.frac_bits)
        b_hh_q = fake_quant_int32_fixed(self.b_hh, self.frac_bits)

        # Gates pre-activation
        gates = F.linear(x_q, W_ih_q, b_ih_q) + F.linear(h_q, W_hh_q, b_hh_q)
        gates = fake_quant_int32_fixed(gates, self.frac_bits)  # match fixed-point add saturation vibe

        i, f, g, o = gates.chunk(4, dim=1)

        # Nonlinearities (still float, but inputs are quantized)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Optionally quantize activations too (helps match quantized inference)
        i = fake_quant_int32_fixed(i, self.frac_bits)
        f = fake_quant_int32_fixed(f, self.frac_bits)
        g = fake_quant_int32_fixed(g, self.frac_bits)
        o = fake_quant_int32_fixed(o, self.frac_bits)

        # Cell update and hidden
        c_new = f * c_q + i * g
        c_new = fake_quant_int32_fixed(c_new, self.frac_bits)

        h_new = o * torch.tanh(c_new)
        h_new = fake_quant_int32_fixed(h_new, self.frac_bits)

        return h_new, (h_new, c_new)


class QuantStackedLSTM(nn.Module):
    """
    Stack of QuantLSTMCell layers.
    For your VHDL single-cell design, keep num_layers=1.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, frac_bits: int):
        super().__init__()
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(QuantLSTMCell(in_size, hidden_size, frac_bits))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frac_bits = frac_bits

    def forward(self, x: torch.Tensor, state=None):
        """
        x: [B, T, input_size]
        returns:
          last_hidden: [B, hidden_size]
        """
        B, T, _ = x.shape
        if state is None:
            h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
            c = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = state

        for t in range(T):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                out, (h[layer], c[layer]) = cell(inp, (h[layer], c[layer]))
                inp = out  # feed to next layer

        return h[-1]  # last layer last timestep hidden


class QuantLSTMClassifier(nn.Module):
    """
    Quantized LSTM (stacked) + final linear classifier neuron.
    Output is a logit for binary classification.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, frac_bits: int):
        super().__init__()
        self.rnn = QuantStackedLSTM(input_size, hidden_size, num_layers, frac_bits)
        self.fc_w = nn.Parameter(torch.empty(1, hidden_size))
        self.fc_b = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.fc_w, -1.0 / math.sqrt(hidden_size), 1.0 / math.sqrt(hidden_size))
        self.frac_bits = frac_bits

    def forward(self, x: torch.Tensor):
        h_last = self.rnn(x)  # [B, H]
        h_last = fake_quant_int32_fixed(h_last, self.frac_bits)

        W_q = fake_quant_int32_fixed(self.fc_w, self.frac_bits)
        b_q = fake_quant_int32_fixed(self.fc_b, self.frac_bits)

        logit = F.linear(h_last, W_q, b_q)  # [B, 1]
        logit = fake_quant_int32_fixed(logit, self.frac_bits)
        return logit


# -----------------------------
# Export helpers
# -----------------------------

def to_int32_fixed(x: torch.Tensor, frac_bits: int) -> np.ndarray:
    """Convert float tensor to int32 fixed-point integer array."""
    scale = float(2**frac_bits)
    q = torch.round(x * scale)
    q = torch.clamp(q, INT32_MIN, INT32_MAX)
    return q.to(torch.int32).cpu().numpy()

def export_model_int32(model: QuantLSTMClassifier, export_dir: str, frac_bits: int):
    os.makedirs(export_dir, exist_ok=True)

    meta: Dict = {
        "format": "int32_fixed_point",
        "frac_bits": frac_bits,
        "scale": int(2**frac_bits),
        "int32_min": INT32_MIN,
        "int32_max": INT32_MAX,
        "layers": [],
        "final_linear": {}
    }

    # Export each LSTM layer
    for li, cell in enumerate(model.rnn.cells):
        layer_dict = {
            "layer_index": li,
            "input_size": cell.input_size,
            "hidden_size": cell.hidden_size,
            "W_ih": f"lstm_layer{li}_W_ih_int32.npy",
            "W_hh": f"lstm_layer{li}_W_hh_int32.npy",
            "b_ih": f"lstm_layer{li}_b_ih_int32.npy",
            "b_hh": f"lstm_layer{li}_b_hh_int32.npy",
        }

        np.save(os.path.join(export_dir, layer_dict["W_ih"]), to_int32_fixed(cell.W_ih.data, frac_bits))
        np.save(os.path.join(export_dir, layer_dict["W_hh"]), to_int32_fixed(cell.W_hh.data, frac_bits))
        np.save(os.path.join(export_dir, layer_dict["b_ih"]), to_int32_fixed(cell.b_ih.data, frac_bits))
        np.save(os.path.join(export_dir, layer_dict["b_hh"]), to_int32_fixed(cell.b_hh.data, frac_bits))

        meta["layers"].append(layer_dict)

    # Export final linear
    fc = {
        "W": "final_fc_W_int32.npy",  # shape [1, H]
        "b": "final_fc_b_int32.npy",  # shape [1]
        "in_features": model.fc_w.shape[1],
        "out_features": 1
    }
    np.save(os.path.join(export_dir, fc["W"]), to_int32_fixed(model.fc_w.data, frac_bits))
    np.save(os.path.join(export_dir, fc["b"]), to_int32_fixed(model.fc_b.data, frac_bits))
    meta["final_linear"] = fc

    with open(os.path.join(export_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[EXPORT] Saved int32 fixed-point weights to: {export_dir}")
    print(f"[EXPORT] W,b are stored as integers representing value * 2^{frac_bits}.")


# -----------------------------
# Train / Eval
# -----------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset), correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-size", type=int, default=1, help="Input feature dimension (default: 1)")
    ap.add_argument("--hidden-size", type=int, default=1, help="Hidden size of LSTM cell")
    ap.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTMCell layers (default: 1)")
    ap.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    ap.add_argument("--train-samples", type=int, default=20000, help="Number of training samples")
    ap.add_argument("--val-samples", type=int, default=4000, help="Number of validation samples")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--frac-bits", type=int, default=16, help="Fractional bits for int32 fixed-point")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--export-dir", type=str, default="", help="If set, export quantized int32 weights here at the end")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] device={device}")
    print(f"[INFO] input_size={args.input_size}, hidden_size={args.hidden_size}, num_layers={args.num_layers}")
    print(f"[INFO] frac_bits={args.frac_bits}, seq_len={args.seq_len}")

    train_ds = SumSignDataset(args.train_samples, args.seq_len, args.input_size, seed=args.seed)
    val_ds   = SumSignDataset(args.val_samples,   args.seq_len, args.input_size, seed=args.seed + 1)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = QuantLSTMClassifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        frac_bits=args.frac_bits
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_ld:
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate(model, val_ld, device)
        print(f"Epoch {epoch:03d}/{args.epochs} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

    # Optional export for VHDL
    if args.export_dir:
        export_model_int32(model, args.export_dir, args.frac_bits)


if __name__ == "__main__":
    main()
