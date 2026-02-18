#!/usr/bin/env python3
# test_patchlstm_cifar100.py
#
# Evaluate a Patch-LSTM CIFAR-100 classifier checkpoint:
# - prints test accuracy
# - saves confusion matrix plot
# - saves two 3x3 grids: correct predictions and wrong predictions
#
# IMPORTANT: activations used at test time are chosen ONLY by --use-torch-acts:
# - default: uses acts.sigmoid and acts.tanh (if available)
# - if --use-torch-acts: uses torch.sigmoid and torch.tanh
# This is independent of how the checkpoint was trained.

import argparse
import os
import random
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# ----------------------------
# Optional import of custom activations
# ----------------------------
try:
    import acts  # must expose acts.sigmoid and acts.tanh
    HAS_ACTS = True
except Exception:
    HAS_ACTS = False
    acts = None


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Activations selection (NO STE here; test only)
# ----------------------------
def make_activation_fns(use_torch_acts: bool) -> Tuple[Callable[[torch.Tensor], torch.Tensor],
                                                      Callable[[torch.Tensor], torch.Tensor]]:
    """
    Returns (sigmoid_fn, tanh_fn).
    By default uses acts.sigmoid/acts.tanh (if available), otherwise torch.*.
    """
    if use_torch_acts or not HAS_ACTS:
        return torch.sigmoid, torch.tanh

    if not hasattr(acts, "sigmoid") or not hasattr(acts, "tanh"):
        raise RuntimeError("Custom module 'acts' must provide acts.sigmoid and acts.tanh")
    return acts.sigmoid, acts.tanh


# ----------------------------
# Patchify transform: image (C,H,W) -> sequence (T, F)
# CIFAR: C=3, H=W=32. patch=4 => 64 tokens, each token F=48
# ----------------------------
class CIFARToPatches:
    def __init__(self, patch: int = 4):
        assert 32 % patch == 0, "For CIFAR 32x32, patch must divide 32."
        self.patch = patch

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3,32,32)
        C, H, W = x.shape
        p = self.patch
        gh, gw = H // p, W // p
        x = x.view(C, gh, p, gw, p).permute(1, 3, 0, 2, 4).contiguous()
        x = x.view(gh * gw, C * p * p)
        return x  # (T, F)


# ----------------------------
# Model (same as training script)
# ----------------------------
class CustomLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
        tanh_fn: Callable[[torch.Tensor], torch.Tensor],
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigmoid_fn = sigmoid_fn
        self.tanh_fn = tanh_fn
        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.ih(x_t) + self.hh(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = self.sigmoid_fn(i)
        f = self.sigmoid_fn(f)
        g = self.tanh_fn(g)
        o = self.sigmoid_fn(o)

        c_new = f * c + i * g
        h_new = o * self.tanh_fn(c_new)
        return h_new, c_new


class CustomLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
        tanh_fn: Callable[[torch.Tensor], torch.Tensor],
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        cells: List[nn.Module] = []
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            cells.append(CustomLSTMCell(in_sz, hidden_size, sigmoid_fn, tanh_fn))
        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B, T, H) from top layer
        """
        B, T, _ = x.shape
        hs = [x.new_zeros((B, self.hidden_size)) for _ in range(self.num_layers)]
        cs = [x.new_zeros((B, self.hidden_size)) for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l in range(self.num_layers):
                h, c = self.cells[l](inp, (hs[l], cs[l]))
                hs[l], cs[l] = h, c
                inp = self.drop(h) if (l < self.num_layers - 1) else h
            outputs.append(hs[-1])

        return torch.stack(outputs, dim=1)  # (B,T,H)


class PatchLSTMClassifier(nn.Module):
    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
        tanh_fn: Callable[[torch.Tensor], torch.Tensor],
        dropout: float = 0.1,
        qat: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.qat = qat

        if qat:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        self.token_embed = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        self.lstm = CustomLSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sigmoid_fn=sigmoid_fn,
            tanh_fn=tanh_fn,
            dropout=dropout,
        )

        self.attn = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if self.qat:
            x = self.quant(x)

        x = self.token_embed(x)          # (B,T,E)
        h_seq = self.lstm(x)             # (B,T,H)

        a = self.attn(h_seq).squeeze(-1)           # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,T,1)
        h = (h_seq * w).sum(dim=1)                 # (B,H)

        logits = self.classifier(h)

        if self.qat:
            logits = self.dequant(logits)
        return logits


# ----------------------------
# QAT helpers (for matching checkpoint structures)
# ----------------------------
def prepare_model_for_qat(model: nn.Module, backend: str) -> nn.Module:
    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_int8(model: nn.Module, backend: str) -> nn.Module:
    torch.backends.quantized.engine = backend
    model_cpu = model.to("cpu").eval()
    model_int8 = torch.ao.quantization.convert(model_cpu, inplace=False)
    return model_int8


# ----------------------------
# Data
# ----------------------------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_test_loader(data_dir: Optional[str], batch_size: int, num_workers: int, patch_size: int) -> DataLoader:
    download = data_dir is None
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    if (not download) and (not os.path.exists(data_dir)):
        raise FileNotFoundError(
            f"--data-dir was provided but directory does not exist: {data_dir}\n"
            f"Either create it / place CIFAR-100 there, or omit --data-dir to auto-download."
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        CIFARToPatches(patch=patch_size),
    ])
    test_ds = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ----------------------------
# Utilities for plotting grids
# We reconstruct the original image from patches so we can show a 32x32 RGB image.
# ----------------------------
def unpatchify(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    patches: (T, F) with T=(32/p)^2 and F=3*p*p
    returns: (3,32,32) in the SAME normalized space as input.
    """
    p = patch_size
    T, F = patches.shape
    gh = gw = int(np.sqrt(T))
    assert gh * gw == T, "T must be a perfect square for CIFAR32 patches."
    assert F == 3 * p * p

    x = patches.view(gh, gw, 3, p, p).permute(2, 0, 3, 1, 4).contiguous()
    x = x.view(3, gh * p, gw * p)
    return x


def denormalize(img_chw: torch.Tensor) -> torch.Tensor:
    """
    img_chw: (3,32,32) normalized
    returns: (3,32,32) in [0,1] roughly (clipped)
    """
    mean = torch.tensor(CIFAR100_MEAN, dtype=img_chw.dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=img_chw.dtype).view(3, 1, 1)
    x = img_chw * std + mean
    return x.clamp(0.0, 1.0)


# ----------------------------
# Metrics and plots
# ----------------------------
@torch.no_grad()
def run_inference_collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    max_collect_correct: int = 9,
    max_collect_wrong: int = 9,
):
    model.eval()

    total = 0
    correct = 0

    all_true = []
    all_pred = []

    correct_examples = []  # list of (img_chw_denorm, y_true, y_pred)
    wrong_examples = []

    for x, y in loader:
        x = x.to(device)  # (B,T,F)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += x.size(0)
        correct += int((pred == y).sum().item())

        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

        if len(correct_examples) < max_collect_correct or len(wrong_examples) < max_collect_wrong:
            x_cpu = x.detach().cpu()
            y_cpu = y.detach().cpu()
            p_cpu = pred.detach().cpu()

            for i in range(x_cpu.size(0)):
                img_norm = unpatchify(x_cpu[i], patch_size=patch_size)  # (3,32,32) normalized
                img = denormalize(img_norm)  # (3,32,32) in [0,1] for plotting

                if len(correct_examples) < max_collect_correct and p_cpu[i].item() == y_cpu[i].item():
                    correct_examples.append((img, int(y_cpu[i].item()), int(p_cpu[i].item())))
                elif len(wrong_examples) < max_collect_wrong and p_cpu[i].item() != y_cpu[i].item():
                    wrong_examples.append((img, int(y_cpu[i].item()), int(p_cpu[i].item())))

                if len(correct_examples) >= max_collect_correct and len(wrong_examples) >= max_collect_wrong:
                    break

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = correct / total

    return acc, y_true, y_pred, correct_examples, wrong_examples


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: str, normalize: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if normalize:
        cm_plot = cm.astype(np.float64)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1.0))
    else:
        cm_plot = cm

    plt.figure(figsize=(10, 9))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    # For CIFAR-100, ticks are too many to label nicely; keep sparse ticks
    n = cm.shape[0]
    step = 10
    ticks = np.arange(0, n, step)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grid(examples, title: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = 9
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    fig.suptitle(title)

    for k in range(n):
        ax = axes[k // 3, k % 3]
        ax.axis("off")
        if k < len(examples):
            img, y_true, y_pred = examples[k]
            # img: (3,32,32) in [0,1]
            ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_title(f"T:{y_true}  P:{y_pred}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Checkpoint loading (robust)
# ----------------------------
def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("Checkpoint must be a dict containing at least 'model_state'.")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Test Patch-LSTM CIFAR-100 checkpoint and generate plots.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to CIFAR-100 root. If omitted, CIFAR-100 will be downloaded into ./data.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--use-torch-acts", action="store_true",
                        help="Use torch.sigmoid/tanh instead of acts.sigmoid/tanh (default uses acts if available).")

    parser.add_argument("--out-dir", type=str, default="test_outputs", help="Directory where plots will be saved.")

    # For matching QAT/int8 checkpoint structures
    parser.add_argument("--quant-backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"],
                        help="Quantization backend (only relevant if checkpoint is QAT/INT8).")

    # If config missing in checkpoint, allow manual override
    parser.add_argument("--patch-size", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--normalize-cm", action="store_true", help="Normalize confusion matrix rows.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Choose activations (independent of checkpoint training)
    if (not args.use_torch_acts) and (not HAS_ACTS):
        print("WARNING: Could not import 'acts'. Falling back to torch.sigmoid/tanh.")
    sigmoid_fn, tanh_fn = make_activation_fns(use_torch_acts=args.use_torch_acts)

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    patch_size = int(cfg.get("patch_size", args.patch_size))
    embed_dim = int(cfg.get("embed_dim", args.embed_dim))
    hidden_size = int(cfg.get("hidden_size", args.hidden_size))
    num_layers = int(cfg.get("num_layers", args.num_layers))
    dropout = float(cfg.get("dropout", args.dropout))

    qat_flag = bool(cfg.get("qat", False))

    note = ckpt.get("note", "")
    is_int8_export = isinstance(note, str) and ("INT8" in note or "int8" in note)

    token_dim = 3 * patch_size * patch_size

    model = PatchLSTMClassifier(
        token_dim=token_dim,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=100,
        sigmoid_fn=sigmoid_fn,
        tanh_fn=tanh_fn,
        dropout=dropout,
        qat=qat_flag or is_int8_export,
    )

    if qat_flag or is_int8_export:
        model = prepare_model_for_qat(model, backend=args.quant_backend)

    # If it's INT8 export: convert structure first, then load state_dict
    if is_int8_export:
        model = convert_to_int8(model, backend=args.quant_backend)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("State dict load warnings:")
        if missing:
            print("  Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
        if unexpected:
            print("  Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    # Device policy:
    # - INT8 quantized models typically run on CPU.
    # - Float/QAT-prepared models can run on CUDA if available.
    if is_int8_export:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_loader = build_test_loader(args.data_dir, args.batch_size, args.num_workers, patch_size=patch_size)

    acc, y_true, y_pred, correct_ex, wrong_ex = run_inference_collect(
        model, test_loader, device, patch_size=patch_size, max_collect_correct=9, max_collect_wrong=9
    )

    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=100)

    os.makedirs(args.out_dir, exist_ok=True)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    cmn_path = os.path.join(args.out_dir, "confusion_matrix_normalized.png")
    corr_path = os.path.join(args.out_dir, "examples_correct_3x3.png")
    wrong_path = os.path.join(args.out_dir, "examples_wrong_3x3.png")

    plot_confusion_matrix(cm, cm_path, normalize=False)
    if args.normalize_cm:
        plot_confusion_matrix(cm, cmn_path, normalize=True)

    plot_grid(correct_ex, "Correct predictions (3x3)", corr_path)
    plot_grid(wrong_ex, "Wrong predictions (3x3)", wrong_path)

    print("\n=== Results ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Patch size: {patch_size} | token_dim: {token_dim}")
    print(f"Embed dim: {embed_dim} | hidden size: {hidden_size} | num layers: {num_layers}")
    print(f"Checkpoint QAT flag: {qat_flag}")
    print(f"INT8 export detected: {is_int8_export}")
    print(f"Activations at test time: {'torch.*' if args.use_torch_acts else ('acts.*' if HAS_ACTS else 'torch.* (acts missing)')}")
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"\nSaved plots to: {args.out_dir}")
    print(f" - {cm_path}")
    if args.normalize_cm:
        print(f" - {cmn_path}")
    print(f" - {corr_path}")
    print(f" - {wrong_path}")


if __name__ == "__main__":
    main()
