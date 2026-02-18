#!/usr/bin/env python3
# test_lstm_mnist.py
#
# Evaluate an LSTM MNIST classifier checkpoint:
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
from typing import Callable, Optional, Tuple

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


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
        tanh_fn: Callable[[torch.Tensor], torch.Tensor],
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

        self.cell = CustomLSTMCell(input_size, hidden_size, sigmoid_fn, tanh_fn)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) with T=28, F=28
        if self.qat:
            x = self.quant(x)

        B, T, _ = x.shape
        h = x.new_zeros((B, self.hidden_size))
        c = x.new_zeros((B, self.hidden_size))

        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))

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
class MNISTRowsAsSequence:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # (1,28,28) -> (28,28) interpreted as (T=28,F=28)
        return x.squeeze(0).contiguous()


def build_test_loader(data_dir: Optional[str], batch_size: int, num_workers: int) -> DataLoader:
    download = data_dir is None
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    if (not download) and (not os.path.exists(data_dir)):
        raise FileNotFoundError(
            f"--data-dir was provided but directory does not exist: {data_dir}\n"
            f"Either create it / place MNIST there, or omit --data-dir to auto-download."
        )

    transform = transforms.Compose([transforms.ToTensor(), MNISTRowsAsSequence()])
    test_ds = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ----------------------------
# Metrics and plots
# ----------------------------
@torch.no_grad()
def run_inference_collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_collect_correct: int = 9,
    max_collect_wrong: int = 9,
):
    model.eval()

    total = 0
    correct = 0

    all_true = []
    all_pred = []

    correct_examples = []  # list of (img28x28, y_true, y_pred)
    wrong_examples = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += x.size(0)
        correct += int((pred == y).sum().item())

        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

        # Collect examples for grids (use original image values)
        # x is (B,28,28) on device; move small subset to cpu for plotting.
        if len(correct_examples) < max_collect_correct or len(wrong_examples) < max_collect_wrong:
            x_cpu = x.detach().cpu()
            y_cpu = y.detach().cpu()
            p_cpu = pred.detach().cpu()

            for i in range(x_cpu.size(0)):
                if len(correct_examples) < max_collect_correct and p_cpu[i].item() == y_cpu[i].item():
                    correct_examples.append((x_cpu[i], int(y_cpu[i].item()), int(p_cpu[i].item())))
                elif len(wrong_examples) < max_collect_wrong and p_cpu[i].item() != y_cpu[i].item():
                    wrong_examples.append((x_cpu[i], int(y_cpu[i].item()), int(p_cpu[i].item())))

                if len(correct_examples) >= max_collect_correct and len(wrong_examples) >= max_collect_wrong:
                    break

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = correct / total

    return acc, y_true, y_pred, correct_examples, wrong_examples


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
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

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = str(int(val))
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grid(examples, title: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # We want a 3x3 grid. If fewer than 9 examples exist, leave some empty.
    n = 9
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    fig.suptitle(title)

    for k in range(n):
        ax = axes[k // 3, k % 3]
        ax.axis("off")
        if k < len(examples):
            img, y_true, y_pred = examples[k]
            # img is (28,28) float
            ax.imshow(img.numpy(), interpolation="nearest", cmap="gray")
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
    parser = argparse.ArgumentParser(description="Test LSTM MNIST checkpoint and generate plots.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to MNIST root. If omitted, MNIST will be downloaded into ./data.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--use-torch-acts", action="store_true", help="Use torch.sigmoid/tanh instead of acts.sigmoid/tanh (default uses acts if available).")

    parser.add_argument("--out-dir", type=str, default="test_outputs", help="Directory where plots will be saved.")

    # For matching QAT/int8 checkpoint structures
    parser.add_argument("--quant-backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"], help="Quantization backend (only relevant if checkpoint is QAT/INT8).")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size (used if checkpoint doesn't contain config.hidden_size).")

    parser.add_argument("--normalize-cm", action="store_true", help="Normalize confusion matrix rows.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Choose activations (independent of checkpoint training)
    if (not args.use_torch_acts) and (not HAS_ACTS):
        print("WARNING: Could not import 'acts'. Falling back to torch.sigmoid/tanh.")
    sigmoid_fn, tanh_fn = make_activation_fns(use_torch_acts=args.use_torch_acts)

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    hidden_size = int(cfg.get("hidden_size", args.hidden_size))
    qat_flag = bool(cfg.get("qat", False))

    # Detect exported INT8 checkpoint (from the training script) if present
    note = ckpt.get("note", "")
    is_int8_export = isinstance(note, str) and ("INT8" in note or "int8" in note)

    # Build base model on CPU first
    model = LSTMClassifier(
        input_size=28,
        hidden_size=hidden_size,
        num_classes=10,
        sigmoid_fn=sigmoid_fn,
        tanh_fn=tanh_fn,
        qat=qat_flag or is_int8_export,  # INT8 needs QuantStub/DeQuantStub structure
    )

    if qat_flag or is_int8_export:
        model = prepare_model_for_qat(model, backend=args.quant_backend)

    # If it's INT8 export: convert structure first, then load state_dict (packed params etc.)
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
    # - INT8 quantized models typically run on CPU (PyTorch quantized backends).
    # - Float/QAT-prepared models can run on CUDA if available.
    if is_int8_export:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_loader = build_test_loader(args.data_dir, args.batch_size, args.num_workers)

    acc, y_true, y_pred, correct_ex, wrong_ex = run_inference_collect(
        model, test_loader, device, max_collect_correct=9, max_collect_wrong=9
    )

    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=10)

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
    print(f"Hidden size: {hidden_size}")
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
