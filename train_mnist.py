import argparse
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    # Determinism can reduce speed; keep it off by default.
    torch.backends.cudnn.benchmark = True


# ----------------------------
# STE wrappers (optional)
# Forward uses approx function, backward uses the true derivative.
# Useful if acts.sigmoid/tanh have no backward defined.
# ----------------------------
class _SigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, approx_fn: Callable[[torch.Tensor], torch.Tensor]):
        ctx.save_for_backward(x)
        # forward with approx, but don't trust its autograd
        with torch.no_grad():
            y = approx_fn(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        grad_input = grad_output * s * (1.0 - s)
        return grad_input, None


class _TanhSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, approx_fn: Callable[[torch.Tensor], torch.Tensor]):
        ctx.save_for_backward(x)
        with torch.no_grad():
            y = approx_fn(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        t = torch.tanh(x)
        grad_input = grad_output * (1.0 - t * t)
        return grad_input, None


def make_activation_fns(
    use_torch_acts: bool,
    use_ste: bool,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:

    if use_torch_acts or not HAS_ACTS:
        return torch.sigmoid, torch.tanh

    if not hasattr(acts, "sigmoid") or not hasattr(acts, "tanh"):
        raise RuntimeError("Custom module 'acts' must provide acts.sigmoid and acts.tanh")

    if use_ste:
        def sigmoid_fn(x: torch.Tensor) -> torch.Tensor:
            return _SigmoidSTE.apply(x, acts.sigmoid)

        def tanh_fn(x: torch.Tensor) -> torch.Tensor:
            return _TanhSTE.apply(x, acts.tanh)

        return sigmoid_fn, tanh_fn

    return acts.sigmoid, acts.tanh


# ----------------------------
# A simple LSTM cell (explicit gates) so we can control activations.
# This also makes QAT easier to apply than nn.LSTM in many cases.
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid_fn = sigmoid_fn
        self.tanh_fn = tanh_fn

        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state  # (B,H), (B,H)
        gates = self.ih(x_t) + self.hh(h)  # (B,4H)
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

        # Quant/DeQuant stubs for QAT flow
        if qat:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        self.cell = CustomLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            sigmoid_fn=sigmoid_fn,
            tanh_fn=tanh_fn,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) where T=28, F=28 for MNIST rows-as-timesteps
        """
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
# Training / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total += x.size(0)

    return total_loss / total, total_correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0
    seen = 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * x.size(0)
        seen += x.size(0)

        if log_interval > 0 and step % log_interval == 0:
            avg = running / seen
            print(f"  step {step:5d}/{len(loader)} - train loss: {avg:.4f}")

    return running / seen


# ----------------------------
# Data
# MNIST image: (1,28,28). We convert to (28,28) sequence:
#   x_seq[t] is row t with 28 features.
# ----------------------------
class MNISTRowsAsSequence:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (1,28,28) float in [0,1]
        x = x.squeeze(0)          # (28,28)
        x = x.contiguous()        # (28,28)
        return x                  # interpreted as (T=28, F=28)


def build_loaders(
    data_dir: Optional[str],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    # Rule: download only if path not passed via CLI
    download = data_dir is None
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        MNISTRowsAsSequence(),
    ])

    if (not download) and (not os.path.exists(data_dir)):
        raise FileNotFoundError(
            f"--data-dir was provided but directory does not exist: {data_dir}\n"
            f"Either create it / place MNIST there, or omit --data-dir to auto-download."
        )

    train_ds = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, test_loader


# ----------------------------
# Checkpointing
# ----------------------------
@dataclass
class RunConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    seed: int
    qat: bool
    use_torch_acts: bool
    ste: bool


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: RunConfig,
    best_acc: float,
    epoch: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "best_acc": best_acc,
        "epoch": epoch,
    }
    torch.save(payload, path)


# ----------------------------
# QAT helpers
# ----------------------------
def prepare_model_for_qat(model: nn.Module, backend: str) -> nn.Module:
    """
    Prepares model for QAT. QAT runs with fake-quant during training.
    Note: actual INT8 inference typically runs on CPU backends (fbgemm/qnnpack).
    """
    # Choose quant backend
    torch.backends.quantized.engine = backend

    # Set a default qconfig for QAT (per-channel weights typically)
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # Insert observers/fake-quant modules
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_int8(model: nn.Module, backend: str) -> nn.Module:
    """
    Converts a QAT-prepared model to a quantized INT8 model (CPU).
    """
    torch.backends.quantized.engine = backend
    model_cpu = model.to("cpu").eval()
    model_int8 = torch.ao.quantization.convert(model_cpu, inplace=False)
    return model_int8


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a simple LSTM on MNIST (rows as sequence) with optional QAT.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to MNIST root. If omitted, MNIST will be downloaded into ./data.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--qat", action="store_true", help="Enable Quantization Aware Training (FakeQuant) via torch.ao.quantization.")
    parser.add_argument("--quant-backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"], help="Quantization backend (mostly affects INT8 conversion/inference on CPU).")

    parser.add_argument("--use-torch-acts", action="store_true", help="Use torch.sigmoid/tanh instead of acts.sigmoid/tanh.")
    parser.add_argument("--ste", action="store_true", help="Use straight-through estimator: forward uses acts.*, backward uses true derivatives.")

    parser.add_argument("--save", type=str, default="checkpoints/lstm_mnist.pt", help="Where to save the best checkpoint (state_dict + optimizer + config).")
    parser.add_argument("--export-int8", type=str, default="", help="If set and --qat is enabled, also export a converted INT8 checkpoint to this path.")

    args = parser.parse_args()

    cfg = RunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        qat=bool(args.qat),
        use_torch_acts=bool(args.use_torch_acts),
        ste=bool(args.ste),
    )

    if (not args.use_torch_acts) and (not HAS_ACTS):
        print("WARNING: Could not import 'acts'. Falling back to torch.sigmoid/tanh.")
    if args.ste and args.use_torch_acts:
        print("NOTE: --ste has no effect with --use-torch-acts.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sigmoid_fn, tanh_fn = make_activation_fns(
        use_torch_acts=args.use_torch_acts,
        use_ste=args.ste,
    )

    model = LSTMClassifier(
        input_size=28,
        hidden_size=args.hidden_size,
        num_classes=10,
        sigmoid_fn=sigmoid_fn,
        tanh_fn=tanh_fn,
        qat=bool(args.qat),
    )

    if args.qat:
        # Important: QAT prepares modules with observers/fake-quant.
        # Training can still happen on CUDA, conversion to INT8 is CPU-oriented.
        model = prepare_model_for_qat(model, backend=args.quant_backend)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_epoch = -1

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_interval)
        val_loss, val_acc = evaluate(model, test_loader, device)

        print(f"  train loss: {train_loss:.4f}")
        print(f"  val   loss: {val_loss:.4f} | val acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(args.save, model, optimizer, cfg, best_acc, epoch)
            print(f"  ✅ saved new best to: {args.save} (acc={best_acc*100:.2f}%)")

    print(f"\nDone. Best accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Best checkpoint: {args.save}")

    # Optional: export an INT8-converted model after QAT training
    if args.qat and args.export_int8:
        print("\nConverting to INT8 (CPU) and exporting...")
        model_int8 = convert_to_int8(model, backend=args.quant_backend)

        payload = {
            "model_state": model_int8.state_dict(),
            "config": asdict(cfg),
            "note": "INT8 model converted from QAT-trained model. Intended for CPU inference.",
            "backend": args.quant_backend,
        }
        os.makedirs(os.path.dirname(args.export_int8) or ".", exist_ok=True)
        torch.save(payload, args.export_int8)
        print(f"  ✅ exported INT8 checkpoint to: {args.export_int8}")


if __name__ == "__main__":
    main()
