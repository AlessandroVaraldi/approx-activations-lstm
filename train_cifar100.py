import argparse
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple, List

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
    torch.backends.cudnn.benchmark = True  # faster; not fully deterministic


# ----------------------------
# STE wrappers (optional)
# Forward uses approx function, backward uses the true derivative.
# ----------------------------
class _SigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, approx_fn: Callable[[torch.Tensor], torch.Tensor]):
        ctx.save_for_backward(x)
        with torch.no_grad():
            y = approx_fn(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_output * s * (1.0 - s), None


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
        return grad_output * (1.0 - t * t), None


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
# Patchify transform: image (C,H,W) -> sequence (T, F)
# CIFAR: C=3, H=W=32. With patch=4 => (8*8)=64 tokens, each token has F=3*4*4=48 features.
# ----------------------------
class CIFARToPatches:
    def __init__(self, patch: int = 4):
        assert 32 % patch == 0, "For CIFAR 32x32, patch must divide 32."
        self.patch = patch

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3,32,32)
        C, H, W = x.shape
        p = self.patch
        gh, gw = H // p, W // p  # grid
        # (C, gh, p, gw, p) -> (gh, gw, C, p, p) -> (T, C*p*p)
        x = x.view(C, gh, p, gw, p).permute(1, 3, 0, 2, 4).contiguous()
        x = x.view(gh * gw, C * p * p)
        return x  # (T, F)


# ----------------------------
# Custom LSTM cell (explicit gates) so we can control activations.
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
        h, c = state  # (B,H)
        gates = self.ih(x_t) + self.hh(h)  # (B,4H)
        i, f, g, o = gates.chunk(4, dim=1)

        i = self.sigmoid_fn(i)
        f = self.sigmoid_fn(f)
        g = self.tanh_fn(g)
        o = self.sigmoid_fn(o)

        c_new = f * c + i * g
        h_new = o * self.tanh_fn(c_new)
        return h_new, c_new


# ----------------------------
# Stacked (multi-layer) Custom LSTM
# ----------------------------
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

        # init states per layer
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


# ----------------------------
# Model: Patch embedding + CustomLSTM + pooling + classifier
# ----------------------------
class PatchLSTMClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        token_dim: int,      # F = 3*patch*patch
        embed_dim: int,      # projected token dim
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        sigmoid_fn: Callable[[torch.Tensor], torch.Tensor],
        tanh_fn: Callable[[torch.Tensor], torch.Tensor],
        dropout: float = 0.1,
        qat: bool = False,
    ):
        super().__init__()
        self.qat = qat

        if qat:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        # token projection + small MLP for capacity
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

        # simple attention pooling over time to be a bit stronger than "last token"
        self.attn = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) from CIFARToPatches
        """
        if self.qat:
            x = self.quant(x)

        x = self.token_embed(x)          # (B,T,E)
        h_seq = self.lstm(x)             # (B,T,H)

        # attention pooling: softmax over T
        a = self.attn(h_seq).squeeze(-1)           # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,T,1)
        h = (h_seq * w).sum(dim=1)                 # (B,H)

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
# CIFAR-100 -> patches sequence
# ----------------------------
def build_loaders(
    data_dir: Optional[str],
    batch_size: int,
    num_workers: int,
    patch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    download = data_dir is None
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    if (not download) and (not os.path.exists(data_dir)):
        raise FileNotFoundError(
            f"--data-dir was provided but directory does not exist: {data_dir}\n"
            f"Either create it / place CIFAR-100 there, or omit --data-dir to auto-download."
        )

    # CIFAR normalization (common)
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        CIFARToPatches(patch=patch_size),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        CIFARToPatches(patch=patch_size),
    ])

    train_ds = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_tf)
    test_ds = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=test_tf)

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
    patch_size: int
    embed_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
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
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Patch-LSTM on CIFAR-100 with optional custom activations + STE + QAT."
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset root. If omitted, CIFAR-100 will be downloaded into ./data.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)

    # model capacity
    parser.add_argument("--patch-size", type=int, default=4, choices=[2, 4, 8],
                        help="Patch size. For CIFAR32, 4 -> 64 tokens, 2 -> 256 tokens (heavier), 8 -> 16 tokens.")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)

    # quantization
    parser.add_argument("--qat", action="store_true",
                        help="Enable Quantization Aware Training (FakeQuant) via torch.ao.quantization.")
    parser.add_argument("--quant-backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"],
                        help="Quantization backend (affects INT8 conversion/inference on CPU).")

    # activations
    parser.add_argument("--use-torch-acts", action="store_true",
                        help="Use torch.sigmoid/tanh instead of acts.sigmoid/tanh.")
    parser.add_argument("--ste", action="store_true",
                        help="Use STE: forward uses acts.*, backward uses true derivatives.")

    # io
    parser.add_argument("--save", type=str, default="checkpoints/patchlstm_cifar100.pt",
                        help="Where to save the best checkpoint.")
    parser.add_argument("--export-int8", type=str, default="",
                        help="If set and --qat enabled, also export a converted INT8 checkpoint to this path.")

    args = parser.parse_args()

    cfg = RunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
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
        patch_size=args.patch_size,
    )

    sigmoid_fn, tanh_fn = make_activation_fns(
        use_torch_acts=args.use_torch_acts,
        use_ste=args.ste,
    )

    token_dim = 3 * args.patch_size * args.patch_size  # RGB patches
    model = PatchLSTMClassifier(
        patch_size=args.patch_size,
        token_dim=token_dim,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=100,  # CIFAR-100
        sigmoid_fn=sigmoid_fn,
        tanh_fn=tanh_fn,
        dropout=args.dropout,
        qat=bool(args.qat),
    )

    if args.qat:
        model = prepare_model_for_qat(model, backend=args.quant_backend)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # (opzionale) scheduler per stabilità su CIFAR-100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = -1

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_interval)
        val_loss, val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  lr: {lr_now:.6f}")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val   loss: {val_loss:.4f} | val acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(args.save, model, optimizer, cfg, best_acc, epoch)
            print(f"  ✅ saved new best to: {args.save} (acc={best_acc*100:.2f}%)")

    print(f"\nDone. Best accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Best checkpoint: {args.save}")

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
