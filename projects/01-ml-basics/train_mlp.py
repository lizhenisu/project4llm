import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Metrics:
    loss: float
    accuracy: float


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_dataset(n_samples: int, input_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=generator)
    true_w = torch.randn(input_dim, generator=generator)
    logits = x @ true_w + 0.3 * torch.randn(n_samples, generator=generator)
    y = (logits > 0).float()
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Metrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        pred = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return Metrics(loss=total_loss / total, accuracy=correct / total)


def build_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    dataset = make_dataset(args.samples, args.input_dim, args.seed)
    train_size = int(args.samples * 0.8)
    valid_size = args.samples - train_size
    train_set, valid_set = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)

    model = BinaryMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = total_loss / total
        valid = evaluate(model, valid_loader, criterion)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid.loss:.4f} "
            f"valid_acc={valid.accuracy:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
