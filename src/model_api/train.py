# Hamiltonian Neural Networks (Modernized)
# Original: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)

import argparse
from pathlib import Path

import numpy as np
import torch

from model_api.data import get_dataset
from model_api.models import HNN
from model_api.models import BaselineMLP

# Model and Training Constants
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = PACKAGE_DIR / "weights"

INPUT_DIM = 2
HIDDEN_DIM = 200
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NONLINEARITY = 'tanh'
TOTAL_STEPS = 2000
FIELD_TYPE = 'solenoidal'
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hamiltonian Neural Network")
    parser.add_argument("--name", default="spring", type=str, help="experiment name")
    parser.add_argument("--baseline", action="store_true", help="run baseline MLP instead of HNN")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument("--print_every", default=200, type=int, help="steps between console logs")
    parser.add_argument(
        "--save_dir",
        default=str(DEFAULT_WEIGHTS_DIR),
        type=str,
        help="directory to save trained model weights",
    )
    # parser.add_argument("--input_dim", default=2, type=int, help="dimensionality of input tensor")
    # parser.add_argument("--hidden_dim", default=200, type=int, help="hidden dimension of mlp")
    # parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
    # parser.add_argument("--nonlinearity", default="tanh", type=str, help="neural net nonlinearity")
    # parser.add_argument("--total_steps", default=2000, type=int, help="number of gradient steps")
    # parser.add_argument("--use_rk4", action="store_true", help="integrate derivative with RK4")
    # parser.add_argument("--field_type", default="solenoidal", type=str, help="type of vector field to learn")
    # parser.add_argument("--seed", default=0, type=int, help="random seed")
    return parser.parse_args()

def train(args: argparse.Namespace) -> tuple[HNN, dict[str, list[float]]]:
    
    torch.manual_seed(SEED)

    # Initialize model and optimizer
    output_dim = INPUT_DIM
    model = HNN(baseline=args.baseline).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

    # Prepare dataset tensors on target device
    data = get_dataset(seed=SEED)
    x = torch.tensor(data["x"], dtype=torch.float32, device=DEVICE, requires_grad=True)
    test_x = torch.tensor(data["test_x"], dtype=torch.float32, device=DEVICE, requires_grad=True)
    dxdt = torch.tensor(data["dx"], dtype=torch.float32, device=DEVICE)
    test_dxdt = torch.tensor(data["test_dx"], dtype=torch.float32, device=DEVICE)

    stats: dict[str, list[float]] = {"train_loss": [], "test_loss": []}

    for step in range(TOTAL_STEPS + 1):
        # Train step
        model.train()
        optim.zero_grad(set_to_none=True)

        dxdt_hat = model.time_derivative(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(dxdt, dxdt_hat)
        loss.backward()
        optim.step()

        # Evaluation step
        # Note: torch.no_grad() is intentionally omitted here because HNN.time_derivative()
        # requires autograd to compute gradients of the Hamiltonian with respect to test_x.
        model.eval()
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = criterion(test_dxdt, test_dxdt_hat)

        # Logging
        stats["train_loss"].append(loss.item())
        stats["test_loss"].append(test_loss.item())

        if args.verbose and step % args.print_every == 0:
            print(f"step {step}, train_loss {loss.item():.4e}, test_loss {test_loss.item():.4e}")

    # Final evaluation metrics
    model.eval()
    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2

    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2

    train_se = train_dist.std().item() / np.sqrt(train_dist.shape[0])
    test_se = test_dist.std().item() / np.sqrt(test_dist.shape[0])

    print(
        f"Final train loss {train_dist.mean().item():.4e} +/- {train_se:.4e}\n"
        f"Final test loss {test_dist.mean().item():.4e} +/- {test_se:.4e}"
    )

    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kind = "baseline" if args.baseline else "symmetric"
    save_path = save_dir / f"{kind}.pt"

    # Save CPU tensors so weights deserialize cleanly on both CPU and GPU machines
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state_dict, save_path)

    # Save a small test batch for sample_batch()
    sample_path = save_dir / "sample_batch.pt"
    if not sample_path.exists():
        data = get_dataset(seed=SEED)
        sample_x = torch.tensor(data["test_x"][:8], dtype=torch.float32)
        torch.save(sample_x, sample_path)

    if args.verbose:
        print(f"Saved {kind} model checkpoint to {save_path}")
        print(f"Sample batch available at {sample_path}")