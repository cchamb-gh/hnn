"""Data-efficiency and symmetry evaluation for HNN vs Baseline MLP.

Runs two experiments against the repository's ideal mass-spring setup:
1. Symmetry test comparing the trained HNN ("symmetric") and Baseline MLP
   ("baseline") checkpoints on exact architectural invariants (Liouville
   divergence, Hamiltonian orthogonality) and rollout energy conservation,
   plus non-hard-coded domain symmetries (SO(2) rotation and reflection).
2. Data-efficiency sweep across training-set fractions (4%, 10%, 25%, 50%,
   100% of the 750-point training split) over multiple random seeds.

Usage:
    python -m model_api.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_api.data import get_dataset
from model_api.models import DEVICE, HNN, load_model
from model_api.train import LEARN_RATE, TOTAL_STEPS, WEIGHT_DECAY
from model_api.utils import L2_loss


DEFAULT_FRACTIONS = (0.04, 0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def exact_spring_rollout(y0_batch: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Analytic solution for H(q, p) = q^2 + p^2 (dq/dt = 2p, dp/dt = -2q)."""
    c = np.cos(2.0 * ts)[None, :]
    s = np.sin(2.0 * ts)[None, :]
    q = y0_batch[:, 0:1] * c + y0_batch[:, 1:2] * s
    p = -y0_batch[:, 0:1] * s + y0_batch[:, 1:2] * c
    return np.stack([q, p], axis=-1)


def rk4_rollout(
    model: HNN,
    y0_batch: np.ndarray,
    t_end: float = 30.0,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a batch of phase-space initial states with 4th-order Runge-Kutta."""
    model.eval()
    steps = int(round(t_end / dt))
    ts = np.linspace(0.0, t_end, steps + 1)
    y = torch.tensor(y0_batch, dtype=torch.float32, device=DEVICE)
    traj = [y.detach().cpu().numpy().copy()]

    for _ in range(steps):
        k1 = model.time_derivative(y).detach()
        k2 = model.time_derivative(y + 0.5 * dt * k1).detach()
        k3 = model.time_derivative(y + 0.5 * dt * k2).detach()
        k4 = model.time_derivative(y + dt * k3).detach()
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(y.detach().cpu().numpy().copy())

    return ts, np.stack(traj, axis=1)


def compute_divergence(model: HNN, points: torch.Tensor) -> tuple[float, float]:
    """Compute mean and max |div f| = |d(dq/dt)/dq + d(dp/dt)/dp|."""
    was_training = model.training
    # HNN.time_derivative sets create_graph=self.training; enable train mode so
    # second-order derivatives with respect to inputs remain in the graph.
    model.train()
    x = points.detach().clone().to(DEVICE).requires_grad_(True)
    out = model.time_derivative(x)
    dq_dq = torch.autograd.grad(out[:, 0].sum(), x, retain_graph=True)[0][:, 0]
    dp_dp = torch.autograd.grad(out[:, 1].sum(), x)[0][:, 1]
    div = (dq_dq + dp_dp).abs()
    model.train(was_training)
    return float(div.mean().item()), float(div.max().item())


def compute_hnn_orthogonality(model: HNN, points: torch.Tensor) -> tuple[float, float]:
    """Compute mean and max |dH_hat/dt| = |grad(H_hat) . f_hat| for an HNN."""
    x = points.detach().clone().to(DEVICE).requires_grad_(True)
    _, h_hat = model._potentials(x)
    grad_h = torch.autograd.grad(h_hat.sum(), x)[0]
    f_hat = model.time_derivative(x.detach().clone().requires_grad_(True))
    dh_dt = (grad_h * f_hat).sum(dim=-1).abs()
    return float(dh_dt.mean().item()), float(dh_dt.max().item())


def compute_domain_symmetries(model: HNN, points: torch.Tensor) -> dict[str, float]:
    """Evaluate approximate domain symmetries not hard-coded in HNN: SO(2) and reflection."""
    model.eval()
    x = points.detach().clone().to(DEVICE)
    fx = model.time_derivative(x).detach()

    # SO(2) phase-space rotation equivariance: f(R x) == R f(x)
    thetas = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)[1:]
    rot_mses = []
    for theta in thetas:
        rot = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=torch.float32,
            device=DEVICE,
        )
        f_rx = model.time_derivative(x @ rot.T).detach()
        r_fx = fx @ rot.T
        rot_mses.append(float(((f_rx - r_fx) ** 2).mean().item()))

    # Time-reversal / momentum-reflection symmetry: (q, p) -> (q, -p) implies
    # (dq/dt, dp/dt) -> (-dq/dt, dp/dt)
    refl_in = torch.tensor([1.0, -1.0], dtype=torch.float32, device=DEVICE)
    refl_out = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=DEVICE)
    f_refl = model.time_derivative(x * refl_in).detach()
    refl_mse = float(((f_refl - fx * refl_out) ** 2).mean().item())

    return {
        "so2_rotation_mse": float(np.mean(rot_mses)),
        "momentum_reflection_mse": refl_mse,
    }


def make_eval_initial_states() -> np.ndarray:
    """Deterministic grid of 20 clean initial conditions across radii [0.3, 1.0]."""
    radii = [0.3, 0.5, 0.7, 0.9, 1.0]
    angles = [0.0, 1.0, 2.5, 4.2]
    return np.array(
        [[r * np.cos(a), r * np.sin(a)] for r in radii for a in angles],
        dtype=np.float64,
    )


def evaluate_rollouts(
    model: HNN,
    y0_batch: np.ndarray,
    horizons: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0),
    dt: float = 0.02,
) -> dict[str, Any]:
    """Integrate model trajectories up to max(horizons) and summarize errors."""
    t_max = max(horizons)
    ts, pred_traj = rk4_rollout(model, y0_batch, t_end=t_max, dt=dt)
    true_traj = exact_spring_rollout(y0_batch, ts)

    e0 = np.sum(y0_batch**2, axis=-1, keepdims=True)
    r0 = np.sqrt(e0)
    e_pred = np.sum(pred_traj**2, axis=-1)
    r_pred = np.sqrt(e_pred)

    horizon_stats: dict[str, dict[str, float]] = {}
    for t_h in horizons:
        idx = int(round(t_h / dt)) + 1
        coord_mse = float(np.mean((pred_traj[:, :idx] - true_traj[:, :idx]) ** 2))
        energy_mse = float(np.mean((e_pred[:, :idx] - e0) ** 2))
        mean_max_edrift = float(np.mean(np.max(np.abs(e_pred[:, :idx] - e0), axis=-1)))
        mean_max_rdrift = float(np.mean(np.max(np.abs(r_pred[:, :idx] - r0), axis=-1)))
        final_edrift = float(np.mean(np.abs(e_pred[:, idx - 1] - e0[:, 0])))
        horizon_stats[f"{t_h:g}"] = {
            "coord_mse": coord_mse,
            "energy_mse": energy_mse,
            "mean_max_energy_drift": mean_max_edrift,
            "mean_max_radius_drift": mean_max_rdrift,
            "mean_final_energy_drift": final_edrift,
        }

    # Learned Hamiltonian conservation (for HNN)
    learned_h_stats = None
    if not model.baseline:
        with torch.no_grad():
            pts_flat = torch.tensor(
                pred_traj.reshape(-1, 2), dtype=torch.float32, device=DEVICE
            )
            _, h_flat = model._potentials(pts_flat)
            h_vals = h_flat.cpu().numpy().reshape(len(y0_batch), -1)
            h_diff = h_vals - h_vals[:, 0:1]
            learned_h_stats = {
                "mse": float(np.mean(h_diff**2)),
                "max_abs_drift": float(np.max(np.abs(h_diff))),
            }

    # Save a single representative rollout from y0 = [1.0, 0.0] (index 16)
    rep_idx = 16
    stride = 5  # dt=0.1 for compact plotting
    rep_t = ts[::stride]
    rep_pred = pred_traj[rep_idx, ::stride]
    rep_true = true_traj[rep_idx, ::stride]
    rep_energy = np.sum(rep_pred**2, axis=-1)

    return {
        "horizons": horizon_stats,
        "learned_hamiltonian_drift": learned_h_stats,
        "representative_rollout": {
            "y0": y0_batch[rep_idx].tolist(),
            "t": rep_t.tolist(),
            "q": rep_pred[:, 0].tolist(),
            "p": rep_pred[:, 1].tolist(),
            "true_q": rep_true[:, 0].tolist(),
            "true_p": rep_true[:, 1].tolist(),
            "energy": rep_energy.tolist(),
        },
    }


def run_symmetry_tests() -> dict[str, Any]:
    """Run architectural and rollout symmetry tests on trained and untrained models."""
    sym = load_model("symmetric")
    base = load_model("baseline")
    assert sym.differentiable_model is not base.differentiable_model

    torch.manual_seed(999)
    untrained_sym = HNN(baseline=False).to(DEVICE)
    untrained_base = HNN(baseline=True).to(DEVICE)

    rng = np.random.default_rng(42)
    pts_id = torch.tensor(rng.uniform(-1.0, 1.0, size=(500, 2)), dtype=torch.float32)
    pts_ood = torch.tensor(rng.uniform(-2.0, 2.0, size=(500, 2)), dtype=torch.float32)

    y0_batch = make_eval_initial_states()

    results: dict[str, Any] = {"trained": {}, "untrained": {}}

    for label, model in [("symmetric", sym), ("baseline", base)]:
        div_id_mean, div_id_max = compute_divergence(model, pts_id)
        div_ood_mean, div_ood_max = compute_divergence(model, pts_ood)
        domain_sym = compute_domain_symmetries(model, pts_id)
        rollout = evaluate_rollouts(model, y0_batch, horizons=(3.0, 10.0, 30.0, 100.0))

        entry: dict[str, Any] = {
            "divergence_in_dist": {"mean": div_id_mean, "max": div_id_max},
            "divergence_ood": {"mean": div_ood_mean, "max": div_ood_max},
            "domain_symmetries": domain_sym,
            "rollout": rollout,
        }
        if not model.baseline:
            orth_mean, orth_max = compute_hnn_orthogonality(model, pts_id)
            entry["orthogonality_dH_dt"] = {"mean": orth_mean, "max": orth_max}
        results["trained"][label] = entry

    for label, model in [("symmetric", untrained_sym), ("baseline", untrained_base)]:
        div_id_mean, div_id_max = compute_divergence(model, pts_id)
        entry = {"divergence_in_dist": {"mean": div_id_mean, "max": div_id_max}}
        if not model.baseline:
            orth_mean, orth_max = compute_hnn_orthogonality(model, pts_id)
            entry["orthogonality_dH_dt"] = {"mean": orth_mean, "max": orth_max}
        results["untrained"][label] = entry

    return results


def train_model_on_subset(
    x_train_np: np.ndarray,
    dx_train_np: np.ndarray,
    baseline: bool,
    seed: int,
    total_steps: int = TOTAL_STEPS,
) -> HNN:
    """Train a fresh HNN or Baseline MLP with the repository's default hyperparameters."""
    torch.manual_seed(seed)
    model = HNN(baseline=baseline).to(DEVICE)
    optim = torch.optim.Adam(
        model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY
    )

    x = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    dxdt = torch.tensor(dx_train_np, dtype=torch.float32, device=DEVICE)

    for _ in range(total_steps + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = L2_loss(dxdt, model.time_derivative(x))
        loss.backward()
        optim.step()

    model.eval()
    return model


def run_data_efficiency(
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    total_steps: int = TOTAL_STEPS,
) -> dict[str, Any]:
    """Sweep training-set sizes across multiple seeds for HNN and Baseline MLP."""
    # Fixed clean test set (25 noiseless trajectories = 750 points) for evaluating
    # true vector-field accuracy without the 0.04 input-noise Bayes floor.
    clean_test = get_dataset(seed=1000, samples=50, test_split=0.5, noise_std=0.0)
    clean_x = torch.tensor(clean_test["test_x"], dtype=torch.float32, device=DEVICE)
    clean_dx = torch.tensor(clean_test["test_dx"], dtype=torch.float32, device=DEVICE)

    # Fixed noisy test set matching train.py (noise_std=0.1)
    noisy_test = get_dataset(seed=1000, samples=50, test_split=0.5, noise_std=0.1)
    noisy_x = torch.tensor(noisy_test["test_x"], dtype=torch.float32, device=DEVICE)
    noisy_dx = torch.tensor(noisy_test["test_dx"], dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(42)
    div_pts = torch.tensor(rng.uniform(-1.0, 1.0, size=(300, 2)), dtype=torch.float32)
    y0_batch = make_eval_initial_states()

    per_fraction: list[dict[str, Any]] = []

    for frac in fractions:
        n_points = int(round(frac * 750))
        metrics_by_kind: dict[str, dict[str, list[float]]] = {
            "symmetric": {
                "clean_vf_mse": [],
                "noisy_vf_mse": [],
                "rollout_t3_coord_mse": [],
                "rollout_t30_coord_mse": [],
                "rollout_t30_energy_mse": [],
                "rollout_t30_max_edrift": [],
                "mean_divergence": [],
            },
            "baseline": {
                "clean_vf_mse": [],
                "noisy_vf_mse": [],
                "rollout_t3_coord_mse": [],
                "rollout_t30_coord_mse": [],
                "rollout_t30_energy_mse": [],
                "rollout_t30_max_edrift": [],
                "mean_divergence": [],
            },
        }

        for seed in seeds:
            ds = get_dataset(seed=seed, samples=50, test_split=0.5, noise_std=0.1)
            x_sub = ds["x"][:n_points]
            dx_sub = ds["dx"][:n_points]

            for kind, is_base in [("symmetric", False), ("baseline", True)]:
                model = train_model_on_subset(
                    x_sub, dx_sub, baseline=is_base, seed=seed, total_steps=total_steps
                )
                model.eval()
                clean_pred = model.time_derivative(clean_x).detach()
                noisy_pred = model.time_derivative(noisy_x).detach()
                clean_mse = float(L2_loss(clean_dx, clean_pred).item())
                noisy_mse = float(L2_loss(noisy_dx, noisy_pred).item())

                div_mean, _ = compute_divergence(model, div_pts)
                roll = evaluate_rollouts(model, y0_batch, horizons=(3.0, 30.0))

                bucket = metrics_by_kind[kind]
                bucket["clean_vf_mse"].append(clean_mse)
                bucket["noisy_vf_mse"].append(noisy_mse)
                bucket["rollout_t3_coord_mse"].append(roll["horizons"]["3"]["coord_mse"])
                bucket["rollout_t30_coord_mse"].append(
                    roll["horizons"]["30"]["coord_mse"]
                )
                bucket["rollout_t30_energy_mse"].append(
                    roll["horizons"]["30"]["energy_mse"]
                )
                bucket["rollout_t30_max_edrift"].append(
                    roll["horizons"]["30"]["mean_max_energy_drift"]
                )
                bucket["mean_divergence"].append(div_mean)

        summary_entry: dict[str, Any] = {
            "fraction": frac,
            "percent": int(round(frac * 100)),
            "n_points": n_points,
            "n_trajectories": round(n_points / 30.0, 2),
        }
        for kind in ("symmetric", "baseline"):
            summary_entry[kind] = {}
            for m_name, vals in metrics_by_kind[kind].items():
                arr = np.asarray(vals, dtype=np.float64)
                summary_entry[kind][m_name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "values": [float(v) for v in arr],
                }
        per_fraction.append(summary_entry)
        print(
            f"[{int(round(frac*100)):3d}% | {n_points:3d} pts] "
            f"Clean VF MSE: HNN={summary_entry['symmetric']['clean_vf_mse']['mean']:.5f} "
            f"vs Base={summary_entry['baseline']['clean_vf_mse']['mean']:.5f} | "
            f"t=30 Energy MSE: HNN={summary_entry['symmetric']['rollout_t30_energy_mse']['mean']:.6f} "
            f"vs Base={summary_entry['baseline']['rollout_t30_energy_mse']['mean']:.6f}",
            flush=True,
        )

    return {
        "seeds": list(seeds),
        "total_steps": total_steps,
        "train_noise_std": 0.1,
        "fractions": per_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HNN vs Baseline MLP symmetry and data-efficiency experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="doc/presentation/assets/evaluation_results.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
        help="Training steps per data-efficiency run",
    )
    args = parser.parse_args()

    print("=== Running Symmetry Tests (Saved Checkpoints & Untrained) ===", flush=True)
    symmetry = run_symmetry_tests()
    for kind in ("symmetric", "baseline"):
        tr = symmetry["trained"][kind]
        print(
            f"  Trained {kind:9s} | div(ID)={tr['divergence_in_dist']['mean']:.3e} "
            f"| div(OOD)={tr['divergence_ood']['mean']:.3e} "
            f"| t=30 E-MSE={tr['rollout']['horizons']['30']['energy_mse']:.6f} "
            f"| t=100 E-MSE={tr['rollout']['horizons']['100']['energy_mse']:.6f}",
            flush=True,
        )

    print("\n=== Running Data-Efficiency Sweep ===", flush=True)
    data_eff = run_data_efficiency(total_steps=args.steps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"symmetry": symmetry, "data_efficiency": data_eff}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved evaluation results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
