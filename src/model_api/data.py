# Hamiltonian Neural Networks (Modernized)
# Original: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)

import numpy as np
import scipy.integrate
import torch

solve_ivp = scipy.integrate.solve_ivp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hamiltonian_fn(coords: torch.Tensor) -> torch.Tensor:
    """Computes the Hamiltonian H = p^2 + q^2 for a batch or single state."""
    q, p = torch.chunk(coords, 2, dim=-1)
    H = p**2 + q**2  # Spring Hamiltonian (linear oscillator)
    return H.sum(dim=-1)


def dynamics_fn(t: float | None, coords: np.ndarray) -> np.ndarray:
    """Computes time derivatives (dq/dt, dp/dt) using PyTorch autograd.
    Supports both 1D (single state) and 2D (batched states) NumPy arrays.
    """
    coords_tensor = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    
    # Sum over batch dimension so autograd computes independent gradients per sample
    
    # compute the scalar energy H = p^2 + q^2 for every coordinate pair
    H = hamiltonian_fn(coords_tensor).sum()

    # compute the ordinary gradient: [dH/dq, dH/dp]
    dcoords = torch.autograd.grad(H, coords_tensor)[0]
    
    # split the gradient into its two partial derivative columns
    dqdt, dpdt = torch.chunk(dcoords, 2, dim=-1)

    # apply the symplectic 90-degree rotation: [dH/dp, -dH/dq]
    S = torch.cat([dpdt, -dqdt], dim=-1)
    return S.detach().numpy()


def get_trajectory(
    t_span: tuple[float, float] = (0.0, 3.0),
    timescale: int = 10,
    radius: float | None = None,
    y0: np.ndarray | None = None,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    num_steps = int(timescale * (t_span[1] - t_span[0]))
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)

    # Get initial state
    if y0 is None:
        y0 = rng.uniform(-1.0, 1.0, size=2)
    if radius is None:
        radius = rng.uniform(0.1, 1.0)
    
    y0 = y0 / np.linalg.norm(y0) * radius  # Normalize to target radius

    spring_ivp = solve_ivp(
        fun=dynamics_fn,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=1e-10,
        **kwargs,
    )
    q, p = spring_ivp.y[0], spring_ivp.y[1]

    # Batched gradient computation instead of a Python list comprehension
    dydt = dynamics_fn(None, spring_ivp.y.T).T
    dqdt, dpdt = np.split(dydt, 2, axis=0)

    # Add observation noise
    q = q + rng.standard_normal(q.shape) * noise_std
    p = p + rng.standard_normal(p.shape) * noise_std
    return q, p, dqdt.squeeze(0), dpdt.squeeze(0), t_eval


def get_dataset(
    seed: int = 0,
    samples: int = 50,
    test_split: float = 0.5,
    **kwargs,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, dxs = [], []

    for _ in range(samples):
        x, y, dx, dy, _ = get_trajectory(rng=rng, **kwargs)
        xs.append(np.stack([x, y], axis=-1))
        dxs.append(np.stack([dx, dy], axis=-1))

    x_all = np.concatenate(xs, axis=0)
    dx_all = np.concatenate(dxs, axis=0)

    # Train/test split
    split_ix = int(len(x_all) * test_split)
    return {
        "x": x_all[:split_ix],
        "test_x": x_all[split_ix:],
        "dx": dx_all[:split_ix],
        "test_dx": dx_all[split_ix:],
    }


def get_field(
    xmin: float = -1.2,
    xmax: float = 1.2,
    ymin: float = -1.2,
    ymax: float = 1.2,
    gridsize: int = 20,
) -> dict[str, np.ndarray]:
    b, a = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize),
    )
    ys = np.stack([b.flatten(), a.flatten()], axis=-1)

    # Vectorized evaluation across the entire grid at once
    dydt = dynamics_fn(None, ys)

    return {
        "x": ys,
        "dx": dydt,
    }

def spring_data():
    """Return some sample data from the ideal mass-spring experiment.

    Normally, this would be a subset of training data, but we have only 750 rows
    by default, so we just return all of it.
    """
    
    full_dataset = get_dataset()
    x = torch.tensor(full_dataset['test_x'], dtype=torch.float32).to(DEVICE)
    return x