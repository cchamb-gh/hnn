from importlib.resources import files

import torch
from torch import nn
import numpy as np

# Model and Training Constants
INPUT_DIM = 2
HIDDEN_DIM = 200
LEARN_RATE = 1e-3
NONLINEARITY = 'tanh'
TOTAL_STEPS = 2000
FIELD_TYPE = 'solenoidal'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM):
        super(BaselineMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            nn.init.orthogonal_(l.weight) # use a principled initialization
        
        self.nonlinearity = torch.tanh


    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)



class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim=INPUT_DIM, differentiable_model=BaselineMLP(), field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.register_buffer("M", self.permutation_tensor(input_dim))
        self.field_type = field_type

    def _potentials(self, x):
        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def forward(self, x):
        # Both baseline and symmetric now return the vector field tensor [batch_size, input_dim]
        return self.time_derivative(x)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STYLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STYLE VECTOR FIELD'''
        # Ensure gradients wrt x work even during model.eval() or inside torch.no_grad()
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

            F1, F2 = self._potentials(x)

            conservative_field = torch.zeros_like(x)
            solenoidal_field = torch.zeros_like(x)

            if self.field_type != 'solenoidal':
                dF1 = torch.autograd.grad(F1.sum(), x, create_graph=self.training)[0]
                conservative_field = dF1 @ torch.eye(*self.M.shape, device=x.device, dtype=x.dtype)

            if self.field_type != 'conservative':
                dF2 = torch.autograd.grad(F2.sum(), x, create_graph=self.training)[0]
                solenoidal_field = dF2 @ self.M.to(dtype=x.dtype).t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M




def load_model(kind):
    assert kind == "baseline" or kind == "symmetric", "must specify symmetric or baseline"

    if kind == "baseline":
        model = HNN(baseline=True)
    elif kind == "symmetric":
        model = HNN(baseline=False)

    weight_resource = files("model_api").joinpath(f"weights/{kind}.pt")
    if weight_resource.is_file():
        print(f"loading {kind} from saved weights {str(weight_resource)}")
        with weight_resource.open("rb") as f:
            state_dict = torch.load(f, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)

    return model.to(DEVICE)