# Hamiltonian Neural Networks (Modernized)
# Original: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)

from .data import spring_data

def sample_batch():
    return spring_data()

# L2_loss from the paper
# Note: this seems equivalent to MSELoss. I don't know why it's implemented as a separate function. --Clay
def L2_loss(u, v):
  return (u-v).pow(2).mean()
