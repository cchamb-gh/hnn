# AI 5200 - Project 2 - Hamiltonian Neural Networks
#### Taylor Turner and Clay Chambers

Based directly on the paper [Hamiltonian Neural Networks (2019)](https://arxiv.org/abs/1906.01563v3) by Greydanus et al and the accompanying source code [https://github.com/greydanus/hamiltonian-nn](https://github.com/greydanus/hamiltonian-nn).


## Overview

Hamiltonian Neural Networks are neural networks modified to work with harmonic systems of motion where energy is conserved. The network can learn the function giving the Hamiltonian $H(q, p)$ where q represents position and p represents momentum. HNNs are an improvement over baseline MLPs since they strictly conserve energy whereas a baseline MLP would tend to drift over time.

This repo only models the ideal mass-spring experiment from the Greydanus paper. More work is required to support pendulums and orbits.

## Installation and Usage


### Installation

    pip install git+https://github.com/cchamb-gh/hnn

### Training

#### Train the baseline model

    python -m model_api.train --baseline --verbose

#### Train the symmetric HNN model

    python -m model_api.train --verbose

### Testing

    python -m model_api.test_models

### Symmetry Test & Data-Efficiency Evaluation

    python -m model_api.evaluate
    python -m model_api.plot_results


