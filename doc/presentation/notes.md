# HNNs: Hamiltonian Neural Networks

## Quick Overview of Hamiltonian Mechanics

* a branch of physics concerned with conservation laws and invariances
* describes a system where energy is conserved
* many systems in natural world have these properties
* *Greydanus* paper gives three examples:

    1. mass spring system
    2. pendulum
    3. orbiting bodies

In the simplest form, it's a system of one-dimensional movement of a mass. 

The function H(p, q) is the total energy of the system, which is the sum of kinetic and potential energy.

$H(q, p)$

## Quick Overview of Phase Space Plots

https://www.youtube.com/watch?v=TtDIz0j4oZc&t=176s

## What is a Hamiltonian Neural Network

A HNN is a neural network where we 


two coordinates: time and position can predict position for any future or
historical time

## The data and its symmetry
*We won't know your dataset, so show it: what one example is, how it's stored as
tensors, what the model predicts, and anything about the domain needed to follow
the rest. Then: what can you change about an input without changing the right
answer? Is the network built to be invariant (the output doesn't change) or
equivariant (the output changes in the same way)? Why does this matter for this
kind of data?*

Full credit if We understand your data by the end of it, and you state clearly
what the symmetry is, whether the network is invariant or equivariant, and why
it matters.

## The architecture, component by component, shown in your diagram
*Say what each component does and why it's there. Name the component(s) that
create the symmetry and explain why they create it.*

Full credit if The diagram is clear and matches the code. Each component's
purpose is explained, and you show why the symmetry-creating component creates
the symmetry.

## The backward pass
*How do gradients flow back through the component that creates the symmetry (a
sum, a max, message passing, a group convolution, …)? Where weights are shared,
what happens to their gradients?*

Full credit if You explain how gradients flow through the symmetry-creating
component and what weight sharing does to the gradients.

## Initialialization 

TODO Clay: lookup why nn_models.py:MLP uses orthogonal_(l.weight) initialization

*How is the network initialized? If it's the standard recipe, say so and move
on; if the symmetric component changes the picture, show how.*

Full credit if:  Standard choices are named briefly; anything unusual about this
network is explained. (Same rubric as Training)

## Training
*Loss, optimizer, learning-rate schedule, data and any augmentation. Is
augmentation still needed when the symmetry is built in, and if so, for what?*

Full credit if:  Standard choices are named briefly; anything unusual about this
network is explained. (Same rubric as Initialization)

## Impact of the architecture
*Your learning curves, data-efficiency results and symmetry test against the
baseline. What did the bias buy?*

Full credit if:  Learning curves, data efficiency and the symmetry test, against
a baseline, with a clear statement of what the bias bought.



## What it would be bad at
*Every bias costs something. Tell us what kind of data or task your network
would be bad at because of what it assumes, and why.*

Full credit if: A specific kind of data or task, and why the network's
assumption hurts there.




References:
Pendulum 30deg.gif, Lucas Vieira, Public domain, via Wikimedia Commons
<a href="https://commons.wikimedia.org/wiki/File:Pendulum_30deg.gif">Lucas Vieira</a>, Public domain, via Wikimedia Commons

Mass Spring Animación1.gif, AlvaroLopez12, CC BY-SA 4.0, via Wikimedia Commons
<a href="https://commons.wikimedia.org/wiki/File:Animaci%C3%B3n1.gif">AlvaroLopez12</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>, via Wikimedia Commons

Orbit2.gif, Zhatt, Public domain, via Wikimedia Commons
<a href="https://commons.wikimedia.org/wiki/File:Orbit2.gif">Zhatt</a>, Public domain, via Wikimedia Commons






--------------------------
$\frac{d\textbf{p}}{dt} = -\frac{\partial H}{\partial \textbf{q}}$

$\frac{d\textbf{q}}{dt} = \frac{\partial H}{\partial \textbf{p}},$



TODOs:
[ ] add header to each source file giving credit to Greydanus
[ ] add gen AI disclosure to slide deck
[ ] add loss curves
[ ] train baseline and HNN models and save weights in github repo
[ ] load model from saved weights
[ ] create HPC script
[ ] document HPC script in the README.md