# MLPs-Universal-Approximators

This project is a set of numerical experiments designed to empirically demonstrate the **Universal Approximation Theorem**. The goal was to see how a Multi-Layer Perceptron (MLP) with a simple architecture can "learn" and approximate various continuous functions without ever being given the underlying mathematical formula.

### How it works

The experiments use a standard MLP built in **PyTorch** with two hidden layers (32 neurons each) and `Tanh` activations. The model is trained using the **Adam optimizer** and **MSE Loss** to fit discrete $(x, y)$ data points generated from several target functions.

We tested the model against six distinct behaviors to see how well it adapts:

- **Smooth/Periodic:** $\sin(x)$ and damped oscillations ($\cos(2x) \cdot e^{-x^2/4}$).
- **Non-linear growth:** $x^2 \cdot \sin(x)$.
- **Sharp transitions:** $\tanh(5x)$.
- **Non-differentiable:** $|x|$ (to test how it handles the "pointy" bit at the origin).
- **Discontinuous:** A sawtooth-like function, which serves as a "negative control" since the theorem specifically applies to continuous functions.

### Visualizing Convergence

The code includes a script to capture the training process as it happens. You can find an example in `mlp_convergence.gif`, which shows the neural network's prediction "warping" and eventually snapping to the true function as the loss decreases.

### Setup

To run the experiments yourself, make sure you have the dependencies installed:

```bash
pip install -r requirements.txt
```
