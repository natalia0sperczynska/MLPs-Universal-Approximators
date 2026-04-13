
#Write a piece of code and use it to perform numerical experiments that empirically
#demonstrate that Multi-Layer Perceptrons  can approximate a large class of continuous functions.


#What is a Multi-Layer Perceptron?

#An MLP is a type of neural network consisting of at least three layers: 
# an input layer, one or more hidden layers, and an output layer.
# Each neuron in one layer is connected to every neuron in the next layer. 
# The MLP learns by adjusting the weights of these connections to 
# minimize the error of its predictions, using a process called backpropagation.

import base64

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch # ostrzegam ze dlugo sie instluje
import torch.nn as nn
import torch.optim as optim

from matplotlib import animation
from IPython.display import HTML # jak chcemy pokazac animacje w jupyterze, ale mozna to tez wyrzucic i po prostu zapisac gif-a, a potem pokazac go w raporcie

#wgl mysle ze mozna notebook zrobic z tego zeby wsystkie grafy i gify pokazac
     
seed = 777 # so we can reproduce the same results each time we run the code
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if avaiable, not really necessary for this small experiment but good practice

    
# they were generated using a prompt to .... , and they cover a range of behaviors (periodic, exponential decay, non-differentiable, etc.) to demonstrate the MLP's versatility
# mysle ze mozna napisac ze byl promt niz sie potem upadalc jak nas spyta jak wybralysmy
# name, function, and range pairs to test the MLP's approximation capabilities
TARGET_FUNCTIONS = {
    "sin(x)":             (lambda x: np.sin(x),                        (-6,  6)),
    "cos(2x)·e^(-x²/4)":  (lambda x: np.cos(2*x) * np.exp(-x**2 / 4),  (-4,  4)),
    "|x|":                (lambda x: np.abs(x),                        (-3,  3)),
    "x²·sin(x)":          (lambda x: x**2 * np.sin(x),                 (-5,  5)),
    "tanh(5x)":           (lambda x: np.tanh(5 * x),                   (-2,  2)),
    "sawtooth-like":      (lambda x: x - np.floor(x),                  ( 0,  4)),
}
# to call it we need to do TARGET_FUNCTIONS["sin(x)"][0](x) to get the function, and TARGET_FUNCTIONS["sin(x)"][1] to get the range, we can write a helper function to make this easier, but its not really necessary


# mozna dodac te funkcje do kodu nizej zeby bylo latwiej sie do nich odwolac
def get_function(name):
    fn, (x_min, x_max) = TARGET_FUNCTIONS[name]
    return fn, x_min, x_max

def get_all_functions():
    return {name: get_function(name) for name in TARGET_FUNCTIONS.keys()}

def get_function_range(name):
    _, (x_min, x_max) = TARGET_FUNCTIONS[name]
    return x_min, x_max



def generate_data(function, n=256, x_min=-2.0, x_max=2.0, noise_std=0.0):
    x = np.linspace(x_min, x_max, n).reshape(-1, 1).astype(np.float32) 
    # generate n points between x_min and x_max, reshape to be a column vector, and convert to float32 for better performance with PyTorch
    y = function(x).astype(np.float32)
    # we can leave this or throw it out
    # we can run it twice once with noise and once without to show the MLP's ability to fit both clean and noisy data
    # and write in the report that the tutorial we followed had this so we decided to try it
    if noise_std > 0:
        y = y + np.random.normal(0, noise_std, size=y.shape).astype(np.float32) # add some noise to make the approximation task more realistic

    x_t = torch.tensor(x, device=device)
    y_t = torch.tensor(y, device=device)
    return x, y, x_t, y_t


class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

def train(model, x_t, y_t, steps=2000, lr=1e-3, record_every=25):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {
        "step": [],
        "loss": [],
        "preds": []  # snapshots (for GIF) # if we dont want gif we can throw this out
    }

    for step in range(steps + 1):
        model.train()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % record_every == 0:
            model.eval()
            with torch.no_grad():
                pred_eval = model(x_t).detach().cpu().numpy()
            history["step"].append(step)
            history["loss"].append(float(loss.detach().cpu().item()))
            history["preds"].append(pred_eval)

    return model, history


def animate_training(history, x, y):
    # Prepare for animation
    fig, ax = plt.subplots()
    ax.set_title("MLP learning y = x^2 (convergence)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # True curve
    ax.plot(x.flatten(), y.flatten(), linewidth=2, label="True: x^2")

    # Pred curve (animated)
    pred_line, = ax.plot(x.flatten(), history["preds"][0].flatten(), linewidth=2, label="MLP prediction")

    ax.legend()

    def update(frame_idx):
        pred = history["preds"][frame_idx]
        pred_line.set_ydata(pred.flatten())
        ax.set_title(f"MLP learning y = x^2 — step {history['step'][frame_idx]} | loss {history['loss'][frame_idx]:.4e}")
        return (pred_line,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(history["preds"]), interval=80, blit=True
    )

    # Save GIF
    gif_path = "mlp_convergence_x2.gif"
    anim.save(gif_path, writer="pillow")
    plt.close(fig)


def visualize_results(): # its not only visualization and we also train the models so we can refactor it
    trained = []
    for name, fn in TARGET_FUNCTIONS.items():
        x, y, x_t, y_t = generate_data(fn[0], n=300, x_min=fn[1][0], x_max=fn[1][1])
        m = MLP(hidden=32)
        m, h = train(m, x_t, y_t, steps=2500, lr=1e-3, record_every=100)  # no need dense snapshots here
        with torch.no_grad():
            pred = m(x_t).detach().cpu().numpy()
        trained.append((name, x, y, pred))

     
    fig, axes = plt.subplots(nrows=len(TARGET_FUNCTIONS), ncols=2, figsize=(10, 14))
    fig.suptitle("True Function vs MLP Approximation", fontsize=16)

    # honestly we can change this os theres only the graph with both of them and throw out the one with only the true function, 
    # jak myslisz?

    # also naprawic bedzie trzeba visualiacje bo nachodza na siebie grafy

    for i, (name, x, y, pred) in enumerate(trained):
        ax_true = axes[i, 0]
        ax_nn   = axes[i, 1]

        ax_true.plot(x.flatten(), y.flatten(), linewidth=2)
        ax_true.set_title(f"{name} (True)")
        ax_true.set_xlabel("x")
        ax_true.set_ylabel("y")

        ax_nn.plot(x.flatten(), y.flatten(), linewidth=2, label="True")
        ax_nn.plot(x.flatten(), pred.flatten(), linewidth=2, label="MLP")
        ax_nn.set_title(f"{name} (MLP vs True)")
        ax_nn.set_xlabel("x")
        ax_nn.set_ylabel("y")
        ax_nn.legend()

    plt.tight_layout()
    plt.show()
     

# main function to demonstrate the MLP's ability to approximate continuous functions
# TO DO: we need to run this with different target functions and different hyperparameters
# and to run it with and wihtout noise to show the MLP's ability to fit both clean and noisy data (not mandatory)
def main():

    x, y, x_t, y_t = generate_data(TARGET_FUNCTIONS["sin(x)"][0], n=300, x_min=-2, x_max=2)

    model_sq = MLP(hidden=32)
    model_sq, hist_sq = train(model_sq, x_t, y_t, steps=2500, lr=1e-3, record_every=25)

  #  min(hist_sq["loss"]), hist_sq["loss"][-1] to show the loss decrease over time, we can plot it later
    animate_training(hist_sq, x, y)

    data = Path("mlp_convergence_x2.gif").read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    HTML(f'<img src="data:image/gif;base64,{b64}" />') # This is for notebook

    visualize_results()

     

if __name__ == "__main__":    main()