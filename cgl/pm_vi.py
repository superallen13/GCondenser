import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern"]
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 32
matplotlib.rcParams["font.weight"] = "heavy"
matplotlib.rcParams["axes.axisbelow"] = True


def main(args):
    # Load the matrix from the provided path
    pm = torch.load(args.pm_path)

    # Find the maximum length of the lists
    max_length = max(len(sublist) for sublist in pm)

    # Pad the lists with zeros
    padded_pm = [sublist + [0] * (max_length - len(sublist)) for sublist in pm]

    # Convert to a NumPy array
    pm_array = np.array(padded_pm)

    # Mask the upper triangle
    mask = np.tri(pm_array.shape[0], k=0)  # k=0 to include the diagonal
    pm_array = np.ma.array(pm_array, mask=~mask.astype(bool))

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(8, 8))  # Make the plot square

    # Plot the matrix
    im = ax.imshow(pm_array, aspect="equal")  # Ensure the plot is square

    # Customize the plot
    ax.set_xlabel("$\mathrm{Tasks}$")
    ax.set_ylabel("$\mathrm{Tasks}$")
    im.set_clim(vmin=0, vmax=1)  # Set the color limits

    # Ensure y-axis shows only integer values
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add colorbar
    cbar = fig.colorbar(im, ticks=[0, 0.5, 1.0])
    cbar.ax.tick_params()

    # Save the plot
    fig_path = os.path.join(os.path.dirname(args.pm_path), "pm_vi.pdf")
    plt.savefig(fig_path)
    print(f"Save the visualisation of performance matrix to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pm_path",
        type=str,
        default="./multirun/2024-05-15/10-15-49/1/performance_matrix.pt",
    )
    args = parser.parse_args()

    main(args)
