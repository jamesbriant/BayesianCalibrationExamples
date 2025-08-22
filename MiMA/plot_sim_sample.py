import argparse
import os

import matplotlib.pyplot as plt
from dataloader import load
from plotting import plot_sim_sample


def main(experiment_name: str, data_dir: str, dpi: int = 300):
    kohdataset, tminmax, ycmean = load(file_name=experiment_name, data_dir=data_dir)

    # 6 Plotting
    # 6.0 Create the directory for figures if it doesn't exist
    print("Creating figures directory...")
    os.makedirs(os.path.join("figures", experiment_name), exist_ok=True)
    # if not os.path.exists("figures"):
    #     print("Creating figures directory...")
    #     os.makedirs("figures")

    # if not os.path.exists(f"figures/{experiment_name}"):
    #     print(f"Creating figures/{experiment_name} directory...")
    #     os.makedirs(f"figures/{experiment_name}")

    print("Plotting...")

    # 6.1 Plot sample of the data
    fig, ax = plot_sim_sample(
        kohdataset=kohdataset,
        tminmax=tminmax,
        ycmean=ycmean,
    )
    plt.savefig(f"figures/{experiment_name}/obs-and-sim-sample.png", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    # 0.1. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot simulation and observation data samples."
    )
    # 0.2. Add arguments
    parser.add_argument(
        "experiment_name",
        type=str,
        help="name of config file without extension (e.g., 'config_sin-a' or 'calib8)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where the data files are stored. Default is 'data'.",
    )
    # default dpi=300
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving figures. Default is 300.",
    )
    # 0.3. Parse the arguments
    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        data_dir=args.data_dir,
        dpi=args.dpi,
    )
