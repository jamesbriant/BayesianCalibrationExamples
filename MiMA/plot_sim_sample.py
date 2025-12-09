import argparse
import os

import matplotlib.pyplot as plt
from datahandler import load
from plotting import plot_sim_sample
from utils import load_config_from_path


def main(
    config_path: str,
    dpi: int = 300,
    num_samples: int = None,
    alpha: float = 0.3,
    output_dir: str = None,
):
    config_module = load_config_from_path(config_path)
    experiment_config = config_module.experiment_config
    experiment_name = experiment_config.name

    kohdataset, tminmax, ycmean = load(
        experiment_config=experiment_config, data_root="data"
    )

    # 6 Plotting
    # 6.0 Create the directory for figures if it doesn't exist
    if output_dir:
        save_dir = output_dir
    else:
        print("Creating figures directory...")
        save_dir = os.path.join("figures", experiment_name)

    os.makedirs(save_dir, exist_ok=True)

    print("Plotting...")

    # 6.1 Plot sample of the data
    fig, ax = plot_sim_sample(
        kohdataset=kohdataset,
        tminmax=tminmax,
        ycmean=ycmean,
        num_samples=num_samples,
        alpha=alpha,
    )
    plt.savefig(os.path.join(save_dir, "obs-and-sim-sample.png"), dpi=dpi)
    plt.close()


if __name__ == "__main__":
    # 0.1. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot simulation and observation data samples."
    )
    # 0.2. Add arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    # default dpi=300
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving figures. Default is 300.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of simulation samples to plot. Default is all.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Transparency of simulation lines. Default is 0.3.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the plot",
    )
    # 0.3. Parse the arguments
    args = parser.parse_args()

    main(
        config_path=args.config,
        dpi=args.dpi,
        num_samples=args.num_samples,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
