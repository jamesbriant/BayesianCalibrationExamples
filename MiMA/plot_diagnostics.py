import argparse
import glob
import os

import arviz as az
import matplotlib.pyplot as plt
from kohgpjax.parameters import ModelParameters

from datahandler import load
from plotting import plot_posterior_chains_with_priors
from utils import load_config_from_model_dir, load_model_from_model_dir


def main(model_dir, output_dir=None):
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    experiment_name = experiment_config.name

    if output_dir:
        # If output_dir is provided, look for chains there
        chains_dir = output_dir
    else:
        # Fallback to old behavior
        chains_dir = os.path.join("chains", experiment_name)

    # Find latest nc file
    nc_files = sorted(glob.glob(os.path.join(chains_dir, "*.nc")))
    if not nc_files:
        print(f"No chains found in {chains_dir}")
        return

    latest_nc = nc_files[-1]
    print(f"Loading chains from {latest_nc}")
    idata = az.from_netcdf(latest_nc)

    print("Summary:")
    print(az.summary(idata))

    # Load model and priors for plotting
    kohdataset, tminmax, yc_mean = load(experiment_config, "data")
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)
    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    # Create tracer index dict
    tracer_index_dict = {}
    for i, prior in enumerate(model_parameters.priors_flat):
        tracer_index_dict[prior.name] = i

    # Create diagnostics directory if it doesn't exist
    diag_dir = os.path.join(chains_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # Plot trace with priors
    plot_posterior_chains_with_priors(
        idata,
        config=config_module,
        model_parameters=model_parameters,
        tminmax=tminmax,
        tracer_index_dict=tracer_index_dict,
        figsize=(12, 16),
    )
    plt.savefig(os.path.join(diag_dir, "trace_plot.png"))
    plt.close()

    # Plot autocorrelation
    az.plot_autocorr(idata)
    plt.savefig(os.path.join(diag_dir, "autocorr_plot.png"))
    plt.close()

    # Plot ESS
    az.plot_ess(idata)
    plt.savefig(os.path.join(diag_dir, "ess_plot.png"))
    plt.close()

    # Plot pair (robust to degenerate densities)
    try:
        az.plot_pair(idata, kind="kde", divergences=True)
    except Exception:
        az.plot_pair(idata, kind="scatter", divergences=False)
    plt.savefig(os.path.join(diag_dir, "pair_plot.png"))
    plt.close()

    print(f"Plots saved to {diag_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory containing the chain file"
    )
    args = parser.parse_args()
    main(args.model_dir, args.output_dir)
    main(args.model_dir, args.output_dir)
