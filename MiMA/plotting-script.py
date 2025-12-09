import argparse
import os

import arviz as az
import datahandler
import jax
import matplotlib.pyplot as plt
import numpy as np
from kohgpjax.parameters import ModelParameters
from plotting import (
    gen_GP_test_data,
    gen_plot_test_data,
    gen_x_test_data,
    plot_f_delta,
    plot_f_eta,
    plot_f_zeta,
    plot_pairwise_samples,
    plot_posterior_chains_with_priors,
    plot_sim_sample,
)
from utils import load_config_from_path, load_model_from_config


def main(config_path: str, file_path: str, dpi: int = 300):
    # Load the config file
    config = load_config_from_path(config_path)
    Model, get_ModelParameterPriorDict = load_model_from_config(config)

    # experiment_name = file_path.split("/")[1]
    # file_name = file_path.split("/")[-1].split(".")[0]
    # file_extension = file_path.split("/")[-1].split(".")[-1]
    # print(f"File: {file_path}")

    # 1.1 Load the KOHDataset and normalization information
    kohdataset, tminmax, ycmean = datahandler.load(
        experiment_name=config.FILE_NAME,
        data_dir="data",
    )

    # 2.1 Load the MCMC inference dataset
    chains = az.from_netcdf(file_path)
    print(az.summary(chains))

    # 2.2 Extract the transformed parameters
    posterior_means = chains.posterior.mean()
    print(f"Posterior means: {posterior_means}")

    posterior_means_GP = posterior_means.copy()
    for i in range(config.N_CALIB_PARAMS):
        tmin, tmax = tminmax[f"theta_{i}"]
        # transform the theta values back to the [0,1] scale for the GP model
        theta_i_gp = (posterior_means[f"theta_{i}"].values - tmin) / (tmax - tmin)
        posterior_means_GP[f"theta_{i}"] = theta_i_gp
        print(f"Posterior theta_{i} mean (GP scale): {theta_i_gp}")

    # 3 Create the arrays for plotting
    # 3.1 Get the x variables
    xpred = gen_x_test_data()
    print(xpred.shape)

    # 3.2 Get the theta vector and tile for array
    theta_vec_plot = np.array([p["true_value"] for p in config.PARAMETERS])
    theta_vec_GP = np.array([p["true_value"] for p in config.PARAMETERS])
    for i in range(config.N_CALIB_PARAMS):
        theta_vec_plot[i] = posterior_means[f"theta_{i}"].values
        theta_vec_GP[i] = posterior_means_GP[f"theta_{i}"].values
    print(f"theta_vec: {theta_vec_plot}")
    print(f"theta_vec_GP: {theta_vec_GP}")

    # 3.3 Full x_test array
    x_test, theta_arr_plot = gen_plot_test_data(
        theta_vec=theta_vec_plot,
        num_points=1000,
    )
    print(x_test.shape)
    print(theta_arr_plot.shape)

    # 3.4 extract variables used for GP
    test_GP = gen_GP_test_data(
        theta_vec=theta_vec_GP,
        num_calib_params=config.N_CALIB_PARAMS,
    )
    print(test_GP.shape)

    # 3.5 Generate the full dataset
    dataset = kohdataset.get_dataset(
        theta_vec_GP[: config.N_CALIB_PARAMS].reshape(1, -1)
    )
    print(dataset)

    # 4 Build the GP models
    # 4.2 Get the prior dictionary and create model parameters
    prior_dict = get_ModelParameterPriorDict(config, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    # 4.3 Create the model instance
    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    # 4.4 Get the parameters into the correct format
    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    # cannot assume jax tree and arviz/xarray ordering is the same
    GP_params_flat = {p.name: posterior_means[p.name].values for p in prior_leaves}
    for i in range(config.N_CALIB_PARAMS):
        tmin, tmax = tminmax[f"theta_{i}"]
        GP_params_flat[f"theta_{i}"] = (GP_params_flat[f"theta_{i}"] - tmin) / (
            tmax - tmin
        )  # Normalise theta_i to [0, 1]

    # thetas = jnp.array(
    #     [GP_params_flat[f"theta_{i}"] for i in range(config.N_CALIB_PARAMS)]
    # )
    GP_params = jax.tree.unflatten(prior_tree, GP_params_flat.values())

    # 4.5 Build the GP posterior
    GP_posterior = model.GP_posterior(GP_params)
    # print(f"GP posterior: {GP_posterior}")

    # 5 Generating the predictions
    # 5.1 Predict the eta, zeta, and obs values
    print("Generating predictions...")
    eta_pred = GP_posterior.predict_eta(test_GP, dataset)
    zeta_pred = GP_posterior.predict_zeta(test_GP, dataset)
    obs_pred = GP_posterior.predict_obs(test_GP, dataset)

    eta_pred_m = eta_pred.mean
    eta_pred_cov = eta_pred.covariance_matrix

    zeta_pred_m = zeta_pred.mean
    zeta_pred_cov = zeta_pred.covariance_matrix

    # obs_pred_m = obs_pred.mean
    # obs_pred_cov = obs_pred.covariance_matrix

    # 5.2 Model discrepancy
    delta_gp_m = zeta_pred_m - eta_pred_m
    delta_gp_cov = zeta_pred_cov + eta_pred_cov

    # 6 Plotting
    # 6.0 Create the directory for figures if it doesn't exist
    output_dir = os.path.join("figures", config.FILE_NAME)
    if not os.path.exists(output_dir):
        print(f"Creating figures directory at {output_dir}...")
        os.makedirs(output_dir)

    print("Plotting...")

    # 6.1 Plot sample of the data
    fig, ax = plot_sim_sample(
        kohdataset=kohdataset,
        tminmax=tminmax,
        ycmean=ycmean,
    )
    plt.savefig(os.path.join(output_dir, "obs-and-sim-sample.png"), dpi=dpi)
    plt.close()

    # 6.2 Plot pairwise samples
    axes = plot_pairwise_samples(
        chains,
        [p.name for p in prior_leaves],
    )
    plt.savefig(os.path.join(output_dir, "pairs.png"), dpi=dpi)
    plt.close()

    # 6.3 Plot posterior chains with priors
    tracer_index_dict = {}
    for i, prior in enumerate(model_parameters.priors_flat):
        tracer_index_dict[prior.name] = i
    axes = plot_posterior_chains_with_priors(
        chains,
        config=config,
        model_parameters=model_parameters,
        tminmax=tminmax,
        tracer_index_dict=tracer_index_dict,
        figsize=(9, 20),
    )
    plt.savefig(os.path.join(output_dir, "trace.png"), dpi=dpi)
    plt.close()

    # 6.4 Plot f_eta
    fig, ax = plot_f_eta(
        config=config,
        x_test=x_test,
        test_GP=test_GP,
        thetas_test=theta_arr_plot,
        GP_eta=eta_pred,
        y_translation=ycmean,  # Add the mean of yc to center the observations
    )
    plt.savefig(os.path.join(output_dir, "eta-posterior.png"), dpi=dpi)
    plt.close()

    # 6.5 Plot f_zeta
    plot_f_zeta(
        config=config,
        x_test=x_test,
        test_GP=test_GP,
        thetas_test=theta_arr_plot,
        GP_zeta=zeta_pred,
        GP_zeta_epsilon=obs_pred,
        scatter_xf=kohdataset.Xf,
        scatter_yf=kohdataset.z,
        y_translation=ycmean,  # Add the mean of yc to center the observations
    )
    plt.savefig(os.path.join(output_dir, "zeta-posterior.png"), dpi=dpi)
    plt.close(fig)

    # 6.6 Plot f_delta
    plot_f_delta(
        config=config,
        x_test=x_test,
        test_GP=test_GP,
        thetas_test=theta_arr_plot,
        delta_gp_mean=delta_gp_m,
        delta_gp_cov=delta_gp_cov,
    )
    plt.savefig(os.path.join(output_dir, "discrepancy-posterior.png"), dpi=dpi)
    plt.close()


if __name__ == "__main__":
    # 0.1. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plotting script for KOHGPJax models from MCMC posterior data."
    )
    # 0.2. Add arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration module (e.g., configs/calib8.py)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="chains/",
        help="Path to the file containing the MCMC posterior data.",
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
        config_path=args.config,
        file_path=args.file_path,
        dpi=args.dpi,
    )
