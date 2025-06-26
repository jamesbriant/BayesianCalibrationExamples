import argparse
import arviz as az
import importlib
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from kohgpjax import KOHDataset
from kohgpjax.parameters import ModelParameters
import os

from plotting import (
    plot_f_eta,
    plot_f_delta,
    plot_f_zeta,
    plot_pairwise_samples,
    plot_posterior_chains_with_priors,
    plot_sim_sample,
)
import dataloader

file_name = "sin-ab"

from data.true_funcs import (
    discrepancy,
    eta,
    TrueParams,
    zeta,
)

TP = TrueParams()


def main():
    # 0.1. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plotting script for KOHGPJax models from MCMC posterior data."
    )
    # 0.2. Add arguments
    parser.add_argument(
        "file_path",
        type=str,
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
    # 0.4. Access the arguments
    file_path = args.file_path

    file_name = file_path.split("/")[-1].split(".")[0]
    file_extension = file_path.split("/")[-1].split(".")[-1]
    print(f"File: {file_path}")

    num_calib_params = len(file_name.split("-")[1])
    n_warm_up_iter = int(file_name.split("-")[2][1:])
    n_main_iter = int(file_name.split("-")[3][1:])

    # 1.1 Load the KOHDataset and normalization information
    file_path_csv = file_name.split("-")[1]
    kohdataset, tminmax, ycmean = dataloader.load(
        sim_file_path_csv=f"data/sim-{file_path_csv}.csv",
        obs_file_path_csv=f"data/obs-{file_path_csv}.csv",
        num_calib_params=num_calib_params,
    )

    # 2.1 Load the MCMC inference dataset
    chains = az.from_netcdf(file_path)
    print(az.summary(chains))

    # 2.2 Extract the transformed parameters
    theta_0 = jnp.mean(chains.posterior["theta_0"].values)
    posterior_means = chains.posterior.mean()
    print(f"Posterior means: {posterior_means}")

    # 3 Create the arrays for plotting
    # 3.1 Get the x variables
    x0_pred = np.linspace(0, 4, 1000)
    x1_pred = np.zeros_like(x0_pred)
    xpred = np.vstack((x0_pred, x1_pred)).T
    print(xpred.shape)

    # 3.2 Get the theta vector and tile for array
    theta_vec = TP.get_theta().squeeze()  # TODO: Should this be a jnp.array?
    for i in range(num_calib_params):
        theta_vec[i] = posterior_means[f"theta_{i}"].values
    theta_arr = jnp.tile(theta_vec, (xpred.shape[0], 1))
    print(theta_arr.shape)

    # 3.3 Full x_test array
    x_test = np.hstack((xpred, theta_arr))
    print(x_test.shape)

    # 3.4 extract variables used for GP
    x_test_GP = x_test[:, [0] + list(range(2, 2 + num_calib_params))]
    print(x_test_GP.shape)

    # 3.5 Generate the full dataset
    dataset = kohdataset.get_dataset(theta_vec[:num_calib_params].reshape(1, -1))
    print(dataset)

    # 4 Build the GP models
    # 4.1 Import the model module dynamically based on the file name
    module = importlib.import_module(f"models.sin_{file_name.split('-')[1]}")

    # 4.2 Get the prior dictionary and create model parameters
    prior_dict = module.get_ModelParameterPriorDict(tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    # 4.3 Create the model instance
    model = module.Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    # 4.4 Get the parameters into the correct format
    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    # cannot assume jax tree and arviz/xarray ordering is the same
    GP_params_flat = {p.name: posterior_means[p.name].values for p in prior_leaves}
    thetas = jnp.array([GP_params_flat[f"theta_{i}"] for i in range(num_calib_params)])
    GP_params = jax.tree.unflatten(prior_tree, GP_params_flat.values())

    # 4.5 Build the GP posterior
    GP_posterior = model.GP_posterior(GP_params)
    # print(f"GP posterior: {GP_posterior}")

    # 5 Generating the predictions
    # 5.1 Predict the eta, zeta, and obs values
    print("Generating predictions...")
    eta_pred = GP_posterior.predict_eta(x_test_GP, dataset)
    zeta_pred = GP_posterior.predict_zeta(x_test_GP, dataset)
    obs_pred = GP_posterior.predict_obs(x_test_GP, dataset)

    eta_pred_m = eta_pred.mean
    eta_pred_cov = eta_pred.covariance_matrix

    zeta_pred_m = zeta_pred.mean
    zeta_pred_cov = zeta_pred.covariance_matrix

    # 5.2 Model discrepancy
    delta_gp_m = zeta_pred_m - eta_pred_m
    delta_gp_cov = zeta_pred_cov + eta_pred_cov

    # 6 Plotting
    # 6.0 Create the directory for figures if it doesn't exist
    if not os.path.exists("figures"):
        print("Creating figures directory...")
        os.makedirs("figures")

    print("Plotting...")

    # 6.1 Plot sample of the data
    fig, ax = plot_sim_sample(
        kohdataset=kohdataset,
        tminmax=tminmax,
        ycmean=ycmean,
    )
    plt.savefig(f"figures/{file_name}-obs-and-sim-sample.png", dpi=args.dpi)
    plt.close()

    # 6.2 Plot pairwise samples
    axes = plot_pairwise_samples(
        chains,
        [p.name for p in prior_leaves],
    )
    plt.savefig(f"figures/{file_name}-pairs.png", dpi=args.dpi)
    plt.close()

    # 6.3 Plot posterior chains with priors
    tracer_index_dict = {}
    for i, prior in enumerate(model_parameters.priors_flat):
        tracer_index_dict[prior.name] = i
    TP.get_theta().squeeze()
    axes = plot_posterior_chains_with_priors(
        chains,
        model_parameters=model_parameters,
        tminmax=tminmax,
        tracer_index_dict=tracer_index_dict,
        true_values={"epsilon_precision": 1 / TP.obs_var}
        | {f"theta_{i}": TP.get_theta().squeeze()[i] for i in range(num_calib_params)},
        figsize=(9, 20),
    )
    plt.savefig(f"figures/{file_name}-trace.png", dpi=args.dpi)
    plt.close()

    # 6.4 Plot f_eta
    fig, ax = plot_f_eta(
        x_full=x_test,
        x_GP=x_test_GP[:, 0],
        thetas=thetas,
        thetas_full=theta_arr,
        eta=eta,
        GP_eta=eta_pred,
        y_translation=ycmean,  # Add the mean of yc to center the observations
    )
    plt.savefig(f"figures/{file_name}-eta-posterior.png", dpi=args.dpi)
    plt.close()

    # 6.5 Plot f_zeta
    fig, ax = plot_f_zeta(
        x_full=x_test,
        x_GP=x_test_GP[:, 0],
        zeta=zeta,
        GP_zeta=zeta_pred,
        GP_zeta_epsilon=obs_pred,
        scatter_xf=kohdataset.Xf,
        scatter_yf=kohdataset.z,
        y_translation=ycmean,  # Add the mean of yc to center the observations
    )
    plt.savefig(f"figures/{file_name}-zeta-posterior.png", dpi=args.dpi)
    plt.close(fig)

    # 6.6 Plot f_delta
    fig, ax = plot_f_delta(
        x_full=x_test,
        x_GP=x_test_GP[:, 0],
        thetas=thetas,
        thetas_full=theta_arr,
        zeta=zeta,
        eta=eta,
        delta=discrepancy,
        delta_gp_mean=delta_gp_m,
        delta_gp_cov=delta_gp_cov,
    )
    plt.savefig(f"figures/{file_name}-discrepancy-posterior.png", dpi=args.dpi)
    plt.close()


if __name__ == "__main__":
    main()
