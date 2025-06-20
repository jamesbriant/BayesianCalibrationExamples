from jax import config

config.update("jax_enable_x64", True)

import arviz
import gpjax as gpx
from gpjax.distributions import GaussianDistribution
import jax
import jax.numpy as jnp
import kohgpjax as kgx
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import (
    ParameterPrior,
    ModelParameterPriorDict,
    ModelParameters,
)
import matplotlib.pyplot as plt
import mici
import numpy as np
import numpyro.distributions as dist

from plotting import (
    plot_f_eta,
    plot_f_delta,
    plot_f_zeta,
    plot_pairwise_samples,
    plot_posterior_chains_with_priors,
    plot_sim_sample,
)

file_name = "sin-ab"

print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())

from data.true_funcs import (
    discrepancy,
    eta,
    TrueParams,
    zeta,
)

TP = TrueParams()


def main():
    DATAFIELD = np.loadtxt("data/obs-ab.csv", delimiter=",", dtype=np.float32)
    DATACOMP = np.loadtxt("data/sim-ab.csv", delimiter=",", dtype=np.float32)

    yf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
    yc = jnp.reshape(DATACOMP[:, 0], (-1, 1)).astype(jnp.float64)
    xf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATACOMP[:, 1], (-1, 1)).astype(jnp.float64)
    tc = jnp.reshape(DATACOMP[:, 2:], (-1, 2)).astype(jnp.float64)

    # normalising the output is not required provided they are all of a similar scale.
    # But subtracting the mean is sensible as our GP priors assume zero mean.
    ycmean = jnp.mean(yc)
    yc_centered = yc - ycmean  # Centre so that E[yc] = 0
    yf_centered = yf - ycmean

    # normalising the inputs is not required provided they are all of a similar scale.

    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    # print(f"tmin: {tmin}, tmax: {tmax}")
    tc_normalized = (tc - tmin) / (tmax - tmin)  # Normalize to [0, 1]

    tminmax = {
        "theta_0": (tmin[0], tmax[0]),
        "theta_1": (tmin[1], tmax[1]),
    }

    field_dataset = gpx.Dataset(xf, yf_centered)
    comp_dataset = gpx.Dataset(jnp.hstack((xc, tc_normalized)), yc_centered)

    kohdataset = kgx.KOHDataset(field_dataset, comp_dataset)
    print(kohdataset)

    fig, ax = plot_sim_sample(
        xf=xf,
        yf=yf,
        xc=xc,
        yc=yc,
        tc=tc,
    )
    plt.savefig(f"figures/{file_name}-obs-and-sim-sample.png", dpi=300)
    plt.close(fig)

    # Define the model
    class Model(KOHModel):
        def k_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained["eta"]
            return gpx.kernels.ProductKernel(
                kernels=[
                    gpx.kernels.RBF(
                        active_dims=[0],
                        lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                        variance=jnp.array(1 / params["variances"]["precision"]),
                    ),
                    gpx.kernels.RBF(
                        active_dims=[1],
                        lengthscale=jnp.array(params["lengthscales"]["theta_0"]),
                    ),
                    gpx.kernels.RBF(
                        active_dims=[2],
                        lengthscale=jnp.array(params["lengthscales"]["theta_1"]),
                    ),
                ]
            )

        def k_delta(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained["delta"]
            return gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                variance=jnp.array(1 / params["variances"]["precision"]),
            )

        def k_epsilon(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained["epsilon"]
            return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1 / params["variances"]["precision"]),
            )

    # Define the priors
    # account for the scaling onto [0, 1]
    A0 = (0.25 - tmin[0]) / (tmax[0] - tmin[0])
    B0 = (0.45 - tmin[0]) / (tmax[0] - tmin[0])
    print(f"A0: {A0}, B0: {B0}")
    A1 = (-3.3 - tmin[1]) / (tmax[1] - tmin[1])
    B1 = (-3.0 - tmin[1]) / (tmax[1] - tmin[1])
    print(f"A1: {A1}, B1: {B1}")
    prior_dict: ModelParameterPriorDict = {
        "thetas": {
            "theta_0": ParameterPrior(
                dist.Uniform(low=A0, high=B0),
                name="theta_0",
            ),
            "theta_1": ParameterPrior(
                dist.Uniform(low=A1, high=B1),
                name="theta_1",
            ),
        },
        "eta": {
            "variances": {
                "precision": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=4.0),
                    name="eta_precision",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    dist.Gamma(concentration=4.0, rate=1.4),
                    name="eta_lengthscale_x_0",
                ),
                "theta_0": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=3.5),
                    name="eta_lengthscale_theta_0",
                ),
                "theta_1": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=3.5),
                    name="eta_lengthscale_theta_1",
                ),
            },
        },
        "delta": {
            "variances": {
                "precision": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=0.1),
                    name="delta_precision",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    # dist.Gamma(concentration=4.0, rate=2.0),
                    dist.Gamma(
                        concentration=5.0, rate=0.3
                    ),  # encourage long value => linear discrepancy
                    name="delta_lengthscale_x_0",
                ),
            },
        },
        "epsilon": {  # This is required despite not appearing in the model
            "variances": {
                "precision": ParameterPrior(
                    # dist.Gamma(concentration=12.0, rate=0.025),
                    dist.Normal(loc=420.0, scale=10.0),  # Much more concentrated
                    name="epsilon_precision",
                ),
            },
        },
    }

    model_parameters = ModelParameters(prior_dict=prior_dict)

    # MCMC setup
    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    ##### Mici #####
    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=model.get_KOH_neg_log_pos_dens_func(),
        backend="jax",
    )
    integrator = mici.integrators.LeapfrogIntegrator(system)

    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    prior_means = jax.tree.map(lambda x: x.inverse(x.distribution.mean), prior_leaves)

    # Test the negative log density function
    init_states = np.array(prior_means)  # NOT jnp.array
    print(f"Initial states: {init_states}")

    f = model.get_KOH_neg_log_pos_dens_func()
    f(init_states)

    # Run MCMC
    tracer_index_dict = {}
    for i, prior in enumerate(model_parameters.priors_flat):
        tracer_index_dict[prior.name] = i
    # print(tracer_index_dict)

    seed = 1234
    n_chain = 2
    n_process = 1  # only 1 works on MacOS
    n_warm_up_iter = 60
    n_main_iter = 80
    rng = np.random.default_rng(seed)

    ##### Mici sampler and adapters #####
    # sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=3)
    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_tree_depth=5
    )
    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(0.8),
        mici.adapters.OnlineCovarianceMetricAdapter(),
    ]

    def trace_func(state):
        trace = {key: state.pos[index] for key, index in tracer_index_dict.items()}
        trace["hamiltonian"] = system.h(state)
        return trace

    final_states, traces, stats = sampler.sample_chains(
        n_warm_up_iter,
        n_main_iter,
        [init_states] * n_chain,
        adapters=adapters,
        n_process=n_process,
        trace_funcs=[trace_func],
        monitor_stats=("n_step", "accept_stat", "step_size", "diverging"),
    )

    # Analyse the MCMC output
    # arviz.summary(traces)

    # Transform the chains
    traces_transformed = {}
    for var, trace in traces.items():
        if var == "hamiltonian":
            continue
        index = tracer_index_dict[var]
        traces_transformed[var] = model_parameters.priors_flat[index].forward(
            np.array(trace)
        )
        if var in prior_dict["thetas"].keys():
            trace = traces_transformed[var]
            tmin, tmax = tminmax[var]
            traces_transformed[var] = list((jnp.array(trace) * (tmax - tmin)) + tmin)

    params_transformed_flat = {}
    for var, trace in traces_transformed.items():
        params_transformed_flat[var] = np.mean(
            trace
        )  # This operation is valid across chains when each chain has equal length (I think).
        print(var, ": ", np.mean(trace), "±", np.std(trace))

    params_transformed = jax.tree.unflatten(
        prior_tree, params_transformed_flat.values()
    )

    # Save the chains as csv files
    arviz.convert_to_inference_data(traces).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}-raw.nc"
    )
    arviz.convert_to_inference_data(traces_transformed).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}.nc"
    )

    # print(arviz.summary(traces_transformed))

    # axes = plot_pairwise_samples(
    #     traces_transformed,
    #     var_names=[
    #         *(prior_dict["thetas"].keys()),
    #         "eta_precision",
    #         "eta_lengthscale_x_0",
    #         "delta_precision",
    #         "delta_lengthscale_x_0",
    #         "epsilon_precision",
    #     ],
    # )
    # plt.savefig(f"figures/{file_name}-pairs.png", dpi=300)
    # plt.close()

    # axes = plot_posterior_chains_with_priors(
    #     traces_transformed,
    #     model_parameters=model_parameters,
    #     tracer_index_dict=tracer_index_dict,
    #     true_values={
    #         "theta_0": TP.a,
    #         "theta_1": TP.b,
    #         "epsilon_precision": 1 / TP.obs_var,
    #     },
    #     figsize=(9, 2 * (7)),
    # )
    # plt.savefig(f"figures/{file_name}-trace.png", dpi=300)
    # plt.close()

    # x0_pred = np.linspace(0, 4, 1000)
    # x1_pred = np.zeros_like(x0_pred)
    # xpred = np.vstack((x0_pred, x1_pred)).T
    # print(xpred.shape)

    # thetas = np.array(
    #     [params_transformed_flat[var] for var in prior_dict["thetas"].keys()]
    # )
    # theta_vec = jnp.array([thetas[0], thetas[1], TP.c, TP.d, TP.e])
    # theta_arr = jnp.tile(theta_vec, (xpred.shape[0], 1))
    # print(theta_arr.shape)

    # x_test = np.hstack((xpred, theta_arr))
    # print(x_test.shape)

    # x_test_GP = x_test[:, [0, 2, 3]]
    # print(x_test_GP.shape)

    # dataset = kohdataset.get_dataset(thetas.reshape(1, -1))
    # print(dataset)

    # # Posterior GPs
    # GP_posterior = model.GP_posterior(params_transformed)

    # eta_pred = GP_posterior.predict_eta(x_test_GP, dataset)
    # zeta_pred = GP_posterior.predict_zeta(x_test_GP, dataset)
    # obs_pred = GP_posterior.predict_obs(x_test_GP, dataset)

    # eta_pred_m = eta_pred.mean
    # eta_pred_cov = eta_pred.covariance_matrix

    # zeta_pred_m = zeta_pred.mean
    # zeta_pred_cov = zeta_pred.covariance_matrix

    # # Simulator emulator
    # fig, ax = plot_f_eta(
    #     x_full=x_test,
    #     x_GP=x_test_GP[:, 0],
    #     thetas=thetas,
    #     thetas_full=theta_arr,
    #     eta=eta,
    #     gaussian_distribution=eta_pred,
    # )
    # plt.savefig(f"figures/{file_name}-eta-posterior.png", dpi=300)
    # plt.close()

    # # True process
    # fig, ax = plot_f_zeta(
    #     x_full=x_test,
    #     x_GP=x_test_GP[:, 0],
    #     zeta=zeta,
    #     GP_zeta=zeta_pred,
    #     GP_zeta_epsilon=obs_pred,
    #     scatter_xf=kohdataset.Xf,
    #     scatter_yf=kohdataset.z
    #     + ycmean,  # Add the mean of yc to center the observations
    # )
    # plt.savefig(f"figures/{file_name}-zeta-posterior.png", dpi=300)
    # plt.close(fig)

    # # Model discrepancy
    # delta_gp_m = zeta_pred_m - eta_pred_m
    # delta_gp_cov = zeta_pred_cov + eta_pred_cov

    # fig, ax = plot_f_delta(
    #     x_full=x_test,
    #     x_GP=x_test_GP[:, 0],
    #     thetas=thetas,
    #     thetas_full=theta_arr,
    #     zeta=zeta,
    #     eta=eta,
    #     delta=discrepancy,
    #     delta_gp_mean=delta_gp_m,
    #     delta_gp_cov=delta_gp_cov,
    # )
    # plt.savefig(f"figures/{file_name}-discrepancy-posterior.png", dpi=300)
    # plt.close()


if __name__ == "__main__":
    main()
