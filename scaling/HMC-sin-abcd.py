from jax import config

config.update("jax_enable_x64", True)

import argparse
import arviz
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
from kohgpjax.parameters import ModelParameters
import mici
import numpy as np
import os
from models.sin_abcd import Model, get_ModelParameterPriorDict
from freethreading import process_workers
from argparser import parse_args

file_name = "sin-abcd"

print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())

from data.true_funcs import TrueParams

TP = TrueParams()


def main():
    args = parse_args()
    # 0 Access the arguments
    n_warm_up_iter: int = args.W
    n_main_iter: int = args.N
    seed: int = args.seed
    n_chain: int = args.n_chain
    n_processes: int = args.n_processes
    max_tree_depth: int = args.max_tree_depth

    DATAFIELD = np.loadtxt("data/obs-ab.csv", delimiter=",", dtype=np.float32)
    DATACOMP = np.loadtxt("data/sim-ab.csv", delimiter=",", dtype=np.float32)

    yf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
    yc = jnp.reshape(DATACOMP[:, 0], (-1, 1)).astype(jnp.float64)
    xf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATACOMP[:, 1], (-1, 1)).astype(jnp.float64)
    tc = jnp.reshape(DATACOMP[:, 2:], (-1, 4)).astype(jnp.float64)

    # normalising the output is not required provided they are all of a similar scale.
    # But subtracting the mean is sensible as our GP priors assume zero mean.
    ycmean = jnp.mean(yc)
    yc_centered = yc - ycmean  # Centre so that E[yc] = 0
    yf_centered = yf - ycmean

    # normalising the inputs is not required provided they are all of a similar scale.

    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    tc_normalized = (tc - tmin) / (tmax - tmin)  # Normalize to [0, 1]

    tminmax = {
        "theta_0": (tmin[0], tmax[0]),
        "theta_1": (tmin[1], tmax[1]),
        "theta_2": (tmin[2], tmax[2]),
        "theta_3": (tmin[3], tmax[3]),
    }

    field_dataset = gpx.Dataset(xf, yf_centered)
    comp_dataset = gpx.Dataset(jnp.hstack((xc, tc_normalized)), yc_centered)

    kohdataset = kgx.KOHDataset(field_dataset, comp_dataset)
    print(kohdataset)

    prior_dict = get_ModelParameterPriorDict(tminmax)
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

    ##### Mici sampler and adapters #####
    rng = np.random.default_rng(seed)
    # sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=max_tree_depth)
    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_tree_depth=max_tree_depth
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
        **process_workers(n_processes),
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

    print(arviz.summary(traces_transformed))

    save_chains_to_netcdf(
        traces,
        traces_transformed,
        file_name=file_name,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
    )


def save_chains_to_netcdf(
    traces,
    traces_transformed,
    file_name: str,
    n_warm_up_iter: int,
    n_main_iter: int,
) -> None:
    """
    Save the MCMC traces to NetCDF files.

    Args:
        traces: The raw MCMC traces
        traces_transformed: The transformed MCMC traces
        file_name: The base name for the output files
        n_warm_up_iter: Number of warm-up iterations
        n_main_iter: Number of main iterations
    """
    # check if chains directory exists, if not create it
    if not os.path.exists("chains"):
        os.makedirs("chains")

    arviz.convert_to_inference_data(traces).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}-raw.nc"
    )
    arviz.convert_to_inference_data(traces_transformed).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}.nc"
    )


if __name__ == "__main__":
    main()
