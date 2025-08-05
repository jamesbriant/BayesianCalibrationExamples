import arviz
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
import mici
import numpy as np
from argparser import parse_args
from data.true_funcs import TrueParams
from dataloader import load, save_chains_to_netcdf
from freethreading import process_workers
from jax import config
from kohgpjax.parameters import ModelParameters
from models.sin_a import Model, get_ModelParameterPriorDict

config.update("jax_enable_x64", True)
file_name = "sin-a"

print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())


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
    # div: int = args.D
    # obs_mode: str = args.obs_mode

    data_file_name = file_name.split("-")[1]

    kohdataset, tminmax, ycmean = load(
        sim_file_path_csv=f"data/sim-{data_file_name}.csv",
        obs_file_path_csv=f"data/obs-{data_file_name}.csv",
        num_calib_params=1,  # number of calibration parameters
        x_dim=2,  # number of control/regression variables
    )
    n_sim = kohdataset.num_sim_obs
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
        n_sim=n_sim,
    )


if __name__ == "__main__":
    main()
