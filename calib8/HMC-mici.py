import arviz
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
import mici
import numpy as np
from argparser import get_base_parser
from datahandler import load, save_chains_to_netcdf
from freethreading import process_workers
from jax import config
from kohgpjax.parameters import ModelParameters
from models.calib8 import Model, get_ModelParameterPriorDict
from utils import load_config_from_path

config.update("jax_enable_x64", True)


print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())


def main(
    config_path: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    seed: int,
    n_chain: int,
    n_processes: int,
    max_tree_depth: int,
):
    """Main function to run the MCMC sampling process.
    Args:
        config_path (str): Path to the configuration module.
        n_warm_up_iter (int): Number of warm-up iterations for MCMC.
        n_main_iter (int): Number of main iterations for MCMC.
        seed (int): Random seed for reproducibility.
        n_chain (int): Number of MCMC chains to run.
        n_processes (int): Number of processes to use for parallel computation.
        max_tree_depth (int): Maximum tree depth for the NUTS sampler.
    """
    # Load the config file
    config = load_config_from_path(config_path)
    file_name = config.FILE_NAME

    # Load the dataset
    print(f"Loading dataset from {file_name}...")
    kohdataset, tminmax, yc_mean = load(
        file_name=file_name,
        data_dir="data",
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

    # Transform the chains
    traces_transformed = transform_chains(
        traces, model_parameters, prior_dict, tminmax
    )

    print(arviz.summary(traces_transformed))

    save_chains_to_netcdf(
        raw_traces=traces,
        transformed_traces=traces_transformed,
        file_name=file_name,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        n_sim=n_sim,
        ycmean=yc_mean,
        inference_library_name="mici",
    )


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "--max_tree_depth",
        type=int,
        default=10,
        help="Maximum tree depth for the NUTS sampler.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        n_warm_up_iter=args.W,
        n_main_iter=args.N,
        seed=args.seed,
        n_chain=args.n_chain,
        n_processes=args.n_processes,
        max_tree_depth=args.max_tree_depth,
    )
