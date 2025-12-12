import os
from datetime import datetime

import arviz
import gpjax as gpx
import jax
import jax.random as jr
import kohgpjax as kgx
import mici
import numpy as np
from jax import config
from kohgpjax.parameters import ModelParameters

from argparser import get_base_parser
from datahandler import load, save_chains_to_netcdf, transform_chains
from freethreading import process_workers
from utils import load_config_from_model_dir, load_model_from_model_dir

config.update("jax_enable_x64", True)


print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())


def main(
    model_dir: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    seed: int,
    n_chain: int,
    n_processes: int,
    max_tree_depth: int,
):
    """Main function to run the MCMC sampling process.
    Args:
        model_dir (str): Path to the model directory.
        n_warm_up_iter (int): Number of warm-up iterations for MCMC.
        n_main_iter (int): Number of main iterations for MCMC.
        seed (int): Random seed for reproducibility.
        n_chain (int): Number of MCMC chains to run.
        n_processes (int): Number of processes to use for parallel computation.
        max_tree_depth (int): Maximum tree depth for the NUTS sampler.
    """
    # Load the config file
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    file_name = experiment_config.name
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

    # Determine data root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")

    # Load the dataset
    print(f"Loading dataset from {file_name}...")
    kohdataset, tminmax, yc_mean = load(
        experiment_config=experiment_config,
        data_root=data_root,
    )

    n_sim = kohdataset.num_sim_obs

    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
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

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    prior_means = jax.tree.map(lambda x: x.inverse(x.distribution.mean), prior_leaves)
    # prior_means = [
    #     x.distribution.sample(key=subkey)
    #     for subkey, x in zip(jr.split(subkey, len(prior_leaves)), prior_leaves)
    # ]
    # prior_means = jax.tree.map(
    #     lambda x, subkey: x.inverse(x.distribution.sample(key=subkey)),
    #     prior_leaves,
    #     jr.split(subkey, len(prior_leaves)),
    # )

    # Test the negative log density function
    init_states = np.array(prior_means)  # NOT jnp.array
    print(f"Initial states: {init_states}")

    f = model.get_KOH_neg_log_pos_dens_func()
    print(f(init_states))

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

    # --- Save backend, algorithm, and parameter details ---
    import importlib.metadata
    import json
    import sys

    try:
        mici_version = importlib.metadata.version("mici")
    except Exception:
        mici_version = None
    try:
        jax_version = importlib.metadata.version("jax")
    except Exception:
        jax_version = None
    try:
        numpy_version = importlib.metadata.version("numpy")
    except Exception:
        numpy_version = None
    try:
        gpjax_version = importlib.metadata.version("gpjax")
    except Exception:
        gpjax_version = None
    try:
        kohgpjax_version = importlib.metadata.version("kohgpjax")
    except Exception:
        kohgpjax_version = None

    # Algorithm parameters: sampler class and settings
    def filter_json_serializable(d):
        out = {}
        for k, v in d.items():
            if callable(v):
                continue
            # Try to convert numpy/jax arrays to lists or floats
            try:
                import jax.numpy as jnp
                import numpy as np

                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    v = v.tolist()
                elif isinstance(v, (np.generic, jnp.generic)):
                    v = float(v)
            except Exception:
                pass
            # Only keep simple types
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                out[k] = v
        return out

    algorithm_parameters = {
        "sampler_class": sampler.__class__.__name__,
        "max_tree_depth": max_tree_depth,
        "adapters": [a.__class__.__name__ for a in adapters],
        "adapter_settings": [
            filter_json_serializable(getattr(a, "__dict__", {})) for a in adapters
        ],
    }

    # MCMC parameters
    mcmc_parameters = {
        "n_chain": n_chain,
        "n_warm_up_iter": n_warm_up_iter,
        "n_main_iter": n_main_iter,
        "seed": seed,
        "n_processes": n_processes,
        "model_dir": model_dir,
    }

    # CLI command
    cli_command = " ".join(sys.argv)

    # Library versions
    library_versions = {
        "mici": mici_version,
        "jax": jax_version,
        "numpy": numpy_version,
        "gpjax": gpjax_version,
        "kohgpjax": kohgpjax_version,
    }

    settings = {
        "backend": "mici",
        "algorithm": "DynamicMultinomialHMC",
        "algorithm_parameters": algorithm_parameters,
        "mcmc_parameters": mcmc_parameters,
        "library_versions": library_versions,
        "cli_command": cli_command,
    }

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
    traces_transformed = transform_chains(traces, model_parameters, prior_dict, tminmax)

    print(arviz.summary(traces_transformed))

    # Create a unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_W{n_warm_up_iter}_N{n_main_iter}"

    # Determine output directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "experiments", file_name, run_id)

    print(f"Saving results to {output_dir}")

    # Ensure output directory exists before writing JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mcmc_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    save_chains_to_netcdf(
        raw_traces=traces,
        transformed_traces=traces_transformed,
        file_name=file_name,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        n_sim=n_sim,
        ycmean=yc_mean,
        inference_library_name="mici",
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "--max_tree_depth",
        type=int,
        default=5,
        help="Maximum tree depth for the NUTS sampler.",
    )
    args = parser.parse_args()

    # Use the model_dir positional argument from the base parser
    main(
        model_dir=args.model_dir,
        n_warm_up_iter=args.W,
        n_main_iter=args.N,
        seed=args.seed,
        n_chain=args.n_chain,
        n_processes=args.n_processes,
        max_tree_depth=args.max_tree_depth,
    )
