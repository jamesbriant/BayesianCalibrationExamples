import blackjax
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
from jax import config
from kohgpjax.parameters import ModelParameters

from argparser import get_base_parser
from datahandler import load, save_chains_to_netcdf, transform_chains
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
    max_num_doublings: int,
):
    """Main function to run the MCMC sampling process.
    Args:
        config_path (str): Path to the configuration module.
        n_warm_up_iter (int): Number of warm-up iterations for MCMC.
        n_main_iter (int): Number of main iterations for MCMC.
        seed (int): Random seed for reproducibility.
        n_chain (int): Number of MCMC chains to run.
        max_num_doublings (int): Maximum number of doublings for the NUTS sampler.
    """
    # Load the config and model from the model directory
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    file_name = experiment_config.name
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

    # Determine data root relative to this script
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")

    # Load the dataset
    print(f"Loading dataset from {file_name}...")
    kohdataset, tminmax, yc_mean = load(
        experiment_config=experiment_config,
        data_root=data_root,
    )

    # n_sim = kohdataset.num_sim_obs
    print(kohdataset)

    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    # Negative log posterior density function (expects flat unconstrained vector)
    neg_log_post = model.get_KOH_neg_log_pos_dens_func()

    # --- Define the log probability function ---
    @jax.jit
    def log_prob(params_vec: jnp.ndarray) -> jax.Array:
        """Return log-posterior given flat unconstrained params vector."""
        return -neg_log_post(params_vec)

    # --- Initialize the MCMC sampler ---
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    prior_leaves, _ = jax.tree.flatten(prior_dict)
    prior_means = jax.tree.map(lambda x: x.inverse(x.distribution.mean), prior_leaves)
    initial_position = jnp.array(prior_means)

    # --- Window adaptation for HMC ---
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)

    # Split keys
    key, warmup_key, sample_key = jax.random.split(key, 3)

    # Initial positions for multiple chains: (n_chain, n_params)
    initial_positions = jnp.tile(initial_position, (n_chain, 1))

    # Warmup keys: (n_chain,)
    warmup_keys = jax.random.split(warmup_key, n_chain)

    print(f"Running warmup for {n_chain} chains with {n_warm_up_iter} iterations...")

    # Run warmup vectorized over chains, explicitly JIT-ed
    @jax.jit
    def run_warmup(keys, initial_positions):
        return jax.vmap(warmup.run, in_axes=(0, 0, None))(
            keys, initial_positions, n_warm_up_iter
        )

    (state, parameters), _ = run_warmup(warmup_keys, initial_positions)

    # parameters is now a dict with batched values (n_chain, ...)
    # state is a batched HMCState

    print(
        f"Adapted parameters (first chain): {jax.tree.map(lambda x: x[0], parameters)}",
        "\n",
    )

    # --- Save backend, algorithm, and parameter details ---
    import importlib.metadata
    import json
    import sys

    # Prepare settings dict
    try:
        blackjax_version = importlib.metadata.version("blackjax")
    except Exception:
        blackjax_version = None
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

    # Algorithm parameters
    algorithm_parameters = {
        "max_num_doublings": max_num_doublings,
    }
    # Add adapted parameters (step_size, inverse_mass_matrix, etc.)
    # Since parameters are batched, we convert them to lists (one value per chain)
    for k, v in parameters.items():
        if hasattr(v, "tolist"):
            algorithm_parameters[k] = v.tolist()
        else:
            # Fallback for scalars or other types
            algorithm_parameters[k] = (
                float(v) if isinstance(v, (float, int, jnp.ndarray)) else str(v)
            )

    # MCMC parameters
    mcmc_parameters = {
        "n_chain": n_chain,
        "n_warm_up_iter": n_warm_up_iter,
        "n_main_iter": n_main_iter,
        "seed": seed,
        "model_dir": model_dir,
    }

    # CLI command
    cli_command = " ".join(sys.argv)

    # Library versions
    library_versions = {
        "blackjax": blackjax_version,
        "jax": jax_version,
        "numpy": numpy_version,
        "gpjax": gpjax_version,
        "kohgpjax": kohgpjax_version,
    }

    settings = {
        "backend": "blackjax",
        "algorithm": "nuts",
        "algorithm_parameters": algorithm_parameters,
        "mcmc_parameters": mcmc_parameters,
        "library_versions": library_versions,
        "cli_command": cli_command,
    }

    # --- Sampling ---
    print(f"Running sampling for {n_chain} chains with {n_main_iter} iterations...")

    @jax.jit
    def run_inference(rng_key, initial_state, parameters):
        def step_fn(key, state, params):
            kernel = blackjax.nuts(
                log_prob, max_num_doublings=max_num_doublings, **params
            ).step
            return kernel(key, state)

        # vmap over chains: key(0), state(0), params(0)
        step_fn_vmap = jax.vmap(step_fn)

        def one_step(state, key):
            new_state, info = step_fn_vmap(key, state, parameters)
            return new_state, new_state

        # Generate keys: (num_samples, n_chain)
        keys = jax.random.split(rng_key, n_main_iter)
        keys = jax.vmap(lambda k: jax.random.split(k, n_chain))(keys)

        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    # Run sampling
    states = run_inference(sample_key, state, parameters)

    # states.position is (n_main_iter, n_chain, n_params)
    # We want (n_chain, n_main_iter, n_params)
    positions = jnp.swapaxes(states.position, 0, 1)

    # Build raw traces dict: each param -> [n_chain, n_main_iter]
    tracer_index_dict = {p.name: i for i, p in enumerate(model_parameters.priors_flat)}
    raw_traces = {
        name: jnp.array(positions[:, :, idx]) for name, idx in tracer_index_dict.items()
    }

    # Transform chains to constrained space and un-normalize theta
    transformed_traces = transform_chains(
        raw_traces, model_parameters, prior_dict, tminmax
    )

    # Quick summary
    means = {name: float(jnp.mean(vals)) for name, vals in transformed_traces.items()}
    print("Posterior means (approx across chains):")
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    # Save to NetCDF in experiments/<file_name>/<timestamp>_W<n_warm_up_iter>_N<n_main_iter>
    from datetime import datetime

    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_W{n_warm_up_iter}_N{n_main_iter}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "experiments", file_name, run_id)
    os.makedirs(output_dir, exist_ok=True)

    n_sim = kohdataset.num_sim_obs
    # Save settings to JSON
    with open(os.path.join(output_dir, "mcmc_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    save_chains_to_netcdf(
        raw_traces={k: np.array(v) for k, v in raw_traces.items()},
        transformed_traces={k: np.array(v) for k, v in transformed_traces.items()},
        file_name=file_name,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        n_sim=n_sim,
        ycmean=float(np.array(yc_mean).reshape(-1)[0]),
        inference_library_name="blackjax",
        output_dir=output_dir,
    )
    # def run_mcmc(initial_states, n_samples, sample_key: jax.random.PRNGKey):
    #     def one_step(states, _):
    #         keys = jax.random.split(sample_key, n_chain)
    #         states, infos = jax.vmap(step_fn)(keys, states)
    #         return states, (states, infos)

    #     final_states, (states_hist, infos_hist) = jax.lax.scan(
    #         one_step, initial_states, jnp.arange(n_samples)
    #     )
    #     return states_hist, infos_hist
    # End of sampling and saving


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--max_num_doublings",
        type=int,
        default=5,
        help="Maximum number of doublings for the NUTS sampler.",
    )
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        n_warm_up_iter=args.W,
        n_main_iter=args.N,
        seed=args.seed,
        n_chain=args.n_chain,
        max_num_doublings=args.max_num_doublings,
    )
