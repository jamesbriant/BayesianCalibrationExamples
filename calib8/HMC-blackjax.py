import arviz
import blackjax
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
import numpy as np
from argparser import get_base_parser
from datahandler import load, save_chains_to_netcdf
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

    log_prob = model.get_KOH_neg_log_pos_dens_func()

    # BlackJax NUTS sampler
    nuts = blackjax.nuts(log_prob, max_num_doublings=max_num_doublings)

    # JIT compile the step function
    step_fn = jax.jit(nuts.step)

    # Initialize the sampler
    rng_key = jax.random.PRNGKey(seed)
    initial_states = jax.random.normal(rng_key, (n_chain, model_parameters.n_params))

    # Run the MCMC
    @jax.jit
    def run_mcmc(initial_states, n_samples):
        def one_step(states, _):
            keys = jax.random.split(jax.random.PRNGKey(0), n_chain)
            states, infos = jax.vmap(step_fn)(keys, states)
            return states, (states, infos)

        final_states, (states_hist, infos_hist) = jax.lax.scan(
            one_step, initial_states, jnp.arange(n_samples)
        )
        return states_hist, infos_hist

    # Run warmup and sampling
    warmup_states, warmup_infos = run_mcmc(initial_states, n_warm_up_iter)
    states, infos = run_mcmc(warmup_states, n_main_iter)

    # Reshape the samples
    samples = states.position.reshape(-1, n_chain, model_parameters.n_params)
    traces = {
        param.name: samples[:, :, i]
        for i, param in enumerate(model_parameters.priors_flat)
    }

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
        inference_library_name="blackjax",
    )


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--max_num_doublings",
        type=int,
        default=10,
        help="Maximum number of doublings for the NUTS sampler.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        n_warm_up_iter=args.W,
        n_main_iter=args.N,
        seed=args.seed,
        n_chain=args.n_chain,
        max_num_doublings=args.max_num_doublings,
    )
