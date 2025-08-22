from typing import Dict

import blackjax
import gpjax as gpx
import jax
import kohgpjax as kgx
from argparser import get_base_parser
from datahandler import load
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

    # n_sim = kohdataset.num_sim_obs
    print(kohdataset)

    prior_dict = get_ModelParameterPriorDict(config, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    log_prob_fn = model.get_KOH_neg_log_pos_dens_func()

    # --- Define the log probability function ---
    @jax.jit
    def log_prob(params: Dict[str, jax.Array]) -> jax.Array:
        """Compute the log probability of the parameters."""

        return log_prob_fn(jax.tree.leaves(params))

    # --- Initialize the MCMC sampler ---
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    # initial_states = jax.random.normal(subkey, (n_chain, model_parameters.n_params))
    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    initial_states = jax.tree.map(
        lambda x: x.inverse(x.distribution.mean), prior_leaves
    )
    prior_means = jax.tree.map(lambda x: x.inverse(x.distribution.mean), prior_leaves)
    initial_states = {p.name: m for p, m in zip(prior_leaves, prior_means)}
    # print(f"Initial states: {initial_states}", "\n")
    # print(f"leaves: {jax.tree.leaves(initial_states)}", "\n")

    # --- Window adaptation for HMC ---
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    key, warmup_key, sample_key = jax.random.split(key, 3)
    (state, parameters), _ = warmup.run(
        warmup_key, initial_states, num_steps=n_warm_up_iter
    )

    print(f"Adapted parameters: {parameters}", "\n")
    print(f"Adapted state: {state}", "\n")

    # BlackJax NUTS sampler
    nuts = blackjax.nuts(log_prob, max_num_doublings=max_num_doublings, **parameters)

    # JIT compile the step function
    step_fn = jax.jit(nuts.step)

    # Run the MCMC
    # @jax.jit
    # def run_mcmc(initial_states, n_samples, sample_key: jax.random.PRNGKey):
    #     def one_step(states, _):
    #         keys = jax.random.split(sample_key, n_chain)
    #         states, infos = jax.vmap(step_fn)(keys, states)
    #         return states, (states, infos)

    #     final_states, (states_hist, infos_hist) = jax.lax.scan(
    #         one_step, initial_states, jnp.arange(n_samples)
    #     )
    #     return states_hist, infos_hist
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        """Run the MCMC inference loop."""

        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    # Run sampling
    # warmup_states, warmup_infos = run_mcmc(state, n_warm_up_iter)
    # states, infos = run_mcmc(state, n_main_iter, sample_key)
    states = inference_loop(sample_key, step_fn, state, n_main_iter)
    # type(states) is <class 'blackjax.mcmc.hmc.HMCState'>

    # states.position[num_MCMC_variables][n_main_iter]
    print(states.position)

    # fig, ax = plt.subplots()
    # ax.plot(states.position[10])
    # ax.set_xlabel("Sample Index")
    # plt.show()
    # for item in states:
    #     print(item, "\n\n")

    # # Reshape the samples
    # samples = states.position.reshape(-1, n_chain, model_parameters.n_params)
    # traces = {
    #     param.name: samples[:, :, i]
    #     for i, param in enumerate(model_parameters.priors_flat)
    # }

    # # Transform the chains
    # traces_transformed = transform_chains(traces, model_parameters, prior_dict, tminmax)

    # print(arviz.summary(traces_transformed))

    # save_chains_to_netcdf(
    #     raw_traces=traces,
    #     transformed_traces=traces_transformed,
    #     file_name=file_name,
    #     n_warm_up_iter=n_warm_up_iter,
    #     n_main_iter=n_main_iter,
    #     n_sim=n_sim,
    #     ycmean=yc_mean,
    #     inference_library_name="blackjax",
    # )


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
