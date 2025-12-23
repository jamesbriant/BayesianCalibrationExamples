import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict, ModelParameters

from datahandler import load
from utils import load_config_from_model_dir, load_model_from_model_dir


def load_params_from_chain(chain_path: str):
    """
    Load posterior means from an ArviZ InferenceData NetCDF and return a dict of parameter means.
    If the file does not exist or loading fails, return None.
    """
    try:
        import arviz as az

        idata = az.from_netcdf(chain_path)
        summary = az.summary(idata, kind="stats")
        means = summary["mean"].to_dict()
        return means
    except Exception:
        return None


def build_params_constrained(
    model_parameters: ModelParameters,
    prior_dict: ModelParameterPriorDict,
    means_dict=None,
):
    """
    Build constrained parameter tree from either posterior means (if provided) or prior means.
    """
    # Start from prior means in unconstrained space
    prior_leaves, _ = jax.tree.flatten(prior_dict)
    prior_means_unconstrained = jax.tree.map(
        lambda x: x.inverse(x.distribution.mean), prior_leaves
    )

    # Map back to constrained
    constrained_flat = []
    for p in model_parameters.priors_flat:
        # Use posterior mean if available; otherwise use prior mean
        if means_dict is not None and p.name in means_dict:
            val = np.array(means_dict[p.name])
            constrained_flat.append(val)
        else:
            constrained_flat.append(
                p.forward(
                    np.array(prior_means_unconstrained[p.index])
                    if hasattr(p, "index")
                    else np.array(
                        prior_means_unconstrained[model_parameters.priors_flat.index(p)]
                    )
                )
            )

    params_constrained = model_parameters.unflatten_sample(constrained_flat)
    return params_constrained


def run(model_dir: str, output_dir: str, W=None, N=None, chain_file=None):
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    file_name = experiment_config.name
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")
    # Hack: data root relative to where we run?
    # Actually, data is in root/data. script_dir is root/analysis (after move).
    # So we need ../data. But let caller handle this or fix properly.
    # Since we appended root to path, we know where root is.
    # Better: usage of utils/datahandler load assumes data_root.
    # "data" works if running from root.

    kohdataset, tminmax, yc_mean = load(
        experiment_config=experiment_config,
        data_root="data",
    )

    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
    from kohgpjax.parameters import ModelParameters

    model_parameters = ModelParameters(prior_dict=prior_dict)

    # Optionally load posterior means
    means = None
    chain_path = chain_file
    if chain_path is None and W is not None and N is not None:
        # Try to find latest run dir with W/N
        # Assumes experiments dir is in root
        exp_dir = os.path.join("experiments", file_name)
        if os.path.exists(exp_dir):
            candidates = sorted(
                [d for d in os.listdir(exp_dir) if d.endswith(f"_W{W}_N{N}")]
            )
            if candidates:
                latest = os.path.join(exp_dir, candidates[-1])
                # pick any .nc file
                nc_files = [f for f in os.listdir(latest) if f.endswith(".nc")]
                if nc_files:
                    chain_path = os.path.join(latest, nc_files[0])

    if chain_path is not None and os.path.exists(chain_path):
        means = load_params_from_chain(chain_path)

    # Pass Model class into build_params_constrained via calling scope
    params_constrained = build_params_constrained(model_parameters, prior_dict, means)

    model: KOHModel = Model(model_parameters=model_parameters, kohdataset=kohdataset)
    posterior_GP = model.GP_posterior(params_constrained)

    # Predict on a fixed lat grid (not from data): linearly spaced -90 to 90
    Xf = jnp.linspace(-90.0, 90.0, 180).reshape(-1, 1)
    # Predict components
    print(kohdataset.sim_dataset)
    eta_pred = posterior_GP.predict_eta(Xf, train_data=kohdataset)
    zeta_pred = posterior_GP.predict_zeta(Xf, train_data=kohdataset)
    obs_pred = posterior_GP.predict_obs(Xf, train_data=kohdataset)
    discrepancy = zeta_pred - eta_pred

    # Prepare plot
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(Xf).squeeze()
    ax.plot(x, np.array(eta_pred).squeeze(), label="eta (model component)")
    ax.plot(x, np.array(zeta_pred).squeeze(), label="zeta (simulated)")
    ax.plot(x, np.array(obs_pred).squeeze(), label="obs (prediction)")
    ax.plot(x, np.array(discrepancy).squeeze(), label="zeta - eta (discrepancy)")
    ax.set_xlabel("lat")
    ax.set_ylabel(experiment_config.data.variable_name)
    ax.set_title(f"Predictions for {file_name}")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "predictions.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
