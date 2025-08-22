from typing import Dict, Tuple

import gpjax as gpx
import jax.numpy as jnp
import numpyro.distributions as dist
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict, ParameterPrior


class Model(KOHModel):
    def k_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        params = params_constrained["eta"]
        kernels = [
            gpx.kernels.RBF(
                active_dims=[i + 1],
                lengthscale=jnp.array(params["lengthscales"][f"theta_{i}"]),
            )
            for i in range(0, 8)  # Assuming theta_0 to theta_7
        ]
        kernels.append(
            gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                variance=jnp.array(1 / params["variances"]["precision"]),
            )
        )
        return gpx.kernels.ProductKernel(kernels=kernels)

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


def get_ModelParameterPriorDict(
    config,
    tminmax: Dict[str, Tuple[float, float]],
) -> ModelParameterPriorDict:
    print("Creating ModelParameterPriorDict for calib8 model...")

    # account for the scaling onto [0, 1]

    # tmm0 = tminmax["theta_0"]
    # A0 = (0.25 - tmm0[0]) / (tmm0[1] - tmm0[0])
    # B0 = (0.45 - tmm0[0]) / (tmm0[1] - tmm0[0])
    # print(f"A0: {A0}, B0: {B0}")

    prior_dict: ModelParameterPriorDict = {
        "thetas": {
            # "theta_0": ParameterPrior(
            #     dist.Uniform(low=A0, high=B0),
            #     name="theta_0",
            # ),
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
                # "theta_0": ParameterPrior(
                #     dist.Gamma(concentration=2.0, rate=3.5),
                #     name="eta_lengthscale_theta_0",
                # ),
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
                    # dist.Normal(loc=420.0, scale=10.0),  # Much more concentrated
                    dist.Gamma(
                        concentration=800, rate=2.0
                    ),  # More concentrated, E(X)=400, SD(X)=14.14
                    name="epsilon_precision",
                ),
            },
        },
    }

    for i in range(0, config.N_CALIB_PARAMS):
        param = config.PARAMETERS[i]
        if param["name"] != f"theta_{i}":
            raise ValueError(
                f"Expected control parameter name to be 'theta_{i}', but got '{param['name']}'"
            )
        if "range" not in param:
            raise ValueError(f"Control parameter 'theta_{i}' must have a 'range' key")
        prior_range = param["range"]
        if len(prior_range) != 2:
            raise ValueError(
                f"Control parameter 'theta_{i}' range must be a list of two values, got {prior_range}"
            )
        tmm = tminmax[f"theta_{i}"]
        A = (prior_range[0] - tmm[0]) / (tmm[1] - tmm[0])
        B = (prior_range[1] - tmm[0]) / (tmm[1] - tmm[0])
        prior_dict["thetas"][f"theta_{i}"] = ParameterPrior(
            dist.Uniform(low=A, high=B),
            name=f"theta_{i}",
        )

        prior_dict["eta"]["lengthscales"][f"theta_{i}"] = ParameterPrior(
            dist.Gamma(concentration=2.0, rate=3.5),
            name=f"eta_lengthscale_theta_{i}",
        )

    return prior_dict
