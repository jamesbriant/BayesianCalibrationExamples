from typing import Dict, Tuple

import jax.numpy as jnp
import gpjax as gpx
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ParameterPrior, ModelParameterPriorDict
import numpyro.distributions as dist


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
                gpx.kernels.RBF(
                    active_dims=[3],
                    lengthscale=jnp.array(params["lengthscales"]["theta_2"]),
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


def get_ModelParameterPriorDict(
    tminmax: Dict[str, Tuple[float, float]],
) -> ModelParameterPriorDict:
    print("Creating ModelParameterPriorDict for sin-ab model...")

    tmm0 = tminmax["theta_0"]
    tmm1 = tminmax["theta_1"]
    tmm2 = tminmax["theta_2"]
    A0 = (0.25 - tmm0[0]) / (tmm0[1] - tmm0[0])
    B0 = (0.45 - tmm0[0]) / (tmm0[1] - tmm0[0])
    print(f"A0: {A0}, B0: {B0}")
    A1 = (-3.3 - tmm1[0]) / (tmm1[1] - tmm1[0])
    B1 = (-3.0 - tmm1[0]) / (tmm1[1] - tmm1[0])
    print(f"A1: {A1}, B1: {B1}")
    A2 = (0.6 - tmm2[0]) / (tmm2[1] - tmm2[0])
    B2 = (1.1 - tmm2[0]) / (tmm2[1] - tmm2[0])
    print(f"A2: {A2}, B2: {B2}")

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
            "theta_2": ParameterPrior(
                dist.Uniform(low=A2, high=B2),
                name="theta_2",
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
                "theta_2": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=3.5),
                    name="eta_lengthscale_theta_2",
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
                    # dist.Normal(loc=420.0, scale=10.0),  # Much more concentrated
                    dist.Gamma(
                        concentration=800, rate=2.0
                    ),  # More concentrated, E(X)=400, SD(X)=14.14
                    name="epsilon_precision",
                ),
            },
        },
    }

    return prior_dict
