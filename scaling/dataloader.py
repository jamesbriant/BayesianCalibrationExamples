# ONLY WORKS FOR 1D X VARIABLES
from jax import config

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX

from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp
import kohgpjax as kgx
import gpjax as gpx


def load(
    sim_file_path_csv: str, obs_file_path_csv: str, num_calib_params: int
) -> Tuple[kgx.KOHDataset, Dict[str, Tuple[float, float]], float]:
    DATAFIELD = np.loadtxt(obs_file_path_csv, delimiter=",", dtype=np.float32)
    DATACOMP = np.loadtxt(sim_file_path_csv, delimiter=",", dtype=np.float32)

    yf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
    yc = jnp.reshape(DATACOMP[:, 0], (-1, 1)).astype(jnp.float64)
    xf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATACOMP[:, 1], (-1, 1)).astype(jnp.float64)
    tc = jnp.reshape(DATACOMP[:, 2:], (-1, num_calib_params)).astype(jnp.float64)

    # normalising the output is not required provided they are all of a similar scale.
    # But subtracting the mean is sensible as our GP priors assume zero mean.
    ycmean = jnp.mean(yc)
    yc_centered = yc - ycmean  # Centre so that E[yc] = 0
    yf_centered = yf - ycmean

    # normalising the inputs is not required provided they are all of a similar scale.

    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    # print(f"tmin: {tmin}, tmax: {tmax}")
    tc_normalized = (tc - tmin) / (tmax - tmin)  # Normalize to [0, 1]

    tminmax = {
        f"theta_{i}": (tmin[i], tmax[i]) for i in range(num_calib_params)
    }  # Create a dictionary for each calibration parameter

    field_dataset = gpx.Dataset(xf, yf_centered)
    comp_dataset = gpx.Dataset(jnp.hstack((xc, tc_normalized)), yc_centered)

    kohdataset = kgx.KOHDataset(field_dataset, comp_dataset)

    return kohdataset, tminmax, ycmean
