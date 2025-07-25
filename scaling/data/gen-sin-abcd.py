# Simulation data is generated using Latin hypercube sampling for the calibration parameters.
# The observations are generated by adding noise and discrepancy to the simulation output.
# Assume 1D observations and simulation output.
# Simulation output/observations is the FIRST column of the output CSV file.

import numpy as np
from scipy.stats.qmc import LatinHypercube
from true_funcs import (
    eta,
    TrueParams,
    zeta,
)

x0_range = (0.02, 3.98)
x1_range = (-0.92, 2.98)

t_ranges = np.array(
    [  # calibration parameter ranges
        [0.25, 0.45],  # theta0
        [-3.3, -3.0],  # theta1
        [0.8, 1.2],  # theta2
        [0.6, 1.1],  # theta3
        # [0.4, 0.6],                     # theta4
    ]
)
n_calib_params = t_ranges.shape[0]  # number of calibration parameters
r = 10 * n_calib_params  # number of simulation runs
n_sim = 50  # number of simulation output points
n_obs = 100  # number of observation points


##### RUN SIMULATOR ######
x0_sim = np.linspace(x0_range[0], x0_range[1], n_sim)
x1_sim = 0.0
TP = TrueParams()
t4 = TP.e

# Generate Latin hypercube samples for the calibration parameters
sampler = LatinHypercube(d=n_calib_params)
sample = sampler.random(n=r)  # sample.shape = (r, d)
for i in range(
    n_calib_params
):  # scale sample onto [theta_ranges[i, 0], theta_ranges[i, 1]]
    sample[:, i] = sample[:, i] * (t_ranges[i, 1] - t_ranges[i, 0]) + t_ranges[i, 0]

indexer = np.arange(r)
x0_sim_grid, indicies = np.meshgrid(
    x0_sim,
    indexer,
)
x0_sim_grid = x0_sim_grid.flatten()
indicies = indicies.flatten()
t0_sim_grid = sample[indicies, 0]  # sample[:, 0] is theta0
t1_sim_grid = sample[indicies, 1]  # sample[:, 1] is theta1
t2_sim_grid = sample[indicies, 2]  # sample[:, 2] is theta2
t3_sim_grid = sample[indicies, 3]  # sample[:, 3] is theta3

X_sim = np.array(
    [
        x0_sim_grid,
        np.repeat(x1_sim, r * n_sim),
        t0_sim_grid,
        t1_sim_grid,
        t2_sim_grid,
        t3_sim_grid,
        np.repeat(t4, r * n_sim),
    ]
).T

eta_output = eta(X_sim[:, :2], X_sim[:, 2:])
np.savetxt(
    "sim-abcd.csv",
    np.column_stack(
        (eta_output, X_sim[:, [0, 2, 3, 4, 5]])
    ),  # extract x0 and t0, t1, t2, t3
    delimiter=",",
)

##### GENERATE OBSERVATIONS ######
x0_field = np.linspace(x0_range[0], x0_range[1], n_obs)
x1_field = 0.0

x0_field_grid, x1_field_grid = np.meshgrid(x0_field, x1_field)
x0_field_grid = x0_field_grid.flatten()
x1_field_grid = x1_field_grid.flatten()

X_field = np.array(
    [
        x0_field_grid,
        x1_field_grid,
    ]
).T

obs = zeta(X_field)
obs += np.random.normal(0, np.sqrt(TP.obs_var), n_obs)

np.savetxt(
    "obs-abcd.csv",
    np.column_stack((obs, X_field[:, 0])),  # extract x0
    delimiter=",",
)
