# See 
# https://github.com/architecture-building-systems/CaPyBaRA/blob/master/calibration_toyproblem.py

import os
import numpy as np
from typing import List

def _discrepancy(x):
    # return 1.5*np.sin(x)
    return 0


class Simulator:
    def __init__(
        self, 
        theta: List[float],
    ):
        """
        Parameters
        ----------
        theta : List[float]
        List of parameters of length 6 for the simulator.
        """
        if len(theta) != 6:
            raise ValueError("theta must be a list of length 6.")
        self.theta = theta

    def run(self, x):
        a = np.sin(np.pi*(self.theta[1] + x))
        b = np.pi*np.sin(self.theta[2] + x)
        c = self.theta[3]*np.sin(np.pi*x/self.theta[4])
        return self.theta[0]*(a + b)/12 + c + self.theta[5]
    

class Observations(Simulator):
    def __init__(
        self, 
        theta: List[float], 
        obs_std: float,
    ):
        """
        Parameters
        ----------
        theta : List[float]
        List of parameters of length 6 for the simulator.
        obs_std : float
        Standard deviation of the observation noise.
        """
        self.theta = theta
        self.obs_std = obs_std
        self._simulator = Simulator(theta)

    def generate(self, x):
        y = self._simulator.run(x)
        obs_noise = np.random.normal(0, self.obs_std, y.shape)
        return y + _discrepancy(x) + obs_noise


def generate_sim_and_obs(
        n_obs: int, 
        n_sim: int, 
        n_runs: int, 
        theta_true: List[float],
        obs_std: float
    ) -> None:
    """
    Generate observations and simulations for the toy 2D -> 1D problem.
    
    Parameters
    ----------
    n_obs : int
    Number of observations to generate.
    n_sim : int
    Number of output points per simulation run.
    n_runs : int
    Number of simulations to run.
    theta_true : float
    True parameter values for generating the observations.
    obs_std : float
    Standard deviation of the observation noise.
        
    Returns
    -------
    None
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    #### Generate observations ####
    x = np.linspace(100, 500, n_obs)
    # x = 600*np.random.sample(n_obs)

    obs = Observations(theta_true, obs_std)
    z = obs.generate(x)

    data_obs = np.column_stack((x, z))
    np.save(
        os.path.join(dir_path, 'obs.npy'), 
        data_obs
    )


    #### Generate simulations ####
    x = np.linspace(100, 500, n_sim)

    # Experimental design - LHD with n_runs 
    a = [90, -6.6, -4.4, 90, 450, 135]
    b = [110, -5.4, -3.6, 110, 550, 165]
    design = np.zeros((n_runs, 6))
    for i in range(len(theta_true)):
        t = np.linspace(a[i], b[i]-1, n_runs, endpoint=True) + np.random.rand(n_runs)*(b[i]-1-a[i])/n_runs
        design[:, i] = np.random.permutation(t)

    # run the simulations and save data to file
    for row in design:
        sim = Simulator(row)
        y = sim.run(x)
        data_sim = np.column_stack((x, y))
        np.save(
            os.path.join(
                dir_path,
                f'sim_{row[0]:.2f}_{row[1]:.2f}_{row[2]:.2f}_{row[3]:.2f}_{row[4]:.2f}_{row[5]:.2f}.npy'
            ),
            data_sim
        )

    return


if __name__ == '__main__':
    theta_true = [100/12, -6, -4, 100, 500, 175]
    obs_std = 20
    n_obs = 600
    n_sim = 250
    n_runs = 5

    print(f"Generating observations and running {n_runs} simulations.")

    generate_sim_and_obs(n_obs, n_sim, n_runs, theta_true, obs_std)