import os
import numpy as np
import jax.numpy as jnp

class DataLoader:
    def __init__(
            self, 
            data_field_dir: str,
            data_comp_dir: str,
    ):
        ## Load simulations
        self.xc = np.empty((0,1))
        self.yc = np.empty((0,1))

        initialize = True

        files = os.listdir(data_comp_dir)
        for file in files:
            if file.startswith("sim_") and file.endswith(".npy"):
                filename_parts = file.split("_")
                n_calib_params = len(filename_parts)-1

                if initialize:
                    self.tc = np.empty((0, n_calib_params))
                    initialize = False

                ts = []
                for t in range(n_calib_params):
                    if t+2 < n_calib_params:
                        ts.append(float(filename_parts[t+1]))
                    else:
                        ts.append(float(filename_parts[t+1].split(".n")[0]))

                data_comp = np.load(os.path.join(data_comp_dir, file))
                n_sim_outputs = data_comp.shape[0]

                self.xc = np.vstack((
                    self.xc, 
                    data_comp[:, 0].reshape(-1, 1)
                ))
                self.tc = np.vstack((
                    self.tc, 
                    np.repeat(
                        np.array(ts).reshape(1, n_calib_params),
                        n_sim_outputs, 
                        axis=0
                    )
                ))
                self.yc = np.vstack((
                    self.yc, 
                    data_comp[:, 1].reshape(-1, 1)
                ))

        ## Load observations
        obs = np.load(os.path.join(data_field_dir, "obs.npy"))
        self.xf = obs[:, 0].reshape(-1, 1)
        self.yf = obs[:, 1].reshape(-1, 1)

        #Standardize full response using mean and std of yc
        self.yc_mean = np.mean(self.yc)
        self.t_min = np.min(self.tc, axis=0)
        self.t_max = np.max(self.tc, axis=0)

        self.yc_centered = self.yc - self.yc_mean
        self.yf_centered = self.yf - self.yc_mean
        
        self.tc_normalized = jnp.array((self.tc - self.t_min)/(self.t_max - self.t_min))
       
        self.x_stack = jnp.vstack((self.xf, self.xc), dtype=np.float64)
        self.y = jnp.vstack((self.yf_centered, self.yc_centered), dtype=np.float64)

    def get_data(self):
        return self.x_stack, self.tc_normalized, self.y
        # return self.x_stack, self.tc, self.y

    def transform_y(self, y):
        return y - self.yc_mean

    def inverse_transform_y(self, y):
        return y + self.yc_mean
    