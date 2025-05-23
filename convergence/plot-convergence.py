import os
import numpy as np
import matplotlib.pyplot as plt

N_fixed = {
    'dynamic': 500,
    'static-3': 1000,
}
W_fixed = {
    'dynamic': 250,
    'static-3': 500,
}

if __name__ == '__main__':
    for hmc_mode in ['dynamic', 'static-3']:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        dir = os.path.join(current_file_dir, hmc_mode)
        files = os.listdir(dir)

        ########### fixed N ###########

        params_transformed_means = {}
        params_transformed_stds = {}
        n_warm_up_iters = []

        for file in files:
            if file.split('=')[2] == f'{N_fixed[hmc_mode]}.npz':
                data = np.load(
                    os.path.join(dir, file), 
                    allow_pickle=True,
                )
                means = data['params_transformed_mean'].item()
                stds = data['params_transformed_std'].item()
                n_warm_up_iters.append(int(file.split('=')[1].split('-')[0]))
                for var in means:
                    if var not in params_transformed_means:
                        params_transformed_means[var] = []
                        params_transformed_stds[var] = []
                    params_transformed_means[var].append(means[var])
                    params_transformed_stds[var].append(stds[var])

        # sort by number of warm-up iterations
        sorted_indices = np.argsort(n_warm_up_iters)
        n_warm_up_iters = np.array(n_warm_up_iters)[sorted_indices]
        print(n_warm_up_iters)
        for var in params_transformed_means:
            params_transformed_means[var] = np.array(params_transformed_means[var])[sorted_indices]
            params_transformed_stds[var] = np.array(params_transformed_stds[var])[sorted_indices]

        ################## Make figure ##################
        fig, axes = plt.subplots(4, 2, figsize=(8, 12))

        for i, var in enumerate(params_transformed_means):
            ax = axes.flatten()[i]
            ax.set_xscale('log')
            ax.plot(n_warm_up_iters, params_transformed_means[var], 'o-', label='mean')
            ax.fill_between(
                n_warm_up_iters, 
                np.array(params_transformed_means[var])-np.array(params_transformed_stds[var]), 
                np.array(params_transformed_means[var])+np.array(params_transformed_stds[var]), 
                alpha=0.3
            )
            ax.set_xlabel('Number of warm-up iterations, W')
            ax.set_ylabel('mean')

            ax2 = ax.twinx()
            ax2.plot(n_warm_up_iters, params_transformed_stds[var], 'x--', color='tab:orange', label='std')
            ax2.set_ylabel('std')

            ax.legend(loc=2)
            ax2.legend(loc=1)

            ax.set_title(var)

        axes[0,0].axhline(0.4, color='k', linestyle='--')
        axes[3,0].axhline(400, color='k', linestyle='--')
        fig.suptitle(f'Convergence of parameters for {hmc_mode} sampler\nN={N_fixed[hmc_mode]}')
        plt.tight_layout()
        plt.savefig(os.path.join(current_file_dir, f'params-convergence-fixed-N-{hmc_mode}.png'))
        plt.close()




        # ########### fixed W ###########

        params_transformed_means = {}
        params_transformed_stds = {}
        n_main_iters = []

        for file in files:
            if file.split('=')[1] == f'{W_fixed[hmc_mode]}-N':
                data = np.load(
                    os.path.join(dir, file), 
                    allow_pickle=True,
                )
                means = data['params_transformed_mean'].item()
                stds = data['params_transformed_std'].item()
                n_main_iters.append(int(file.split('=')[2].split('.')[0]))
                for var in means:
                    if var not in params_transformed_means:
                        params_transformed_means[var] = []
                        params_transformed_stds[var] = []
                    params_transformed_means[var].append(means[var])
                    params_transformed_stds[var].append(stds[var])

        # sort by number of main iterations
        sorted_indices = np.argsort(n_main_iters)
        n_main_iters = np.array(n_main_iters)[sorted_indices]
        print(n_main_iters)
        for var in params_transformed_means:
            params_transformed_means[var] = np.array(params_transformed_means[var])[sorted_indices]
            params_transformed_stds[var] = np.array(params_transformed_stds[var])[sorted_indices]

        ################## Make figure ##################
        fig, axes = plt.subplots(4, 2, figsize=(8, 12))

        for i, var in enumerate(params_transformed_means):
            ax = axes.flatten()[i]
            ax.set_xscale('log')
            ax.plot(n_main_iters, params_transformed_means[var], 'o-', label='mean')
            ax.fill_between(
                n_main_iters, 
                np.array(params_transformed_means[var])-np.array(params_transformed_stds[var]), 
                np.array(params_transformed_means[var])+np.array(params_transformed_stds[var]), 
                alpha=0.3
            )
            ax.set_xlabel('Number of iterations, N')
            ax.set_ylabel('mean')

            ax2 = ax.twinx()
            ax2.plot(n_main_iters, params_transformed_stds[var], 'x--', color='tab:orange', label='std')
            ax2.set_ylabel('std')

            ax.legend(loc=2)
            ax2.legend(loc=1)

            ax.set_title(var)

        axes[0,0].axhline(0.4, color='k', linestyle='--')
        axes[3,0].axhline(400, color='k', linestyle='--')
        fig.suptitle(f'Convergence of parameters for {hmc_mode} sampler\nW={W_fixed[hmc_mode]}')
        plt.tight_layout()
        plt.savefig(os.path.join(current_file_dir, f'params-convergence-fixed-W-{hmc_mode}.png'))
        plt.close()