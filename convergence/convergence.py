import sys
from pathlib import Path

import numpy as np
from jax import jit, grad
import mici

from models import toymodel_sin as KOHmodel
from data.dataloader import DataLoader
from kohgpjax.mappings import mapRto01, map01toR, mapRto0inf, map0inftoR


################## Usage ##################
# python path/to/convergence.py n_warm_up_iter n_main_iter hmc_mode [n_steps]
# e.g.
# python path/to/convergence.py 100 100 dynamic
# or
# python path/to/convergence.py 100 100 static 3
# or
# python path/to/convergence.py 100 100 GRWM

def main(n_warm_up_iter, n_main_iter, hmc_mode, n_steps=None):
    current_file_dir = Path(__file__).resolve().parent


    ################## Load data ##################
    field_path = current_file_dir.joinpath('data/toy/field_sin.csv')
    sim_path = current_file_dir.joinpath('data/toy/sim_sin.csv')

    dataloader = DataLoader(field_path, sim_path)
    data = dataloader.get_data()
    model = KOHmodel.Model(*data)

    tmax = dataloader.t_max
    tmin = dataloader.t_min

    ################## MCMC setup ##################
    theta_0 = 0.5

    ell_eta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/3
    ell_eta_1_0 = 0.3 # np.sqrt(np.var(dataloader.tc))/3
    ell_delta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/5

    lambda_eta_0 = 1
    lambda_delta_0 = 30
    lambda_epsilon_0 = 400
    lambda_epsilon_eta_0 = 10000

    period_delta_0 = 2

    init_states = np.array([[
        map01toR(theta_0), 
        map0inftoR(ell_eta_0_0),
        map0inftoR(ell_eta_1_0),
        map0inftoR(ell_delta_0_0),
        map0inftoR(lambda_eta_0),
        map0inftoR(lambda_delta_0),
        map0inftoR(lambda_epsilon_0),
        map0inftoR(lambda_epsilon_eta_0),
        map0inftoR(period_delta_0),
    ]])

    param_transform_mici_to_gpjax = lambda x: [
        [ # theta (calibration) parameters
            mapRto01(x[0]),
        ],
        [ # lengthscale parameters
            mapRto0inf(x[1]), 
            mapRto0inf(x[2]), 
            mapRto0inf(x[3]),
        ],
        [ # lambda (variance) parameters
            mapRto0inf(x[4]), 
            mapRto0inf(x[5]), 
            mapRto0inf(x[6]), 
            mapRto0inf(x[7]),
        ],
        [ # period parameters
            mapRto0inf(x[8]),
        ]
    ]

    jitted_neg_log_posterior_density = jit(
        model.get_KOH_neg_log_pos_dens_func(
            param_transform_mici_to_gpjax
        )
    )
    grad_neg_log_posterior_density = jit(grad(
        model.get_KOH_neg_log_pos_dens_func(
            param_transform_mici_to_gpjax
        )
    ))

    def neg_log_pos_dens(x):
        return np.asarray(jitted_neg_log_posterior_density(x))

    def grad_neg_log_pos_dens(x):
        return np.asarray(grad_neg_log_posterior_density(x))
    if hmc_mode == 'GRWM':
        grad_neg_log_pos_dens = lambda q: q * 0

    ##### Mici #####
    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_pos_dens,
        grad_neg_log_dens=grad_neg_log_pos_dens,
    )
    integrator = mici.integrators.LeapfrogIntegrator(system)


    ################## Run MCMC ##################
    seed = 1234
    n_chain = 1
    rng = np.random.default_rng(seed)

    ##### Mici sampler and adapters #####
    acceptance_rate = 0.8
    if hmc_mode == 'GRWM':
        sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=1)
        acceptance_rate = 0.234
    elif hmc_mode == 'dynamic':
        sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)
    else:
        sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_steps)

    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(acceptance_rate),
        mici.adapters.OnlineCovarianceMetricAdapter()
    ]

    def trace_func(state):
        return {
            'm_theta': state.pos[0], 
            'm_ell_eta_0': state.pos[1], 
            'm_ell_eta_1': state.pos[2],
            'm_ell_delta_0': state.pos[3],
            'm_lambda_eta': state.pos[4],
            'm_lambda_delta': state.pos[5],
            'm_lambda_epsilon': state.pos[6],
            'm_lambda_epsilon_eta': state.pos[7],
            'm_period_delta': state.pos[8],
            'hamiltonian': system.h(state)
        }


    params_transformed_mean = {}
    params_transformed_std = {}

    final_states, traces, stats = sampler.sample_chains(
        n_warm_up_iter, 
        n_main_iter, 
        init_states, 
        adapters=adapters, 
        n_process=n_chain, # only 1 works on MacOS
        trace_funcs=[trace_func]
    )

    for var, trace in traces.items():
        if var == 'hamiltonian':
            continue
        var_name = var.split('m_')[1]
        if var_name == 'theta':
            params_transformed_mean[var_name] = np.mean(mapRto01(trace[0])*(tmax-tmin) + tmin)
            params_transformed_std[var_name] = np.std(mapRto01(trace[0])*(tmax-tmin) + tmin)
        elif var_name.startswith('ell'):
            params_transformed_mean[var_name] = np.mean(mapRto0inf(trace[0]))
            params_transformed_std[var_name] = np.std(mapRto0inf(trace[0]))
        elif var_name.startswith('lambda'):
            params_transformed_mean[var_name] = np.mean(mapRto0inf(trace[0]))
            params_transformed_std[var_name] = np.std(mapRto0inf(trace[0]))


    np.savez(
        current_file_dir.joinpath(f'W={n_warm_up_iter}-N={n_main_iter}.npz'),
        params_transformed_mean=params_transformed_mean, 
        params_transformed_std=params_transformed_std
    )

if __name__ == '__main__':
    ################## Parse arguments ##################
    n_warm_up_iter = int(sys.argv[1])
    n_main_iter = int(sys.argv[2])
    hmc_mode = sys.argv[3]
    if hmc_mode == 'dynamic':
        sampler_name = 'dynamic'
    elif hmc_mode == 'GRWM':
        sampler_name = 'GRWM'
    elif hmc_mode == 'static':
        n_steps = int(sys.argv[4])
        sampler_name = f'static-{n_steps}'
    else:
        raise ValueError('Invalid sampler')
    
    main(
        n_main_iter=n_main_iter, 
        n_warm_up_iter=n_warm_up_iter, 
        hmc_mode=sampler_name,
        n_steps=n_steps if hmc_mode == 'static' else None
    )