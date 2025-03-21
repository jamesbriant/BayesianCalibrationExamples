import os
import sys

import numpy as np
from jax import jit, grad
import mici

from models import toy6d as KOHmodel
from data.dataloader import DataLoader
from kohgpjax.mappings import mapRto01, map01toR, mapRto0inf, map0inftoR

from truefuncs import zeta
eta = zeta


def main(n_warm_up_iter, n_main_iter, hmc_mode, n_steps=None):
    ################## Load data ##################
    dataloader = DataLoader(
        os.path.join(os.getcwd(), 'data/6d/'),
        os.path.join(os.getcwd(), 'data/6d/'),
    )
    print(os.path.join(os.getcwd(), 'data/6d/'))
    data = dataloader.get_data()
    model = KOHmodel.Model(*data)

    tmax = dataloader.t_max
    tmin = dataloader.t_min

    ################## MCMC setup ##################
    theta_0 = 0.5
    theta_1 = 0.5
    theta_2 = 0.5
    theta_3 = 0.5
    theta_4 = 0.5
    theta_5 = 0.5

    ell_eta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/3
    ell_eta_1_0 = 0.3 # np.sqrt(np.var(dataloader.tc))/3
    ell_eta_2_0 = 0.3 # np.sqrt(np.var(dataloader.tc))/3
    ell_eta_3_0 = 0.3
    ell_eta_4_0 = 0.3
    ell_eta_5_0 = 0.3
    ell_eta_6_0 = 0.3
    ell_delta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/5

    lambda_eta_0 = 0.5
    lambda_delta_0 = 10000 # currently no discrepancy so make this very small
    lambda_epsilon_0 = 10
    lambda_epsilon_eta_0 = 10000

    init_states = np.array([[
        map01toR(theta_0), 
        map01toR(theta_1),
        map01toR(theta_2),
        map01toR(theta_3),
        map01toR(theta_4),
        map01toR(theta_5),
        map0inftoR(ell_eta_0_0),
        map0inftoR(ell_eta_1_0),
        map0inftoR(ell_eta_2_0),
        map0inftoR(ell_eta_3_0),
        map0inftoR(ell_eta_4_0),
        map0inftoR(ell_eta_5_0),
        map0inftoR(ell_eta_6_0),
        map0inftoR(ell_delta_0_0),
        map0inftoR(lambda_eta_0),
        map0inftoR(lambda_delta_0),
        map0inftoR(lambda_epsilon_0),
        map0inftoR(lambda_epsilon_eta_0),
    ]])


    param_transform_mici_to_gpjax = lambda x: [
        [ # theta (calibration) parameters
            mapRto01(x[0]),
            mapRto01(x[1]),
            mapRto01(x[2]),
            mapRto01(x[3]),
            mapRto01(x[4]),
            mapRto01(x[5]),
        ],
        [ # lengthscale parameters
            mapRto0inf(x[6]), 
            mapRto0inf(x[7]), 
            mapRto0inf(x[8]),
            mapRto0inf(x[9]),
            mapRto0inf(x[10]),
            mapRto0inf(x[11]),
            mapRto0inf(x[12]),
            mapRto0inf(x[13]),
        ],
        [ # lambda (variance) parameters
            mapRto0inf(x[14]), 
            mapRto0inf(x[15]), 
            mapRto0inf(x[16]), 
            mapRto0inf(x[17]),
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
            'm_theta_0': state.pos[0], 
            'm_theta_1': state.pos[1],
            'm_theta_2': state.pos[2],
            'm_theta_3': state.pos[3],
            'm_theta_4': state.pos[4],
            'm_theta_5': state.pos[5],
            'm_ell_eta_0': state.pos[6], 
            'm_ell_eta_1': state.pos[7],
            'm_ell_eta_2': state.pos[8],
            'm_ell_eta_3': state.pos[9],
            'm_ell_eta_4': state.pos[10],
            'm_ell_eta_5': state.pos[11],
            'm_ell_eta_6': state.pos[12],
            'm_ell_delta_0': state.pos[13],
            'm_lambda_eta': state.pos[14],
            'm_lambda_delta': state.pos[15],
            'm_lambda_epsilon': state.pos[16],
            'm_lambda_epsilon_eta': state.pos[17],
            'hamiltonian': system.h(state)
        }

    final_states, traces, stats = sampler.sample_chains(
        n_warm_up_iter, 
        n_main_iter, 
        init_states, 
        adapters=adapters, 
        n_process=n_chain, # only 1 works on MacOS
        trace_funcs=[trace_func]
    )

    for var, trace in traces.items():
        print(var, ": ", np.mean(trace[0]), '±', np.std(trace[0]))


    traces_transformed = {}
    for var, trace in traces.items():
        if var == 'hamiltonian':
            continue
        var_name = var.split('m_')[1]
        if var_name.startswith('theta'):
            calib_var_num = int(var_name.split('_')[1])
            traces_transformed[var_name] = mapRto01(trace[0])*(tmax[calib_var_num]-tmin[calib_var_num]) + tmin[calib_var_num]
        elif var_name.startswith('ell'):
            traces_transformed[var_name] = mapRto0inf(trace[0])
        elif var_name.startswith('lambda'):
            traces_transformed[var_name] = mapRto0inf(trace[0])
        elif var_name == 'period':
            traces_transformed[var_name] = mapRto0inf(trace[0])

    params_transformed = {}
    for var, trace in traces_transformed.items():
        params_transformed[var] = np.mean(trace)
        print(var, ": ", np.mean(trace), '±', np.std(trace))


    np.savez(
        f'toy6d-{hmc_mode}-W={n_warm_up_iter}-N={n_main_iter}.npz', 
        **traces_transformed
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