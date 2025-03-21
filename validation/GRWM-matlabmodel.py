from models import matlabmodel as KOHmodel
from data.dataloadermatlab import DataLoader
from mappings import mapRto01, map01toR, mapRto0inf, map0inftoR
from MATLAB_mappings import ell2rho, beta2ell
from jax import jit

import numpy as np

import arviz
import mici


dataloader = DataLoader('data/matlab/simple_field_medium.csv', 'data/matlab/simple_comp.csv')
data = dataloader.get_data() # loads normalised/standardised data
model = KOHmodel.MatlabModel(*data)

tmax = dataloader.t_max
tmin = dataloader.t_min



init_states = np.array([[
    map01toR(0.4257), 
    map0inftoR(beta2ell(51.5551)), #these are the beta values!!!
    map0inftoR(beta2ell(3.5455)), 
    # map0inftoR(beta2ell(2)), 
    map0inftoR(0.25557), 
    map0inftoR(37.0552), 
    map0inftoR(10030.5142), 
    map0inftoR(79548.2126)
]])

param_transform_mici_to_gpjax = lambda x: [
    [ # theta (calibration) parameters
        mapRto01(x[0])
    ],
    [ # lengthscale parameters
        mapRto0inf(x[1]), 
        mapRto0inf(x[2]), 
    ],
    [ # lambda (variance) parameters
        mapRto0inf(x[3]), 
        mapRto0inf(x[4]), 
        mapRto0inf(x[5]), 
        mapRto0inf(x[6])
    ]
]

jitted_neg_log_posterior_density = jit(
    model.get_KOH_neg_log_pos_dens_func(
        param_transform_mici_to_gpjax
    )
)

def neg_log_pos_dens(x):
    return np.asarray(jitted_neg_log_posterior_density(x))

##### Mici #####
system = mici.systems.EuclideanMetricSystem(
    neg_log_dens=neg_log_pos_dens,
    grad_neg_log_dens=lambda q: q * 0,
)
integrator = mici.integrators.LeapfrogIntegrator(system)

seed = 1234
n_chain = 1 # only 1 works on MacOS
n_warm_up_iter = 4000
n_main_iter = 10000
rng = np.random.default_rng(seed)

##### Mici sampler and adapters #####
sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=1)
adapters = [
    mici.adapters.DualAveragingStepSizeAdapter(0.234),
    mici.adapters.OnlineCovarianceMetricAdapter()
]

def trace_func(state):
    return {
        'theta': state.pos[0], 
        'ell_eta_1': state.pos[1], 
        'ell_eta_2': state.pos[2],
        'lambda_eta': state.pos[3],
        'lambda_delta': state.pos[4],
        'lambda_epsilon': state.pos[5],
        'lambda_epsilon_eta': state.pos[6],
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



traces_transformed = {}
for var, trace in traces.items():
    if var == 'theta':
        traces_transformed[var] = mapRto01(trace[0])*(tmax-tmin) + tmin
    elif var.startswith('ell'):
        traces_transformed[var] = mapRto0inf(trace[0])
    elif var.startswith('lambda'):
        traces_transformed[var] = mapRto0inf(trace[0])

params_transformed = {}
for var, trace in traces_transformed.items():
    params_transformed[var] = np.mean(trace)
    print(var, ": ", np.mean(trace), '±', np.std(trace))


print("Parameters transformed: ", params_transformed)

# Save the trace
# arviz.to_netcdf(traces, 'output/mici_trace.nc')
# arviz.to_netcdf(traces_transformed, 'output/mici_trace_transformed.nc')

# print ESS
print("ESS: ", arviz.ess(traces_transformed))