from jax import config
config.update("jax_enable_x64", True)

import arviz
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
from kohgpjax.parameters import (
    ParameterPrior,
    PriorDict,
    ModelParameters,
)
# import matplotlib.pyplot as plt
import mici
import numpy as np
import numpyro.distributions as dist

print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)

# plot_style = {
#     'mathtext.fontset': 'cm',
#     'font.family': 'serif',
#     'axes.titlesize': 10,
#     'axes.labelsize': 10,
#     'xtick.labelsize': 6,
#     'ytick.labelsize': 6,
#     'legend.fontsize': 8,
#     'legend.frameon': False,
#     'axes.linewidth': 0.5,
#     'lines.linewidth': 0.5,
#     'axes.labelpad': 2.,
#     'figure.dpi': 150,
# }

# from data.true_funcs import (
#     discrepancy,
#     eta,
#     TrueParams,
#     zeta,
# )
# TP = TrueParams()

def main():
    DATAFIELD = np.loadtxt('data/obs-a.csv', delimiter=',', dtype=np.float32)
    DATACOMP = np.loadtxt('data/sim-a.csv', delimiter=',', dtype=np.float32)

    yf = jnp.reshape(DATAFIELD[:, 0], (-1,1)).astype(jnp.float64)
    yc = jnp.reshape(DATACOMP[:, 0], (-1,1)).astype(jnp.float64)
    xf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATACOMP[:, 1], (-1,1)).astype(jnp.float64)
    tc = jnp.reshape(DATACOMP[:, 2], (-1,1)).astype(jnp.float64)

    # normalising the output is not required provided they are all of a similar scale.
    # But subtracting the mean is sensible as our GP priors assume zero mean.
    ycmean = jnp.mean(yc)
    ycstd = jnp.std(yc)
    yc_standardized = (yc - ycmean)/ycstd # standardized so that E[yc] = 0, V[yc] = 1
    yf_standardized = (yf - ycmean)/ycstd

    # normalising the inputs is not required provided they are all of a similar scale.
    xcmin = jnp.min(xc)
    xcmax = jnp.max(xc)
    # print(f"xcmin: {xcmin}, xcmax: {xcmax}")
    xc_normalized = (xc - xcmin)/(xcmax - xcmin) # Normalize to [0, 1]
    xf_normalized = (xf - xcmin)/(xcmax - xcmin) # Normalize to [0, 1]

    tmin = jnp.min(tc)
    tmax = jnp.max(tc)
    # print(f"tmin: {tmin}, tmax: {tmax}")
    tc_normalized = (tc - tmin)/(tmax - tmin) # Normalize to [0, 1]

    field_dataset = gpx.Dataset(xf_normalized, yf_standardized)
    comp_dataset = gpx.Dataset(jnp.hstack((xc_normalized, tc_normalized)), yc_standardized)

    kohdataset = kgx.KOHDataset(field_dataset, comp_dataset)
    # print(kohdataset)


    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(xf, yf_centered, label='Observations')
    # # ax.plot(xf, zeta(xf), label='True function')
    # rng = np.random.default_rng()
    # ts = rng.permutation(np.unique(tc))[:5]
    # for t in ts:
    #     rows = tc==t
    #     ax.plot(xc[rows], yc_centered[rows], '--', label=f'Simulator t={t:.2f}')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.legend()
    # plt.show()


    class Model(kgx.KOHModel):
        def k_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained['eta']
            return gpx.kernels.ProductKernel(
                kernels=[
                    gpx.kernels.RBF(
                        active_dims=[0],
                        lengthscale=jnp.array(params['lengthscales']['x_0']),
                        variance=jnp.array(1/params['variances']['precision'])
                    ), 
                    gpx.kernels.RBF(
                        active_dims=[1],
                        lengthscale=jnp.array(params['lengthscales']['theta_0']),
                    )
                ]
            )
        
        def k_delta(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained['delta']
            return gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(params['lengthscales']['x_0']),
                    variance=jnp.array(1/params['variances']['precision'])
                )
        
        def k_epsilon_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
            params = params_constrained['epsilon_eta']
            return gpx.kernels.White(
                    active_dims=[0],
                    variance=jnp.array(1/params['variances']['precision'])
                )
        

    # account for the scaling onto [0, 1]
    A = (0.25 - tmin)/(tmax - tmin)
    B = (0.45 - tmin)/(tmax - tmin)
    # print(f"A: {A}, B: {B}")
    prior_dict: PriorDict = {
        'thetas': {
            'theta_0': ParameterPrior(
                dist.Uniform(low=A, high=B),
                name='theta_0',
            ),
        },
        'eta': {
            'variances': {
                    'precision': ParameterPrior(
                    dist.LogNormal(loc=0.0, scale=0.2), # mean near 1
                    name='eta_precision',
                ),
            },
            'lengthscales': {
                'x_0': ParameterPrior(
                    dist.LogNormal(loc=-1.0, scale=0.2), # mean near 0.37
                    name='eta_lengthscale_x_0',
                ),
                'theta_0': ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=3.5), # mean near 0.57
                    name='eta_lengthscale_theta_0',
                ),
            },
        },
        'delta': {
            'variances': {
                'precision': ParameterPrior(
                    dist.LogNormal(loc=-2.0, scale=0.5), # mean near 0.14
                    name='delta_precision',
                ),
            },
            'lengthscales': {
                'x_0': ParameterPrior(
                    dist.LogNormal(loc=-0.7, scale=0.5), # mean near 0.5, larger than eta's x_0
                    name='delta_lengthscale_x_0',
                ),
            },
        },
        'epsilon': { # This is required despite not appearing in the model
            'variances': {
                'precision': ParameterPrior(
                    dist.LogNormal(loc=0.0, scale=0.05), # mean near 1
                    name='epsilon_precision',
                ),
            },
        },
        'epsilon_eta': {
            'variances': {
                'precision': ParameterPrior(
                    dist.Gamma(concentration=10.0, rate=0.001),
                    name='epsilon_eta_precision',
                ),
            },
        },
    }

    model_parameters = ModelParameters(
        prior_dict=prior_dict
    )


    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    ##### Mici #####
    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=model.get_KOH_neg_log_pos_dens_func(),
        backend="jax",
    )
    # system = mici.systems.GaussianEuclideanMetricSystem(
    #     neg_log_dens=model.get_KOH_neg_log_pos_dens_func(),
    #     backend="jax",
    # )
    # system = mici.systems.RiemannianMetricSystem(
    #     neg_log_dens=model.get_KOH_neg_log_pos_dens_func(),
    #     backend="jax",
    #     metric=???
    # )
    integrator = mici.integrators.LeapfrogIntegrator(system)

    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    prior_means = jax.tree.map(
        lambda x: x.inverse(x.distribution.mean), prior_leaves
    )

    init_states = np.array(prior_means) # NOT jnp.array
    # print(f"Initial states: {init_states}")

    f = model.get_KOH_neg_log_pos_dens_func()
    f(init_states)

    tracer_index_dict = {}
    for i, prior in enumerate(model_parameters.priors_flat):
        tracer_index_dict[prior.name] = i
    # print(tracer_index_dict)


    seed = 1234
    n_chain = 2
    n_process = 1 # only 1 works on MacOS
    n_warm_up_iter = 150
    n_main_iter = 50
    rng = np.random.default_rng(seed)

    ##### Mici sampler and adapters #####
    # sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=3)
    # sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng, max_tree_depth=5)
    sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng, max_tree_depth=3)
    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(0.8),
        mici.adapters.OnlineCovarianceMetricAdapter(),
    ]

    def trace_func(state):
        trace = {
            key: state.pos[index] for key, index in tracer_index_dict.items()
        }
        trace['hamiltonian'] = system.h(state)
        return trace

    final_states, traces, stats = sampler.sample_chains(
        n_warm_up_iter, 
        n_main_iter, 
        [init_states] * n_chain, 
        adapters=adapters, 
        n_process=n_process, 
        trace_funcs=[trace_func],
        monitor_stats=("n_step", "accept_stat", "step_size", "diverging")
    )

    print(arviz.summary(traces))

    # with plt.style.context(plot_style):
    #     axes = arviz.plot_pair(
    #         traces,
    #         var_names=["theta_0", "epsilon_eta_precision", "epsilon_precision", "eta_lengthscale_x_0"],
    #         figsize=(6, 6)
    #     )
    #     axes[0, 0].figure.tight_layout()


    traces_transformed = {}
    for var, trace in traces.items():
        if var == 'hamiltonian':
            continue
        index = tracer_index_dict[var]
        traces_transformed[var] = model_parameters.priors_flat[index].forward(np.array(trace))
        if var == 'theta_0':
            trace = traces_transformed[var]
            traces_transformed[var] = list((jnp.array(trace) * (tmax - tmin)) + tmin)

    # rescale the outputs back to the original scale
    traces_transformed['eta_lengthscale_x_0'] = list(
        jnp.array(traces_transformed['eta_lengthscale_x_0']) * (xcmax - xcmin)
    )
    traces_transformed['delta_lengthscale_x_0'] = list(
        jnp.array(traces_transformed['delta_lengthscale_x_0']) * (xcmax - xcmin)
    )
    traces_transformed['eta_lengthscale_theta_0'] = list(
        jnp.array(traces_transformed['eta_lengthscale_theta_0']) * (tmax - tmin)
    )
    traces_transformed['eta_precision'] = list(
        jnp.array(traces_transformed['eta_precision']) * ycstd
    )
    traces_transformed['delta_precision'] = list(
        jnp.array(traces_transformed['delta_precision']) * ycstd
    )
    traces_transformed['epsilon_precision'] = list(
        jnp.array(traces_transformed['epsilon_precision']) * ycstd
    )
    traces_transformed['epsilon_eta_precision'] = list(
        jnp.array(traces_transformed['epsilon_eta_precision']) * ycstd
    )

    params_transformed_flat = {}
    for var, trace in traces_transformed.items():
        params_transformed_flat[var] = np.mean(trace) # This operation is valid across chains when each chain has equal length (I think).
        print(var, ": ", np.mean(trace), '±', np.std(trace))

    params_transformed = jax.tree.unflatten(
        prior_tree,
        params_transformed_flat.values()
    )


    print(arviz.summary(traces_transformed))

    # with plt.style.context(plot_style):
    #     axes = arviz.plot_pair(
    #         traces_transformed,
    #         var_names=["theta_0", "epsilon_eta_precision", "epsilon_precision", "eta_lengthscale_x_0"],
    #         figsize=(6, 6)
    #     )
    #     axes[0, 0].figure.tight_layout()


    # with plt.style.context(plot_style):
    #     axes = arviz.plot_trace(
    #         traces_transformed,
    #         figsize=(9, 2 * (7)),
    #         legend=True,
    #         compact=False,
    #         lines=(
    #             ('theta_0', {}, 0.4),
    #             ('epsilon_precision', {}, 1/TP.obs_var),
    #         )
    #     )
    # for i in range(axes.shape[0]):
    #     left, right = axes[i, 0].get_xlim()
    #     left, right = left*0.9, right*1.1
    #     x = np.linspace(left, right, 1000)
    #     title = axes[i, 0].get_title()
    #     prior_dist = model_parameters.priors_flat[tracer_index_dict[title]].distribution
    #     pdf = jnp.exp(prior_dist.log_prob(x))
    #     axes[i, 0].plot(x, pdf, color='red', linestyle='--', label='Prior')
    #     axes[i, 0].legend()
    # plt.show()
    # plt.show()

if __name__ == '__main__':
    main()