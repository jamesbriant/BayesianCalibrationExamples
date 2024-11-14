import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Data
######################################################################
chain_lengths = np.array([ # Units: number of posterior samples
    100, 200, 500, 750, 1000,
    1500, 2000, 3000, 4000, 5000,
    7500, 10000, 14000, 20000, 30000,
    50000
])
elapsed_times_mcmc_gpjax = np.array([ # Units: seconds
    196.749, 206.402, 261.075, 304.549, 333.062,
    426.566, 517.982, 681.971, 818.537, 1033.813,
    1445.583, 1873.658, 2545.666, 3559.045, 5225.075,
    8650.231
])
elapsed_times_mcmc_matlab = np.array([ # Units: seconds
    51.862924, 59.392028, 74.717352, 87.437760, 96.925808,
    119.241484, 147.016532, 194.317163, 238.705580, 286.932739,
    422.004247, 519.850071, 753.612832, 987.540140, 1542.575931,
    2374.717598
])
elapsed_times_script_real_gpjax = np.array([ # Units: seconds
    15.776, 14.408, 17.183, 19.512, 21.091, 26.246, 31.244, 
    39.766, 46.824, 59.873, 135.567, 145.185, 222.839, 
    313.837, 442.295, 746.097
])
elapsed_times_script_user_gpjax = np.array([ # Units: seconds
    3*60 + 23.744, 3*60 + 32.671, 4*60 + 27.582, 5*60 + 10.794, 
    5*60 + 38.867, 7*60 + 12.060, 8*60 + 42.511, 11*60 + 25.403, 
    13*60 + 41.169, 17*60 + 15.083, 24*60 + 3.845, 31*60 + 9.166, 
    42*60 + 17.340, 59*60 + 4.438, 86*60 + 38.572, 143*60 + 23.136
])
elapsed_times_script_sys_gpjax = np.array([ # Units: seconds
    3.958, 4.644, 4.451, 4.681, 5.125, 5.444, 6.457, 7.527, 
    8.276, 9.604, 12.549, 15.315, 19.243, 25.498, 37.373, 
    57.995
])
elapsed_times_script_real_matlab = np.array([ # Units: seconds
    60.435, 67.886, 83.403, 95.810, 105.398, 127.890, 155.570, 
    143.096, 247.400, 295.479, 430.592, 528.379, 761.995, 
    1037.395, 1551.646, 2383.677
])
elapsed_times_script_user_matlab = np.array([ # Units: seconds
    14*60 + 30.385, 16*60 + 31.175, 20*60 + 37.642, 23*60 + 38.138, 
    27*60 + 21.636, 33*60 + 12.787, 40*60 + 32.155, 53*60 + 22.903, 
    66*60 + 46.846, 81*60 + 29.264, 113*60 + 0.516, 143*60 + 42.825, 
    201*60 + 15.651, 278*60 + 41.728, 417*60 + 51.356, 670*60 + 59.694
])
elapsed_times_script_sys_matlab = np.array([ # Units: seconds
    4*60 + 24.725, 5*60 + 1.510, 6*60 + 12.442, 7*60 + 14.783, 
    8*60 + 10.197, 10*60 + 6.109, 11*60 + 58.220, 15*60 + 53.915, 
    20*60 + 4.200, 24*60 + 10.107, 33*60 + 54.486, 43*60 + 16.135, 
    61*60 + 59.992, 82*60 + 14.305, 126*60 + 29.707, 196*60 + 28.536
])

######################################################################
# Plot
######################################################################

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

fig.suptitle('Elapsed Times (script files)')

for axis in axes.flatten():
    # axis.grid(True)
    # axis.set_xscale('log')
    # axis.set_yscale('log')
    pass

ax = axes.flatten()[0]
ax.plot(chain_lengths, elapsed_times_script_real_gpjax, 'o-', label='Real (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_real_matlab, 'o-', label='Real (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('"REAL"')
ax.legend()

ax = axes.flatten()[1]
ax.plot(chain_lengths, elapsed_times_script_user_gpjax, 'o-', label='User (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_user_matlab, 'o-', label='User (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('"USER"')
ax.legend()

ax = axes.flatten()[2]
ax.plot(chain_lengths, elapsed_times_script_sys_gpjax, 'o-', label='Sys (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_sys_matlab, 'o-', label='Sys (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('"SYS"')
ax.legend()

ax = axes.flatten()[3]
ax.plot(chain_lengths, elapsed_times_mcmc_gpjax, 'o-', label='MCMC (GPJax)')
ax.plot(chain_lengths, elapsed_times_mcmc_matlab, 'o-', label='MCMC (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('Elapsed Time (mcmc files)')
ax.legend()
ax.tick_params(color='tab:red', labelcolor='tab:red')
for spine in ax.spines.values():
    spine.set_edgecolor('tab:red')

plt.tight_layout()
plt.show()
plt.close()





fig, axes = plt.subplots(2, 2, figsize=(9, 8))

fig.suptitle('Elapsed Times')

for axis in axes.flatten():
    # axis.grid(True)
    # axis.set_xscale('log')
    # axis.set_yscale('log')
    pass

ax = axes.flatten()[0]
ax.plot(chain_lengths, elapsed_times_script_real_gpjax, 'o-', label='Real (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_real_matlab, 'o-', label='Real (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('"REAL"')
ax.legend()

ax = axes.flatten()[1]
ax.plot(chain_lengths, elapsed_times_mcmc_gpjax, 'o-', label='MCMC (GPJax)')
ax.plot(chain_lengths, elapsed_times_mcmc_matlab, 'o-', label='MCMC (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('Elapsed Time (mcmc files)')
ax.legend()
ax.tick_params(color='tab:red', labelcolor='tab:red')
for spine in ax.spines.values():
    spine.set_edgecolor('tab:red')

ax = axes.flatten()[2]
ax.plot(chain_lengths, elapsed_times_mcmc_gpjax, 'o-', label='MCMC (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_real_gpjax, 'x--', label='Real (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_user_gpjax, 'x--', label='User (GPJax)')
ax.plot(chain_lengths, elapsed_times_script_sys_gpjax, 'x--', label='Sys (GPJax)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('Elapsed Time (GPJax)')
ax.legend()

ax = axes.flatten()[3]
ax.plot(chain_lengths, elapsed_times_mcmc_matlab, 'o-', label='MCMC (MATLAB)')
ax.plot(chain_lengths, elapsed_times_script_real_matlab, 'x--', label='Real (MATLAB)')
ax.plot(chain_lengths, elapsed_times_script_user_matlab, 'x--', label='User (MATLAB)')
ax.plot(chain_lengths, elapsed_times_script_sys_matlab, 'x--', label='Sys (MATLAB)')
ax.set_xlabel('Chain Length')
ax.set_ylabel('Elapsed Time (s)')
ax.set_title('Elapsed Time (MATLAB)')
ax.legend()

plt.tight_layout()
plt.show()
plt.close()