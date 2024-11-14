from validation.truefuncs import zeta
import numpy as np

####### observations #######
n = 1000        # number of observation points
sigma = 0.05    # observation noise standard deviation
theta = 0.4     # calibration parameter true value

x = np.linspace(0.02, 9.98, n)
f = zeta(x, theta)

filename_extensions = []
def const_discrepancy(x):
    filename_extensions.append('_const')
    disc = 0.14
    return disc
def growing_discrepancy(x):
    filename_extensions.append('_growing')
    rate = 0.14
    return np.exp(rate*x)/10
def sin_discrepancy(x):
    filename_extensions.append('_sin')
    disc = 0.14
    return disc*np.sin(x*np.pi)
def growingsin_discrepancy(x):
    filename_extensions.append('_growingsin')
    rate = 0.14
    return np.exp(rate*x)/10 * 0.14*np.sin(x*np.pi)

###### OPTIONAL DISCREPANCY ######
# f += const_discrepancy(x)
# f += growing_discrepancy(x)
# f += sin_discrepancy(x)
# f += growingsin_discrepancy(x)

obs = f + np.random.normal(0, sigma, n)
data = np.column_stack((x, obs))

np.savetxt(f"toy/field_big{''.join(filename_extensions)}.csv", data, delimiter=',')




####### simulation #######
n = 500 # number of simulation output points
m = 10  # number of simulation runs

x = np.linspace(0.02, 9.98, n)
def LHD(n, xmin, xmax):
    x = np.linspace(xmin, xmax, n, endpoint=False)
    return x + np.random.uniform(0, 1, n)*(xmax-xmin)/n
ts = LHD(m, 0, 1)
print(ts)



sim_output = []
for i, t in enumerate(ts):
    f = zeta(x, t)
    sim_output.append(f)



data = np.empty((len(ts)*len(x), 3))
data[:, 0] = np.tile(x, len(ts))

data[:, 1] = np.repeat(ts, len(x))
data[:, 2] = np.array(sim_output).flatten()

np.savetxt(f"toy/sim_big{''.join(filename_extensions)}.csv", data, delimiter=',')