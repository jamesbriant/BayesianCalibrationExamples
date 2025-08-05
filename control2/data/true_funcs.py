import numpy as np


# True parameter values
class TrueParams:
    """True parameters for the simulator."""

    a = 0.4
    b = -3.14
    c = 1.0
    d = 1.0
    e = 0.5
    obs_var = 0.05**2

    def get_theta(self) -> np.ndarray:
        """Get the true parameter values as a numpy array."""
        return np.array(
            [
                self.a,
                self.b,
                self.c,
                self.d,
                self.e,
            ]
        ).reshape(1, -1)


def eta(
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """eta function for the simulator.

    :param x: input array
    :param t: parameter array
    :return: output array
    """
    x0 = x[:, 0]
    x1 = x[:, 1]
    t0 = t[:, 0]
    t1 = t[:, 1]
    t2 = t[:, 2]
    t3 = t[:, 3]
    t4 = t[:, 4]

    return t0 * np.sin(t1 * x0 + t4 * x1 + t2) + t3


def zeta(x: np.ndarray) -> np.ndarray:
    """zeta function for the simulator.

    :param x: input array
    :return: output array
    """
    theta = TrueParams().get_theta()
    true_params = np.repeat(
        np.array(theta),
        x.shape[0],
        axis=0,
    )
    return eta(x, true_params) + discrepancy(x)


def discrepancy(x: np.ndarray) -> np.ndarray:
    """discrepancy function for the simulator.

    :param x: input array
    :return: output array
    """
    x0 = x[:, 0]
    x1 = x[:, 1]
    return np.exp(0.14 * x0 - 0.14 * x1) / 10
