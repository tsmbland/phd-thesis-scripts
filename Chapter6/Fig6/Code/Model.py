import numpy as np
from polaritymodel import pdeRK
from numba import njit


class Model:
    def __init__(self, Dp, kPA, kon, k, e_wd, psi, pP, xsteps, deltax, Tmax, beta):
        self.Dp = Dp
        self.kPA = kPA
        self.kon = kon
        self.k = k
        self.e_wd = e_wd
        self.psi = psi
        self.pP = pP
        self.xsteps = xsteps
        self.deltax = deltax
        self.Tmax = Tmax
        self.beta = beta

        # Results
        self.X0 = [np.zeros(self.xsteps), ]
        self.X = None

    def run(self):
        # PDE function
        def pde_func(X):
            return dxdt_pde(np.array(X), Dp=self.Dp, kPA=self.kPA, kon=self.kon, k=self.k, e_wd=self.e_wd,
                            psi=self.psi, pP=self.pP, xsteps=self.xsteps, deltax=self.deltax, beta=self.beta)

        # Run
        soln, time, solns, times = pdeRK(dxdt=pde_func, X0=self.X0, Tmax=self.Tmax, deltat=0.01,
                                         t_eval=np.arange(0, self.Tmax + 0.0001, 1))
        self.X = soln


@njit(cache=True)
def calc_koff(m, k, e_wd):
    return k / np.sqrt(2 * e_wd * m + np.sqrt(4 * e_wd * m + 1) + 1)


@njit(cache=True)
def diffusion(concs, dx=1):
    d = concs[:-2] - 2 * concs[1:-1] + concs[2:]
    return d / (dx ** 2)


@njit(cache=True)
def mon(t, e_wd):
    return (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


@njit(cache=True)
def dim(t, e_wd):
    return t - (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


@njit(cache=True)
def dxdt_pde(X, Dp, kPA, kon, k, e_wd, psi, pP, xsteps, deltax, beta):
    # Species
    Pm = X[0]
    Pc = pP - psi * np.mean(Pm)

    # Reaction matrices
    r = np.zeros((4, xsteps))

    # P reactions
    r[0] = kon * Pc
    r[1] = calc_koff(Pm, k, e_wd) * Pm
    r[2] = kPA * (mon(Pm, e_wd) + (1 - beta) * dim(Pm, e_wd))
    r[3, 1:-1] = Dp * diffusion(Pm, deltax)
    r[3, 0], r[3, -1] = r[3, 1], r[3, -2]

    # PDEs
    dPm = r[0] - r[1] - r[2] + r[3]

    return [dPm, ]
