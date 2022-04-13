import numpy as np
from scipy.integrate import odeint
from polaritymodel import pdeRK
from numba import njit


class Model:
    def __init__(self, Da, Dp, kAP, kPA, konA, konP, koffA, k, e_wd, psi, pA, pP, xsteps, deltax, Tmax, beta, eAnt):
        self.Da = Da
        self.Dp = Dp
        self.kAP = kAP
        self.kPA = kPA
        self.konA = konA
        self.konP = konP
        self.koffA = koffA
        self.k = k
        self.e_wd = e_wd
        self.psi = psi
        self.pA = pA
        self.pP = pP
        self.xsteps = xsteps
        self.deltax = deltax
        self.Tmax = Tmax
        self.beta = beta
        self.eAnt = eAnt

        # Results
        self.X0 = None
        self.X = None

    def initiate(self):

        # ODE function
        def ode_func(X, t):
            return dxdt_ode(X, kAP=0, kPA=0, konA=self.konA, konP=self.konP, koffA=self.koffA, k=self.k, e_wd=self.e_wd,
                            psi=self.psi, eAnt=self.eAnt, pA=self.pA, pP=self.pP, beta=self.beta)

        # Run
        soln = odeint(ode_func, (0, 0), t=np.linspace(0, 10000, 100000))[-1]

        # Polarise
        A0 = soln[0] * 2 * np.r_[np.ones([self.xsteps // 2]), np.zeros([self.xsteps // 2])]
        P0 = soln[1] * 2 * np.r_[np.zeros([self.xsteps // 2]), np.ones([self.xsteps // 2])]
        self.X0 = [A0, P0]

    def initiate2(self):

        # ODE function
        def ode_func(X, t):
            return dxdt_ode(X, kAP=self.kAP, kPA=self.kPA, konA=self.konA, konP=self.konP, koffA=self.koffA,
                            k=self.k, e_wd=self.e_wd, psi=self.psi, eAnt=self.eAnt, pA=self.pA, pP=self.pP,
                            beta=self.beta)

        # Run
        soln = odeint(ode_func, (self.pA / self.psi, 0), t=np.linspace(0, 10000, 100000))[-1]

        # Polarise (slightly)
        A0 = soln[0] * np.linspace(1.01, 0.99, self.xsteps)
        P0 = soln[1] * np.linspace(0.99, 1.01, self.xsteps)
        self.X0 = [A0, P0]

    def run(self, kill_uni=False, t_eval=None):
        if t_eval is None:
            t_eval = np.arange(0, self.Tmax + 0.0001, 1)

        # PDE function
        def pde_func(X):
            return dxdt_pde(np.array(X), Da=self.Da, Dp=self.Dp, kAP=self.kAP, kPA=self.kPA, konA=self.konA,
                            konP=self.konP, koffA=self.koffA, k=self.k, e_wd=self.e_wd, psi=self.psi, eAnt=self.eAnt,
                            pA=self.pA, pP=self.pP, xsteps=self.xsteps, deltax=self.deltax, beta=self.beta)

        # Kill when uniform
        if kill_uni:
            def killfunc(X):
                if np.sum(X[0] > X[1]) == len(X[0]) or np.sum(X[0] > X[1]) == 0:
                    return True
                return False
        else:
            killfunc = None

        # Run
        soln, time, solns, times = pdeRK(dxdt=pde_func, X0=self.X0, Tmax=self.Tmax, deltat=0.01,
                                         t_eval=t_eval, killfunc=killfunc)
        self.X = soln
        return soln, time, solns, times


@njit(cache=True)
def calc_koff(m, k, e_wd):
    return k / np.sqrt(2 * e_wd * m + np.sqrt(4 * e_wd * m + 1) + 1)


@njit(cache=True)
def mon(t, e_wd):
    return (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


@njit(cache=True)
def dim(t, e_wd):
    return t - (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


@njit(cache=True)
def dxdt_ode(X, kAP, kPA, konA, konP, koffA, k, e_wd, psi, eAnt, pA, pP, beta):
    Am = X[0]
    Pm = X[1]
    Ac = pA - psi * Am
    Pc = pP - psi * Pm

    r = np.zeros(6)
    r[0] = konA * Ac
    r[1] = koffA * Am
    r[2] = konP * Pc
    r[3] = calc_koff(Pm, k, e_wd) * Pm
    r[4] = kAP * (Pm ** eAnt) * Am
    r[5] = kPA * (Am ** eAnt) * (mon(Pm, e_wd) + (1 - beta) * dim(Pm, e_wd))

    dAm = r[0] - r[1] - r[4]
    dPm = r[2] - r[3] - r[5]
    return dAm, dPm


@njit(cache=True)
def diffusion(concs, dx=1.):
    d = concs[:-2] - 2 * concs[1:-1] + concs[2:]
    return d / (dx ** 2)


@njit(cache=True)
def dxdt_pde(X, Da, Dp, kAP, kPA, konA, konP, koffA, k, e_wd, psi, eAnt, pA, pP, xsteps, deltax, beta):
    # Species
    Am = X[0]
    Pm = X[1]
    Ac = pA - psi * np.mean(Am)
    Pc = pP - psi * np.mean(Pm)

    # Reaction matrices
    rp = np.zeros((4, xsteps))
    ra = np.zeros((4, xsteps))

    # A reactions
    ra[0] = konA * Ac  # ensures slight imbalance
    ra[1] = koffA * Am
    ra[2] = kAP * (Pm ** eAnt) * Am
    ra[3, 1:-1] = Da * diffusion(Am, deltax)
    ra[3, 0], ra[3, -1] = ra[3, 1], ra[3, -2]

    # P reactions
    rp[0] = konP * Pc
    rp[1] = calc_koff(Pm, k, e_wd) * Pm
    rp[2] = kPA * (Am ** eAnt) * (mon(Pm, e_wd) + (1 - beta) * dim(Pm, e_wd))
    rp[3, 1:-1] = Dp * diffusion(Pm, deltax)
    rp[3, 0], rp[3, -1] = rp[3, 1], rp[3, -2]

    # PDEs
    dAm = ra[0] - ra[1] - ra[2] + ra[3]
    dPm = rp[0] - rp[1] - rp[2] + rp[3]

    return dAm, dPm
