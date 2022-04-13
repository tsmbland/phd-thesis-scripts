import sys
import os

home_direc = os.path.dirname(os.path.realpath(__file__))
sys.path.append(home_direc)
sys.path.append(home_direc + '/../../../polaritymodel')
save_direc = home_direc + '/../../../../ModelData/dimer_kinetic_pPAR_dimerisation_only_alpha_theta_scheme4/'

from Model import Model
from polaritymodel import ParamSpace2D
import numpy as np
import itertools
import copy

print(sys.argv[1])

"""


"""

# Generic parameter set
BaseM = Model(Da=0.1, Dp=0.1, kAP=0.001, kPA=0.001, konA=0.1, konP=0.1, koffA=0.01, k=0.01, e_wd=0.1, psi=0.1, eAnt=1,
              pA=1, pP=1, xsteps=100, deltax=0.5, Tmax=10000, beta=0)

dosages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

p1_range = (-3, 0)  # log_theta
p2_range = (-4, -2)  # log_alpha

frac = 0.6
beta = 0.8


def calc_k_e(dos_base, kon_base, koff_base, psi_base, frac):
    m_base = (kon_base * dos_base) / (koff_base + kon_base * psi_base)
    e_wd = frac / (m_base * (frac - 1) ** 2)
    k = koff_base * np.sqrt(2 * e_wd * m_base + np.sqrt(4 * e_wd * m_base + 1) + 1)
    return k, e_wd


def mon(t, e_wd):
    return (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


def dim(t, e_wd):
    return t - (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


"""
Dosage imbalance P

"""

if int(sys.argv[1]) in range(0, len(dosages)):
    BaseM.pP = dosages[int(sys.argv[1])]


    def func(log_theta, log_alpha):
        m = copy.deepcopy(BaseM)
        alpha = 10 ** log_alpha
        theta = 10 ** log_theta

        # Parameters
        k, e_wd = calc_k_e(dos_base=1, kon_base=m.konP, koff_base=alpha, psi_base=m.psi, frac=frac)
        m.koffA = alpha
        m.k = k
        m.e_wd = e_wd
        m.beta = beta
        _m_base = (m.konP * 1) / (alpha + m.konP * m.psi)
        m.kPA = theta / (1 - frac * beta)
        m.kAP = theta

        # Run
        m.initiate()
        m.run(kill_uni=True)
        a, p = m.X[0], m.X[1]

        # Calculate state (polarised or not)
        if np.sum(a > p) == len(a):
            # A dominant
            return 1
        elif np.sum(a > p) == 0:
            # P dominant
            return 1
        else:
            # Polarised
            return 2

"""
Spontaneous

"""

if int(sys.argv[1]) == (len(dosages)):

    def func(log_theta, log_alpha):
        m = copy.deepcopy(BaseM)
        alpha = 10 ** log_alpha
        theta = 10 ** log_theta

        # Parameters
        k, e_wd = calc_k_e(dos_base=1, kon_base=m.konP, koff_base=alpha, psi_base=m.psi, frac=frac)
        m.koffA = alpha
        m.k = k
        m.e_wd = e_wd
        m.beta = beta
        _m_base = (m.konP * 1) / (alpha + m.konP * m.psi)
        m.kPA = theta / (1 - frac * beta)
        m.kAP = theta

        # Run
        m.initiate2()
        m.run(kill_uni=False)
        a, p = m.X[0], m.X[1]

        # Calculate state (polarised or not)
        if np.sum(a > p) == len(a):
            # A dominant
            return 1
        elif np.sum(a > p) == 0:
            # P dominant
            return 1
        else:
            # Polarised
            return 2

"""
ASI P

"""

if int(sys.argv[1]) == (len(dosages) + 1):

    def func(log_theta, log_alpha):
        m = copy.deepcopy(BaseM)
        alpha = 10 ** log_alpha
        theta = 10 ** log_theta

        # Parameters
        k, e_wd = calc_k_e(dos_base=1, kon_base=m.konP, koff_base=alpha, psi_base=m.psi, frac=frac)
        m.koffA = alpha
        m.k = k
        m.e_wd = e_wd
        m.beta = beta
        _m_base = (m.konP * 1) / (alpha + m.konP * m.psi)
        m.kPA = theta / (1 - frac * beta)
        m.kAP = theta

        # Run
        m.initiate()
        m.run(kill_uni=True)
        a, p = m.X[0], m.X[1]

        # Return state (ASI category)
        if np.sum(a > p) == len(a):
            # A dominant
            return 1
        elif np.sum(a > p) == 0:
            # P dominant
            return 1
        else:
            # Polarised
            ant = np.mean(p[:50])
            post = np.mean(p[50:])
            asi = np.abs((ant - post) / (2 * (ant + post)))
            if asi < 0.2:
                return 2
            elif asi < 0.35:
                return 3
            elif asi < 0.45:
                return 4
            elif asi < 0.49:
                return 5
            else:
                return 6

###############################################################################################


ParamSpace2D(func, p1_range=p1_range, p2_range=p2_range, cores=32, resolution0=11, resolution_step=2,
             n_iterations=7, direc=save_direc + sys.argv[1], parallel=True, replace=False).run()
