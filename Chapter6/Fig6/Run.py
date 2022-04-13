import sys
import os

home_direc = os.path.dirname(os.path.realpath(__file__))
sys.path.append(home_direc)
sys.path.append(home_direc + '/../../../polaritymodel')
save_direc = home_direc + '/../../../../ModelData/dimer_kinetic_antagonism_effective_exponent_scheme4/'

from Model import Model
from polaritymodel import ParamSpace2D
import numpy as np
import copy
import itertools

# print(sys.argv[1])

"""
Investigating the effects of membrane binding non-linearity and differential antagonism on effective antagonism exponent

- e vs beta

"""

# Generic parameter set
BaseM = Model(Dp=0, kPA=0.001, kon=0.1, k=0.01, e_wd=0.1, psi=0.1, pP=1,
              xsteps=2, deltax=1, Tmax=1000, beta=0)

koff_vals = [0.0001, 0.001, 0.01]
kant_vals = np.linspace(0, 0.1, 4)
koff_kant_vals = list(itertools.product(koff_vals, kant_vals))

# Parameter range boundaries
p1_boundaries = np.linspace(0.01, 0.99, 5)  # frac
p2_boundaries = (0, 0.25, 0.5, 0.75, 1)  # beta

# Split parameter ranges
param_range_groups = []
for i in range(len(p1_boundaries) - 1):
    for j in range(len(p2_boundaries) - 1):
        p1_range = [p1_boundaries[i], p1_boundaries[i + 1]]
        p2_range = [p2_boundaries[j], p2_boundaries[j + 1]]
        param_range_groups.append([p1_range, p2_range])
len_param_range_groups = len(param_range_groups)


# print(len(koff_kant_vals) * len(param_range_groups)) #= 192


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
Cyt / mem

"""

for i, (koff, kant) in enumerate(koff_kant_vals):
    if int(sys.argv[1]) in range(i * len_param_range_groups, (i + 1) * len_param_range_groups):
        _koff = koff
        _kant = kant
        prange = param_range_groups[int(sys.argv[1]) - i * len_param_range_groups]
        p1_range = prange[0]
        p2_range = prange[1]


        def func(frac, beta):
            m = copy.deepcopy(BaseM)

            # Parameters
            k, e_wd = calc_k_e(dos_base=1, kon_base=m.kon, koff_base=_koff, psi_base=m.psi, frac=frac)
            m.k = k
            m.e_wd = e_wd
            m.beta = beta
            _m_base = (m.kon * 1) / (_koff + m.kon * m.psi)
            m.kPA = _kant / (1 - frac * beta)

            # Run
            m.run()
            p_mem = m.X[0][0]
            p_cyt = m.pP - m.psi * m.X[0][0]

            # Calculate cyt / mem
            return p_cyt / p_mem

###############################################################################################


ParamSpace2D(func, p1_range=p1_range, p2_range=p2_range, cores=32, resolution0=11, resolution_step=2,
             n_iterations=1, direc=save_direc + sys.argv[1], parallel=True, replace=True).run()
