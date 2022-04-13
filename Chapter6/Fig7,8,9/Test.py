from Model import *
import matplotlib.pyplot as plt
import numpy as np


def calc_k_e(dos_base, kon_base, koff_base, psi_base, frac):
    m_base = (kon_base * dos_base) / (koff_base + kon_base * psi_base)
    e_wd = frac / (m_base * (frac - 1) ** 2)
    k = koff_base * np.sqrt(2 * e_wd * m_base + np.sqrt(4 * e_wd * m_base + 1) + 1)
    return k, e_wd


def mon(t, e_wd):
    return (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


def dim(t, e_wd):
    return t - (-1 + np.sqrt((1 + 4 * t * e_wd))) * (1 / e_wd) / 2


k, e_wd = calc_k_e(dos_base=1, kon_base=0.1, koff_base=0.001, psi_base=0.1, frac=0.9)

BaseM = Model(Da=0.1, Dp=0.1, kAP=1, kPA=1, konA=0.1, konP=0.1, koffA=0.01, k=k, e_wd=e_wd, psi=0.1, eAnt=1, pA=1, pP=1,
              xsteps=100, deltax=0.5, Tmax=1000, beta=1)

kant = 1
_m_base = (BaseM.konP * 1) / (0.01 + BaseM.konP * BaseM.psi)
BaseM.kPA = kant / (mon(_m_base, e_wd) + (1 - BaseM.beta) * dim(_m_base, e_wd))
BaseM.kAP = kant / _m_base

BaseM.initiate()
BaseM.run(kill_uni=True)

plt.plot(BaseM.X[0])
plt.plot(BaseM.X[1])
plt.show()
