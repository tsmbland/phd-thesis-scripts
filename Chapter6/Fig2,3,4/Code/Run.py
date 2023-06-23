import sys
import os

home_direc = os.path.dirname(os.path.realpath(__file__))
sys.path.append(home_direc)
sys.path.append(home_direc + '/../..')  # for access to model code
sys.path.append(home_direc + '/../../../../../polaritymodel')  # for access to polaritymodel package
save_direc = home_direc + '/../../../../../../ModelData/goehring_kant_koff/'

from Model import Model
from polaritymodel import ParamSpace2D
import numpy as np
import copy

sim_number = int(sys.argv[1])
# sim_number = 0
print(sim_number)

"""



"""

# Generic parameter set
BaseM = Model(Da=0.1, Dp=0.1, konA=0.1, koffA=0.01, konP=0.1, koffP=0.0101, kAP=0.001, kPA=0.001,
              eAP=2, ePA=2, xsteps=100, Tmax=10000, deltax=0.5, psi=0.1, pA=1, pP=1)

dosages = [0.2, 0.4, 0.6, 0.8, 1]
p1_range = (-3, 0)  # kant
p2_range = (-3, -1)  # koff

"""
Dosage imbalance

"""

if sim_number in range(0, len(dosages)):
    dosage = dosages[sim_number]
    BaseM.pP = dosage


    def func(kant, koff):
        m = copy.deepcopy(BaseM)
        m.kAP = 10 ** kant
        m.kPA = 10 ** kant
        m.koffP = (10 ** koff) * 1.01
        m.koffA = 10 ** koff
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
ASI

"""

if sim_number in range(len(dosages), len(dosages) + 1):

    def func(kant, koff):
        m = copy.deepcopy(BaseM)
        m.kAP = 10 ** kant
        m.kPA = 10 ** kant
        m.koffP = (10 ** koff) * 1.01
        m.koffA = 10 ** koff
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

"""
Spontaneous

"""

if sim_number in range(len(dosages) + 1, len(dosages) + 2):

    def func(kant, koff):
        m = copy.deepcopy(BaseM)
        m.kAP = 10 ** kant
        m.kPA = 10 ** kant
        m.koffP = (10 ** koff) * 1.01
        m.koffA = 10 ** koff
        m.initiate2()
        m.run(kill_uni=False)
        a, p = m.X[0], m.X[1]

        # Return state (polarised or not)
        if np.sum(a > p) == len(a):
            # A dominant
            return 1
        elif np.sum(a > p) == 0:
            # P dominant
            return 1
        else:
            # Polarised
            return 2

###############################################################################################


ParamSpace2D(func, p1_range=p1_range, p2_range=p2_range, cores=32, resolution0=11, resolution_step=2,
             n_iterations=7, direc=save_direc + sys.argv[1], parallel=True, replace=False).run()
