from Model import *
import matplotlib.pyplot as plt

BaseM = Model(Dp=0, kPA=0.01, kon=0.1, k=0.01, e_wd=0.1, psi=0.1, pP=1,
              xsteps=2, deltax=1, Tmax=1000, beta=1)

BaseM.run()

print(BaseM.X[0][0])
plt.plot(BaseM.X[0])
plt.show()
