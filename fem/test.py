import sys
sys.path.append('..')
from fem.lfcnsp import LocalFunctionSpace   
from fem.transf import *
import numpy as np

# Create an object of the LFCNSP class
# Test p2d1
dimen = 2
porder = 2
lfsp = LocalFunctionSpace(polysp='P', nsd=dimen, porder=porder)
#lfhc = LocalFunctionSpace('P', dimen, porder)

xe = np.array([[0, 0], [0.5, 0.15], [1.0, 0.7],[-0.3, 0.5], [0.3,0.75],[-0.25,1.2]]).T

# plot the simplex element using xe
import matplotlib.pyplot as plt
plt.figure()
plt.plot(xe[0], xe[1], '-bo') # '-o' option to plot lines connecting the points
# connect the last point to the first point
plt.plot([xe[0, -1], xe[0, 0]], [xe[1, -1], xe[1, 0]], '-bo')


transf = ElementTransformation()
transf.eval_transf_quant(xe, lfsp)

V_e = (1 * lfsp.wqv[:,None] * transf.detG).sum()
print('V_e:', V_e)


C_e = 1 / V_e * (transf.Ge.T * lfsp.wqv[:,None] * transf.detG).sum(axis=0)
print('C_e:', C_e)

S_e = (1 * lfsp.wqf[:,None] * transf.sigf).sum()
print('S_e:', S_e)

plt.show()