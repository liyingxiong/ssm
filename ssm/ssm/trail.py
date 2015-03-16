import numpy as np
from scipy.sparse import csc_matrix

Bars = np.array([[10., 10., 0., 10., 1., 100000.],
                 [30., 10., 0., 10., 1., 100000.],
                 [10., 30., 0., 10., 1., 100000.],
                 [30., 30., 0., 10., 1., 100000.]])
nBars = len(Bars)

# representation of the pipe positions in terms of node coordinates
coord = np.dstack((Bars[:, 0] - 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                   Bars[:, 1] - 0.5 * Bars[:, 3] * np.sin(Bars[:, 2]),
                   Bars[:, 0] + 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                   Bars[:, 1] + 0.5 * Bars[:, 3] * np.sin(Bars[:, 2])))
bars = np.copy(Bars)
bars[:, 0:4] = coord

X, Y = np.meshgrid(range(20 + 2), range(20 + 2))
xcoord = X.flatten(order='F') * 2
ycoord = Y.flatten(order='F') * 2
# print xcoord
# print ycoord

xi = bars[:, 0]
yi = bars[:, 1]
xj = bars[:, 2]
yj = bars[:, 3]

lbars = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)

# print lbars
r = 15.

di2 = (xcoord[:, None]-xi[None, :])**2 + (ycoord[:, None]-yi[None,:])**2
idxi = np.where(di2 <= r ** 2)

di = np.zeros_like(di2)
Di = np.zeros_like(di2)
aDiadi = np.zeros_like(di2)  # derivative of Di
aDiaxi = np.zeros_like(di2)  # derivative of Di
awaxi = np.zeros_like(di2)

di[idxi] = np.sqrt(di2[idxi]) / r
# aDiadi[idxi] = (1 - di[idxi]) ** 3. * (-20 * di[idxi])
aDiaxi[idxi] = (1 - di[idxi]) ** 3. * (-20 * di[idxi]) * (xi[None, :]-xcoord[:, None])[idxi] / (di[idxi]*r**2)

Di[idxi] = (1. - Di[idxi]) ** 4. * (4. * Di[idxi] + 1.)

awaxi = (aDiaxi * np.sum(Di, axis=0) - Di * np.sum(aDiaxi, axis=0))
awaxi = awaxi / np.sum(Di, axis=0) ** 2

print awaxi
# sum = np.array([4., 2, 3, 4])
#
# print aDiaxi * sum
# print xi[None,:]-xcoord[:, None]


# Di[idxi] = (1.-Di[idxi])**4. * (4.*Di[idxi] + 1.)
# print Di
# print np.sum(Di)
# Di = Di / np.sum(Di, axis=0)
# print Di
# di[idxi] = 0.
# dj = (xcoord[:, None]-xj[None, :])**2 + (ycoord[:, None]-yj[None, :])**2
# dj[dj>r**2]= 0.
#
# Di = np.sqrt(di)/r
# print Di
# Dj = np.sqrt(dj)/r
#
# Di = (1.-Di)**4. * (4.*Di + 1.)
# Dj = (1.-Dj)**4. * (4.*Dj + 1.)
# Di = Di / np.sum(Di, axis=0)
# Dj = Dj / np.sum(Dj, axis=0)
#
# for i in range(len(bars)):
#     t = np.vstack((Di[:, i], Dj[:, i]))
#     nonzero = np.nonzero(t)[1]
#     t = t.T[nonzero]
#     print t


#
# print Di, Dj
# print np.sum(di, axis=0)
# print di
# Di = di[idx]
# print Di
# print Di
# di = (xcoord-xi[None, :])**2 + (ycoord-yi)**2
# dj = (xcoord-xj)**2 + (ycoord-yj)**2
