'''
input
bars -- the array containing the configuration (position, length, cross-sectional area)
        of the pipes
xcoord -- array containing the x coordinates of the nodes of the continuum mesh
ycoord -- array containing the y coordinates of the nodes of the continuum mesh
nelx -- number of elements in x direction
nely -- number of elements in y direction
r -- the influence radius
output
KBar -- the equivalent conduction matrix of the pipes
akax_list, akay_list, akaa_list -- sensitivities of the objective function
'''
import numpy as np
from scipy.sparse import coo_matrix


def stiffness_spreading(bars, xcoord, ycoord, nelx, nely, r):

    xi = bars[:, 0]
    yi = bars[:, 1]
    xj = bars[:, 2]
    yj = bars[:, 3]

    Ci, aCiaxi, aCiayi = transform(xi, yi, xcoord, ycoord, r)
    Cj, aCjaxj, aCjayj = transform(xj, yj, xcoord, ycoord, r)

    lbars = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
    factor = bars[:, 4] * bars[:, 5] / lbars
    ke = np.array([[1., -1.],
                   [-1., 1.]])

    keq_flat = np.array([])
    row_flat = np.array([])
    col_flat = np.array([])
    # derivative of the equivalent conduction matrix with respect to x
    akax_list = []
    # derivative of the equivalent conduction matrix with respect to y
    akay_list = []
    # derivative of the equivalent conduction matrix with respect to angle
    akaa_list = []

    for i in range(len(bars)):
        C = np.vstack((Ci[:, i], Cj[:, i]))
        nonzero = np.unique(np.nonzero(C)[1])
        row_idx = np.repeat(nonzero, len(nonzero))
        col_idx = np.tile(nonzero, len(nonzero))
        row_flat = np.hstack((row_flat, np.copy(row_idx).flatten()))
        col_flat = np.hstack((col_flat, np.copy(col_idx).flatten()))
        k = factor[i] * ke

        # calculate the elements in the equivalent conduction matrix
        C = C.T[nonzero]
        keq_flat = np.hstack((keq_flat, C.dot(k).dot(C.T).flatten()))

        # calculate the sensitivities of the equivalent matrix with respect to
        # x
        aCax = np.vstack((aCiaxi[:, i], aCjaxj[:, i])).T[nonzero]
        ax = aCax.dot(k).dot(C.T)
        akaxi = coo_matrix(((ax + ax.T).flatten(), (row_idx, col_idx)),  shape=(
            (nelx + 1) * (nely + 1), (nelx + 1) * (nely + 1))).tocsc()
        akax_list.append(akaxi)
#         print type(akax_list[0])

        # calculate the sensitivities of the equivalent matrix with respect to
        # y
        aCay = np.vstack((aCiayi[:, i], aCjayj[:, i])).T[nonzero]
        ay = aCay.dot(k).dot(C.T)
        akayi = coo_matrix(((ay + ay.T).flatten(), (row_idx, col_idx)),  shape=(
            (nelx + 1) * (nely + 1), (nelx + 1) * (nely + 1))).tocsc()
        akay_list.append(akayi)

        # calculate the sensitivities of the equivalent matrix with respect to
        # angle
        aCiaa = 0.5 * \
            aCiaxi[:, i] * (yj[i] - yi[i]) - 0.5 * \
            aCiayi[:, i] * (xj[i] - xi[i])
        aCjaa = -0.5 * \
            aCjaxj[:, i] * (yj[i] - yi[i]) + 0.5 * \
            aCjayj[:, i] * (xj[i] - xi[i])
        aCaa = np.vstack((aCiaa, aCjaa)).T[nonzero]
        aa = aCaa.dot(k).dot(C.T)
        akaai = coo_matrix(((aa + aa.T).flatten(), (row_idx, col_idx)), shape=(
            (nelx + 1) * (nely + 1), (nelx + 1) * (nely + 1))).tocsc()
        akaa_list.append(akaai)

#     row_idx = np.array(row_list).flatten()
#     col_idx = np.array(col_list).flatten()
#     keq_flat = np.array(keq_list).flatten()
#     print np.array(keq_list).shape
#     print row_idx.shape
#     print col_idx.shape

#     try:
    keq = coo_matrix((keq_flat, (row_flat, col_flat)),
                     shape=((nelx + 1) * (nely + 1), (nelx + 1) * (nely + 1)))
#     except:
#         print keq_list

    return keq.tocsc(), akax_list, akay_list, akaa_list


def transform(xi, yi, xcoord, ycoord, r):
    '''return the transformation matrix and its sensitivities'''
    di2 = (xcoord[:, None]-xi[None, :])**2. + (ycoord[:, None]-yi[None,:])**2.
    idxi = np.where(di2 <= r ** 2.)

    di = np.zeros_like(di2)
    Di = np.zeros_like(di2)
    aDiaxi = np.zeros_like(di2)  # derivative of Di
    aDiayi = np.zeros_like(di2)  # derivative of Di

    di[idxi] = np.sqrt(di2[idxi]) / r
    Di[idxi] = (1. - di[idxi]) ** 4. * (4. * di[idxi] + 1.)  # RBF

    aDiaxi[idxi] = (1-di[idxi])**3.*(-20*di[idxi]) * (xi[None, :]-xcoord[:, None])[idxi] / (di[idxi]*r**2+np.finfo(np.double).tiny)
    awaxi = (aDiaxi * np.sum(Di, axis=0) - Di *
             np.sum(aDiaxi, axis=0)) / np.sum(Di, axis=0) ** 2.

    aDiayi[idxi] = (1-di[idxi])**3.*(-20*di[idxi]) * (yi[None, :]-ycoord[:, None])[idxi] / (di[idxi]*r**2+np.finfo(np.double).tiny)
    awayi = (aDiayi * np.sum(Di, axis=0) - Di *
             np.sum(aDiayi, axis=0)) / np.sum(Di, axis=0) ** 2.

    Ci = Di / np.sum(Di, axis=0)

    return Ci, awaxi, awayi

if __name__ == '__main__':

    nelx = 40
    nely = 40
    X, Y = np.meshgrid(range(nelx + 1), range(nely + 1))
    xcoord = X.flatten(order='F')
    ycoord = Y.flatten(order='F')
    r = 4

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
    Keq, akax_lst, akay_lst, akaa_lst = stiffness_spreading(
        bars, xcoord, ycoord, nelx, nely, r)

    print akax_lst[0]


#     bars = np.array([[0.5, 0.5, 2.5, 2.5, 1., 1.],
#                      [0.5, 0.5, 2.5, 2.5, 0., 0.]])
#
#     bars2 = np.array([[0.501, 0.499, 2.499, 2.501, 1., 1.],
#                       [0.5, 0.5, 2.5, 2.5, 0., 0.]])
#
#     keq1, akax1, akay1, akaa1 = stiffness_spreading(
#         bars, x, y, nelx, nely, 1.4)
#     keq2, akax2, akay2, akaa2 = stiffness_spreading(
#         bars2, x, y, nelx, nely, 1.4)
#     np.set_printoptions(precision=4, linewidth=1000)
# print keq1.todense()
# print keq2.todense()
#     print (keq2.todense() - keq1.todense()) / 0.001
#     print akaa1[0].todense()

#     xi = np.array([0.5, 0.5])
#     yi = np.array([0.5, 0.5])
# #
# xi2 = np.array([0.505, 0.505])
# yi2 = np.array([0.505, 0.505])
#
#     xj = np.array([2.5, 2.5])
# xj2 = np.array([2.505, 2.505])
#     yj = np.array([2.5, 2.5])
#
# #
#     ci, axi, ayi = transform(xi, yi, x, y, 1.4)
# c2, ax2, ay2 = transform(xi2, yi, x, y, 1.4)
# cj2, axj2, ayj2 = transform(xj2, yj, x, y, 1.4)
#     cj, axj, ayj = transform(xj, yj, x, y, 1.4)
#
#     C = np.vstack((ci[:, 0], cj[:, 0]))
#     k = np.array([[1., -1.],
#                   [-1., 1.]]) / (2 * np.sqrt(2))
#     aCax = np.vstack((axi[:, 0], axj[:, 0]))
#
# print C.T.dot(k).dot(C)
#
#     print aCax.T.dot(k).dot(C)
#     print C.T.dot(k).dot(aCax)

#     print aCax.T.dot(k).dot(C) + C.T.dot(k).dot(aCax)

#
#     np.set_printoptions(suppress=True)

#
#     print c1, '\n', c2
#     print ay1, '\n', (c2 - c1) / 0.05
