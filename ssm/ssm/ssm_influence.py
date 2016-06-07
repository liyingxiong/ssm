'''
Study of the influence of spreading radius on OBJ and the singularities
'''
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# from stiffness_spreading import stiffness_spreading
import numpy as np


def main(nelx, nely):

    # size of the design domain
    element_size = 40. / nelx
    x_size = nelx * element_size
    y_size = nely * element_size

    # the coordinates of the nodes
    X, Y = np.meshgrid(range(nelx + 1), range(nely + 1))
    xcoord = X.flatten(order='F') * element_size
    ycoord = Y.flatten(order='F') * element_size

    # conductive coefficient of the continuum
    Ec = 1.0

    xPhys = np.ones(nely * nelx, dtype=float)

    # dofs:
    ndof = (nelx + 1) * (nely + 1)

    # radius of the influence circle
    cs = 1e-3

    # initial design for the pipe configurations
    Bars = np.array([[10, 10, np.pi / 4., 20 * np.sqrt(2), 1., 10000]])
#     Bars = np.array([[5, 10, np.pi / 3., 10 * np.sqrt(5), 1., 0]])

    # representation of the pipe positions in terms of node coordinates
    coord = np.dstack((Bars[:, 0] - 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                       Bars[:, 1] - 0.5 * Bars[:, 3] * np.sin(Bars[:, 2]),
                       Bars[:, 0] + 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                       Bars[:, 1] + 0.5 * Bars[:, 3] * np.sin(Bars[:, 2])))
    bars = np.copy(Bars)
    bars[:, 0:4] = coord

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk(nelx)
    edofMat = np.zeros((nelx * nely, 4), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el,:] = np.array([n1 + 1, n2 + 1, n2, n1])

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((4, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 4))).flatten()

    dofs = np.arange((nelx + 1) * (nely + 1))
#     fixed = np.array([0])
    fixed = np.arange(nelx + 1)
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

#     f[:] = 0.1 * element_size ** 2
#     f[:] = 0.1
    f[(nelx + 1) * (nely + 1) - 1] = 1e3

    # Setup and solve FE problem
    sK = ((KE.flatten()[np.newaxis]).T * (xPhys * Ec)).flatten(order='F')
    # conduction matrix of the continuum
    Kconti = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    # equivalent conduction matrix of the pipes (bars) and its derivatives
    Keq, akax_lst, akay_lst, akaa_lst = stiffness_spreading(
        bars, xcoord, ycoord, nelx, nely, cs)
    # overall conduction matrix
    K = Kconti + Keq
    # Remove constrained dofs from matrix
    K = deleterowcol(K, fixed, fixed)
    # Solve system
    u[free, 0] = spsolve(K, f[free, 0])

    ce = np.ones(nely * nelx)

    ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 4), KE)
             * u[edofMat].reshape(nelx * nely, 4)).sum(1)
    obj = 0.5 * ((xPhys * Ec) * ce).sum()

    print obj

    plt.figure()
    plt.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'r', lw=6)
    plt.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'ko')
    CS = plt.contour(
        X * element_size, Y * element_size, u.reshape((nely + 1, nelx + 1), order='F'))
    plt.clabel(CS, inline=1, fontsize=10)
#     plt.show()


def stiffness_spreading(bars, xcoord, ycoord, nelx, nely, r):

    xi = bars[:, 0]
    yi = bars[:, 1]
    xj = bars[:, 2]
    yj = bars[:, 3]

    Ci, aCiaxi, aCiayi = transform(xi, yi, xcoord, ycoord, 1e-3)
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
    di2 = (xcoord[:, None]-xi[None,:])**2. + (ycoord[:, None]-yi[None,:])**2.
    idxi = np.where(di2 <= r ** 2.)

    di = np.zeros_like(di2)
    Di = np.zeros_like(di2)
    aDiaxi = np.zeros_like(di2)  # derivative of Di
    aDiayi = np.zeros_like(di2)  # derivative of Di

    di[idxi] = np.sqrt(di2[idxi]) / r
    Di[idxi] = (1. - di[idxi]) ** 4. * (4. * di[idxi] + 1.)  # RBF

    aDiaxi[idxi] = (1-di[idxi])**3.*(-20*di[idxi]) * (xi[None,:]-xcoord[:, None])[idxi] / (di[idxi]*r**2+np.finfo(np.double).tiny)
    awaxi = (aDiaxi * np.sum(Di, axis=0) - Di *
             np.sum(aDiaxi, axis=0)) / np.sum(Di, axis=0) ** 2.

    aDiayi[idxi] = (1-di[idxi])**3.*(-20*di[idxi]) * (yi[None,:]-ycoord[:, None])[idxi] / (di[idxi]*r**2+np.finfo(np.double).tiny)
    awayi = (aDiayi * np.sum(Di, axis=0) - Di *
             np.sum(aDiayi, axis=0)) / np.sum(Di, axis=0) ** 2.

    Ci = Di / np.sum(Di, axis=0)

    return Ci, awaxi, awayi


def lk(nelx):
    element_size = 40. / nelx
    factor = 1000. / element_size ** 2
    return factor * np.array([[2. / 3., -1. / 6., -1. / 3., -1. / 6.],
                              [-1. / 6., 2. / 3., -1. / 6., -1. / 3.],
                              [-1. / 3., -1. / 6., 2. / 3., -1. / 6.],
                              [-1. / 6., -1. / 3., -1. / 6., 2. / 3.]])


def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep,:]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A

if __name__ == '__main__':
    main(40, 40)
    main(50, 50)
    main(60, 60)
    main(80, 80)
    plt.show()