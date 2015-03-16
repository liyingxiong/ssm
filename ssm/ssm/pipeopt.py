'''
Created on Feb 22, 2015

@author: Li Yingxiong
'''
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from mmasub import mmasub
from stiffness_spreading import stiffness_spreading
import time as t


def main(nelx, nely, volfrac, penal, rmin, ft):

    # size of the design domain
    element_size = 1.
    x_size = nelx * element_size
    y_size = nely * element_size

    # the coordinates of the nodes
    X, Y = np.meshgrid(range(nelx + 1), range(nely + 1))
    xcoord = X.flatten(order='F') * element_size
    ycoord = Y.flatten(order='F') * element_size

    # Max and min conductive coefficient
    Emin = 1e-8
    Emax = 1.0

    # dofs:
    ndof = (nelx + 1) * (nely + 1)

    # radius of the influence circle
    cs = 3.0

# parameters for MMA
#     a0 = 1.
#     a = np.zeros(1,dtype=float)
#     c = 1e6*np.ones(1,dtype=float)
#     d = np.zeros(1,dtype=float)

    # Allocate design variables (as array), initialize and allocate sens.
#     x = volfrac * np.ones(nely * nelx, dtype=float)
#     xold = x.copy()
#     xold2 = x.copy()
#     xPhys = x.copy()
#     xmin = np.zeros(nely * nelx, dtype=float)
#     xmax = np.ones(nely * nelx, dtype=float)
#     low = xmin.copy()
#     upp = xmax.copy()

# g = 0  # must be initialized to use the NGuyen/Paulino OC approach
#     dc = np.zeros((nely, nelx), dtype=float)

    # initial design for the pipe configurations
    Bars = np.array([[10., 10., 0., 10., 1., 100000.],
                     [30., 10., 0., 10., 1., 100000.],
                     [10., 30., 0., 10., 1., 100000.],
                     [30., 30., 0., 10., 1., 100000.]])
#     Bars = np.array([[2.5, 2.5, np.pi / 4., 3., 1., 100000]])
    nBars = len(Bars)

    # representation of the pipe positions in terms of node coordinates
    coord = np.dstack((Bars[:, 0] - 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                       Bars[:, 1] - 0.5 * Bars[:, 3] * np.sin(Bars[:, 2]),
                       Bars[:, 0] + 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                       Bars[:, 1] + 0.5 * Bars[:, 3] * np.sin(Bars[:, 2])))
    bars = np.copy(Bars)
    bars[:, 0:4] = coord

    # the parameters for MMA
    # continuum
    Pm = 1  # number of constraints
    Pn = nely * nelx  # number of design variables
    Pxmax = np.ones(Pn, dtype=float)
    Pxmin = np.zeros(Pn, dtype=float)
    # initial density design
    Pxval = volfrac * np.ones(nely * nelx, dtype=float)
    xPhys = np.copy(Pxval)
#     Pdf0dx2 = np.zeros(Pn, dtype=float)
#     Pxold1 = Pxval
#     Pxold2 = Pxval
#     Plow = Pxmin
#     Pupp = Pxmax
# todo
#     Pdfdx2 = np.zeros(Pn)
    # pipes(bars)
    Bm = 8 * nBars  # number of constraints
    Bn = 3 * nBars  # number of variables
    Bxmax = np.tile(np.array([x_size, y_size, 10 * np.pi]), nBars)
    Bxmin = np.tile(np.array([0., 0., -10 * np.pi]), nBars)
    Bxval = Bars[:, 0:3].flatten()
#     Bdf0dx2 = np.zeros(Bn, dtype=float)
    #continuum and pipes(bars)
    m = Pm + Bm
    n = Pn + Bn

    a0 = 1.
    a = 1e-3 * np.ones(m, dtype=float)
    c = 1e6 * np.ones(m, dtype=float)
    d = np.ones(m, dtype=float)
    xval = np.hstack((Pxval, Bxval))
    xmax = np.hstack((Pxmax, Bxmax))
    xmin = np.hstack((Pxmin, Bxmin))
    upp = xmax
    low = xmin
    xold1 = xval
    xold2 = xval
#     df0dx2 = np.hstack((Pdf0dx2, Bdf0dx2))
#     dfdx2 = np.zeros((m, n))

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 4), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([n1 + 1, n2 + 1, n2, n1])

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((4, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 4))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the coo matrix
    # format
    nfilter = nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2)
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - \
                        np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support
    dofs = np.arange((nelx + 1) * (nely + 1))
#     fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
#     fixed=np.union1d(np.array([0]), np.array([nely+1]))
#     fixed = np.array([0])
#     fixed = dofs[0:nely + 1]
    fixed = np.array([0])
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load
#     f[1,0]=-1
    f[:] = 0.1

    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
              interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0), origin='lower', extent=(0, x_size, 0, y_size))
    ax.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'r', lw=4)
    ax.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'ko')
#     plt.xlim((0, x_size))
#     plt.ylim((0, y_size))

    fig.show()

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    Pdf0dx = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    fval = np.zeros(m)  # values of the constraint functions

    while loop < 20:
        t1 = t.time()

        loop = loop + 1

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys)
                                              ** penal * (Emax - Emin))).flatten(order='F')
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

        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 4), KE)
                 * u[edofMat].reshape(nelx * nely, 4)).sum(1)
        obj = 0.5 * ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
        Pdf0dx[:] = 0.5 * (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx)

        # Sensitivity filtering:
        if ft == 0:
            Pdf0dx[:] = np.asarray(
                (H * (Pxval * Pdf0dx))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, Pxval)
        elif ft == 1:
            Pdf0dx[:] = np.asarray(H * (Pdf0dx[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # derivatives of the objective function
        Bdf0dx = np.zeros(Bn)
        for iBar in range(nBars):
            #             print akax_lst[iBar].shape
            #             print u[:].shape
            #             print np.dot(akax_lst[iBar].dot(u).T, u)
            Bdf0dx[3 * iBar] = -0.5 * np.dot(akax_lst[iBar].dot(u).T, u)
            Bdf0dx[3 * iBar + 1] = -0.5 * np.dot(akay_lst[iBar].dot(u).T, u)
            Bdf0dx[3 * iBar + 2] = -0.5 * np.dot(akaa_lst[iBar].dot(u).T, u)
        df0dx = np.hstack((Pdf0dx, Bdf0dx))

        # constraint of the continuum volume fraction
        fval[0] = np.sum(xPhys) - volfrac * nelx * nely
        fval[1:m] = np.hstack((bars[:, 0] - x_size, bars[:, 1] - y_size, bars[:, 2] -
                               x_size, bars[:, 3] - y_size, -bars[:, 0], -bars[:, 1], -bars[:, 2], -bars[:, 3]))

        dfdx = np.zeros((n, m))
        # derivatives of the volume constraint
        dfdx[0:Pn, 0] = dv[:]
        dw = np.vstack((0.5 * Bars[:, 3] * np.sin(Bars[:, 2]), -0.5 * Bars[:, 3] * np.cos(
            Bars[:, 2]), -0.5 * Bars[:, 3] * np.sin(Bars[:, 2]), 0.5 * Bars[:, 3] * np.cos(Bars[:, 2])))
        # derivatives of the bar position constraints
        for ibar in range(nBars):
            dfdx[Pn + 3 * ibar, [ibar + 1, ibar + 2 * nBars + 1, ibar +
                                 4 * nBars + 1, ibar + 6 * nBars + 1]] = [1., 1., -1., -1.]
            dfdx[Pn + 3 * ibar + 1, [ibar + nBars + 1, ibar + 3 * nBars + 1,
                                     ibar + 5 * nBars + 1, ibar + 7 * nBars + 1]] = [1., 1., -1., -1.]
            dfdx[Pn + 3 * ibar + 2, ibar +
                 1::nBars] = np.hstack((dw[:, ibar], -dw[:, ibar]))

        # define the move limits
        Bxmax = np.vstack(
            (Bars[:, 0] + 2., Bars[:, 1] + 2., Bars[:, 2] + 10. * np.pi / 180.)).flatten(order='F')
        Bxmin = np.vstack(
            (Bars[:, 0] - 2., Bars[:, 1] - 2., Bars[:, 2] - 10. * np.pi / 180.)).flatten(order='F')
        if loop > 50:
            xmax = np.hstack((Pxmax, Bxmax))
            xmin = np.hstack((Pxmin, Bxmin))
        else:
            Pxmax1 = Pxval + 0.001
            Pxmin1 = Pxval - 0.001
            xmax = np.hstack((Pxmax1, Bxmax))
            xmin = np.hstack((Pxmin1, Bxmin))

        # MMA
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
            mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2,
                   obj, df0dx, fval, dfdx.T, low, upp, a0, a, c, d)
        xold2 = xold1
        xold1 = xval
        xval = xmma

        Pxval = xval[0:Pn]
        # Filter design variables
        if ft == 0:
            xPhys[:] = Pxval
        elif ft == 1:
            xPhys[:] = np.asarray(H * Pxval[np.newaxis].T / Hs)[:, 0]

        # extract the bar positions
        Bars[:, 0:3] = np.reshape(xval[Pn:], (nBars, 3))
        coord = np.dstack((Bars[:, 0] - 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                           Bars[:, 1] - 0.5 * Bars[:, 3] * np.sin(Bars[:, 2]),
                           Bars[:, 0] + 0.5 * Bars[:, 3] * np.cos(Bars[:, 2]),
                           Bars[:, 1] + 0.5 * Bars[:, 3] * np.sin(Bars[:, 2])))
        bars[:, 0:4] = coord

        # Compute the change by the inf. norm
#         change = np.linalg.norm(
# Pxval.reshape(nelx * nely, 1) - xold1[0:Pn].reshape(nelx * nely, 1),
# np.inf)

        # Plot to screen
        plt.cla()
        ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
                  interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0), origin='lower', extent=(0, x_size, 0, y_size))
        ax.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'r', lw=6)
        ax.plot((bars[:, 0], bars[:, 2]), (bars[:, 1], bars[:, 3]), 'ko')
        plt.xlim((0, x_size))
        plt.ylim((0, y_size))
        fig.show()

        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, (volfrac * nelx * nely) / (nelx * nely), change))
        print t.time() - t1

    # Make sure the plot stays and that the shell remains
    plt.show()
    raw_input("Press any key...")


def lk():
    return 1000. * np.array([[2. / 3., -1. / 6., -1. / 3., -1. / 6.],
                             [-1. / 6., 2. / 3., -1. / 6., -1. / 3.],
                             [-1. / 3., -1. / 6., 2. / 3., -1. / 6.],
                             [-1. / 6., -1. / 3., -1. / 6., 2. / 3.]])


def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


# def oc(nelx, nely, x, volfrac, dc, dv, g):
#     l1 = 0
#     l2 = 1e9
#     move = 0.2
# reshape to perform vector operations
#     xnew = np.zeros(nelx * nely)
#
#     while (l2 - l1) / (l1 + l2) > 1e-3:
#         lmid = 0.5 * (l2 + l1)
#         xnew[:] = np.maximum(0.0, np.maximum(
#             x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-
#         gt = g + np.sum((dv * (xnew - x)))
#         if gt > 0:
#             l1 = lmid
#         else:
#             l2 = lmid
#     return (xnew, gt)

if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40
    volfrac = 0.4
    rmin = 1.3
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens

#     import sys
#     if len(sys.argv) > 1:
#         nelx = int(sys.argv[1])
#     if len(sys.argv) > 2:
#         nely = int(sys.argv[2])
#     if len(sys.argv) > 3:
#         volfrac = float(sys.argv[3])
#     if len(sys.argv) > 4:
#         rmin = float(sys.argv[4])
#     if len(sys.argv) > 5:
#         penal = float(sys.argv[5])
#     if len(sys.argv) > 6:
#         ft = int(sys.argv[6])
    import profile

    profile.run("main(nelx, nely, volfrac, penal, rmin, ft)")
