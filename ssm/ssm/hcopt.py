from __future__ import division
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from mmasub import mmasub
import mma


def main(nelx, nely, volfrac, penal, rmin, ft):

    # Max and min conductive coefficient
    Emin = 1e-8
    Emax = 1.0

    # dofs:
    ndof = (nelx + 1) * (nely + 1)
    X, Y = np.meshgrid(range(nelx + 1), range(nely + 1))

    # parameters for MMA
    a0 = 1.
    a = np.zeros(1, dtype=float)
    c = 1e6 * np.ones(1, dtype=float)
    d = np.zeros(1, dtype=float)
    fmax = np.zeros(1, dtype=float)
    m = 1
    n = nelx * nely

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xold2 = x.copy()
    xPhys = x.copy()
    xmin = np.zeros(nely * nelx, dtype=float)
#     xmin = 0.
    xmax = np.ones(nely * nelx, dtype=float)
    low = xmin.copy()
#     low = np.zeros(nely * nelx)
    upp = xmax.copy()

    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    dc = np.zeros((nely, nelx), dtype=float)

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 4), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([n1 + 1, n2 + 1, n2, n1])

#     print edofMat
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
#     fixed = dofs[10200]
    fixed = []
#     fixed = dofs[-nely - 1::]
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load
#     f[1,0]=-1
#     f[:] = 3e-6
    f[0:(nely + 1) * 50] = 3e-2
    f[-(nely + 1) * 50::] = -3e-2

    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig.show()

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)
    while loop < 200:
        loop = loop + 1

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys)
                                              ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = deleterowcol(K, fixed, fixed)
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])

        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 4), KE)
                 * u[edofMat].reshape(nelx * nely, 4)).sum(1)
        obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce

        fval = np.sum(xPhys) - volfrac * nelx * nely
        dv[:] = np.ones(nely * nelx)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray(
                (H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # Optimality criteria
#         xold[:]=x
#         (x[:],g)=oc(nelx,nely,x,volfrac,dc,dv,g)

        # MMA
#         xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
#             mmasub(1, nelx * nely, loop, x, xmin, xmax, xold,
#                    xold2, obj, dc, fval, dv, low, upp, a0, a, c, d)

        xmma, ymma, zmma, lam = mma.mmasub(
            loop, m, x.copy(), xold.copy(), xold2.copy(
            ), xmin, xmax, low, upp, a, c, obj, fval, fmax,
            dc, dv, n)

        xold2 = xold.copy()
        xold = x.copy()
        x = xmma.copy()

        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

        # Compute the change by the inf. norm
        change = np.linalg.norm(
            x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

        # Plot to screen
#         plt.cla()
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        plt.savefig('D:\ssm\%s.png' % loop)
        fig.canvas.draw()
#         CS = plt.contour(X, Y, u.reshape((nelx + 1, nely + 1)))
#         ax.clabel(CS, inline=1, fontsize=10)
#         plt.show()

        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, (g + volfrac * nelx * nely) / (nelx * nely), change))

    # Make sure the plot stays and that the shell remains
    plt.show()
    raw_input("Press any key...")


def lk():
    return np.array([[2. / 3., -1. / 6., -1. / 3., -1. / 6.],
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


def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx * nely)

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)

if __name__ == "__main__":
    # Default input parameters
    nelx = 200
    nely = 100
    volfrac = 0.4
    rmin = 3.5
    penal = 3.0
    ft = 1  # ft==0 -> sens, ft==1 -> dens

    import sys
    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        rmin = float(sys.argv[4])
    if len(sys.argv) > 5:
        penal = float(sys.argv[5])
    if len(sys.argv) > 6:
        ft = int(sys.argv[6])

    main(nelx, nely, volfrac, penal, rmin, ft)
