import numpy as np
from scipy.sparse import csr_matrix, spdiags
from subsolv import subsolv


def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    '''
    INPUT:
    m     = The number of general constraints.
    n     = The number of variables x_j.
    iter  = Current iteration number ( =1 the first time mmasub is called).
    xval  = Column vector with the current values of the variables x_j.
    xmin  = Column vector with the lower bounds for the variables x_j.
    xmax  = Column vector with the upper bounds for the variables x_j.
    xold1 = xval, one iteration ago (provided that iter>1).
    xold2 = xval, two iterations ago (provided that iter>2).
    f0val = The value of the objective function f_0 at xval.
    df0dx = Column vector with the derivatives of the objective function
            f_0 with respect to the variables x_j, calculated at xval.
    fval  = Column vector with the values of the constraint functions f_i,
            calculated at xval.
    dfdx  = (m x n)-matrix with the derivatives of the constraint functions
            f_i with respect to the variables x_j, calculated at xval.
            dfdx(i,j) = the derivative of f_i with respect to x_j.
    low   = Column vector with the lower asymptotes from the previous
            iteration (provided that iter>1).
    upp   = Column vector with the upper asymptotes from the previous
            iteration (provided that iter>1).
    a0    = The constants a_0 in the term a_0*z.
    a     = Column vector with the constants a_i in the terms a_i*z.
    c     = Column vector with the constants c_i in the terms c_i*y_i.
    d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

    OUTPUT:
    xmma  = Column vector with the optimal values of the variables x_j
            in the current MMA subproblem.
    ymma  = Column vector with the optimal values of the variables y_i
            in the current MMA subproblem.
    zmma  = Scalar with the optimal value of the variable z
            in the current MMA subproblem.
    lam   = Lagrange multipliers for the m general MMA constraints.
    xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
     mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    zet   = Lagrange multiplier for the single constraint -z <= 0.
     s    = Slack variables for the m general MMA constraints.
    low   = Column vector with the lower asymptotes, calculated and used
            in the current MMA subproblem.
    upp   = Column vector with the upper asymptotes, calculated and used
            in the current MMA subproblem.
    '''
    epsimin = 1e-7
    raa0 = 0.00001
    albefa = 0.1
    asyinit = 0.5  # 0.2e-2;
    asyincr = 1.2
    asydecr = 0.7
    eeen = np.ones(n, dtype=float)
    eeem = np.ones(m, dtype=float)
    zeron = np.zeros(n, dtype=float)

    # Calculation of the asymptotes low and upp :
    if iter < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta :
    zzz = low + albefa * (xval - low)
    alfa = np.maximum(zzz, xmin)
    zzz = upp - albefa * (upp - xval)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b.
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1
    #
    p0 = zeron
    q0 = zeron
    p0 = df0dx.clip(min=0)
    q0 = (-df0dx).clip(min=0)
    # p0(find(df0dx > 0)) = df0dx(find(df0dx > 0));
    # q0(find(df0dx < 0)) = -df0dx(find(df0dx < 0));
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2
    #
#     P = csr_matrix((m,n))
#     D = csr_matrix((m,n))
#     print 'd=', type(D)
#     Q = csr_matrix((m,n))
    P = dfdx.clip(min=0)
#     print P.size
#     print 'dfdx=', type(dfdx)
#     print 'xmamiinv', type(xmamiinv)
    Q = (-dfdx).clip(min=0)
    # P(find(dfdx > 0)) = dfdx(find(dfdx > 0));
    # Q(find(dfdx < 0)) = -dfdx(find(dfdx < 0));
    PQ = 0.001 * (P + Q) + raa0 * eeem[:, None] * xmamiinv[None, :]
#     print xmamiinv.shape
#     d = np.tensordot(eeem, xmamiinv, axes=0)
#     print 'i', d.shape

#     print 'PQ=', type(PQ)
    P = P + PQ
    Q = Q + PQ
#     print P.size
#     P = spdiags(ux2,0,n,n).T.dot(P)
    P = csr_matrix(P) * spdiags(ux2, 0, n, n)
#     Q = spdiags(xl2,0,n,n).T.dot(Q)
    Q = csr_matrix(Q) * spdiags(xl2, 0, n, n)
#     F = np.dot(P,uxinv)+np.dot(Q, xlinv)
#     print 'F=', type(F)
#     print 'fval=', type(fval)
    b = P.dot(uxinv) + Q.dot(xlinv) - fval

    # Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = \
        subsolv(m, n, epsimin, low, upp, alfa,
                beta, p0, q0, P, Q, a0, a, b, c, d)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
