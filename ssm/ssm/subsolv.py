'''
Created on Dec 2, 2014

@author: Li Yingxiong
'''
import numpy as np
from scipy.sparse import spdiags, csr_matrix


def subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d):
    '''
    INPUT:
    m : integer
    n : integer
    epsimin : float 1e-7
    low : Column vector 
    upp : Column vector
    alfa : Column vector
    beta : Column vector
    p0 : Column vector length n
    q0 : Column vector length n
    P : m x n sparse matrix
    Q : m x n sparse matrix
    a0 : constants
    a,b,c,d : Column vector
    '''
      
    een = np.ones(n,dtype=float)
    eem = np.ones(m,dtype=float)
    epsi = 1.
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa+beta)
    y = eem
    z = 1.
    lam = eem
    xsi = een/(x-alfa)
    xsi = np.amax((xsi,een), axis=0)
    eta = een/(beta-x)
    eta = np.amax((eta,een), axis=0)
    mu  = np.amax((eem,0.5*c), axis=0)
    zet = 1.
    s = eem
    itera = 0
    
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        
        
        plam = p0 + P.T.dot(lam) 
        qlam = q0 + Q.T.dot(lam) 
        gvec = P.dot(uxinv1) + Q.dot(xlinv1)
        dpsidx = plam/ux2 - qlam/xl2 
    
        rex = dpsidx - xsi + eta
        rey = c + d*y - mu - lam
        rez = a0 - zet - np.dot(a, lam)
        relam = gvec - a*z - y + s - b
        rexsi = xsi*(x-alfa) - epsvecn
        reeta = eta*(beta-x) - epsvecn
        remu = mu*y - epsvecm
        rezet = zet*z - epsi
        res = lam*s - epsvecm
    
        residu1 = np.hstack((rex, rey, rez))
        residu2 = np.hstack((relam, rexsi, reeta, remu, rezet, res))
        residu = np.hstack((residu1, residu2))
        residunorm = np.sqrt(np.dot(residu,residu))
        residumax = max(abs(residu))
    
        ittt = 0
        while residumax > 0.9*epsi and ittt < 100:
            ittt=ittt + 1.
            itera=itera + 1.
        
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0 + P.T.dot(lam) 
            qlam = q0 + Q.T.dot(lam) 
            gvec = P.dot(uxinv1) + Q.dot(xlinv1)
            GG = P*spdiags(uxinv2,0,n,n) - Q*spdiags(xlinv2,0,n,n)
            dpsidx = plam/ux2 - qlam/xl2 
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - np.dot(a, lam) - epsi/z
            dellam = gvec - a*z - y - b + epsvecm/lam
            diagx = plam/ux3 + qlam/xl3
            diagx = 2*diagx + xsi/(x-alfa) + eta/(beta-x)
            diagxinv = een/diagx
            diagy = d + mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
        
            if m < n:
                blam = dellam + dely/diagy - GG.dot(delx/diagx)
                bb = np.hstack((blam, delz))
                Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG.T
                AA = np.vstack(( np.hstack((Alam.todense(), a[:, None])), \
                                 np.hstack((a, -zet/z)) ))
                solut = np.linalg.solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m]
                dx = -delx/diagx - GG.T.dot(dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam + dely/diagy
                Axx = spdiags(diagx,0,n,n) + GG.T*spdiags(diaglamyiinv,0,m,m)*GG
                azz = zet/z + np.dot(a, a/diaglamyi)
                axz = -GG.T.dot(a/diaglamyi)
                bx = delx + GG.T.dot(dellamyi/diaglamyi)
                bz  = delz - np.dot(a, dellamyi/diaglamyi)
                AA = np.vstack(( np.hstack((Axx.todense(), axz[:, None])), \
                                 np.hstack((axz, azz)) ))
                bb = np.hstack((-bx, -bz))
                solut = np.linalg.solve(AA, bb)
                dx  = solut[0:n]
                dz = solut[n]
                dlam = GG.dot(dx)/diaglamyi - dz*(a/diaglamyi) + dellamyi/diaglamyi
        
            dy = -dely/diagy + dlam/diagy
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu  = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet*dz/z
            ds   = -s + epsvecm/lam - (s*dlam)/lam
            xx  = np.hstack((y, z, lam, xsi, eta, mu, zet, s))
            dxx = np.hstack((dy, dz, dlam, dxsi, deta, dmu, dzet, ds))
            
            stepxx = -1.01*dxx/xx
            stmxx  = max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = max(stepbeta)
            stmalbe  = max(stmalfa,stmbeta)
            stmalbexx = max(stmalbe,stmxx)
            stminv = max(stmalbexx,1.)
            steg = 1./stminv
        
            xold   =   x
            yold   =   y
            zold   =   z
            lamold =  lam
            xsiold =  xsi
            etaold =  eta
            muold  =  mu
            zetold =  zet
            sold   =   s
        
            itto = 0
            resinew = 2*residunorm
            while resinew > residunorm and itto < 50:
                itto = itto+1
            
                x   =   xold + steg*dx
                y   =   yold + steg*dy
                z   =   zold + steg*dz
                lam = lamold + steg*dlam
                xsi = xsiold + steg*dxsi
                eta = etaold + steg*deta
                mu  = muold  + steg*dmu
                zet = zetold + steg*dzet
                s   =   sold + steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0 + P.T.dot(lam) 
                qlam = q0 + Q.T.dot(lam) 
                gvec = P.dot(uxinv1) + Q.dot(xlinv1)
                dpsidx = plam/ux2 - qlam/xl2 
            
                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - np.dot(a, lam)
                relam = gvec - a*z - y + s - b
                rexsi = xsi*(x-alfa) - epsvecn
                reeta = eta*(beta-x) - epsvecn
                remu = mu*y - epsvecm
                rezet = zet*z - epsi
                res = lam*s - epsvecm
            
                residu1 = np.hstack((rex, rey, rez))
                residu2 = np.hstack((relam, rexsi, reeta, remu, rezet, res))
                residu = np.hstack((residu1, residu2))
                resinew = np.sqrt(np.dot(residu, residu))
                steg = steg/2.

            residunorm=resinew
            residumax = max(abs(residu))
            steg = 2.*steg
        
        epsi = 0.1*epsi
    
    return x, y, z, lam, xsi, eta, mu, zet, s