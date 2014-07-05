from traits.api import HasTraits, Float, Int, Array, Property, \
    cached_property, List
import numpy as np
from scipy.sparse import bsr_matrix

class HeatPipe(HasTraits):
    
    x=Float
    y=Float
    L=Float
    theta=Float
    _lamda = Float
    A = Float
    
    c_m = Property(depends_on = 'L, theta, _lambda, A') #conduction matrix
    @cached_property
    def _get_conduction_matrix(self):
        return self._lambda*self.A/self.L*np.array([[1, -1], [-1, 1]])
        

class HeatConductionOptimization(HasTraits):
    
#=============================================================================
# specify the parameters
#=============================================================================    
    #the volume fraction
    V0 = Float(0.4)
    #finite element discretization 
    nelx = Int(40)
    nely = Int(40)
    #spreading radius
    r = Float(3)
    #conductivity of the background mesh
    BG_lambda = Float(1000)
    #the array containing the density variables
    BG_density = Array
    #penalty factor
    penalty = Float(3.)
    # the heat pipes
    pipes = List

#=============================================================================
#the conduction matrix of a background mesh element
#=============================================================================        
    ECM = Property(depends_on = 'BG_lambda')
    @cached_property
    def _get_ECM(self):
        return self.BG_lambda*np.array([[2./3., -1./6., 1./3., -1./6.],
                                        [-1./6., 2./3., -1./6., 1./3.],
                                        [1./3., -1./6., 2./3., -1./6.],
                                        [-1./6., 1./3., -1./6., 2./3.]])
        
#=============================================================================
# the background mesh conduction matrix
#=============================================================================        
    BGCM = Property(depends_on = 'nelx, nely, BG_lambda, BG_density, Penalty')
    @cached_property
    def _get_BGCM(self):
        BCGM = np.zeros(((self.nelx+1)*(self.nely+1), (self.nelx+1)*(self.nely+1)))
        for x in range(self.nelx):
            for y in range(self.nely):
                nodes = np.array([x*(self.nely+1)+y, (x+1)*(self.nely+1)+y, \
                                  (x+1)*(self.nely+1)+y+1, x*(self.nely+1)+y+1])
                BCGM[nodes[:, np.newaxis], nodes[np.newaxis, :]] += \
                    self.BG_density[x, y]**self.penalty*self.ECM
        return bsr_matrix(BCGM)
    
#=============================================================================
# Shepard interpolation, return the transformation matrix
#=============================================================================        
    def shepard(self, xi, yi, xj, yj):
        X, Y = np.meshgrid(range(self.nelx+1), range(self.nely+1))
        x, y = X.flatten('F'), Y.flatten('F')
        di = (x-xi)**2 + (y-yi)**2
        dj = (x-xj)**2 + (y-yj)**2
        Di = di[di<=self.r**2]
        Dj = dj[dj<=self.r**2]
        Di = np.sqrt(Di)/self.r
        Dj = np.sqrt(Dj)/self.r
        Di = (1-Di)**4 * (4*Di + 1)
        Dj = (1-Dj)**4 * (4*Dj + 1)
        row = np.hstack([np.zeros_like(Di), np.ones_like(Dj)])
        col = np.hstack([np.where(di<=self.r**2)[0], np.where(dj<=self.r**2)[0]])
        return bsr_matrix((np.hstack([Di/np.sum(Di), Dj/np.sum(Dj)]), (row, col)))

#=============================================================================
# equvalient conduction matrix of the heat pipes
#=============================================================================        
    PECM = Property(depends_on='pipes')
    @cached_property
    def _get_PECM(self):
        PECM = bsr_matrix(((self.nelx+1)*(self.nely+1),(self.nelx+1)*(self.nely+1)))
        for pipe in self.pipes:
            C = self.shepard(pipe.xi, pipe.yi, pipe.xj, pipe.yj)
            pecm = C.T*bsr_matrix(pipe.c_m)*C
            PECM += pecm
        return pecm
    
            

    
    
        
        
        
    
    
if __name__ == '__main__':
    
    hco = HeatConductionOptimization(nelx=2,
                                     nely=2,
                                     BG_density=np.array([[1, 1], [1, 1]]),
                                     r=1.5)
    
    print hco.shepard(0, 0, 2, 2)
