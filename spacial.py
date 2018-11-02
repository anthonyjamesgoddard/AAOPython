import meshline as ml
import FEMhelper as fh
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp

#number of nodes
N = 10
#constant source term
h = 1
#final time
T = 1.0;
#number of time steps
Tn = 2
#timestep
tau = T/Tn
#mesh of the interval
interval = ml.mesh(0,1,N,0.1)


#the mass and stiffness matrices
#we only need to solve for the interior nodes
#the outer nodes are both zero
mass, stiff = fh.generateMassStiff(interval)
mass = mass[1:,1:]
stiff = stiff[1:,1:]

#Right hand side
b1 =0.5*h*(np.diff(interval[:-1])+np.diff(interval[1:]))
b = np.zeros(N)
b[1:-1] = b1
mass2 = ssp.csr_matrix((N,N))
mass2[1:-1,1:-1] = mass
#right hand side for first time step
RHS0 = mass2*np.sin(2*np.pi*interval)+tau*b
#right hand side for remaining time steps
RHSi = tau*b

#creation of the block matrices
A0 = ssp.csr_matrix((N,N))
A0[1:-1,1:-1] = mass+tau*stiff
A1 = ssp.csr_matrix((N,N))
A1[1:-1,1:-1] = -1*mass
A1[0,0] =1;A1[-1,-1] = 1
A0[0,0] =1;A0[-1,-1] = 1

#We build the large block bi-diag matrix
temp1 = ssp.block_diag([A0]*Tn).tocsr()
temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
temp3 = ssp.csr_matrix((N*Tn,N*Tn))
temp3[N:,:-N] = temp2
A = temp3+temp1

#contruct the big right hand side
B = np.tile(RHSi,Tn)
B[:N] = RHS0

#the precond
temp1 = ssp.block_diag([A0]*Tn).tocsr()
temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
temp3 = ssp.csr_matrix((N*Tn,N*Tn))
temp3[N:,:-N] = temp2
precon = temp3+temp1

precon[:N,-N:] =1.1*A1
#plt.imshow(precon.todense())
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

counter = gmres_counter()
#moment of truth

AA= (mass*stiff).todense()
BB= (stiff*mass).todense()

print BB-AA

M_x = lambda x: la.spsolve(precon,x)
M=la.LinearOperator((N*Tn,N*Tn),M_x)
#x,info = la.gmres(A,B,callback=counter,tol=1e-10,M=M)
EIGZ,EIGVECZ=np.linalg.eig(M*A.todense())
plt.plot(EIGZ)
#print info
#plt.plot(interval,x[1*N:2*N],'o-')
#print counter.niter
plt.show()

