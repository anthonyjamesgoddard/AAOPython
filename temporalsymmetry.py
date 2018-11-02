import meshline as ml
import FEMhelper as fh
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import scipy as sp
import matplotlib.cm as cm
#number of nodes
N = 30
#constant source term
h = 1
#final time
T = 1.0;
#number of time steps
Tn = 100
#timestep
tau = T/Tn
#mesh of the interval
interval = ml.mesh(0,1,N,0.0)
times,timesteps,pert = ml.chebyTime(Tn)
#times,timesteps,pert = ml.timeDis(T,Tn,0)
print timesteps
print pert

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
stiff2 = ssp.csr_matrix((N,N))
mass2[1:-1,1:-1] = mass
stiff2[1:-1,1:-1] = stiff
#right hand side for first time step
RHS0 = mass2*np.sin(2*np.pi*interval)+tau*b
#right hand side for remaining time steps
RHSi = tau*b

#creation of the block matrices
A0 = ssp.csr_matrix((N,N))
A0list = [ssp.csr_matrix((N,N)) for i in timesteps]

for i in range(Tn):
	A0list[i][1:-1,1:-1] = mass+timesteps[i]*stiff
	A0list[i][0,0] =1; A0list[i][N-1,N-1] = 1

A1 = ssp.csr_matrix((N,N))
A1[1:-1,1:-1] = -1*mass

#We build the large block bi-diag matrix
temp1 = ssp.block_diag(A0list).tocsr()
temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
temp3 = ssp.csr_matrix((N*Tn,N*Tn))
temp3[N:,:-N] = temp2
A = temp3+temp1

#contruct the big right hand side
B = np.tile(RHSi,Tn)
B[:N] = RHS0



# THE APPROXIMATE PRECONDITIONER

mass3 = ssp.csr_matrix((N,N))
mass3[0,0] = 1;
mass3[N-1,N-1] = 1;
mass3[1:-1,1:-1] = mass
temp1 = ssp.kron(np.eye(Tn),mass3)
temp2 = ssp.kron(np.eye(Tn),stiff2)
temp3 = ssp.kron(np.roll(np.eye(Tn),1,axis=0),mass2)

precon = temp1+tau*temp2-temp3

BB = ssp.kron(np.diag(pert),stiff2)
PP = precon.todense()
BB = BB.todense()

precon1 = PP.transpose()*PP 

P = sp.linalg.sqrtm(precon1)
PP1 = np.linalg.inv(P)

P=PP1- PP1*(BB)*PP1


counter = fh.gmres_counter()
#moment of truth
##
hankdata = np.zeros(N*Tn)
hankdata[-1] = 1
Y=sp.linalg.hankel(hankdata)
Ys = ssp.csr_matrix(Y)

# Obtain symmetric preconditioner


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


x,info = la.minres(Ys*A,Ys*B,tol=1e-5,show=True,M=P)
#EIGZ,EIGVECZ=np.linalg.eig(P*A.todense())
#listOfEVecs = np.hsplit(EIGVECZ,N*Tn);
#angles = []
#for e1 in listOfEVecs:
#	for e2 in listOfEVecs:
#		angles.append(np.real(np.asscalar(angle_between(e1.flatten(),e2))))

#angles=list(set(angles))
#angles.sort()
#angles = [a for a in angles if a>0]
#plt.plot(range(len(angles)),angles,'ro')

#x,info = la.gmres(A,B,tol=1e-10,callback=counter,M=P)
#plt.plot(EIGZ,'g^')
#plt.title('Angles between eigvecs of preconditioned system. n = l = 15,delta = 0.9')
#plt.title('Eigenvalues of preconditioned system')
#print info
#plt.plot(interval,x[1*N:2*N],'o-')
#print counter.niter


