import meshline as ml
import FEMhelper as fh
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import scipy as sp


# returns A and the preconditioner for the wave equation
# the preconditioner has already been inverted

# this function returns the eigenvalues and eigenvectors

def monolithicAAOBD2(N,Tn):
    h = 0
    #final time
    T = 1.0;
    #timestep
    tau = T/Tn
    #mesh of the interval
    interval = ml.mesh(0,1,N,0)
    times,timesteps,pert = ml.timeDis(T,Tn,0)

    # CONTRUCTION OF MASS AND STIFFNESS MATRICES.
    mass, stiff = fh.generateMassStiff(interval)
    mass = mass[1:,1:]
    stiff = stiff[1:,1:]
    mass2 = ssp.csr_matrix((N,N))
    stiff2 = ssp.csr_matrix((N,N))
    mass2[1:-1,1:-1] = mass
    stiff2[1:-1,1:-1] = stiff

    # CONSTRUCTION OF MATRIX A.
    A0list = [ssp.csr_matrix((N,N)) for i in timesteps]
    for i in range(Tn):
            A0list[i] = mass2+timesteps[i]*timesteps[i]*stiff2
            A0list[i][0,0] =1; A0list[i][N-1,N-1] = 1

    A0list[0] = ssp.eye(N)

    A1 = ssp.csr_matrix((N,N))
    A2 = ssp.csr_matrix((N,N))
    A1= -2*mass2
    A2 = mass2
    temp1 = ssp.block_diag(A0list).tocsr()
    temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
    temp3 = ssp.block_diag([A2]*(Tn-2)).tocsr()
    temp4 = ssp.csr_matrix((N*Tn,N*Tn))
    temp5 = ssp.csr_matrix((N*Tn,N*Tn))
    temp4[N:,:-N] = temp2
    temp5[2*N:,:-2*N] = temp3
    A = temp5+temp4+temp1

    # CONSTRUCTION OF RIGHT HAND SIDE.
    h =h*np.ones(N-2)
    b = np.zeros(N)
    b[1:-1] = h
    RHS0 = np.sin(2*np.pi*interval)
    RHS1 = -mass2*np.sin(2*np.pi*interval)+timesteps[1]*timesteps[1]*mass2*b
    RHSLIST = [np.zeros(N) for i in timesteps]
    #right hand side for remaining time steps
    for i in range(Tn):
        RHSLIST[i] = timesteps[i]*timesteps[i]*mass2*b
    RHSLIST[0] = RHS0
    RHSLIST[1] = RHS1
    B = np.concatenate(RHSLIST,axis=0)

    # CONSTRUCTION OF THE EXACT PRECONDITIONER

    precon = temp5+temp4+temp1
    precon[:N,-N:] =A1
    precon[:N,-2*N:-N] =A2
    precon[N:2*N,-N:] =A2
    # Construct a linear operator that computes P^-1 * x.
    P = precon.todense()
    Pinv = np.linalg.inv(P);

    # OBTAIN EIGENVALUES OF THE PRECONDITONED SYSTEM


    return A, Pinv

# we actually need to implement the details for BD4... and central differences.
# to get the analsis
def monolithicAAOBD4(N,Tn):
    h = 0
    #final time
    T = 1.0;
    #timestep
    tau = T/Tn
    #mesh of the interval
    interval = ml.mesh(0,1,N,0)
    times,timesteps,pert = ml.timeDis(T,Tn,0)

    # CONTRUCTION OF MASS AND STIFFNESS MATRICES.
    mass, stiff = fh.generateMassStiff(interval)
    mass = mass[1:,1:]
    stiff = stiff[1:,1:]
    mass2 = ssp.csr_matrix((N,N))
    stiff2 = ssp.csr_matrix((N,N))
    mass2[1:-1,1:-1] = mass
    stiff2[1:-1,1:-1] = stiff

    # CONSTRUCTION OF MATRIX A.
    A0list = [ssp.csr_matrix((N,N)) for i in timesteps]
    for i in range(Tn):
            A0list[i] = mass2+timesteps[i]*timesteps[i]*stiff2
            A0list[i][0,0] =1; A0list[i][N-1,N-1] = 1

    A0list[0] = ssp.eye(N)

    A1 = ssp.csr_matrix((N,N))
    A2 = ssp.csr_matrix((N,N))
    A1= -2*mass2
    A2 = mass2
    temp1 = ssp.block_diag(A0list).tocsr()
    temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
    temp3 = ssp.block_diag([A2]*(Tn-2)).tocsr()
    temp4 = ssp.csr_matrix((N*Tn,N*Tn))
    temp5 = ssp.csr_matrix((N*Tn,N*Tn))
    temp4[N:,:-N] = temp2
    temp5[2*N:,:-2*N] = temp3
    A = temp5+temp4+temp1

    # CONSTRUCTION OF RIGHT HAND SIDE.
    h =h*np.ones(N-2)
    b = np.zeros(N)
    b[1:-1] = h
    RHS0 = np.sin(2*np.pi*interval)
    RHS1 = -mass2*np.sin(2*np.pi*interval)+timesteps[1]*timesteps[1]*mass2*b
    RHSLIST = [np.zeros(N) for i in timesteps]
    #right hand side for remaining time steps
    for i in range(Tn):
        RHSLIST[i] = timesteps[i]*timesteps[i]*mass2*b
    RHSLIST[0] = RHS0
    RHSLIST[1] = RHS1
    B = np.concatenate(RHSLIST,axis=0)

    # CONSTRUCTION OF THE EXACT PRECONDITIONER

    precon = temp5+temp4+temp1
    precon[:N,-N:] =A1
    precon[:N,-2*N:-N] =A2
    precon[N:2*N,-N:] =A2
    # Construct a linear operator that computes P^-1 * x.
    P = precon.todense()
    Pinv = np.linalg.inv(P);

    # OBTAIN EIGENVALUES OF THE PRECONDITONED SYSTEM


    return A, Pinv


