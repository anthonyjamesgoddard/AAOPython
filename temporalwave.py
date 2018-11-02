'''
****************************
TODO 
****************************


1.          MODIFY THIS CODE TO USE BD4
2.          MODIFY THIS CODE TO RETURN THE RHS IN ADDITION TO THE MONOLITHIC MATRIX AND PRECONDITIONER


****************************
'''


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


    # CONSTRUCTION OF THE EXACT PRECONDITIONER

    precon = temp5+temp4+temp1
    precon[:N,-N:] =A1
    precon[:N,-2*N:-N] =A2
    precon[N:2*N,-N:] =A2
    # Construct a linear operator that computes P^-1 * x.
    P = precon.todense()
    Pinv = np.linalg.inv(P);

    return A, Pinv

# this does the same as the above function but instead implements the central
# difference formula to deal with the time derivative

def monolithicAAOCD(N,Tn):
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
            A0list[i] = timesteps[i]*timesteps[i]*stiff2-2*mass2
            A0list[i][0,0] =1; A0list[i][N-1,N-1] = 1

    A0list[0] = ssp.eye(N)

    A1 = ssp.csr_matrix((N,N))
    A2 = ssp.csr_matrix((N,N))
    A1 = mass2
    temp1 = ssp.block_diag(A0list).tocsr()
    temp2 = ssp.block_diag([A1]*(Tn-1)).tocsr()
    temp4 = ssp.csr_matrix((N*Tn,N*Tn))
    temp5 = ssp.csr_matrix((N*Tn,N*Tn))
    temp4[N:,:-N] = temp2
    temp5[:-N,N:] = temp2
    A = temp5+temp4+temp1

    # CONSTRUCTION OF THE EXACT PRECONDITIONER

    precon = temp5+temp4+temp1
    precon[:N,-N:] =A1
    precon[-N:,:N] =A1
    # Construct a linear operator that computes P^-1 * x.
    P = precon.todense()
    Pinv = np.linalg.inv(P);

    return A, Pinv


