import numpy as np
import scipy.sparse as ssp


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def generateMassStiff(linemesh):
    #stepsizes
    stepsizes = np.diff(linemesh)
    #the three strips that will be placed
    #on the diagonals of a matrix
    
    #number of elements
    NE = len(stepsizes)

    aStiff = np.zeros(NE)
    bStiff = np.zeros(NE-1)
    cStiff = np.zeros(NE-1)

    aMass = np.zeros(NE)
    bMass = np.zeros(NE-1)
    cMass = np.zeros(NE-1)

    eMass = np.zeros((2,2))
    eStiff = np.zeros((2,2))
    
    #We fill in the interior elements
    for i in range(NE-1):
        eStiff[0,0]=1./stepsizes[i] ; eStiff[0,1]=-eStiff[0,0]
        eStiff[1,0]=eStiff[0,1]     ; eStiff[1,1]= eStiff[0,0]

        eMass[0,0]=stepsizes[i]/3.; eMass[0,1]=0.5*eMass[0,0]
        eMass[1,0]=eMass[0,1]     ; eMass[1,1]=    eMass[0,0]

        aStiff[i] = aStiff[i]+eStiff[0,0]
        bStiff[i] = bStiff[i]+eStiff[0,1]
        cStiff[i] = cStiff[i]+eStiff[1,0]
        aStiff[i+1] = aStiff[i+1] + eStiff[1,1]  

        aMass[i] = aMass[i]+eMass[0,0]
        bMass[i] = bMass[i]+eMass[0,1]
        cMass[i] = cMass[i]+eMass[1,0]
        aMass[i+1] = aMass[i+1] + eMass[1,1]  

    aStiff[-1] = aStiff[-1] + 1./stepsizes[-1]
    aMass[-1] = aMass[-1] + stepsizes[-1]/3.
    S = ssp.diags([aStiff,bStiff,cStiff],[0,1,-1]).tocsr() 
    M = ssp.diags([aMass,bMass,cMass],[0,1,-1]).tocsr()
    return M,S
