import temporalwave as twave
import numpy as np
import matplotlib.pyplot as plt

# Example data
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'Eigenvalue Index',fontsize=14)
plt.ylabel(r'Eigenvalues',fontsize=16)

A,Pinv= twave.monolithicAAOCD(32,32)
w,v= np.linalg.eig(Pinv*A)
plt.plot(w[:192],'x')
plt.savefig('CD',dpi=400)
plt.show()
