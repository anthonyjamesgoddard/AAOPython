import temporalwave as twave
import numpy as np
import matplotlib.pyplot as plt


# Example data
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.xlabel(r'Eigenvalue Index',fontsize=14)
plt.ylabel(r'Eigenvalues',fontsize=16)
A,Pinv= twave.monolithicAAO(32,32)

w,v= np.linalg.eig(Pinv*A)

plt.plot(w,'x')
plt.savefig('tex_demo',dpi=400)
plt.show()


