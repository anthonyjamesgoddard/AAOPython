import temporalwave as twave
import numpy as np
import matplotlib.pyplot as plt


# Example data
A,Pinv= twave.monolithicAAOCD(96,96)

w,v= np.linalg.eig(Pinv*A)

plt.plot(w[:192],'x')
plt.savefig('CD96')
plt.show()


