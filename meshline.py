import numpy as np

def mesh(begin,end,N,delta):
    if delta ==0:
        return np.linspace(begin,end,N)
    else:
        randz=(delta/N)*(np.random.random_sample(N)-0.5)
        randz[0] = 0;randz[-1] =0
        inter=np.linspace(begin,end,N)
        return randz+inter

def timeDis(T,N,delta):
    if delta == 0:
        return (np.linspace(0,T,N+1),[1.0*T/N for i in range(N)],np.zeros(N))
    else:
        times = mesh(0,T,N+1,delta)
        timesteps = np.diff(times)
        
        return (times,timesteps,timesteps - [1.0*T/N for i in range(N)])

def expTimeDis(N):
    times = mesh(0,1,N+1,0)
    times = np.exp(times)
    timesteps = np.diff(times)
    return (times,timesteps,timesteps - [1.0/N for i in range(N)])

def chebyTime(N):
	x = range(N,-1,-1)
	y = [i*(1.0/N)*np.pi for i in x]
	z = 0.5*np.cos(y) + 0.5
	timesteps = np.diff(z)
	return (z,timesteps,timesteps - [1.0/N for i in range(N)])
