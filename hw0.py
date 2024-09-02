import random
import numpy as np
def draw_samples(n):
    values = [1, 0, -1]
    probabilities = [0.25, 0.45, 0.30]
    samples = random.choices(values, weights=probabilities, k=n)
    samples = np.array(samples)
    return samples
  
def sum_squares(N):
    N=np.array(N,np.float128)
    ans1=np.float128(0)
    for i in N:
        ans1+=i*i
    return np.float64(ans1)

def troublemakers(n):
    N=np.array([1.0,1.0],dtype=np.float64)
    for _ in range(n):
        
        N[1]+=0.35*N[0]
        N[0]-=0.35*N[0]
        N[0]+=N[1]*0.2
        N[1]-=0.2*N[1]
    return N
