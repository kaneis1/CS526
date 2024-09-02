import random
import numpy as np
def draw_samples(n):
    samples = []
    for _ in range(n):
        r=random.random()
        if r<=0.25:
            x=1
        elif r<=0.7:
            x=0
        else:
            x=-1
        samples.append(x)
    return samples
  
def sum_squares(N):
    N=np.array(N,np.int64)
    ans1=0
    for i in N:
        i=i*i
    for i in N:
        ans1+=i    
    return ans1

def troublemakers(n):
    N=np.array([100000000.0000000000,100000000.0000000000],dtype=np.float64)
    for _ in range(n):
        
        N[1]+=0.35*N[0]
        N[0]-=0.35*N[0]
        N[0]+=N[1]*0.2
        N[1]-=0.2*N[1]
    return N/100000000.0000000000
