import random
import numpy as np
def draw_samples(n):
 for _ in range(n):
    r=random.random()
    if r<=0.25:
        x=1
    elif r<=0.7:
        x=0
    else:
        x=-1
    print(x)
  
def sum_squares(N):
    ans=0
    for i in N:
        ans+=i*i
    print(ans)

def troublemakers(n):
    N=np.array([1.0,1.0],dtype=float)
    for _ in range(n):
        N[1]+=0.35*N[0]
        N[0]*=0.65
        N[0]+=N[1]*0.2
        N[1]*=0.8
    print(N)
    
    
