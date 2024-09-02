import numpy as np
N=np.array([1.0,1.0],dtype=float)
def troublemakers(n):
    for _ in range(n):
        N[1]+=0.35*N[0]
        N[0]*=0.65
        N[0]+=N[1]*0.2
        N[1]*=0.8
    print(N)

if __name__ == '__main__':
    n=int(input())
    troublemakers(n)
    
    
