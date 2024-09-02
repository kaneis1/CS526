import numpy as np
N=np.array([])
def sum_squares(N):
    ans=0
    for i in N:
        ans+=i*i
    print(ans)

if __name__ == '__main__':
    n=int(input())
    for i in range(n):
        x=int(input())
        N=np.append(N,x)
    
    sum_squares(N)
