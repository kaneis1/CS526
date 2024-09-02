import random

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
     
if __name__ == '__main__':
    
    x = int(input())
    
    draw_samples(x)
    
    
