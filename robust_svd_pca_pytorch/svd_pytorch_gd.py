import torch as th
import numpy as np
from scipy import linalg
import timeit
def genData(m,n,k):

    d = []
    for i in range(m):
        d.append(np.random.randint(0,10,size=n))
    d= np.array(d,dtype = np.float)
    U,S,V = np.linalg.svd(d)
    D = np.dot(U[:,0:k]*S[0:k],V[:,0:k].T)
    return D,S[0:k]

def svd_torch_gd(X,k,a=0.01,itn=1000):
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    X=np.asarray(X)
    m,n=X.shape
    U,_ = genData(m,k,k)
    V,_ = genData(n,k,k)
    F=np.linalg.norm(X)

    U=linalg.orth(U)
    V=linalg.orth(V)

    U=U*(F**0.5/np.linalg.norm(U))
    V=V*(F**0.5/np.linalg.norm(V))
    a=a/F

    X = th.Tensor(X).to(DEVICE)
    U = th.Tensor(U).to(DEVICE)
    V = th.Tensor(V).to(DEVICE)

    stop = False
    cot = 0
    t1=0
    while not stop:

        E = th.mm(U,(V.T))-X
        B = (th.mm(U.T,U)-th.mm(V.T,V))
        U = U -(th.mm(E,V) + th.mm(U,B))*a
        V = V -(th.mm(E.T,U) - th.mm(V,B))*a

        t = (0.5*th.norm(th.mm(U,V.T)-X)**2+0.25*th.norm(th.mm(U.T,U)-th.mm(V.T,V))**2)/F**2

        # print("%dth loss = %.8f"%(cot,t))
        if cot==0:
            t1=t
        else:
            if(abs(t-t1)<1e-5):
                t1 =t
                stop = True
            else :
                t1 =t
        if not cot < itn:
            stop = True
        cot+=1

    print(t1)
    return [U.cpu().numpy(),V.cpu().numpy()]


if __name__ == '__main__':
    D,S = genData(2000,2000,2)

    def func1():
        np.linalg.svd(D)

    def func2():
        U,V=svd_torch_gd(D,2,a=0.1)
        print(D[0:5,0:5],"\n\n",(np.dot(U,V.T)-D)[0:5,0:5])
        print(np.dot(U.T,U),"\n\n",np.dot(V.T,V),"\n\n",S)
    def func3():
        th.svd(D)
    print("func1",timeit.timeit(func1,number=1))
    print("func2",timeit.timeit(func2,number=1))
    print(timeit.timeit(func3,number=1))    