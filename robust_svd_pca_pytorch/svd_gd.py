import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy
from scipy import linalg
def genData(m,n,k):

    d = []
    for i in range(m):
        d.append(np.random.randint(0,10,size=n))
    d= np.array(d,dtype = np.float)
    U,S,V = np.linalg.svd(d)
    D = np.dot(U[:,0:k]*S[0:k],V[:,0:k].T)
    return D,S[0:k]

def svd_gd(X,k,a=0.01,itn=1000):

    X=np.asarray(X)
    m,n=X.shape
    U,_ = genData(m,k,k)
    V,_ = genData(n,k,k)
    F=np.linalg.norm(X)

    U=linalg.orth(U)
    V=linalg.orth(V)

    U=U*(F**0.5)
    V=V*(F**0.5)
    a=a/F

    stop = False
    cot = 0
    t1=0
    while not stop:
        E = np.dot(U,V.T)-X

        B = (np.dot(U.T,U)-np.dot(V.T,V))
        U = U -(np.dot(E,V) + np.dot(U,B))*a
        V = V -(np.dot(E.T,U) - np.dot(V,B))*a

        t = (0.5*np.linalg.norm(np.dot(U,V.T)-X)**2+0.25*np.linalg.norm(np.dot(U.T,U)-np.dot(V.T,V))**2)/F**2

        print("%dth loss = %.8f"%(cot,t))
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
    return [U,V]

if __name__ == '__main__':
    D,S = genData(2000,1000,2)

    def func1():
        np.linalg.svd(D)

    def func2():
        U,V=svd_gd(D,2,a=0.1)
        print(D[0:5,0:5],"\n\n",(np.dot(U,V.T)-D)[0:5,0:5])
        print(np.dot(U.T,U),"\n\n",np.dot(V.T,V),"\n\n",S)
    # def func3():
    #     import lansvd as ls
    #     [U,Sg,V,x,y] = ls.lansvd(D,2)
    # print("func1",timeit.timeit(func1,number=1))
    print("func2",timeit.timeit(func2,number=1))
    # print(timeit.timeit(func3,number=1))