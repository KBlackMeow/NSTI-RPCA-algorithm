import numpy as np 
import lansvd  as ls
def Soft(tau, X):
    t= np.abs(X) - tau
    a= np.sign(X) * np.where(t<0,0,t)
    return a


def NSTI(Y,r,alpha,max_iter=30,tol=0.01):

    Y_norm = np.linalg.norm(Y,ord=1)
    (d1,d2)=Y.shape
    
    [U,Sg,V,x,y] = ls.lansvd(Y,r)
    U=U[:,0:r]
    V=V[:,0:r]
    Sg=Sg[0:r]
    U=U*(Sg**0.5)
    V=V*(Sg**0.5)
    converged = False
    t=0
    err=[]
    while not converged:

        t=t+1
        YminusUV=Y-np.dot(U,V.T)
        tos=np.sum(np.abs(YminusUV))/(d1*d2)*alpha
        S=Soft(tos,YminusUV)
        E = YminusUV - S

        hess= np.linalg.inv(np.dot(V.T,V))
        Unew = U + np.dot(np.dot(E,V),hess)
        hess= np.linalg.inv(np.dot(U.T,U))
        Vnew = V + np.dot(np.dot(U.T,E).T,hess)

        U=Unew
        V=Vnew

        err .append((np.linalg.norm(S,ord =1)+np.linalg.norm(E,ord ='fro'))/Y_norm)
 
        print('\rclearIter no. %d err %f '%(t,err[-1]),end="")
        if t>=max_iter:
            print("max iter reached")
            converged=True
        if err[-1] <=tol:
            print("error reached")
            converged=True

        if t >1 and abs(err[-2]-err[-1])/err[-1] <=0.00001 :
            print("no improment")
            converged=True

    ret =np.dot(U,V.T)
    return ret ,Y-ret

if __name__ == '__main__':

    B=np.random.randint(-3,4,size=(5000))
    C=np.ones(shape=(5000,5000))

    E=np.asarray([i for i in range(-2500,2500)])
    C[1]=C[1]+np.sign(E)*2
    D=1.0*(np.diag(B)+C)

    r  = 2
    step_const = 10
    max_iter   = 1000
    tol        = 0.02
    tos=2
    L, S = NSTI(D, r,0.1, max_iter,tol)

    print(L[0:5,0:5])