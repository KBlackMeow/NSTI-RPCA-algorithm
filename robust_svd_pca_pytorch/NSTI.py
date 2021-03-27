from decimal import Decimal
import numpy as np 
import svd_pytorch_gd as tsvd
import torch as th
def Soft(tau, X):
    t= (th.abs(X) - tau).double()
    return (th.sign(X).float()*th.where(t<0.,0.,t)).float()


def NSTI(Y,r,alpha,max_iter=30,tol=0.01):
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    Y_norm = np.linalg.norm(Y)
    (d1,d2)=Y.shape
    
    [U,V] = tsvd.svd_torch_gd(Y,r,0.1)
    Y=th.Tensor(Y).to(DEVICE)
    U=th.Tensor(U).to(DEVICE)
    V=th.Tensor(V).to(DEVICE)
    converged = False
    t=0
    err=[]
    p = th.Tensor([(d1*d2)*alpha]).to(DEVICE)
    while not converged:

        t=t+1
        YminusUV=Y- th.mm(U,V.T)

        tos=(th.sum(th.abs(YminusUV))/p)
        S=Soft(tos,YminusUV)

        E = YminusUV - S
        
        hess= th.inverse(th.mm(V.T,V))
        Unew = U + th.mm(th.mm(E,V),hess)
        hess= th.inverse(th.mm(U.T,U))
        Vnew = V + th.mm(th.mm(E.T,U),hess)

        U=Unew
        V=Vnew

        err.append((th.norm(S,p=1)+th.norm(E,p='fro')).double()/Y_norm)
 
        print('\rclearIter no. %d err %f '%(t,err[-1]),end="")
        if t>=max_iter:
            print("max iter reached")
            converged=True
        if err[-1] <=tol:
            print("error reached")
            converged=True

        if t >1 and abs(err[-2]-err[-1])/err[-1] <=0.01 :
            print("no improment")
            converged=True

    L =th.mm(U,V.T)
    S = Y-L
    return L.cpu().numpy() ,S.cpu().numpy()

if __name__ == '__main__':

    B=np.random.randint(-3,4,size=(5000))
    C=np.ones(shape=(5000,5000))

    E=np.asarray([i for i in range(-2500,2500)])
    C[1]=C[1]+np.sign(E)*2
    D=1.0*(np.diag(B)+C)

    r  = 2
    max_iter   = 1000
    tol        = 0.0002

    L, S = NSTI(D, r,0.05, max_iter,tol)

    print(L[0:5,0:5])