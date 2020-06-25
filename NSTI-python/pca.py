import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import NSTI
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# newX = pca.fit_transform(X)
# print(X)
# print(newX)
# print(pca.explained_variance_ratio_)


def genData(show=False):

    a = 2
    b = 10
    x = (np.random.rand(100)-0.5)*20
    err = np.random.randn(100)
    y = x*a +err
    if show :
        plt.scatter(x,y,s=50)                #绘制一系列点
        plt.title("Square Numbers",fontsize=24)        #给图标指定标题
        plt.xlabel("Value",fontsize=14)          #为x轴设置标题
        plt.ylabel("Square of value",fontsize=14)             #为y轴设置标题
        plt.tick_params(axis='both',which='major',labelsize=14)         #设置刻度标记大小
        plt.show()
    return x,y


if __name__ == "__main__":
    x,y = genData(False)
    plt.scatter(x,y,s=5,label= "Data")
    d=np.array([x,y]).T
    pca = PCA(n_components=1)
    xn = pca.fit_transform(d)
    xn = xn.dot(pca.components_)
    plt.plot(xn[:,0],xn[:,1],linewidth=2,c='r',label = "PCA") 
    # plt.scatter(xn[:,0],xn[:,1],s=20)                #绘制一系列点
    # plt.title("Square Numbers",fontsize=24)        #给图标指定标题
    plt.xlabel("X",fontsize=14)          #为x轴设置标题
    plt.ylabel("Y",fontsize=14)             #为y轴设置标题
    plt.tick_params(axis='both',which='major',labelsize=14)         #设置刻度标记大小
    plt.legend()
    plt.show()

    xe=x
    ye=y
    for i in range(100):
        if i % 15 ==0:
            ye[i] += (np.random.rand(1)-0.5)*100

    plt.scatter(xe,ye,s=5,label = "Data") 
    d=np.array([xe,ye]).T
    xn = pca.fit_transform(d)
    xn = xn.dot(pca.components_)
    plt.plot(xn[:,0],xn[:,1],linewidth=2,c='r',label = "PCA")   
            #为y轴设置标题
    plt.tick_params(axis='both',which='major',labelsize=14)         #设置刻度标记大小
    #rp
    d,s = NSTI.NSTI(d,1,1,300,0.02)
    plt.plot(d[:,0],d[:,1],linewidth=2,c='y',label= "RPCA") 
    # plt.title("PCA & RPCA",fontsize=24)        #给图标指定标题
    plt.xlabel("X",fontsize=14)          #为x轴设置标题
    plt.ylabel("Y",fontsize=14) 
    plt.tick_params(axis='both',which='major',labelsize=14)         #设置刻度标记大小
    plt.legend()
    plt.show()