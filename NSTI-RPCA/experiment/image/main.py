from PIL import Image
import numpy as np
import os

def sparse_mask(a,shape):
    ret = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
             if np.random.rand(1)>a:
                ret[i,j]=0
    return ret

def process(path,a):
    I = Image.open(path)
    L = I.convert('L')
    L = np.asarray(L)
    mask = sparse_mask(a,L.shape)
    x= (np.random.randint(low=0,high=2,size=mask.shape)*254).astype('uint8')
    L=np.where(mask,x,L)
    L=Image.fromarray(L,mode='L')
    return L

if __name__ == '__main__':
    
    
    # for i in range(10):
    #     L = process('000011.jpg')
    #     L.save('image1/%04d.jpg'%(i+1))
    fileList = os.listdir('data10')
    for i,v in enumerate(fileList):
        I = Image.open('data10/'+v)
        L = I.convert('L')
        L.save('src/'+"%04d.jpg"%(i+1))
        # for j in range(10):
        #     L = process('data10/'+v,0.2)
        #     L.save('image10/%02d%02d.jpg'%(i+1,j+1))