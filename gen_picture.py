import os
import numpy as np
from PIL import Image
import traceback
import cv2
directory = 'D:/exam/'
traindata  = 'D:/exam/train/picture'
sample = os.path.join(directory,'48-5','negative5')
labelfilepath  = os.path.join(directory,'48-5','negative5.txt')
filenames = os.listdir(traindata)
if not os.path.exists(sample):
    os.makedirs(sample)
labelfile = open(labelfilepath,'w')
a = 1
for (k,filename) in enumerate(filenames) :
    if (a >= 25000):
        break
    allfilename = os.path.join(traindata,filename)
    img = Image.open(allfilename)
    print(filename)
    width,height = img.size
    for  i in  range(0,width,200):
        if (a >= 25000):
                break
        for  j in  range(0,height,200):
            img = Image.open(allfilename)
            img = img.crop([i,j,i+200,j+200])
            img = img.resize((48,48))
            img.save(os.path.join(sample,'{0}.jpg'.format(a)))
            labelfile.write("negative5/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(a))
            labelfile.flush()
            if (a >= 25000):
                break
            a = a+1
        
    
            
    
        

