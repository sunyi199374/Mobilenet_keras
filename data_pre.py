import numpy as np 
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import sys


# change the data from image to numpy
def read_image(imageName):
    im = Image.open(imageName).convert('RGB')
    imResize = im.resize((256,256), Image.ANTIALIAS)
    data = np.array(imResize)
    return data

def load_image(dir):
    images = []
    labels = []
    # Read folders under train
    text = os.listdir(dir)
    #print(text,"123123123")
    # label image with folder name
    for textPath in text:
        for fn in os.listdir(os.path.join(dir,textPath)):
            fd = os.path.join(dir,textPath, fn)
            images.append(read_image(fd))
            labels.append(textPath)
    #print(labels)
    X = np.array(images)
    y = np.array(list(map(int, labels)))

    #X=X/255
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    pickle.dump( ((X_train, X_test), (y_train, y_test)), open( "data.p", "wb" ) )
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    path = str(sys.argv[1])
    print(path)
    load_image(path)