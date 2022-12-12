import pandas as pd
import numpy as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr=0']
y=pd.read_csv('labels.csv')('labels')
print(pd.series(y).value.counts())
classes=['A','B'.'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)
x,y=fetch_openml('mnist_784', version=1, return_x_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=3500,test_size=500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0

clf=LogisticRegression(solver=saga,
multi_class='multinomial').fit(x_train_scaled, y_train)

def getPrediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw.resize((22,30), Image.ANTIALIAS)
    pixel_filter=20
    min_pixel= np.percentile(image_bw-image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min-pixel,0,255)
    nax_pixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scaled=
    np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_predict