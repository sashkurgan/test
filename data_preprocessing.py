import pandas as pd
import numpy as np
from PIL import Image
import os
data=pd.DataFrame(columns=['ch1','ch2','ch3','name','clas'])

def identification_image(name):
    img=Image.open(f"dataset/train/{name}")
    img=img.resize((224,224))
    content=np.asarray(img)
    print(content.shape)
    channel1=content[0]
    channel2=content[1]
    channel3=content[2]
    if 'as' in name:
        clas=0
    else:
        clas=1
    return [channel1, channel2, channel3, name, clas]

images_asian=os.listdir('dataset/train/Asian')
images_african=os.listdir('dataset/train/African')
for i in range(len(images_asian)):
    name=f"Asian/{images_asian[i]}"

    data.loc[ len(data.index )] = identification_image(name)



for i in range(len(images_african)):
    name=f"African/{images_african[i]}"
    data.loc[len(data.index)] = identification_image(name)

data.to_csv('dataset/train/data.csv')