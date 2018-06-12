import cv2
import os
import numpy as np
import random as shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

TRAIN_DIR= '/Users/aniketdwivedi/Downloads/train'
TEST_DIR=  '/Users/aniketdwivedi/Downloads/test'

#Macintosh HD/Users/aniketdwivedi/Downloads/


IMG_SIZE=50
LR=0.001


#for defining labels to the Image
def label_img(img):
	word_label=img.split('.')[-3]
	if word_label=='cat' :return [1,0]
	elif word_label=='dog' :return [0,1]


#For making Images in readable format that can be fed to the our neural network
def create_train_data():
	train_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label=label_img(img)
		path= os.path.join(TRAIN_DIR,img)
		img= cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		train_data.append([np.array(img),np.array(label)])

	return train_data


def create_test_data():
	test_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		path=os.path.join(TEST_DIR,img)
		img_num=img.split('.')[0]
		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		test_data.append([np.array(img),img_num])

	return test_data

#Making training data

train_data=create_train_data()


#Making nneural net

convnet=input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],name='input')

convnet=conv_2d(convnet,32,2, activation='relu')
convnet=max_pool_2d(convnet, 2)

convnet=conv_2d(convnet,64, 2, activation='relu')
convnet=max_pool_2d(convnet, 2)


convnet=conv_2d(convnet,32,2, activation='relu')
convnet=max_pool_2d(convnet, 2)

convnet=conv_2d(convnet,64, 2, activation='relu')
convnet=max_pool_2d(convnet, 2)

convnet=conv_2d(convnet,32,2, activation='relu')
convnet=max_pool_2d(convnet, 2)

convnet=conv_2d(convnet,64, 2, activation='relu')
convnet=max_pool_2d(convnet, 2)

#for fully connected layer
convnet=fully_connected(convnet,1024, activation='relu')
convnet=dropout(convnet, 0.8)


#for output layer
convnet=fully_connected(convnet, 2, activation='softmax')
convnet=regression(convnet, optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

model=tflearn.DNN(convnet)





train=train_data[0:-500]
test=train_data[-500:]

X=np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
Y=[i[1] for i in train ]


test_x =np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
test_y=[i[1] for i in test ]




model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input':test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id='cats vs dogs')

model.save('cats_Dog_tflearn.model')


test_data=create_test_data()

#lets test it

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
	#cat=[1,0]
	#dog=[0,1]

	img_num=data[0]
	img_data=data[0]
	y=fig.add_subplot(3,4,num+1)
	orig=img_data
	data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)

	model_out=model.predict([data])[0]
	if np.argmax(model_out)==1 : str_label='Dog'
	else: str_label='Cat'

	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)

plt.show()

for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
