#set the matplotlib backend so figures can be saved in the backf=ground
import matplotlib
matplotlib.use('Agg')

#import the necessary packages
from CNN.trafficsign import Traffic
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform, exposure, io
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import random 
import os

def load_split(basepath, csvPath):
    #initialize the list of data and labels
    data = []
    labels = []

    #load the contents of the CSV file, remove the forst line(because its a CSV header)
    # and shuffle the rows( otherwise all example of a particular class will be in sequential order )
    rows = open(csvPath).read().strip().split('\n')[1:]
    random.shuffle(rows)

    print(rows[:3])

    #loop over the rows of the CSV file
    for (i, row) in enumerate(rows):
        # check to see if we should show a status update
        if i > 0 and i % 1000 == 0:
            print('[info] processed {} total images'.format(i))
        
        # split the row into components and then grab the class ID and image path
        (label, imagePath) = row.strip().split(',')[-2:]
        print(label, imagePath)

        #Drive the full path to the image file and load it
        imagePath = os.path.sep.join([basepath, imagePath])
        image = io.imread(imagePath)

        #resize the image to be 32 * 32 pixels, ignoring aspect ratio, and then
        #perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
        image = transform.resize(image, (32,32))
        image = exposure.equalize_adapthist(image, clip_limit= 0.1)

        #update the list of data and labels
        data.append(image)
        labels.append(int(label))

    #Convert the data and label to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    #return a tuple of the data and labels
    return (data, labels)


#Construct the argument parser and parse the argument
ap = argparse.ArgumentParser() 
ap.add_argument('-d','--datasets',required=True,help='path to Datasets')
ap.add_argument('-m','--model',required=True,help='path to output model')
ap.add_argument('-p','--plot',type=str,default='plot.png',help='path to training history plot')
args = vars(ap.parse_args())

#initialize the number of epochs to train for, base learning and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS= 64

#load the label name
labelName = open('Meta.csv').read().strip().split('\n')
labelName = [l.split(',')[1] for l in labelName]

#derive the path to the training and testing csv path
trainPath = os.path.sep.join([args['datasets'],'Train.csv'])
testPath = os.path.sep.join([args['datasets'],'Test.csv'])

#load the training and testing data
print('[info] loading training and testing data...')
(trainX, trainY) = load_split(args['datasets'],trainPath) 
(testX,testY) = load_split(args['datasets'],testPath)

#scale tyhe data to the range of [0,1]
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

#one-hot encode the training and testing labels
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

#calculate the total number of images in each class and initialize a dictionary to store the class weights
classTotals = trainY.sum(axis=0)
classWeight = dict()

#loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = Traffic.build(width=32, height=32, depth=3,
	classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelName))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# plot the training loss and accuracy
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
