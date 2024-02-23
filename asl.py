from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import os
import cv2 #open cv


#Loading data and Variables setup
TRAIN_DATA_PATH = os.path.join("SLR", "asl") #SLR\asl

NUM_OF_LETTERS = 26 #class
IMAGE_SIZE = 50 #for resizing every image
NUM_OF_CHANNELS = 1 #1 - grey scale, 3 - RGB
NUM_OF_DENSE_LAYER_NODES = (IMAGE_SIZE * IMAGE_SIZE) // 2 #rule of thumb in neural network design for using half of the total number of pixels as the number of nodes

LABELS = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n','o', 'p', 'q','r', 's', 't','u', 'v', 'w', 'x', 'y', 'z']



#data preprocessing color to grey, resize
def create_data(DATA_PATH):
    x=[] #images
    y=[] #labels
    paths=[]
    for label in LABELS:
        path = os.path.join(DATA_PATH, label) #SLR\asl\a
        label_name = LABELS.index(label) #label encoding
        for img in os.listdir(path): #all the files in this directory will be as a list
            p=os.path.join(path, img) #SLR\asl\z\z_48_rotate_10.jpeg
            paths.append(p)
            print(p)
            try:
                img_array = cv2.imread(p) #reads an image in RGB format
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                x.append(img_array)
                y.append(label_name)
            except Exception as e:
                pass
    return x,y


#data creation
X_train, y_train= create_data(TRAIN_DATA_PATH)

NUM_OF_TRAIN_IMAGES = len(X_train) #x_train is a list in which every element is a 2D array representing a grayscale image.
X_train=np.array(X_train) #height,width,channel 2D - 3D
X_train = X_train.reshape(NUM_OF_TRAIN_IMAGES, IMAGE_SIZE , IMAGE_SIZE, NUM_OF_CHANNELS) #3D - 4D

print(len(X_train))
print(X_train.shape)

X_train = X_train.astype("float32")
X_train /= 255.0 #normalising to 0,1
y_train = to_categorical(y_train, NUM_OF_LETTERS)


# Saving Data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

X_train=np.load("X_train.npy")
y_train=np.load("y_train.npy")


# Creating CNN Model

model = Sequential() #output of one layer - input of another layer

#conv2D(number of output parameter - same bcoz image, kernel)
model.add(Conv2D(IMAGE_SIZE, (3, 3), padding = "same", input_shape = (IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS), activation = "relu"))
model.add(Conv2D(IMAGE_SIZE, (3, 3), activation = "relu")) #we dont need to specify input size as it takes the size of output of previous layer
model.add(MaxPooling2D(pool_size = (2, 2))) #takes an important feature for every 2 X 2 matrix
model.add(Dropout(0.25)) # 1. dropout to avoid overfitting

model.add(Conv2D(2 * IMAGE_SIZE, (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(2 * IMAGE_SIZE, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #1D

model.add(Dense(NUM_OF_DENSE_LAYER_NODES, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_OF_LETTERS, activation = "softmax"))


# Compiling CNN Model

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()


# Training Model

history = model.fit(
    X_train,
    y_train,
    batch_size = 52,
    epochs = 8,
    shuffle = True
)


# Saving Model
model.save('model.h5')

