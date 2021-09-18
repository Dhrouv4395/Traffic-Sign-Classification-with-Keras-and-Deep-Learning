#import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense

class Traffic:
    @staticmethod
    def build(width, height, depth, classes):
        #initialize the model along with the input shape to be "Channel last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        channelDim = -1

        # CONV => RELU => BN => POOL
        model.add(Conv2D(8, (5, 5), padding='Same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #first set of (CONV => RELU => CONV => RELU) *2 => POOL
        model.add(Conv2D(16, (3, 3), padding ='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3,3), padding="same"))
        model.add(Activation('relu')) 
        model.add(BatchNormalization(axis=channelDim))
        model.add(Conv2D(32, (3,3), padding="same"))
        model.add(Activation('relu')) 
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #First set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        #Second set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        #return the constructed network architecture
        return model 