from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.models import Model
import keras
import numpy as np
import cv2

IMAGE_SIZE = 784 # 28 * 28 pixels

# Scale of downscaling/ upscaling
SCALE = 4 # [0.25, 0.5, 1, 2, 4]
 
# Basic encoder layers
def encoder(input_image, code_dimention):
    layer1 = Dense(128, activation='relu')(input_image)
    layer2 = Dense(64, activation='relu')(layer1)
    layer3 = Dense(code_dimention, activation='sigmoid')(layer2)
    return layer3
 
# Basic decoder layers
def decoder(encoded_image):
    layer1 = Dense(64, activation='relu')(encoded_image)
    layer2 = Dense(128, activation='relu')(layer1)
    if SCALE > 1:
        layer3 = Dense(IMAGE_SIZE*(2**(SCALE)), activation='sigmoid')(layer2)
    else:
        layer3 = Dense(IMAGE_SIZE*(0.5**((1/SCALE))), activation='sigmoid')(layer2)
    return layer3

# Set input size
input_image = Input(shape=(IMAGE_SIZE, ))

# Setup model for training
model = Model(input_image, decoder(encoder(input_image, 32)))
model.compile(loss='mean_squared_error', optimizer='nadam')
  
# Setup mnist data
(x_train, _), (x_test, _) = mnist.load_data()
  
# Scale training data based on SCALE
x_train_resize = np.zeros((x_train.shape[0],int(x_train.shape[1]*SCALE),int(x_train.shape[2]*SCALE) ))#, np.zeros((28*SCALE,28*SCALE)))
for im in range(len(x_train_resize)):
    x_train_resize[im] = cv2.resize(x_train[im], dsize=(int(28*SCALE), int(28*SCALE)), interpolation=cv2.INTER_CUBIC)

x_train = x_train.astype('float32') /255.0
x_train_resize = x_train_resize.astype('float32') /255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train_resize = x_train_resize.reshape((len(x_train_resize), np.prod(x_train_resize.shape[1:])))


# Scale test data based on SCALE
x_test_resize = np.zeros((x_test.shape[0],int(x_test.shape[1]*SCALE), int(x_test.shape[2]*SCALE) ))#, np.zeros((28*SCALE,28*SCALE)))
for im in range(len(x_test_resize)):
    x_test_resize[im] = cv2.resize(x_test[im], dsize=(int(28*SCALE), int(28*SCALE)), interpolation=cv2.INTER_CUBIC)

x_test = x_test.astype('float32') /255.0
x_test_resize = x_test_resize.astype('float32') /255.0

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_test_resize = x_test_resize.reshape((len(x_test_resize), np.prod(x_test_resize.shape[1:])))
  
# # Start Training
# model.fit(x_train, x_train_resize, batch_size=128, epochs=10, validation_data=(x_test, x_test_resize), shuffle=True)

# # Save the Model
# model.save('autoencoder_mnist')

# Load model for inference  
model = keras.models.load_model('autoencoder_mnist')

# Generate output on 10 samples
for i in range(1,10):
    
    # Inference
    array = model.predict(x_test[i-1:i])
    array  = np.reshape(array, (int(28*SCALE), int(28*SCALE)))

    # Groundtruth
    inp = np.reshape(x_test[i-1:i], (28, 28))

    # Display output
    cv2.imshow('input', inp)
    cv2.imshow('output', array)
    cv2.waitKey(0)

