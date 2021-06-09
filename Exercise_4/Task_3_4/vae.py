

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import random

class VAE(keras.Model):
    def __init__(self, encoder, decoder,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # self.total_loss_hist= []
        # self.reconstruction_loss_hist = []
        # self.latent_loss_hist = []
        

    def train_step(self, data):
        with tf.GradientTape() as tape:
            
            z_mean, z_std, z = self.encoder(data)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, self.decoder(z)), axis=(1, 2)))
            latent_loss = -0.5 * (1 + z_std - tf.square(z_mean) - tf.exp(z_std))
            latent_loss = tf.reduce_mean(tf.reduce_sum(latent_loss, axis=1))
            total_loss = reconstruction_loss + latent_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # print(type(reconstruction_loss))
        # print(reconstruction_loss.shape)
        # self.total_loss_hist.append(total_loss)
        # self.reconstruction_loss_hist.append(reconstruction_loss)
        # self.latent_loss_hist.append(latent_loss)
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
        }

def encoder(latent_space=2):

	mnist_input = keras.Input(shape=(28, 28, 1)) # Mnist consists of 28x28 pixel images
	flattened_input = layers.Reshape((784,), name="Reshape")(mnist_input)  
	#MI_input = keras.Input(shape=(2,1))# MI building data consists of 2x1 positional data
	#flattened_input = layers.Reshape((2,), name="Reshape")(MI_input)  
	hidden_1_e = layers.Dense(256, activation="relu")(flattened_input)
	hidden_2_e = layers.Dense(256, activation="relu")(hidden_1_e)
	encoder_mean = layers.Dense(latent_space, name = 'encoder_mean')(hidden_2_e)
	encoder_std = layers.Dense(latent_space, name = 'encoder_std')(hidden_2_e)

	# Sampling z from x s.t. z = mu + L*epsilon
	batch = tf.shape(encoder_mean)[0]
	dimension = tf.shape(encoder_mean)[1]
	epsilon = tf.keras.backend.random_normal(shape=(batch, dimension))
	sample_z = encoder_mean + tf.sqrt(tf.exp(encoder_std)) * epsilon
	#encoder = keras.Model(mnist_input, [encoder_mean, encoder_std, sample_z], name="encoder")
	encoder = keras.Model(MI_input, [encoder_mean, encoder_std, sample_z], name="encoder")
	#encoder.summary()
	return encoder


def decoder(latent_space=2):
	latent_input = keras.Input(shape=(latent_space,)) # z's input
	hidden_1_d = layers.Dense(256, activation="relu")(latent_input)
	hidden_2_d = layers.Dense(256, activation="relu")(hidden_1_d)
	decoder_output = layers.Dense(28*28, activation="sigmoid", name="decoder_output")(hidden_2_d)
	decoder_output = layers.Reshape((28, 28, 1), name="Reshape")(decoder_output)  
	# decoder_output = layers.Dense(2, activation="sigmoid", name="decoder_output")(hidden_2_d)
	# decoder_output = layers.Reshape((2, 1), name="Reshape")(decoder_output) 
	decoder = keras.Model(latent_input, decoder_output, name="decoder")
	#decoder.summary()
	return decoder

def loadData():
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	x_train = np.expand_dims(x_train, -1).astype("float32") / 255
	x_test = np.expand_dims(x_test, -1).astype("float32") / 255

	return x_train,x_test,y_train,y_test

def plotLoss(latent_space):
    # Using readlines()
    if latent_space == 2:
    	log_file = 'output_2.txt'
    else:
    	log_file = 'output_32.txt'
    loss_log = open(log_file, 'r')
    Lines = loss_log.readlines()
    total_loss = []
    reconstructed_loss = []
    latent_loss = []

    is_odd_line = False
    # # Strips the newline character
    for line in Lines:
        if is_odd_line:
            line_array = line.split(' ')
            total_loss.append(float(line_array[7]))
            reconstructed_loss.append(float(line_array[10]))
            latent_loss.append(float(line_array[13]))
            is_odd_line = False
        else:
            is_odd_line = True
    #print(total_loss)
    y1 = total_loss
    x = np.linspace(1,len(total_loss),len(total_loss))
    
    y2 = reconstructed_loss
    x = np.linspace(1,len(reconstructed_loss),len(reconstructed_loss))

    # plot lines
    plt.title("In "+str(latent_space)+" dimensonal latent space case")
    plt.plot(x,y1, label = "total_loss")
    plt.plot(x,y2, label = "reconstructed_loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def plotLatentRepresentation(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.suptitle("Latent Space Digit Distribution")
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()




def plotReconstruction(x_test,y_test,vae):
    
    digitNumber = 15
    x_rand_indexes=random.sample(range(x_test.shape[0]), digitNumber)
    x_random = [x_test[i] for i in x_rand_indexes]
    y_random = [y_test[i] for i in x_rand_indexes]
    x_random = np.expand_dims(x_random, -1).astype("float32") / 255
    
    _,_,encoder_output = vae.encoder.predict(x_random)
    # print(encoder_output.shape)
    # print(encoder_output[0])
    reconstructed = vae.decoder.predict(encoder_output)
    
    digit_size = 28

    plt.figure(figsize=(6, 7))
    plt.suptitle("Reconstructed & Original Images")
    for i in range(1,6):
        for j in range(1,4):
            index = (i-1)*3+j-1
            digit = reconstructed[index].reshape(digit_size, digit_size)

            ax = plt.subplot(5,6,index+1+(i-1)*3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Rec_"+str(index+1+(i-1)*3))
            plt.imshow(digit)
        for j in range(1,4):
            index = (i-1)*3+j-1
            digit_org = x_test[x_rand_indexes[index]].reshape(digit_size,digit_size)

            ax2 = plt.subplot(5,6,(i)*3 + index+1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title("Org_"+str(index+1+(i-1)*3))
            plt.imshow(digit_org)
    plt.show()

def plotGeneration(latent_space,decoder):
    
    encoder_output = np.random.normal(size=(15,latent_space)) #Random 15 z samples
    reconstructed = decoder.predict(encoder_output)
    
    digit_size = 28

    plt.figure(figsize=(6, 9))
    plt.suptitle("Generated Images")
    for i in range(1,6):
        for j in range(1,4):
            index = (i-1)*3+j-1
            digit = reconstructed[index].reshape(digit_size, digit_size)

            ax = plt.subplot(5,3,index+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Img "+str(index+1))
            plt.imshow(digit)

    plt.show()

def plotAll(vae, x_test, y_test,latent_space):
	plotLatentRepresentation(vae, x_test, y_test)
	plotReconstruction(x_test,y_test,vae)
	plotGeneration(latent_space,vae.decoder)

















	