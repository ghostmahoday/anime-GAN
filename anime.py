import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Conv2DTranspose, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2



def load_anime(data_path):
    images = []
    files = os.listdir(path)
    
    for i in files:
        img = cv2.imread(path+i)
        images.append(img)
    
    images = images.astype('float32')
    images = images/255.0
    return images


def real_data(dataset, n_samples=100):
    indexes = np.random.randint(0, dataset.shape[0],n_samples)
    x_real = dataset[indexes]
    y_real = np.ones((n_samples,1))
    return x_real, y_real 


def fake_data(gen_model, n_samples=100, latent_dim =100):
    x_fake = random_noise(n_samples, latent_dim)
    x_fake = gen_model.predict(x_fake)
    y_fake = np.zeros((n_samples,1))
    return x_fake, y_fake


def random_noise(n_samples, latent_dim):
    x_fake = np.random.randn(n_samples*latent_dim)
    x_fake = x_fake.reshape((n_samples, latent_dim))
    return x_fake
    

def g_model(latent_dim=100):
    
    model = Sequential()
    model.add(Dense(128*7*7, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    model.add(Conv2DTranspose(128, (4,4), strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128,(4,4),strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid'))
    return model


def d_model():
    model=Sequential()
    model.add(Conv2D(64, (3,3), strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    opt= Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def gan_model(g_model, d_model):
    d_model.trainable=False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model    


def train(n_samples, latent_dim, epochs, batch_size):
    gen_model = g_model(100)
    dis_model = d_model()
    gan = gan_model(gen_model, dis_model)
    
    dataset = load_anime('./data/')
    no_of_batches_per_epoch = int(dataset.shape[0]/batch_size)
    half_batchsize = int(batch_size/2)
    
    for i in range(epochs):
        for j in range(no_of_batches_per_epoch):
            x_real, y_real = real_data(dataset, half_batchsize)
            x_fake, y_fake = fake_data(gen_model, 100, half_batchsize)
            X = np.vstack(x_real, x_fake)
            Y = np.vstack(y_real, y_fake)
            
            dis_loss, _  = dis_model.train_on_batch(X,Y)
            X_gan = random_noise(batch_size, latent_dim)
            Y_gan = np.ones((batch_size, latent_dim))
            gan_loss, _ = gan.train_on_batch(X_gan, Y_gan)
            
            print("epoch: ",i+1,"\t",j+1,"/",batch_size,"\t",'d_loss= ',d_loss,'\t','g_loss= ',g_loss,sep='')
            
        if i%5==0:
            n=10
            x_fake_random, _ = fake_data(100,100)
            for k in range(100):
                plt.axis('off')
                plt.subplot(n,n,k+1)
                plt.imshow(x_fake_random[k].reshape(28,28))
            plt.save('./output/'+i+'.jpg')
            gan.save('gan'+i+'.h5')
            
            
    
    
train(100, 100, 50, 256)

y = fake_data()

