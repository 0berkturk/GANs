import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from gan_functions import train_BCE, Discriminator, Generator, to_device, get_default_device,train_WGan, train_Wgan_GP, gradient_penalty
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

BATCH_SIZE=64
epochs=15

device = get_default_device()
print('GPU State:', device)


data_dir = "D:/ams-02_data/data"

xval_loader = torch.load(data_dir + "/x_train_part1.pt")

xval_loader = xval_loader.numpy()[:,0,:,:]/50000
xval_loader = torch.from_numpy(xval_loader[:, [0, 1, 4, 5, 8, 9, 12, 13, 16, 17], :].reshape(-1,1,10,72))

yval_loader = torch.load(data_dir + "/y_train_part1.pt")
train_loader= TensorDataset(xval_loader,yval_loader)
train_loader= DataLoader(train_loader,batch_size=BATCH_SIZE, shuffle=True)




WGan_GP = 0
WGan = 1
BCE= 0

if (WGan_GP == 1):
    print("Wgan_gp")
    noise_dim = 72
    critic_iterations = 5
    lambda_gp = 10
    lr=5e-5

    generator = Generator(noise_dim,32,1).to(device)
    critic = Discriminator(1,32).to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    start_time = time.time()

    for epoch in range(epochs):
        epoch += 1
        times=0
        for data in (train_loader):
            times += 1
            d_loss, g_loss = train_Wgan_GP(data,critic_iterations,noise_dim,generator,critic,gradient_penalty,opt_critic,opt_gen,lambda_gp)

            if times % 10 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader),
                                                                            d_loss.item(), g_loss.item()))

                noise = torch.randn(4,1,1,noise_dim, device=device)
                fake_inputs = generator(noise)
                imgs_numpy = fake_inputs.data.cpu().numpy()
                for i in range(4):

                    a = imgs_numpy[i][0]

                    plt.figure()
                    cmap = plt.get_cmap('plasma')
                    cmap.set_under('white')
                    plt.imshow(a, interpolation='nearest', vmin=0.01, cmap=cmap)
                    plt.colorbar()
                    plt.show()

    print('Training Finished.')
    print('Cost Time: {}s'.format(time.time() - start_time))




if (WGan == 1):
    print("Wgan")
    noise_dim = 72
    critic_iterations = 5
    weight_clip = 0.01
    lr=1e-4

    generator = Generator(noise_dim,32,1).to(device)
    critic = Discriminator(1,32).to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=lr)
    opt_critic = optim.Adam(critic.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        epoch += 1
        times = 0
        for data in (train_loader):
            times += 1
            d_loss, g_loss = train_WGan(data,critic_iterations,noise_dim,generator,critic,opt_critic,opt_gen,weight_clip)

            if times % 10 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader),
                                                                            d_loss.item(), g_loss.item()))
                noise = torch.randn(1, 1, 1, noise_dim, device=device)
                fake_inputs = generator(noise)
                imgs_numpy = fake_inputs.data.cpu().numpy()

                for i in range(1):
                    a = imgs_numpy[i][0]
                    plt.figure()
                    cmap = plt.get_cmap('plasma')
                    cmap.set_under('white')
                    plt.imshow(a, interpolation='nearest', vmin=0.01, cmap=cmap)
                    plt.colorbar()
                    plt.show()




    print('Training Finished.')
    print('Cost Time: {}s'.format(time.time() - start_time))



if(BCE==1):
    print("BCE")
    noise_dim = 100
    lr = 0.0002

    G = Generator(noise_dim,32,1).to(device)
    D = Discriminator(1,32).to(device)


    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


    start_time=time.time()

    for epoch in range(epochs):
        epoch+=1
        times=0
        for data in (train_loader):
            times += 1
            d_loss,g_loss =train_BCE(data, g_optimizer, d_optimizer, D, G, noise_dim)

            if times % 100 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader),
                                                                            d_loss.item(), g_loss.item()))
                noise = torch.randn(2, noise_dim, 1, 1, device=device)
                fake_inputs = G(noise)
                imgs_numpy = fake_inputs.data.cpu().numpy()

                for i in range(2):
                    a = imgs_numpy[i][0]
                    plt.figure()
                    cmap = plt.get_cmap('plasma')
                    cmap.set_under('white')
                    plt.imshow(a, interpolation='nearest', vmin=0.01, cmap=cmap)
                    plt.colorbar()
                    plt.show()



    print('Training Finished.')
    print('Cost Time: {}s'.format(time.time() - start_time))

