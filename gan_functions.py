import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
class Generator2(nn.Module):
    def __init__(self,nz,ngf,nc):
        super().__init__()
        self.step1=nn.Sequential(
            ## nz*1*1
            nn.ConvTranspose2d(nz, ngf * 8,kernel_size= 4,stride= 1,padding= 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ## (ngf*8)*4*4
            nn.ConvTranspose2d(ngf*8, ngf * 16, kernel_size=(3,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            ## (ngf*4)*7*8
            nn.ConvTranspose2d(ngf * 16, ngf *8, kernel_size=(4,4), stride=(1,2), padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ## (ngf*4)*8*16
            nn.ConvTranspose2d(ngf * 8, ngf*4 , kernel_size=4, stride=(1,2), padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            ## (ngf*4)*9*32
            nn.ConvTranspose2d(ngf * 4, ngf*2, kernel_size=4, stride=(1,2), padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            ## (ngf*4)*10*64
            nn.ConvTranspose2d(ngf * 2, nc, kernel_size=(3,9), stride=(1,1), padding=(1,0), bias=False),

            ## (1)*10*72
            nn.ReLU()
        )
    def forward(self,x):
        out=self.step1(x)
        out = out.view(-1,1,10, 72)
        return out

class Generator(nn.Module):
    def __init__(self,nz,ngf,nc):
        super().__init__()

        self.step1=nn.Sequential(
            ## nz*1*1
            nn.Conv2d(1, ngf * 8,kernel_size= (2,3),stride= (1,1),padding= (1,1), bias=True),
            nn.ReLU())

        self.step2 = nn.Sequential(nn.Conv2d( ngf * 8, ngf * 4, kernel_size=(2, 3), stride=(1, 1), padding=(2, 1), bias=True),
            nn.ReLU())

        self.step3 = nn.Sequential(
            nn.Conv2d( ngf * 4, ngf * 2, kernel_size=(2, 3), stride=(1, 1), padding=(2, 1), bias=False),
            nn.ReLU())

        self.step4 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf,kernel_size= (2,3),stride= (1,1),padding= (1,1), bias=False),
            nn.LayerNorm(72),
            nn.ReLU())
        self.step5 = nn.Sequential(
            nn.Conv2d(ngf, 1,kernel_size= (2,3),stride= (1,1),padding= (1,1), bias=False),
            nn.LayerNorm(72),
            nn.ReLU())
        img_size_x = int(((72 + 2 * 1 - 3) / 1) + 1)
        print(img_size_x)
        self.step6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.LayerNorm(72),
            nn.ReLU())
        self.step7= nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.LayerNorm(72),
            nn.ReLU())


    def forward(self,x):
       # print(x.shape)
        out=self.step1(x)
       # print(out.shape)

        out = self.step2(out)
        #print(out.shape)

        out = self.step3(out)
       # print(out.shape)

        out = self.step4(out)


        out = self.step5(out)

        out = self.step6(out)
        #out = self.step7(out)


        out = out.view(-1,1,10, 72)
        return out
device=torch.device("cuda")
noise_dim=72
noise = torch.randn(4,1,1,noise_dim, device=device)
generator = Generator(noise_dim,32,1).to(device)
generator(noise)


class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super().__init__()

        self.model= nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=(1,2), padding=(1,0), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 9 x 35
            nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=(1,2), padding=(0,0), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 17
            nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=(1,2), padding=(0,0), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 5 x 8
            nn.Conv2d(ndf*4, ndf*8, kernel_size=(4,4), stride=(1,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            )
    def forward(self,x):
        output=self.model(x)
        return output

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

plt.rcParams['image.cmap'] = 'gray'



# Discriminator Loss => BCELoss
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)


def train_BCE(data,g_optimizer,d_optimizer,D,G,noise_dim):
    real_inputs, label= to_device(data,device)

    len1 = real_inputs.shape[0]

    sigmoid=nn.Sigmoid()
    real_outputs = D(real_inputs).reshape(-1,1)
    real_label = torch.ones(real_inputs.shape[0], 1).to(device)

    noise = torch.randn(len1, noise_dim, 1, 1, device=device)
    fake_inputs = G(noise)
    fake_outputs = D(fake_inputs).reshape(-1,1)
    fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

    outputs = sigmoid(torch.cat((real_outputs, fake_outputs), 0))
    targets = torch.cat((real_label, fake_label), 0)


    ##train the discriminator
    d_optimizer.zero_grad()
    # Backward propagation
    d_loss = d_loss_function(outputs, targets)

    a=d_loss
    d_loss.backward()
    d_optimizer.step()

    ### Now Generator
    noise = torch.randn(len1, noise_dim, 1, 1, device=device)

    fake_inputs = G(noise)
    fake_outputs = sigmoid(D(fake_inputs)).reshape(-1,1)

    g_loss = g_loss_function(fake_outputs)
    b=g_loss
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return a,b


def train_WGan(data,critic_iterations,noise_dim,G,critic,opt_critic,opt_gen,weight_clip):
    real_inputs, label= to_device(data,device)

    len1 = real_inputs.shape[0]


    for _ in range(critic_iterations):
        noise = torch.randn(len1, 1, 1, noise_dim, device=device)
        fake_inputs = G(noise).to(device)
        critic_real = critic(real_inputs).reshape(-1)
        critic_fake = critic(fake_inputs).reshape(-1)
        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))  # to min
        a=loss_critic
        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        for p in critic.parameters():
            p.data.clamp_(-weight_clip, weight_clip)
    ## train generator: min -E(critic(gen_fake))
    output = critic(fake_inputs).reshape(-1)
    loss_gen = - torch.mean(output)
    b=loss_gen

    G.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    return a,b

def train_Wgan_GP(data,critic_iterations,noise_dim,G,critic,gradient_penalty,opt_critic,opt_gen,lambda_gp):
    real_inputs, label= to_device(data,device)

    len1 = real_inputs.shape[0]
    for _ in range(critic_iterations):
        noise = torch.randn(len1, noise_dim, 1, 1, device=device)
        fake_inputs = G(noise).to(device)
        critic_real = critic(real_inputs).reshape(-1)
        critic_fake = critic(fake_inputs).reshape(-1)

        gp = gradient_penalty(critic, real_inputs, fake_inputs, device)

        loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp)  # to min
        a= loss_critic

        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

    ## train generator: min -E(critic(gen_fake))
    output = critic(fake_inputs).reshape(-1)
    loss_gen = - torch.mean(output)
    b=loss_gen

    G.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    return a,b

def gradient_penalty(critic,real,fake,device):
    b,c,h,w= real.shape
    epsilon = torch.rand((b,1,1,1)).repeat(1,c,h,w).to(device)
    interpolated_images = real*epsilon +fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty
