import pdb #used for debugging directly from terminal instead of IDE
import numpy as np
from tqdm.auto import tqdm #creates and shows progress bars
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid

#defines how all images are transformed (not sure wether imagechanger is function, object, etc)
imagechanger = transforms.Compose([transforms.Resize(255), #resizes images, shorter size of 255, aspect ratio maintained
                                 transforms.CenterCrop(224), #crops image to 224x224 (no idea what part it crops)
                                 transforms.ToTensor()]) #converts to pytorch tensor

#saves the images then transforms them as per above defined imagechanger process
dataset = datasets.ImageFolder('/content/drive/MyDrive/kaggle/datasets/Data/Train', transform=imagechanger)


#shows the image
def show(tensor, ch=1, size=(512, 512), num=25):
    data = tensor.detach().cpu().view(-1, ch, *size)
    grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()

#hyper parameters
epoch = 1000
cur_iter = 0
info_iter = 300 #frequency of printing image
checkpoint_freq = 25
alpha = 0.2 #leaky relu parameter
z_dim = 64 #dimension of noise vector fed into Generator
lr = 0.001
batch_size = 2048
if torch.cuda.is_available(): device = "cuda"
else: device = "cpu"
h_dim = 120 #number nodes in a hidden layer in generator
o_dim = 150526 #dimension of vector generator outputs, 150528=3*224*224
inp_nodes = 150528 #dimension of input to discriminator
hidden_dim = 256 #number of nodes in a hidden layer in discriminator
accumulation_steps = 32 #accumulated gradient, implemented to avoid memory restriction

#loss function; BCEWithLogitsLoss() sigmoid + binary cross entropy;
loss = nn.BCEWithLogitsLoss()

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#the processes to carry out to go to next layer of nodes
def genBlock(inp_nodes, out_nodes):
    return nn.Sequential(
        nn.Linear(inp_nodes, out_nodes),
        nn.BatchNorm1d(out_nodes), #carries out batch normalization which makes the model more 'stable'
        nn.ReLU()
    )

#fed to the generator which converts it into meaningful image, can not be constant otherwise would generate same image everytime
def gen_noise(batch_size, z_dim):
    return torch.randn(batch_size, z_dim).to(device)

class Generator(nn.Module):
    def __init__(self, z_dim=64, o_dim=150528, h_dim=120):
        super().__init__()
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),
            genBlock(h_dim, h_dim * 2),
            genBlock(h_dim * 2, h_dim * 4),
            genBlock(h_dim * 4, h_dim * 8),
            genBlock(h_dim * 8, o_dim), # The output layer of the generator should have o_dim output features.
            nn.Sigmoid(),
        )

    def forward(self, noise):
        #the generator outsputs flattened vector, this is converted to form 3*224*224
        #view function of pytorch helps
        return self.gen(noise).view(-1, 3, 224, 224)

#process to undergo from one layer of nodes to next in discriminator
def discBlock(inp_nodes, out_nodes):
    return nn.Sequential(
        nn.Linear(inp_nodes, out_nodes),
        nn.LeakyReLU(alpha)
    )

class Discriminator(nn.Module):
    def __init__(self, inp_dim=150528, hidden_dim=256):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.disc = nn.Sequential(
            discBlock(3*224*224, hidden_dim * 4), #input layer of the discriminator will expect 3*224*224 features
            discBlock(hidden_dim * 4, hidden_dim * 2),
            discBlock(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image.view(image.size(0), -1)) #flattens the image inside the discriminator's forward method.


#optimizers
gen = Generator(z_dim).to(device) #generator class object
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr) #adam

disc = Discriminator().to(device) #discriminator class object
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def gen_loss(loss_func, gen, disc, batch_size, z_dim):
    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    target = torch.ones_like(pred).to(device)
    return loss_func(pred, target)

def disc_loss(loss_func, gen, disc, batch_size, z_dim, real):
    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach()) #.detach() prevents gradient from changing generators weights
    disc_fake_target = torch.zeros_like(disc_fake).to(device)
    disc_fake_loss = loss_func(disc_fake, disc_fake_target).to(device)

    disc_real = disc(real)
    disc_real_target = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_target)

    return (disc_fake_loss + disc_real_loss) / 2

checkpoint_path = '/content/drive/My Drive/Colab Notebooks/model_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
start_epoch = checkpoint['epoch'] + 1
cur_iter = checkpoint['cur_iter']

gen.load_state_dict(checkpoint['gen_state_dict'])
disc.load_state_dict(checkpoint['disc_state_dict'])
gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])

#training
mean_gen_loss = 0
mean_disc_loss = 0
for epoch in range(start_epoch, epoch):
    print(epoch)
    mean_disc_loss_list = []
    mean_gen_loss_list = []
    iters_list = []

    for real_image, _ in enumerate(tqdm(dataloader)):
      try:
        cur_batch_size = len(real_image)
        real_image = real_image.to(device) #loading real_image to device
        disc_opt.zero_grad()
        cur_batch_size = len(real_image)
        disc_losses = disc_loss(loss, gen, disc, cur_batch_size, z_dim, real_image)
        disc_losses.backward()
        if cur_iter % accumulation_steps == 0:
          disc_opt.step()
          disc_opt.zero_grad()

        gen_opt.zero_grad()
        gen_losses = gen_loss(loss, gen, disc, cur_batch_size, z_dim)
        gen_losses.backward()
        gen_opt.step()

        mean_disc_loss += disc_losses.item() / info_iter
        mean_gen_loss += gen_losses.item() / info_iter
        mean_disc_loss_list.append(mean_disc_loss)
        mean_gen_loss_list.append(mean_gen_loss)

        #prints real and ai generated images after every info_iter iterations
        if (cur_iter+1) % info_iter == 0 and cur_iter > 0:
          try:
            with torch.no_grad():
              fake_noise = gen_noise(cur_batch_size, z_dim)
              fake = gen(fake_noise)
              show(real_image)
              show(fake)
              print(f"{epoch} : step {cur_iter}, Generator loss : {mean_gen_loss}, Discriminator Loss : {mean_disc_loss}")
              mean_gen_loss, mean_disc_loss = 0, 0

          except:
            print("error in generating image")

        if (cur_iter + 1) % checkpoint_freq == 0 and cur_iter > 0:
          print(f"Saving checkpoint at iteration {cur_iter}")
          torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_opt_state_dict': gen_opt.state_dict(),
            'disc_opt_state_dict': disc_opt.state_dict(),
            'cur_iter': cur_iter,
          }, checkpoint_path)
          print(f"Checkpoint saved at iteration {cur_iter}")

        iters_list.append(cur_iter)
        cur_iter += 1

      except RuntimeError as e:
        if 'out of memory' in str(e):
          print("GPU out of memory. Saving checkpoint and exiting")
          torch.save(
              {
              'epoch': epoch,
              'gen_state_dict': gen.state_dict(),
              'disc_state_dict': disc.state,
              'disc_state_dict': disc.state_dict(),
              'gen_opt_state_dict': gen_opt.state_dict(),
              'disc_opt_state_dict': disc_opt.state_dict(),
              'cur_iter': cur_iter,
              }, checkpoint_path
          )
