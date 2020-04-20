



#########################################
#				 Imports 				#
#########################################

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from IPython.display import HTML

from PIL import Image
import natsort

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


#########################################
#			 Properties 				#
#########################################

# Root directory for dataset
dataroot = "/home/dcor/ronmokady/workshop20/celebA"

# directory for weights file
Save_PATH = "/savefolder_AE"

# Number of workers for dataloader
workers = 8

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
img_channels = 3

# Size of latent vector (middle vector - flatten)
middle_vector_size = 256

# Size of image (image_res x image_res)
image_res = 128

# Number of training epochs
num_epochs = 100

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#
shuffle_dataset = True

#
test_size = 20000
train_size = len([name for name in os.listdir(dataroot) if os.path.isfile(os.path.join(dataroot, name))]) - test_size

#########################################
#			 HyperParameters 			#
#########################################
# Learning rate for optimizers
lr = 0.00305 #0.001

# Adam optimizers hyperparameters
adam_beta1 = 0.9 #0.9
adam_beta2 = 0.9 #0.999
adam_eps = 1e-08 #1e-08
adam_weight_decay = 0 #0

#Leaky RELU slope
LR_slope = 0.0056 #0.01

# BatchNorm hyperparameters
batch_norm_eps = 0.000003 #1e-05
batch_norm_momentum = 0.3 #0.1


#########################################
#				 Using cuda				#
#########################################
if ngpu>0 and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


#########################################
#		 Loading and Preprocessing 		#
#########################################

class CelebADataset(Dataset):
    """CelebA dataset."""

    def __init__(self, images_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.all_imgs = os.listdir(images_dir)
        self.all_imgs = natsort.natsorted(self.all_imgs)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.all_imgs[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            tensor_image = self.transform(image)
            
        return tensor_image

def init_Data():
    # Create the dataset
    dataset = CelebADataset(dataroot,
                            transform=transforms.Compose([
                                   transforms.Resize((image_size, image_size)),
                                   transforms.ToTensor()
                               ]))
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(manualSeed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[test_size:], indices[:test_size]
    
    # Create data samplers: (Could (maybe) use Subset)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              #shuffle=True,
                                              sampler=train_sampler,
                                              num_workers=workers,
                                              drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              #shuffle=True,
                                              sampler=test_sampler,
                                              num_workers=workers,
                                              drop_last=True)
    
    # Plot some training images
    #real_batch = next(iter(train_loader))
    #plt.figure(figsize=(8,8))
    #plt.axis("off")
    #plt.title("Training Images")
    #plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    return train_loader, test_loader

#########################################
#			 Define Network 			#
#########################################

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# AutoEncoder Code - "Main" architecture
class AutoEncoder(nn.Module):
    def __init__(self, ngpu=ngpu,
                 LR_slope=LR_slope,
                 batch_norm_eps=batch_norm_eps,
                 batch_norm_momentum=batch_norm_momentum):
        super(AutoEncoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is an image 128x128, going into a convolution
            nn.Conv2d(img_channels, 32, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(32, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size 32x64x64
            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(64, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size 64x32x32
            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(128, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size 128x16x16
            nn.Conv2d(128, 128, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(128, eps=batch_norm_eps, momentum=batch_norm_momentum),
            #  size 128x8x8
            nn.Flatten(),
            nn.Linear(8192, middle_vector_size, bias=True),
            nn.Tanh()
            # size 256x1
        )
        self.decoder = nn.Sequential(
            nn.Linear(middle_vector_size, 8192, bias=True),
            Reshape(batch_size, 128, 8, 8),
            nn.LeakyReLU(LR_slope, True),
            #
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(128, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(64, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(32, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(16, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size
            nn.ConvTranspose2d(16, 3, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True)
            # size
        )

    def forward(self, input):
        #print(input.shape)
        x = self.encoder(input)
        #print(x.shape)
        y = self.decoder(x)
        #print(y.shape)
        return y

# AutoEncoder Code - secondary architecture
"""
class AutoEncoder(nn.Module):
    def __init__(self, ngpu=ngpu,
                 LR_slope=LR_slope,
                 batch_norm_eps=batch_norm_eps,
                 batch_norm_momentum=batch_norm_momentum):
        super(AutoEncoder, self).__init__()
        in_channels = img_channels
        dec_channels = 32
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is an image 128x128, going into a convolution
            nn.Conv2d(in_channels, dec_channels, 5, stride=2, padding=2, bias=True, groups=1),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size dec_channelsx64x64
            nn.Conv2d(dec_channels, dec_channels*2, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*2, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*2)x32x32
            nn.Conv2d(dec_channels*2, dec_channels*4, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*4, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*4)x16x16
            nn.Conv2d(dec_channels*4, dec_channels*8, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*8, eps=batch_norm_eps, momentum=batch_norm_momentum),
            #  size (dec_channels*8)x8x8
            nn.Conv2d(dec_channels*8, dec_channels*16, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*16, eps=batch_norm_eps, momentum=batch_norm_momentum),
            #  size (dec_channels*16)x4x4
            nn.Conv2d(dec_channels*16, dec_channels*32, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*32, eps=batch_norm_eps, momentum=batch_norm_momentum),
            #  size (dec_channels*32)x2x2

            nn.Flatten(),
            nn.Linear((dec_channels*32)*2*2, middle_vector_size, bias=True),
            # size 256x1
        )
        self.decoder = nn.Sequential(
            nn.Linear(middle_vector_size, (dec_channels*32)*2*2, bias=True),
            Reshape(batch_size, (dec_channels*32), 2, 2),
            nn.LeakyReLU(LR_slope, True),
            # size (dec_channels*32)x2x2
            nn.ConvTranspose2d(dec_channels*32, dec_channels*16, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*16, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*16)x4x4
            nn.ConvTranspose2d(dec_channels*16, dec_channels*8, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*8, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*8)x8x8
            nn.ConvTranspose2d(dec_channels*8, dec_channels*4, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*4, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*4)x16x16
            nn.ConvTranspose2d(dec_channels*4, dec_channels*2, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels*2, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size (dec_channels*2)x32x32
            nn.ConvTranspose2d(dec_channels*2, dec_channels, 4, 2, 1, bias=True),
            nn.LeakyReLU(LR_slope, True),
            nn.BatchNorm2d(dec_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
            # size dec_channelsx64x64
            nn.ConvTranspose2d(dec_channels, in_channels, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # size 3x128x128
        )

    def forward(self, input):
        #print(input.shape)
        x = self.encoder(input)
        #print(x.shape)
        y = self.decoder(x)
        #print(y.shape)
        return y
"""
#########################################

def init_AutoEncoder(isLoad = False,
                     isTrain = True,
                     ngpu=ngpu,
                     lr=lr,
                     LR_slope=LR_slope,
                     batch_norm_eps=batch_norm_eps,
                     batch_norm_momentum=batch_norm_momentum,
                     adam_beta1=adam_beta1,
                     adam_beta2=adam_beta2,
                     adam_eps=adam_eps,
                     adam_weight_decay=adam_weight_decay):
    # Create the AutoEncoder
    if isLoad:
        netAE = load_weights(isTrain = True)
    else:
        netAE = AutoEncoder(ngpu, LR_slope, batch_norm_eps, batch_norm_momentum)
        #netAE.apply(weights_init)

    netAE = netAE.to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netAE = nn.DataParallel(netAE, list(range(ngpu)))
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    ######netAE.apply(weights_init)##############
    
    # Print the model
    #print(netAE)
    return netAE

#########################################
#			 Run Training 				#
#########################################

def run_Train(isLoad = False,
                isParamSerach = False,
                num_epochs=num_epochs,
                ngpu=ngpu,
                lr=lr,
                LR_slope=LR_slope,
                batch_norm_eps=batch_norm_eps,
                batch_norm_momentum=batch_norm_momentum,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                adam_eps=adam_eps,
                adam_weight_decay=adam_weight_decay):
    
    train_loader, _ = init_Data()
    
    netAE = init_AutoEncoder(isLoad, isTrain = True)
    
    # Construct loss function and Optimizer
    criterion = nn.MSELoss() #nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(netAE.parameters(),
                           lr = lr,
                           betas = (adam_beta1, adam_beta1),
                           eps = adam_eps,
                           weight_decay = adam_weight_decay)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        train_loss = 0.0
        
        # For each batch in the train_loader
        for batch_idx, images in enumerate(train_loader, 0):
            # get images and move them to the device (cuda if availabe)
            images_device = images.to(device)
            # clear gradients
            optimizer.zero_grad()
            # forward pass for all images in batch
            outputs = netAE(images_device)
            # calculate the loss
            loss = criterion(outputs, images_device)
            # backword pass
            loss.backward()
            # update parameters
            optimizer.step()
            
            # update running training loss
            train_loss += loss.item()*images.size(0)

        train_loss = train_loss/train_size
        """print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            epoch,
            train_loss,
            TestLoss(netAE)
            ))"""
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss,
            ))
        if not isParamSerach:
            save_weights(netAE)
    print('Finished Training')

    return train_loss

#########################################
#			 Save and Load 				#
#########################################
# Save
def save_weights(model):
    torch.save(model.state_dict(), "weightsAE_try.pt")
    print("Model saved.")

# Load
def load_weights(isTrain):
    model = AutoEncoder(ngpu)
    model.load_state_dict(torch.load("weightsAE_try_load.pt",  map_location=device))
    
    if isTrain:
        model.train() # for train
    else:
        model.eval()  # for test 
    
    return model

#########################################
#			      Test     				#
#########################################

def run_Test(isLoad=True):
    _, test_loader = init_Data()

    netAE = init_AutoEncoder(isLoad, isTrain=False)

    # Construct loss function and Optimizer
    criterion = nn.MSELoss()  # nn.MSELoss(reduction='sum')

    test_loss = 0.0
    # For each batch in the test_loader
    for batch_idx, images in enumerate(test_loader, 0):
        # get images and move them to the device (cuda if availabe)
        images_device = images.to(device)
        # forward pass for all images in batch
        outputs = netAE(images_device)
        # calculate the loss
        loss = criterion(outputs, images_device)
        # update running training loss
        test_loss += loss.item() * images.size(0)

        images_device = images_device.detach()
        outputs = outputs.detach()

        ## Save plot ##
        # Input plot
        plt.figure(figsize=(8, 8), dpi=200)
        plt.axis("off")
        plt.title("Test input images")
        #plt.imshow(np.transpose(vutils.make_grid(images_device[:104], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.imshow(np.transpose(vutils.make_grid(images_device[:6], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig("plt_{}_input.jpg".format(batch_idx), quality=100, format='jpg', optimize=True)
        plt.close('all')

        # Output plot
        plt.figure(figsize=(8, 8), dpi=200)
        plt.axis("off")
        plt.title("Test output images")
        #plt.imshow(np.transpose(vutils.make_grid(outputs[:104], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.imshow(np.transpose(vutils.make_grid(outputs[:6], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig("plt_{}_output.jpg".format(batch_idx), quality=100, format='jpg', optimize=True)
        plt.close('all')

        if batch_idx % 10 == 0:
            print("Saved {} batches".format(batch_idx+1))

    test_loss = test_loss / test_size
    print("Test Loss: {}.".format(test_loss))

def TestLoss(netAE):
    _, test_loader = init_Data()

    # Construct loss function and Optimizer
    criterion = nn.MSELoss()  # nn.MSELoss(reduction='sum')

    test_loss = 0.0
    # For each batch in the test_loader
    for batch_idx, images in enumerate(test_loader, 0):
        # get images and move them to the device (cuda if availabe)
        images_device = images.to(device)
        # forward pass for all images in batch
        outputs = netAE(images_device)
        # calculate the loss
        loss = criterion(outputs, images_device)
        # update running training loss
        test_loss += loss.item() * images.size(0)

        images_device = images_device.detach()
        outputs = outputs.detach()

    test_loss = test_loss / test_size
    return test_loss

#########################################
#		Hyperparameters Search 			#
#########################################
import pandas as pd
def saveLosses(losses, tryIndex=0):
    df_ans = pd.DataFrame.from_dict(losses, orient='index')
    df_ans.to_csv('losses_{}.csv'.format(tryIndex))

def param_search(num_epochs, tryIndex,
                lr_=[lr],
                LR_slope_=[LR_slope],
                batch_norm_eps_=[batch_norm_eps],
                batch_norm_momentum_=[batch_norm_momentum],
                adam_beta1_=[adam_beta1],
                adam_beta2_=[adam_beta2],
                adam_eps_=[adam_eps],
                adam_weight_decay_=[adam_weight_decay]):

    losses = {}
    for lr in lr_:
        for LR_slope in LR_slope_:
            for batch_norm_eps in batch_norm_eps_:
                for batch_norm_momentum in batch_norm_momentum_:
                    for adam_beta1 in adam_beta1_:
                        for adam_beta2 in adam_beta2_:
                            for adam_eps in adam_eps_:
                                for adam_weight_decay in adam_weight_decay_:
                                    losses[(lr, LR_slope, batch_norm_eps, batch_norm_momentum, adam_beta1,
                                            adam_beta2, adam_eps, adam_weight_decay)] = run_Train(num_epochs=num_epochs,
                                                lr=lr,
                                                LR_slope=LR_slope,
                                                batch_norm_eps=batch_norm_eps,
                                                batch_norm_momentum=batch_norm_momentum,
                                                adam_beta1=adam_beta1,
                                                adam_beta2=adam_beta2,
                                                adam_eps=adam_eps,
                                                adam_weight_decay=adam_weight_decay,
                                                isLoad=True,
                                                isParamSerach=True)

    saveLosses(losses, tryIndex)
    return min(losses, key=losses.get)






#########################################
#			     MAIN   				#
#########################################


isLoad = False

def main():
    run_Train(isLoad, num_epochs=200,
                lr=0.00305,
                LR_slope=0.00056,
                batch_norm_eps=0.000003,
                batch_norm_momentum=0.3,
                adam_beta1=adam_beta1,
                adam_beta2=0.9,
                adam_eps=adam_eps,
                adam_weight_decay=adam_weight_decay)
    #run_Test()
    #print(param_search(7, tryIndex=2, lr_=[0.003, 0.004], batch_norm_momentum_=[0.01,0.05,0.2], adam_beta1_=[0.9,0.8], adam_beta2_=[0.999,0.9]))

if __name__ == "__main__":
    main()

