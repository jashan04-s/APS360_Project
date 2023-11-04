import torch.nn as nn
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import cv2


class LABImageDataset(Dataset):
    def __init__(self, L_data, ab_data):
        self.L_data = L_data
        self.ab_data = ab_data

    def __len__(self):
        return len(self.L_data)

    def __getitem__(self, idx):
        L = self.L_data[idx]
        ab = self.ab_data[idx]
        return L, ab

def rgb_image(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.zeros(shape)
    img[:,:,0] = l[:,:,0]
    img[:,:,1:]= ab
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def get_data_loader(batch_size = 64):
   

    npArray_ab = np.load("./ab/ab/ab1.npy")
    npArray_l = np.load("l/gray_scale.npy")

   
    npArray_ab = npArray_ab[:1000]
    # npArray_ab = npArray_ab.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    npArray_l = npArray_l[:1000]
    # new_shape = (1000, 224, 224,1)
    # npArray_l = npArray_l.reshape(new_shape)
    
    lab_dataset = LABImageDataset(npArray_l, npArray_ab)
    print(npArray_ab.shape)
    print(npArray_l.shape)
    relevant_indices = np.arange(1,999)
    
    np.random.seed(1000)

    np.random.shuffle(relevant_indices)

    split = int(len(relevant_indices) * 0.8)

    split_in_val_test = split + int(len(relevant_indices) * 0.1)

    train_indices = relevant_indices[:split]
    val_indices = relevant_indices[split: split_in_val_test]
    test_indices = relevant_indices[split_in_val_test:]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(val_indices)
   
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = train_sampler)
    val_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = val_sampler)
    test_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = test_sampler)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_data_loader(1)


def display(img):
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(img)
    plt.show()

def display_colored(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


for images, labels in train_loader:
   display(images[:][0])
   print(images[:][0].shape)
   break

""" BUILDING A MODEL """

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(3, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 2, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        return x
    
model = Net()    
# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1000
for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    save = 0
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, labels = data
        save = images
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        images = images.float()
        labels = labels.float()
        outputs = model(images)
        # outputs_to_print = (outputs.swapaxes(0,1).swapaxes(1, 2)).detach().numpy()
        # input_to_print = (images.swapaxes(0,1).swapaxes(1, 2)).detach().numpy()
        # display(rgb_image(input_to_print, outputs_to_print))
        # print(outputs.shape)
        outputs = outputs.float()
        labels = labels.permute(3, 1, 2, 0)
        labels = labels.squeeze(dim=3)
        
        # calculate the loss
        loss = criterion(outputs, labels)
        loss = loss.float()
        outputs = outputs.float()

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    outputs = model(images)
    outputs_to_print = (outputs.swapaxes(0,1).swapaxes(1, 2)).detach().numpy()
    input_to_print = (images.swapaxes(0,1).swapaxes(1, 2)).detach().numpy()
    
    display(rgb_image(input_to_print, outputs_to_print))
    input_to_print = np.squeeze(input_to_print,axis=-1)
    display(input_to_print)



