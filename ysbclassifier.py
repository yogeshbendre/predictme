import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#import matplotlib.pyplot as plt
import numpy as np
# functions to show an image



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
use_cuda = torch.cuda.is_available()
mydevice="cpu"
if use_cuda:
  net = net.cuda()
  print("GPU Device found, congrats, you will have fast training...!!!")
  mydevice = "gpu"
else:
  print("No GPU found, training may be slow, hang on...!!!")
  mydevice = "cpu"

def print_info(stmt):
  print(mydevice+": "+stmt)

def print_gpu_stats():
  if use_cuda:
    t = torch.cuda.get_device_properties(0).total_memory/1048576
    r = torch.cuda.memory_reserved(0)/1048576
    a = torch.cuda.memory_allocated(0)/1048576
    f = r-a
    print_info("Total GPU Memory: "+str(t)+" MB")
    print_info("Reseverd GPU Memory: "+str(r)+" MB")
    print_info("Allocated GPU Memory: "+str(a)+" MB")
    print_info("Free GPU Memory: "+str(f)+" MB")
  else:
    print_info(" No GPU Device")


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

import time

t0_total = time.time()
total_epochs=1
for epoch in range(total_epochs):  # loop over the dataset multiple times
    t0 = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if use_cuda:
          
          inputs, labels = data[0].to('cuda'), data[1].to('cuda')
        else:
          
          inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print_gpu_stats()
        # print statistics
        #print("Batch: "+str(i))
        running_loss += loss.item()
        if (i+1) % 100 == 0:    
            print_info('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print_info('Epoch {} took {} seconds'.format((epoch+1), (time.time() - t0)))
    print_gpu_stats()	

print_info('Finished Training')
totalimages = len(trainloader.dataset)
print_info('Training Stats')
print_info('Number of images: '+str(totalimages))
print_info('Number of images per batch: '+str(batch_size))
print_info('Number of batches: '+str(totalimages/batch_size))
print_info('Number of epochs: '+ str(total_epochs))
print_info('Complete training took {} seconds'.format((time.time() - t0_total)))
print("Training completed with device: "+mydevice)






