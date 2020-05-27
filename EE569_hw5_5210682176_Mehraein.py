#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[2]:


trainset = torchvision.datasets.CIFAR10(root='./datasets',
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())


# In[3]:


trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=64, #default is 1  #64?
                                         shuffle=True,
                                         num_workers=2) #use 2 subprocesses for data loading


# In[4]:


testset = torchvision.datasets.CIFAR10(root='./datasets',
                                       train=False, #this will download test set
                                       download=True,
                                       transform=transforms.ToTensor())


# In[5]:


testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1000, #default is 1 #1000?
                                         shuffle=False, #false for testset
                                         num_workers=2)


# In[6]:


labels = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[7]:


import matplotlib.pyplot as plt
import numpy as np


# In[8]:


images_batch, labels_batch = iter(trainloader).next()


# In[9]:


images_batch.shape


# In[10]:


#we need to reshape data
#use a torchvision utility to make a grid of images in this batch
img = torchvision.utils.make_grid(images_batch)
#it places images side by side


# In[11]:


img.shape 
#8 images placed side_by_side, 2 pixel padding between images and the edges of the grid
#number of channels, height, width


# In[12]:


#we need to make the channel the last one for matplotlib, matplotlib requires the channels to be the 3rd dimension
np.transpose(img, (1,2,0)).shape
#height, width, number of channels


# In[13]:


plt.imshow(np.transpose(img, (1,2,0)))
plt.axis('off')
plt.show()


# In[14]:


import torch.nn as nn
import torch.nn.functional as F #contains logSoftmax function
import torch.optim as optim


# In[15]:


in_size = 3 #input is the 3 channels of the image
hid1_size=64 #6 for LeNet-5   #16 CNN  #HW32
hid2_size=128 #16 for LeNet-5  #32 CNN  #HW64
#out_size1=400
out_size = len(labels)

k_conv_size =5 #5*5 convolution kernel


# In[16]:


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(  #groups of sequential layers
            nn.Conv2d(in_size, hid1_size, k_conv_size , stride =1, padding =0),
            #nn.Conv2d(in_size, hid1_size, k_conv_size), 
            nn.BatchNorm2d(hid1_size), #normalize the outputs of this layer for one batch so they have 0 mean and unit variance
            #only need the number of channels, the batch size, height and width of the input can be inferred
            nn.ReLU(), #pass the output to the activation function
          
            nn.AvgPool2d(kernel_size=2)) #max pool layer with a 2*2 kernel
        
        self.layer2 = nn.Sequential(  
            nn.Conv2d(hid1_size, hid2_size, k_conv_size, stride =1, padding =0), 
            nn.BatchNorm2d(hid2_size), 
            nn.ReLU(), 
            #nn.Dropout(0.5),
            #nn.Dropout2d(0.5),
            nn.AvgPool2d(kernel_size=2)) 
        
        #self.layer3=nn.Sequential(
        #    nn.Linear(hid2_size * k_conv_size * k_conv_size, out_size1),
        #    nn.ReLU(),
        #    nn.Dropout(0.5),
            #nn.Dropout2d(0.5),
         #   nn.Linear (out_size1, out_size2))
        
        
        #number of features to represent one image: 32*5*5
        #the size of each image after passing through 2 convolutional and 2 pooling layers is 5*5
        self.fc1 = nn.Linear(hid2_size * k_conv_size * k_conv_size, 200) #120 for LeNet-5 #200HW
        self.fc2 = nn.Linear(200,100) #(120,84) #(200,100)HW
        self.fc3 = nn.Linear(100, out_size) #output_size= classification categories  #84 #100 HW
        
        
    def forward(self, x): #x are input
        out = self.layer1(x)
        out = self.layer2(out)
        #take next out for in_size = 1
        out = out.reshape(out.size(0), -1) #reshape the output so each image is represented as a 1D vector to feed into the linear layer
        out = torch.sigmoid(self.fc1(out))   #torch.sigmoid #F.relu HW
        out = torch.sigmoid(self.fc2(out)) #then output of first layer passes to the second layer
                                      #torch.sigmoid #F.relu HW
        out = self.fc3 (out) # no activation function needed for this layerr: Last layer is a linear layer with no activation 
        #out = self.layer3 (out)
        #mathematically equivalent to log(softmax(x))
        #performing those operations separately is numerically unstable
        # this single function is a better alternative
        return F.log_softmax(out, dim=-1) #dimension along which softmax will be computed, 
                                        #here we allow the function to infer the right dimension
        
        
        


# In[17]:


#ready to instantiate and train out convolution network
model = ConvNet()


# In[18]:


learning_rate = 0.001
#momentum = 0.9
criterion = nn.CrossEntropyLoss() #measure of distance between prob distribution 
#criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=learning_rate)
#optimizer = optim.SGD(model.parameters(),
                      #lr=learning_rate,
                       # momentum = momentum)


# In[19]:


#train_accuracy.clear() 
#loss_values.clear()


# In[20]:


total_step = len(trainloader)
num_epochs = 40 
loss_values = list()
training_accuracy = list()
loss_per_epoch = list()
loss_per_epoch = list()
training_accuracy_per_epoch = list()
loss_save = list()
training_acc_save = list()


# In[21]:


import time


# In[22]:



start_training=time.time()


# In[ ]:





# In[24]:



for epoch in range(num_epochs):
    
   
   
    total_train=0
    correct_train = 0
    
    for i, (images, labels) in enumerate(trainloader,0): #it gives us one batch of images at a time
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
       
        
        _,pred_train=torch.max(outputs.data, 1)
        

    
        total_train += labels.nelement()
        correct_train += pred_train.eq(labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        
        loss_values.append(loss.item())
        training_accuracy.append(train_accuracy)
        
        loss_save.append(loss.item())
        training_acc_save.append(train_accuracy)
        
        if (i+1) % 200 == 0:
            
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} '
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item() ))
         
    
    loss_per_epoch.append(sum(loss_values)/len(loss_values))
    loss_values.clear()
    #print ('loss_per_epoch', loss_per_epoch)
    training_accuracy_per_epoch.append(sum(training_accuracy)/len(training_accuracy))
    training_accuracy.clear()
    #print ('training accuracy per epoch',training_accuracy_per_epoch )
                


# In[ ]:


stop_training=time.time()


# In[ ]:


r=len(loss_save)
x = (range (1,r+1))

plt.plot(x,loss_save)
plt.xlabel('iteration')
plt.ylabel('train_Loss')
plt.show()


# In[ ]:



print ('loss_per_epoch', loss_per_epoch)
print ('training accuracy per epoch',training_accuracy_per_epoch)


# In[ ]:


x = (range (1,num_epochs+1))

plt.plot(x,loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('train_Loss')
#plt.ylim (2.3, 2.4)
plt.show()


# In[ ]:


#loss_values.clear()


# In[ ]:


r= len (training_acc_save)
y = (range (1,r+1))
plt.plot(y, training_acc_save)
plt.xlabel('iteration')
plt.ylabel('train_accuracy')


# In[ ]:


#y = np.linspace(0, 1/total_step, num_epochs)
#y = (range (1,len(training_accuracy_per_epoch)))
y = (range (1,num_epochs+1))
plt.plot(y, training_accuracy_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('train_accuracy')
plt.xlim(0,len(training_accuracy_per_epoch))


# In[ ]:


#training_accuracy.clear()  


# In[ ]:


testing_accuracy = list()


# In[ ]:


start_test=time.time()


# In[ ]:


#set our model to evaluation or prediction mode
model.eval()
#we don't want grad to be calculated during prediction
with torch.no_grad():
    correct = 0
    total = 0
    
    
    total_test=0
    correct_test = 0
    
    for images, labels in testloader:
        outputs = model(images)
        
        
        
        _,predicted = torch.max(outputs.data, 1)
        #total number of test instances
        total += labels.size(0)
        #calculate total number of correct predictions
        
        correct += (predicted == labels).sum().item()
        
        
        total_test += labels.nelement()
        correct_test += predicted.eq(labels).sum().item()
        test_accuracy = 100 * correct_test / total_test
        
        
        testing_accuracy.append(test_accuracy)
        
        
        
    print ('Accuracy of the model on the 10000 test images: {}%'
          .format(100 *correct /total))
        


# In[ ]:


stop_test=time.time()


# In[ ]:


y = range (0,len(testing_accuracy))
#y=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
#y=[1,5,10,15,20,25,30,35,40,45]
plt.plot(y, testing_accuracy)
plt.xlabel('Batches')
plt.ylabel('test_accuracy')
#plt.ylim(70,80)


# In[ ]:


#loss_values_test = list()


# In[ ]:


#testing_accuracy = list()


# In[ ]:


#testing_accuracy.clear()


# In[ ]:


print(testing_accuracy)


# In[ ]:


#inference time


# In[ ]:


inference_start=time.time()


# In[ ]:


sample_img, _ = testset[24]
sample_img.shape


# In[ ]:


sample_img = np.transpose(sample_img, (1, 2, 0))
sample_img.shape


# In[ ]:


m, M = sample_img.min(), sample_img.max()
sample_img = (1/(abs(m) * M)) * sample_img + 0.5 


# In[ ]:


plt.imshow(sample_img)


# In[ ]:


test_img, test_label = testset[23]
test_img = test_img.reshape(-1, 3, 32, 32)


# In[ ]:





# In[ ]:


out_predict = model(test_img)
_,predicted = torch.max(out_predict.data, 1)


# In[ ]:


inference_stop=time.time()


# In[ ]:


print("Actual Label : ", test_label)
print("Predicted Label : ", predicted.item())


# In[ ]:


total_training_time= stop_training-start_training
total_test_time= stop_test-start_test
total_inference_time= inference_stop-inference_start

print('total_training_time in S:', total_training_time)
print('total_test_time in S:',total_test_time)
print('total_inference_time in S:', total_inference_time)
print('total_training_time in min:', total_training_time/60)
print('total_test_time in min:',total_test_time/60)
print('total_inference_time in min:', total_inference_time/60)


# In[ ]:




