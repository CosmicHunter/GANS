from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable


# Hyperparameters
batch_size = 64
img_size = 64


"""
Transforms are common image transformations. They can be chained together using Compose
torchvision.transforms.Compose(transforms) :- Composes several transforms together.
Parameters := transforms (list of Transform objects) – list of transforms to compose.



torchvision.transforms.Normalize(mean, std, inplace=False)
Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) 
and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the
input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel

"""
transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])


# Loading the dataset
dataset = torchvision.datasets.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.

"""
Dataset – It is mandatory for a DataLoader class to be constructed with a dataset first. PyTorch Dataloaders support two kinds of datasets:
    Map-style datasets – These datasets map keys to data samples. Each item is retrieved by a __get_item__() method implementation.
    Iterable-style datasets – These datasets implement the __iter__() protocol. 
    Such datasets retrieve data in a stream sequence rather than doing random reads as in the case of map datasets

num_workers – Number of sub-processes needed for loading the data.
"""
# We use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2) 

# Defining the weights_init function that takes the input a neural network nn and that will initialize all its weights.

def weights_init(neuralnet):
    classname = neuralnet.__class__.__name__
    if classname.find('Conv') != -1:
        neuralnet.weight.data.normal_(0.0, 0.02)  # Fills the weight Tensor with values drawn from the normal distribution N(mean =0.0,std2 = 0.02)
    elif classname.find('BatchNorm') != -1:
        neuralnet.weight.data.normal_(1.0, 0.02) 
        neuralnet.bias.data.fill_(0)  # Fill the bias tensor with zeros
        

# Creating the Generator
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator , self).__init__()  # Calling the constructor of the parent class i.e nn.module
        
        """
        torch.nn.Sequential(*args)
        A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        Alternatively, an ordered dict of modules can also be passed in.
        
        
        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        Applies a 2D transposed convolution operator over an input image composed of several input planes . 
        out_channels is the number of feature_maps in the output.
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        
        
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) 
        
        """
        self.main = nn.Sequential(
            # First we will add an inverse convolutional module that will generate image
            nn.ConvTranspose2d(in_channels=100, out_channels = 512, kernel_size = 4,stride = 1 , padding = 0 ,bias = False),
            # Now we will  normalise all the features along the 512 feature maps
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4,stride = 2,padding = 1 ,bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4,stride = 2,padding = 1 ,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4,stride = 2,padding = 1 ,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 4,stride = 2,padding = 1 ,bias = False), # Since the output generated image will have 3 channels
            # so out_channels=3
            nn.Tanh()
            )
    
    # Forward_prop function will have input a random vector of size 100 that will simply represent noise    
    def forward(self,input):
        output = self.main(input)
        return output


# Creating the Generator object
gen = Generator() 
#initializing the weights
gen.apply(weights_init)  # apply just applies the function passed to our generator object

# Defining the Discriminator
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator , self).__init__()
        self.main = nn.Sequential(
            # This takes input an image so first layer is simply going to be a convolution
            # inChannels = 3 as generator generates 3 channel image as output
            nn.Conv2d(in_channels = 3, out_channels = 64 , kernel_size = 4, stride = 2 , padding=1, bias = False),
            nn.LeakyReLU(0.2,inplace = True) ,  # The argument is the negative slope
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4,stride  = 2,padding =1,bias = False ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4,stride = 2,padding =1 ,bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4,stride = 2,padding =1 ,bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace = True),
            
            #Output Layer
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size =4,stride = 1,padding = 0,bias=  False),
            nn.Sigmoid()
            )
        
    def forward(self,input):
        output = self.main(input)
        # We know that at the output of a convolutional neural network we need to flatten the convolutions into a single dimension
        # The view function is meant to reshape the tensor.
        """
        If there is any situation that you don't know how many rows you want but are sure of the number of columns, 
        then you can specify this with a -1. (Note that you can extend this to tensors with more dimensions. Only one 
        of the axis value can be -1). This is a way of telling the library: "give me a tensor that has these many columns 
        and you compute the appropriate number of rows that is necessary to make this happen. We basically know one dimesnion and that is 
        the batch size.
        """
        return output.view(-1)

# Creating the Discriminator object

discriminator = Discriminator()
discriminator.apply(weights_init)

# Training the DCGANS

# We will use the BCE loss that is the binary cross entropy loss
"""
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
Creates a criterion that measures the Binary Cross Entropy between the target and the output:
"""

eval_criteria = nn.BCELoss()
optimizer_d = optim.Adam(discriminator.parameters(),lr = 0.0002,betas = (0.5,0.999))
# betas (Tuple[python:float, python:float], optional) – coefficients used for computing running averages of gradient and its square
 
optimizer_gen = optim.Adam(gen.parameters(),lr = 0.0002,betas = (0.5,0.999))

epochs = 25
for epoch in range(epochs):
    # go through each minibatch
    # We will get minibatch from dataloader object
    # 2nd parameter of enumerate is 
    # Start: the index value from which the counter is 
    #          to be started, by default it is 0 
    for i,data in enumerate(dataloader,0): # 0 tells it that where the index of the loop will start i.e i will start from 0
         #  1st Step : Updating the wts of the network of discriminator
         discriminator.zero_grad()  # Initialize the gradients w.r.t wts to zero
        
         # Training the discriminator with a real image of the dataset
         real_img , temp =  data
        # print(temp) 
         # Pytorch network only accepts Torch type Variable
         input_imgs = Variable(real_img)
         
         """ for each of the real images of the dataset we need to set the target = 1  for the discriminator, WE will have a one for each of 
         the input image of the minibatch 
         input_imgs contains minibatch number of images
         input_imgs.size()[0] gives the size of the minibatch
         """
         target = torch.ones(input_imgs.size()[0])
         target = Variable(target)
         output = discriminator(input_imgs)
         
         # Getting the error
         error_real_imgs = eval_criteria(output,target)  # this is the error for discriminator on how it classifies real images 
         
         
         # Training the discriminator to recognize the generated images
         # Generator will take input of 100 random flattened noise vector and generate an image its just the opposite of the simple CNN
         noise_vec = torch.randn(input_imgs.size()[0],100,1,1) # 100 , 1, 1 in this 1,1 are just fake dimensions , its like 100 feature maps of 1x1
         noise_vec = Variable(noise_vec)
         # pass this to generator network
         generated_imgs = gen(noise_vec)
         target = torch.zeros(input_imgs.size()[0])
         target = Variable(target)
         # feeding these images to discriminator
         output = discriminator(generated_imgs.detach())
         """
         We can save some memory . generated_imgs is a torch variable. Hence it contains the tensor of the predicitions and the gradients
         But we will not use these gradients after backpropogating the error inside the neural net. So we can detach the gradient of this 
         torch variable . Since here we do not want gradients w.r.t to generator network , since here we are training discriminator
         
         """
         # getting the error for discriminator on how it classifies generated images that are not real
         error_gen_imgs  = eval_criteria(output, target) # We compute the loss between the prediction (output) and the target (equal to 0).
         

         # Backpropagating the total error
         error_discriminator = error_real_imgs + error_gen_imgs
         error_discriminator.backward() # Backprop
         optimizer_d.step()  # Updates the weights
         
         
         # 2nd Step: Updating the weights of the neural network of the generator
         gen.zero_grad()
         # We need to train the generator in order to produce images that are like real so the target should be one
         target = torch.ones(input_imgs.size()[0])
         target = Variable(target)
         
         # Getting the output of the discriminator when the input is fake images
         output = discriminator(generated_imgs)
         # Here we need to update the weights of the generator so no need to detac the gradient
         
         # Getting the error
         error_generator = eval_criteria(output,target)
         error_generator.backward()
         optimizer_gen.step()
         
         print('[%d/%d][%d/%d] LOSS_D : %0.4f LOSS_G : %0.4f' % (epoch,epochs,i,len(dataloader), error_discriminator.data[0],error_generator.data[0]))
         if i%100 ==0:
             torchvision.utils.save_image(real_img, '%s/real_samples.png' % './results',normalize = True)
             # normalize (bool, optional) – If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False.
             gen_imgs = gen(noise_vec)
             torchvision.utils.save_image(gen_imgs.data, '%s/generated_samples_epoch_%03d.png' % ('./results',epoch),normalize = True)
     
             
torch.save(discriminator.state_dict(),'.\Module 3 - GANs')
torch.save(gen.state_dict(),'.\Module 3 - GANs')
