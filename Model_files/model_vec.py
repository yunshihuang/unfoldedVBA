"""
unfoldedVBA model classes
"""

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from Model_files.blindVBA_vec import cardan
from Model_files.blindVBA_vec_RGB import cardan_RGB
from torch.nn.modules.loss import _Loss
from math import ceil
import os   
from Model_files.modules import *
from Model_files.tools import *
from Model_files.UNet import *
import PyTorch_ssim  
    
class SSIM_loss(_Loss):
    """
    Defines the SSIM training loss.
    """
    def __init__(self): 
        super(SSIM_loss, self).__init__()
        self.ssim = PyTorch_ssim.SSIM()
 
    def forward(self, input, target):
        """
        Computes the training loss.
        Parameters
        ----------
      	    input  (torch.FloatTensor): restored images
            target (torch.FloatTensor): ground-truth images
        Returns
        -------
       	    (torch.FloatTensor): SSIM loss, size 1 
        """
        return -self.ssim(input,target)
        
class diff_h(_Loss):
    """
    Defines the MSE training loss.
    """
    def __init__(self): 
        super(diff_h, self).__init__()
 
    def forward(self, input, target):
        """
        Computes the training loss.
        Parameters
        ----------
            input  (torch.FloatTensor): restored blur
            target (torch.DoubleTensor): true blur
        Returns
        -------
            (torch.FloatTensor): MSE loss, size 1 
        """
        target = target
        if len(target.shape)==2:
            output = (torch.norm(input[:,:]-target[:,:],'fro')**2)
        else:
            batch = target.shape[0]
            sizeh = target.shape[1]
            loss = nn.MSELoss()
            output = loss(input, target)*(sizeh**2)
        return output
    
class Cnn_bar(nn.Module):
    """
    Predicts the parameter xi.
    Attributes
    ----------
        conv2, conv3 (torch.nn.Conv2d): 2-D convolution layer
        lin          (torch.nn.Linear): fully connected layer
        avg       (torch.nn.AVgPool2d): average layer
        soft       (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self):
        super(Cnn_bar, self).__init__()
        self.conv3  = nn.Conv2d(1, 1, 3,padding=1)
        self.conv2  = nn.Conv2d(1, 1, 2)
        self.lin1   = nn.Linear(64, 16)
        self.lin3   = nn.Linear(16, 1)
        self.avg    = nn.AvgPool2d(2, 2)
        self.soft   = nn.Softplus()


    def forward(self, x):
        """
        Computes the parameter xi.
        Parameters
        ----------
        x (torch.FloatTensor): images
        Returns
        -------
        xi (torch.FloatTensor)
        """
        x = self.soft(self.conv2(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.soft(self.lin1(x))
        x = self.soft(self.lin3(x))
        x = x.view(-1)*1e8
        return x

    
# one block of unfolded VBA    
class Block(torch.nn.Module):
    """
    One layer in unfoldedVBA.
    Attributes
    ----------
        cnn_bar        (Cnn_bar): computes the parameter xi
        soft (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self):
        super(Block, self).__init__()
        self.soft         = nn.Softplus()
        self.cnn_bar      = Cnn_bar()

    def forward(self,y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype):
        batch = h0_vec.shape[0]
        sizeh = h0_vec.shape[1]
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(dtype)
        t = t.type(dtype)
        hhat_vec=T_vec@h_vec+t
        hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
        xi = self.cnn_bar(hhat_vec.unsqueeze(-3).type(dtype))

        print('xi is {}'.format(xi))
        
        
        return cardan.apply(y_vec,mk0_vec,h0_vec,gvar,xi,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype,self.training)
    
    
class myModel(torch.nn.Module):
    """
    unfolded VBA model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of unfolded VBA layers

    """
    def __init__(self,dtype,KmtK0_dict,KmtKn_dict,nL):
        super(myModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.loss_fun = diff_h()
        self.dtype = dtype
        self.KmtKn_dict = KmtKn_dict 
        self.KmtK0_dict = KmtK0_dict
        for i in range(nL):
            self.Layers.append(Block())
        
    def GradFalse(self,block,mode):
        print('GradFalse is used')
        """
        Initializes current layer's parameters with previous layer's parameters, fixes the parameters of the previous layers.
        Parameters
        ----------
      	    block (int): block-1 is the layer to be trained
            mode  (str): 'greedy' if training one layer at a time, 'all_layer' if training all the layers together
        """ 
        if block>=1:
            if mode=='greedy':
                self.Layers[block].load_state_dict(self.Layers[block-1].state_dict())
            elif mode=='all_layers':
                for j in range(0,block):
                    #print('block is',block)
                    self.Layers[j].load_state_dict(self.Layers[j].state_dict()) 
            if mode!='all_layers':       
                for i in range(0,block):
                    self.Layers[i].eval()
                    for p in self.Layers[i].parameters():
                        p.requires_grad = False
                        p.grad          = None 

           
        
    def forward(self,y_vec,x0_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,mode,block=0):
        newMh = int(sio.loadmat('Model_files/useful_tools.mat')['newMh'])
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        batch = y_vec.shape[0]
        sizeh = h0_vec.shape[1]
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(self.dtype)
        t = t.type(self.dtype)
        loss_fun = diff_h() 
        if mode=='first_layer' or mode=='greedy':
            mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[block](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
        elif mode=='all_layers':
            for i in range(0,len(self.Layers)):
                if i ==0:
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                else:
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk_vec,h0_vec,gvar,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)             
        elif mode=='test':
            for i in range(0,len(self.Layers)):
                if i==0:
                    print('This is block ',i)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                    hhat_vec=T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
                    loss = loss_fun(hhat_vec,h0_vec)
                    print('loss for Layer {} is {}'.format(i,loss))
                else:
                    print('This is block ',i)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk_vec,h0_vec,gvar,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                    hhat_vec=T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
                    loss = loss_fun(hhat_vec,h0_vec)
                    print('loss for Layer {} is {}'.format(i,loss))
        if mode=='first_layer' or mode=='greedy':
            return mk_vec.detach(),diagSigma_vec.detach(),newmh_vec,newSigmah_vec.detach(),Gammap_vec.detach(),LAMBDAk_vec.detach() 
        else:
            return mk_vec,diagSigma_vec.detach(),newmh_vec.detach(),newSigmah_vec.detach(),Gammap_vec.detach(),LAMBDAk_vec.detach()
    
class nn_sigma(nn.Module):
    """
    Predicts the noise level sigma.
    Attributes
    ----------
        soft       (torch.nn.Softplus): Softplus activation function
        tanh           (torch.nn.Tanh): Tanh activation function)
    """
    def __init__(self,device):
        super(nn_sigma, self).__init__()
        self.device  = device
        self.soft   = nn.Softplus()
        self.tanh   = nn.Tanh()
        Haar_filt   = np.array(((0.5,-0.5),(-0.5,0.5)))    # Haar
        [Haar_filt] = TensorFilter([Haar_filt])
        self.Haar   = MyConv2d(Haar_filt,'batch',pad_type='circular',padding=1,stride=2)
        if device == '1':
            self.a_k    = nn.Parameter(torch.tensor(0.80).cuda()) #a_k
            self.b_k    = nn.Parameter(torch.tensor(-10.0).cuda()) #b_k
        else:
            self.a_k    = nn.Parameter(torch.tensor(0.80)) #a_k
            self.b_k    = nn.Parameter(torch.tensor(-10.0)) #b_k

    def forward(self, x, dtype):
        """
        Computes the noise level \sigma.
        Parameters
        ----------
      	    x (torch.FloatTensor): images, size n*c*h*w 
            Haar(MyConv2d object): 2-D convolution operator computing Haar wavelet diagonal coefficients
        Returns
        -------
       	    (torch.FloatTensor): the noise level, size n*1*1*1
        """
        # \hat{sigma}(y)
        x = x.float()
        y  = torch.abs(self.Haar(x)).view(x.data.shape[0],-1).data/0.6745
        std_approx = torch.topk(y,ceil(y.shape[1]/2),1)[0][:,-1].type(dtype)
        gvar = (self.soft(self.a_k)*std_approx+self.soft(self.b_k))**2
        beta = 1.0/gvar
        return beta
 
        
# one block of unfolded VBA with the estimation of the noise level   
class Block_RGB(torch.nn.Module):
    def __init__(self,dtype):
        super(Block_RGB, self).__init__()
        self.soft         = nn.Softplus()
        self.cnn_bar      = Cnn_bar()
        if dtype == torch.cuda.FloatTensor:
            device = '1'
            self.nn_sigma = nn_sigma(device)
        else:
            device = '0'
            self.nn_sigma = nn_sigma(device)
    def forward(self,y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype):

        batch = h0_vec.shape[0]
        sizeh = h0_vec.shape[1]
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(dtype)
        t = t.type(dtype)
        hhat_vec=T_vec@h_vec+t
        hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
        xi = self.cnn_bar(hhat_vec.unsqueeze(-3).type(dtype))*0.03
        print('xi is {}'.format(xi))
        image = y_vec.reshape(batch,1,256,256) # the blurred image
        image = image.repeat(1,3,1,1) # repeat it to 3 channels
        beta = self.nn_sigma(image,dtype)

        
        return cardan_RGB.apply(y_vec,mk0_vec,h0_vec,beta,xi,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype,self.training)
    
# the first block of unfolded VBA with the estimation of the noise level   
class Block_RGB0(torch.nn.Module):
    def __init__(self,dtype):
        super(Block_RGB0, self).__init__()
        self.soft         = nn.Softplus()
        self.cnn_bar      = Cnn_bar()
        if dtype == torch.cuda.FloatTensor:
            device = '1'
            self.nn_sigma = nn_sigma(device)
        else:
            device = '0'
            self.nn_sigma = nn_sigma(device)
    def forward(self,y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype):
        batch = h0_vec.shape[0]
        sizeh = h0_vec.shape[1]
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(dtype)
        t = t.type(dtype)
        hhat_vec=T_vec@h_vec+t
        hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
        xi = self.cnn_bar(hhat_vec.unsqueeze(-3).type(dtype))
        print('xi is {}'.format(xi))
        image = y_vec.reshape(batch,1,256,256) # the blurred image
        image = image.repeat(1,3,1,1) # repeat it to 3 channels
        beta = self.nn_sigma(image,dtype)
        
        return cardan_RGB.apply(y_vec,mk0_vec,h0_vec,beta,xi,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype,self.training)
            
    
class myModel_RGB(torch.nn.Module):
    """
    unfolded VBA model with the estimation of noise level
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of unfolded VBA layers

    """
    def __init__(self,dtype,KmtK0_dict,KmtKn_dict,nL):
        super(myModel_RGB,self).__init__()
        self.Layers   = nn.ModuleList()
        self.loss_fun = diff_h()
        self.dtype = dtype
        self.KmtKn_dict = KmtKn_dict 
        self.KmtK0_dict = KmtK0_dict
        for i in range(nL):
            if i == 0:
                self.Layers.append(Block_RGB0(self.dtype))
            else:
                self.Layers.append(Block_RGB(self.dtype))
        
    def GradFalse(self,block,mode):
        print('GradFalse is used')
        """
        Initializes current layer's parameters with previous layer's parameters, fixes the parameters of the previous layers.
        Parameters
        ----------
      	    block (int): block-1 is the layer to be trained
            mode  (str): 'greedy' if training one layer at a time, 'last_layers_lpp' if training the last 10 layers + lpp
        """
        if block>=1:
            if mode=='greedy':
                self.Layers[block].load_state_dict(self.Layers[block-1].state_dict())
            elif mode=='all_layers':
                for j in range(0,block):
                    self.Layers[j].load_state_dict(self.Layers[j].state_dict()) 
            if mode!='all_layers':       
                for i in range(0,block):
                    self.Layers[i].eval()
                    for p in self.Layers[i].parameters():
                        p.requires_grad = False
                        p.grad          = None 

           
        
    def forward(self,y_vec,x0_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec, mode,block=0):
        newMh = int(sio.loadmat('Model_files/useful_tools.mat')['newMh'])
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        sizeh = 9
        batch = y_vec.shape[0]
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(self.dtype)
        t = t.type(self.dtype)
        loss_fun = diff_h() 
        if mode=='first_layer' or mode=='greedy':
            mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[block](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
        elif mode=='all_layers':
            for i in range(0,len(self.Layers)):
                print('This is block ',i)
                if i ==0:
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                else:
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk_vec,h0_vec,gvar,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)             
        elif mode=='test':
            for i in range(0,len(self.Layers)):
                if i==0:
                    print('This is block ',i)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk0_vec,h0_vec,gvar,h_vec,Ch_vec,gamma_vec,lambda_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                    hhat_vec=T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
                    for j in range(batch):
                        loss_ = loss_fun(hhat_vec[j,:,:],h0_vec[j,:,:])
                        print('The RMSE for image {} is {}'.format(j,loss_))
                else:
                    print('This is block ',i)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec = self.Layers[i](y_vec,mk_vec,h0_vec,gvar,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec,self.KmtK0_dict,self.KmtKn_dict,self.dtype)
                    hhat_vec=T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
                    for j in range(batch):
                        loss_ = loss_fun(hhat_vec[j,:,:],h0_vec[j,:,:])
                        print('The RMSE for image {} is {}'.format(j,loss_))
        if mode=='first_layer' or mode=='greedy':
            return mk_vec.detach(),diagSigma_vec.detach(),newmh_vec,newSigmah_vec.detach(),Gammap_vec.detach(),LAMBDAk_vec.detach() 
        else:
            return mk_vec,diagSigma_vec.detach(),newmh_vec.detach(),newSigmah_vec.detach(),Gammap_vec.detach(),LAMBDAk_vec.detach()
    
    
class myLastLayer(torch.nn.Module):
    """
    Post-processing layer.
    Attributes
    ----------
        conv1-conv9  (torch.nn.Conv2d): convolutional layers
        bn1-bn7 (torch.nn.BatchNorm2d): batchnorm layers
        relu           (torch.nn.ReLU): ReLU activation layer
        n_channels: the number of input channels
        n_classes: the number of output channels
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(myLastLayer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.dropout = nn.Dropout(0.5)  # dropout
        
    def forward(self,x):
        """
        Parameters
        ----------
            x (torch.FloatTensor): input images, size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): output of the post-processing layer, size n*c*h*w
        """
        if self.training==True:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x5 = self.dropout(x5)  # dropout
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            x= torch.sigmoid(x)
            return x

        else:
            # we use .detach() to avoid computing and storing the gradients since the model is being tested
            # it allows to save memory
            x1 = self.inc(x.detach())
            x2 = self.down1(x1.detach())
            x3 = self.down2(x2.detach())
            x4 = self.down3(x3.detach())
            x5 = self.down4(x4.detach())
            x5 = self.dropout(x5.detach())  # dropout
            x = self.up1(x5.detach(), x4.detach())
            x = self.up2(x.detach(), x3.detach())
            x = self.up3(x.detach(), x2.detach())
            x = self.up4(x.detach(), x1.detach())
            x = self.outc(x.detach())
            x = torch.sigmoid(x.detach())
            return x

class myLastLayer_RGB(torch.nn.Module):
    """
    Post-processing layer of RGB images.
    Attributes
    ----------
        conv1-conv9  (torch.nn.Conv2d): convolutional layers
        bn1-bn7 (torch.nn.BatchNorm2d): batchnorm layers
        relu           (torch.nn.ReLU): ReLU activation layer
    """
    def __init__(self):
        super(myLastLayer_RGB, self).__init__()
        
        self.conv1    = nn.Conv2d(3, 64, 3, dilation=1, padding=1)
        self.conv2    = nn.Conv2d(64, 64, 3, dilation=2, padding=2)
        self.conv3    = nn.Conv2d(64, 64, 3, dilation=3, padding=3)
        self.conv4    = nn.Conv2d(64, 64, 3, dilation=4, padding=4)
        self.conv5    = nn.Conv2d(64, 64, 3, dilation=5, padding=5)
        self.conv6    = nn.Conv2d(64, 64, 3, dilation=4, padding=4)
        self.conv7    = nn.Conv2d(64, 64, 3, dilation=3, padding=3)
        self.conv8    = nn.Conv2d(64, 64, 3, dilation=2, padding=2)
        self.conv9    = nn.Conv2d(64, 3, 3, dilation=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.relu     = nn.ReLU()
        
    def forward(self,x):
        """
        Parameters
        ----------
            x (torch.FloatTensor): input images, size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): output of the post-processing layer, size n*c*h*w
        """
        if self.training==True:
            return self.conv9(self.relu(self.bn7(
                        self.conv8(self.relu(self.bn6(
                            self.conv7(self.relu(self.bn5(
                               self.conv6(self.relu(self.bn4(
                                   self.conv5(self.relu(self.bn3(
                                       self.conv4(self.relu(self.bn2(
                                           self.conv3(self.relu(self.bn1(
                                               self.conv2(self.relu(self.conv1(x))))))))))))))))))))))))

        else:
            # we use .detach() to avoid computing and storing the gradients since the model is being tested
            # it allows to save memory
            r=self.conv1(x.detach())
            r=self.relu(r.detach())
            r=self.conv2(r.detach())
            r=self.bn1(r.detach())
            r=self.relu(r.detach())
            r=self.conv3(r.detach())
            r=self.bn2(r.detach())
            r=self.relu(r.detach())
            r=self.conv4(r.detach())
            r=self.bn3(r.detach())
            r=self.relu(r.detach())
            r=self.conv5(r.detach())
            r=self.bn4(r.detach())
            r=self.relu(r.detach())
            r=self.conv6(r.detach())
            r=self.bn5(r.detach())
            r=self.relu(r.detach())
            r=self.conv7(r.detach())
            r=self.bn6(r.detach())
            r=self.relu(r.detach())
            r=self.conv8(r.detach())
            r=self.bn7(r.detach())
            r=self.relu(r.detach())
            r=self.conv9(r.detach())
            return r
