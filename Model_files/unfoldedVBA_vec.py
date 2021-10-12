import glob
import os
import gc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import pickle
from IPython.display import clear_output
from Model_files.model_vec import *
from Model_files.modules import *
from PIL import Image
from math import ceil
from tqdm import tqdm
import sys
import scipy
from scipy import signal
from Model_files.tools import *
from Model_files.settings0 import * 
        
class VBA_class(nn.Module):
    """
    Includes the main training and testing methods of unfoldedVBA.
    Attributes
    ----------
        name_kernel            (str): blur kernel name
        noise_std_range       (list): minimal and maximal pixel values
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
        path_save              (str): path to the folder dedicated to saved models
        mode                   (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one, 'lpp' if training the post-processing layer, 'all_layers' if training all the layers together + lpp, 'test' if testing the model (default is 'first_layer')
        lr_first_layer (numpy array): the learning rate to train the first layer 
        lr_greedy      (numpy array): the learning rate to train the following layers    
        lr_lpp         (numpy array): the learning rate to train the post-procesing layer
        lr_N_N         (numpy array): the learning rate to train all the layers + lpp together
        nb_epochs             (list): list of three integers, number of epochs for training the first layer, the remaining layers, lpp and N-N training       
        nb_blocks              (int): number of unfolded iterations  
        batch_size            (list): list of three integers, number of images per batch for training, validation and testing, respectively          
        device                 (str): 'CPU' if training the model using CPU, 'GPU' if training the model using GPU 
        model              (myModel): unfoldedVBA layers    
        last_layer     (myLastLayer): post-processing layer
        train_loader    (DataLoader): loader for the training set
        val_loader      (DataLoader): loader for the validation set
        size_train             (int): number of images in the training set
        size_val               (int): number of images in the validation set
    """
    def __init__(self, test_conditions, folders, mode='first_layer', 
                 lr_first_layer=5e-3, lr_greedy=5e-3, lr_lpp=1e-3, lr_N_N = 5e-5,
                 nb_epochs=[10,10,30,6], nb_blocks=7, batch_size=[10,10,10], device = 'GPU'):

        super(VBA_class, self).__init__()           
        self.name_kernel, self.noise_std_range, self.im_size, self.im_range     = test_conditions
        self.path_test, self.path_trainsets, self.path_valsets, self.path_save  = folders
        self.mode               = mode 
        # training information
        self.lr_first_layer     = lr_first_layer
        self.lr_greedy          = lr_greedy
        self.lr_lpp             = lr_lpp
        self.lr_N_N             = lr_N_N
        self.nb_epochs          = nb_epochs
        self.nb_blocks          = nb_blocks
        self.batch_size         = batch_size
        
        self.loss_fun_mh = diff_h() # MSE(h,\hat{h})
        self.loss_fun_mk = SSIM_loss() # SSIM(x,\hat{x})
   
        if device == 'CPU':
            self.dtype        = torch.FloatTensor 
        elif device == 'GPU':
            self.dtype        = torch.cuda.FloatTensor
        # the unfoldedVBA layers    
        if mode == 'test':
            self.model        = myModel(self.dtype,0,0,self.nb_blocks).type(self.dtype)
        else:
            KmtK0_dict, KmtKn_dict = init_settings0(mode)
            self.model        = myModel(self.dtype,KmtK0_dict,KmtKn_dict,self.nb_blocks).type(self.dtype)
        
        # the post-processing layer
        self.last_layer   = myLastLayer(1,1).type(self.dtype)
     
    def CreateLoader(self,block=0):
        """
        According to the mode, creates the appropriate loader for the training and validation sets.
        """
        # the first layer
        if block==0:    
            train_data        = MyTrainset(folder = self.path_trainsets)
            val_data          = MyTrainset(folder = self.path_valsets)
        else:
            #else, creates a loader loading output of the previous layer
            folder_temp = os.path.join(self.path_save,'ImagesLastBlock')
            train_data  = MyDataset_OneBlock(
                folder_trueimage           = os.path.join(folder_temp,'train','block_'+str(block-1),'trueimage'),
                folder_blurredimage        = os.path.join(folder_temp,'train','block_'+str(block-1),'blurredimage'),
                folder_trueblur            = os.path.join(folder_temp,'train','block_'+str(block-1),'trueblur'),
                folder_noise_std           = os.path.join(folder_temp,'train','block_'+str(block-1),'noise_std'),
                folder_mk                  = os.path.join(folder_temp,'train','block_'+str(block-1),'mk_vec'),
                folder_diagSigma           = os.path.join(folder_temp,'train','block_'+str(block-1),'diagSigma_vec'),
                folder_newmh               = os.path.join(folder_temp,'train','block_'+str(block-1),'newmh_vec'),
                folder_newSigmah           = os.path.join(folder_temp,'train','block_'+str(block-1),'newSigmah_vec'),
                folder_Gammap              = os.path.join(folder_temp,'train','block_'+str(block-1),'Gammap_vec'),
                folder_LAMBDAk             = os.path.join(folder_temp,'train','block_'+str(block-1),'LAMBDAk_vec'), dtype = self.dtype)
            val_data  = MyDataset_OneBlock(
                folder_trueimage           = os.path.join(folder_temp,'val','block_'+str(block-1),'trueimage'),
                folder_blurredimage        = os.path.join(folder_temp,'val','block_'+str(block-1),'blurredimage'),
                folder_trueblur            = os.path.join(folder_temp,'val','block_'+str(block-1),'trueblur'),
                folder_noise_std           = os.path.join(folder_temp,'val','block_'+str(block-1),'noise_std'),
                folder_mk                  = os.path.join(folder_temp,'val','block_'+str(block-1),'mk_vec'),
                folder_diagSigma           = os.path.join(folder_temp,'val','block_'+str(block-1),'diagSigma_vec'),
                folder_newmh               = os.path.join(folder_temp,'val','block_'+str(block-1),'newmh_vec'),
                folder_newSigmah           = os.path.join(folder_temp,'val','block_'+str(block-1),'newSigmah_vec'),
                folder_Gammap              = os.path.join(folder_temp,'val','block_'+str(block-1),'Gammap_vec'),
                folder_LAMBDAk             = os.path.join(folder_temp,'val','block_'+str(block-1),'LAMBDAk_vec'), dtype = self.dtype)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size[0], shuffle=True)
        self.val_loader   = DataLoader(val_data, batch_size=self.batch_size[1], shuffle=False)
    def CreateFolders(self,block):
        """
        Creates directories for saving results.
        """
       
        if self.mode=='first_layer' or self.mode=='greedy':
            name = 'block_'+str(block)
            if not os.path.exists(os.path.join(self.path_save,name)):
                os.makedirs(os.path.join(self.path_save,name,'training'))
        elif self.mode=='all_layers':
            name = 'block_'+str(0)+'_'+str(self.nb_blocks-1)
            if not os.path.exists(os.path.join(self.path_save,name)):
                os.makedirs(os.path.join(self.path_save,name,'training'))        
        elif self.mode=='lpp':
            name = 'lpp'
            if not os.path.exists(os.path.join(self.path_save,name)):
                os.makedirs(os.path.join(self.path_save,name,'training'))        
        if self.mode!='test':
            folder = os.path.join(self.path_save,'ImagesLastBlock')
            if not os.path.exists(folder):
                subfolders    = ['train','val']
                subsubfolders = ['trueimage','blurredimage','trueblur','noise_std','mk_vec','diagSigma_vec','newmh_vec','newSigmah_vec','Gammap_vec','LAMBDAk_vec']
                paths         = [os.path.join(folder, sub, subsub) for sub in subfolders for subsub in subsubfolders]
                for path in paths:
                    os.makedirs(path)
                    
        
    def train(self,block=0):
        """
        The training of unfoldedVBA.
        Parameters
        ----------
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """  
        folder_save = os.path.join(self.path_save,'epoch')
        if self.mode=='first_layer':
            print('====================A new training starts!=============')
            # trains the first layer
            print('=================== Block number {} ==================='.format(0))
            # to store results
            loss_epochs_train  =  np.zeros(self.nb_epochs[0])
            loss_epochs_val  =  np.zeros(self.nb_epochs[0])
            loss_min_val      =  float('Inf')
            self.CreateFolders(0)
            folder = os.path.join(self.path_save,'block_'+str(0))
            self.CreateLoader(block=0)
            # defines the optimizer
            lr        = self.lr_first_layer #learnig rate
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
#                 #==========================================================================================================
            # for the first layer
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[0]): 
                print('This is epoch {} '.format(epoch))
                # sets training mode
                self.model.Layers[0].train()
                gc.collect()
                # goes through all minibatches
                print('This is traning stage')
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, h] = minibatch            # get the minibatch
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names))      
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    batch = x_true.shape[0]
                    sizex = x_true.shape[1]
                    sizeh = h.shape[1]
                    SNR_init = 0
                    SNR_temp = 0
                    init = Initialization(batch,sizex,sizeh,self.dtype)
                    self.T_vec,self.t,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec     = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode) 
                    hhat_vec=self.T_vec@newmh_vec+self.t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))#the restored kernel of 
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
#                         loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
#                         print('The RMSE is {}'.format(loss_))
                    loss = self.loss_fun_mh(hhat_vec,h)
                    print('The loss over all batches are {}'.format(loss))
                    loss_epochs_train[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() #performs a parameter update

                # tests on validation set
                print('This is validation stage')
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, h] = minibatch            # gets the minibatch
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names)) 
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    batch = x_true.shape[0]
                    sizex = x_true.shape[1]
                    sizeh = h.shape[1]
                    SNR_init = 0
                    SNR_temp = 0
                    init = Initialization(batch,sizex,sizeh,self.dtype)
                    __,__,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode) 
                    hhat_vec=self.T_vec@newmh_vec+self.t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
#                         loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
#                         print('The RMSE is {}'.format(loss_))
                    # computes loss on validation set
                    loss = self.loss_fun_mh(hhat_vec,h)
                    print('The loss over all batches are {}'.format(loss))
                    loss_current_val += torch.Tensor.item(loss)
                    loss_epochs_val[epoch] += torch.Tensor.item(loss)
                    
                if loss_min_val>loss_current_val:
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal1.pt'))
                    loss_min_val = loss_current_val
                    
                #save the results for each epoch
                folder_results_train = os.path.join(folder_save,'block'+str(block),'train')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_train):
                    os.makedirs(folder_results_train)
                with open(folder_results_train+'/loss_epoch_train.txt', "a") as file_object:
                    if epoch == 0:
                        file_object.write('------------------A new test-------------------------------')
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_train[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                folder_results_val = os.path.join(folder_save,'block'+str(block),'val')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_val):
                    os.makedirs(folder_results_val)
                with open(folder_results_val+'/loss_epoch_val.txt', "a") as file_object:
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_val[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
    #==========================================================================================================
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Block 0 is done.')
            self.save_OneBlock(block=0)
            print('-----------------------------------------------------------------')
            
            
            # calls the same function to start training of the next layer
            self.mode = 'greedy'
            self.train(block=1)
            

#===========================================================================================================
    
        
        elif self.mode=='greedy':
            print('This is greedy processing')
            # trains the next layer
            print('=================== Block number {} ==================='.format(block))
            # to store results
            loss_epochs_train       =  np.zeros(self.nb_epochs[1])
            loss_epochs_val       =  np.zeros(self.nb_epochs[1])
            loss_min_val      =  float('Inf')
            self.CreateFolders(block)
            folder = os.path.join(self.path_save,'block_'+str(block))
            self.CreateLoader(block=block)
            # puts first blocks in evaluation mode: gradient is not computed
            self.model.GradFalse(block,self.mode)
            # defines the optimizer
            lr        = self.lr_greedy
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[1]):
                print('This is epoch {}'.format(epoch))
                self.model.Layers[block].train() # training mode
                gc.collect()
                print('This is traning stage')
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch                         # gets the minibatch
                    h_vec = h_vec.type(self.dtype)
                    Ch_vec = Ch_vec.type(self.dtype)
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names))      
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    batch = x_true.shape[0]
                    sizex = x_true.shape[1]
                    sizeh = h.shape[1]
                    SNR_init = 0
                    SNR_temp = 0
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,estimatedimage_vec,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode,block=block) 
                    hhat_vec=self.T_vec@newmh_vec+self.t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
#                         loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
#                         print('The RMSE is {}'.format(loss_))
                    
                    # Computes and prints loss
                    loss = self.loss_fun_mh(hhat_vec,h)
                    print('The loss over all batches are {}'.format(loss))
                    loss_epochs_train[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # tests on validation set
                print('This is validation stage')
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch           # gets the minibatch
                    h_vec = h_vec.type(self.dtype)
                    Ch_vec = Ch_vec.type(self.dtype)
                    if names =='.ipynb_checkpoints': continue
                    print('The name is ',names)     
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    SNR_init = 0
                    SNR_temp = 0
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,estimatedimage_vec,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode,block=block) 
                    hhat_vec=self.T_vec@newmh_vec+self.t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
#                         loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
#                         print('The RMSE is {}'.format(loss_))
                    loss = self.loss_fun_mh(hhat_vec,h)
                    print('The loss over all batches are {}'.format(loss))
                    loss_epochs_val[epoch] += torch.Tensor.item(loss)
                    # computes loss on validation set
                    loss_current_val += torch.Tensor.item(self.loss_fun_mh(hhat_vec, h))

                if loss_min_val>loss_current_val:
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal1.pt'))
                    loss_min_val = loss_current_val

                # save the results for each epoch
                folder_results_train = os.path.join(folder_save,'block'+str(block),'train')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_train):
                    os.makedirs(folder_results_train)
                with open(folder_results_train+'/loss_epoch_train.txt', "a") as file_object:
                    if epoch == 0:
                        file_object.write('------------------A new test-------------------------------')
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_train[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                folder_results_val = os.path.join(folder_save,'block'+str(block),'val')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_val):
                    os.makedirs(folder_results_val)
                with open(folder_results_val+'/loss_epoch_val.txt', "a") as file_object:
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_val[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                    #==========================================================================================================
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Block {} is done.'.format(block))
            self.save_OneBlock(block=block)
            print('-----------------------------------------------------------------')
            
            #calls the same function to start training of next block 
            if block==self.nb_blocks-1:
                self.mode = 'lpp'
                self.train()
            else:   
                self.train(block=block+1)

###############################################################################################################
        elif self.mode=='lpp':
            # trains the post-processing layer
            print('start the post-processing layer')
            # to store results
            loss_epochs_train       =  np.zeros(self.nb_epochs[2])
            loss_epochs_val       =  np.zeros(self.nb_epochs[2])
            loss_min_val      =  float('Inf')
            self.CreateFolders(self.nb_blocks)
            folder = os.path.join(self.path_save,'lpp')
            self.CreateLoader(block=self.nb_blocks)
            # puts first blocks in evaluation mode: gradient is not computed
            self.model.GradFalse(self.nb_blocks,self.mode) 
            # defines the optimizer
            lr        = self.lr_lpp
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==============================================================================================
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[2]):
                self.last_layer.train() #training mode
                gc.collect()
                print('This is traning stage')
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch                         # gets the minibatch
                    batch = x_true.shape[0]
                    sizex = x_true.shape[1]
                    sizeh = h.shape[1]
                    estimatedimage_vec_ = estimatedimage_vec.reshape(batch,1,sizex,sizex)
                    x_true_ = x_true.reshape(batch,1,sizex,sizex)
                    x_blurred_ = x_blurred.reshape(batch,1,sizex,sizex)
                    h_vec = h_vec.type(self.dtype)
                    Ch_vec = Ch_vec.type(self.dtype)
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names))      
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    SNR_init = 0
                    SNR_temp = 0
                    mk_vec_ = self.last_layer(estimatedimage_vec_)
                    mk_vec = mk_vec_.reshape(batch,sizex,sizex)
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
                    # Computes and prints loss
                    loss                = self.loss_fun_mk(mk_vec_, x_true_)
                    print('The SSIM over all batches are {}'.format(-loss))
                    loss_epochs_train[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))

                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # tests on validation set
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                print('This is validation stage')
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch                         # gets the minibatch
                    estimatedimage_vec_ = estimatedimage_vec.reshape(batch,1,sizex,sizex)
                    x_true_ = x_true.reshape(batch,1,sizex,sizex)
                    x_blurred_ = x_blurred.reshape(batch,1,sizex,sizex)
                    h_vec = h_vec.type(self.dtype)
                    Ch_vec = Ch_vec.type(self.dtype)
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names))      
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    SNR_init = 0
                    SNR_temp = 0
                    mk_vec_ = self.last_layer(estimatedimage_vec_)
                    mk_vec = mk_vec_.reshape(batch,sizex,sizex)
#                     for j in range(batch):
#                         print('This is batch {}'.format(j))
#                         SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
#                         SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
#                         print('The initial SNR is {}'.format(SNR_init))
#                         print('The current SNR is {}'.format(SNR_temp))
                    # Computes and prints loss
                    loss                = self.loss_fun_mk(mk_vec_, x_true_)
                    print('The SSIM over all batches are {}'.format(-loss))
                    loss_epochs_val[epoch] += torch.Tensor.item(loss)
                    loss_current_val += torch.Tensor.item(self.loss_fun_mh(mk_vec_, x_true_))
                    

                if loss_min_val>loss_current_val:
                    torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing_MinLossOnVal.pt'))
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal.pt'))
                    loss_min_val = loss_current_val

                # save the results for each epoch
                folder_results_train = os.path.join(folder_save,'lpp','train')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_train):
                    os.makedirs(folder_results_train)    
                with open(folder_results_train+'/SSIM_epoch_train.txt', "a") as file_object:
                    if epoch == 0:
                        file_object.write('------------------A new test-------------------------------')
                    file_object.write('The loss for epoch {} is {}'.format(epoch,-loss_epochs_train[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                folder_results_val = os.path.join(folder_save,'lpp','val')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_val):
                    os.makedirs(folder_results_val)
                with open(folder_results_val+'/SSIM_epoch_val.txt', "a") as file_object:
                    file_object.write('The loss for epoch {} is {}'.format(epoch,-loss_epochs_val[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                
                
                
            # training of greedy approach is finished
            print('-----------------------------------------------------------------')
            print('Training of lpp is done.')
            print('-----------------------------------------------------------------')
            return 


##############################################################################################################       
        elif self.mode=='all_layers':
            # start the N-N training
            # trains several blocks as one
            print('=================== Block number {} to Block number {} ==================='.format(0,self.nb_blocks-1))
            # to store results
            loss_epochs_train       =  np.zeros(self.nb_epochs[1])
            loss_epochs_val       =  np.zeros(self.nb_epochs[1])
            loss_min_val      =  float('Inf')
            self.CreateFolders(self.nb_blocks-1)
            folder = os.path.join(self.path_save,'block_'+str(0)+'_'+str(self.nb_blocks-1))
            self.CreateLoader(0)
            # puts first blocks in evaluation mode: gradient is not computed
            self.model.GradFalse(self.nb_blocks,self.mode)
            # defines the optimizer
            lr        = self.lr_N_N #learnig rate
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr,weight_decay=1e-4)
             #==========================================================================================================
            # for the first layer
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[1]): 
                print('This is epoch {} '.format(epoch))
                # sets training mode
                for k in range(0,self.nb_blocks):
                    self.model.Layers[k].train() #training mode
                gc.collect()
                print('This is training stage')
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, h] = minibatch            # get the minibatch
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names))    
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    batch = x_true.shape[0]
                    sizex = x_true.shape[1]
                    sizeh = h.shape[1]
                    SNR_init = 0
                    SNR_temp = 0
                    init = Initialization(batch,sizex,sizeh,self.dtype)
                    T_vec,t,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec     = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode) 
                    hhat_vec = T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))#the restored kernel of 
                    x_true_ = x_true.reshape(batch,1,sizex,sizex)
                    mk_vec_ = mk_vec.reshape(batch,1,sizex,sizex)
                    mk_vec_ = self.last_layer(mk_vec_)
                    mk_vec = mk_vec_.reshape(batch,sizex,sizex)
                    for j in range(batch):
                        print('This is batch {}'.format(j))
                        SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
                        SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
                        print('The initial SNR is {}'.format(SNR_init))
                        print('The current SNR is {}'.format(SNR_temp))
                        loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
                        print('The RMSE is {}'.format(loss_))
                    loss                = self.loss_fun_mk(mk_vec_, x_true_)
                    print('The SSIM over all batches are {}'.format(-loss))
                    loss_epochs_train[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    #sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() #performs a parameter update
                    
                # tests on validation set
                print('This is validation stage')
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, h] = minibatch            # gets the minibatch
                    if names =='.ipynb_checkpoints': continue
                    print('The name is {} '.format(names)) 
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    h            = Variable(h.type(self.dtype),requires_grad=False)
                    SNR_init = 0
                    SNR_temp = 0
                    init = Initialization(batch,sizex,sizeh,self.dtype)
                    T_vec,t,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                    mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode) 
                    hhat_vec = T_vec@newmh_vec+t
                    hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
                    x_true_ = x_true.reshape(batch,1,sizex,sizex)
                    mk_vec_ = mk_vec.reshape(batch,1,sizex,sizex)
                    mk_vec_ = self.last_layer(mk_vec_)
                    mk_vec = mk_vec_.reshape(batch,sizex,sizex)
                    for j in range(batch):
                        print('This is batch {}'.format(j))
                        SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
                        SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
                        print('The initial SNR is {}'.format(SNR_init))
                        print('The current SNR is {}'.format(SNR_temp))
                        loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
                        print('The RMSE is {}'.format(loss_))
                    loss                = self.loss_fun_mk(mk_vec_, x_true_)
                    print('The SSIM over all batches are {}'.format(-loss))
                    loss_current_val += torch.Tensor.item(loss)
                    loss_epochs_val[epoch] += torch.Tensor.item(loss)
                    
                if loss_min_val>loss_current_val:
                    torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing_MinLossOnVal.pt'))
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal.pt'))
                    loss_min_val = loss_current_val
                # save the results for each epoch
                folder_results_train = os.path.join(folder_save,'block0_'+str(self.nb_blocks-1),'train')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_train):
                    os.makedirs(folder_results_train)
                with open(folder_results_train+'/loss_epoch_train.txt', "a") as file_object:
                    if epoch == 0:
                        file_object.write('------------------A new test-------------------------------')
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_train[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                folder_results_val = os.path.join(folder_save,'block0_'+str(self.nb_blocks-1),'val')
                # create the path if it does not exist 
                if not os.path.exists(folder_results_val):
                    os.makedirs(folder_results_val)
                with open(folder_results_val+'/loss_epoch_val.txt', "a") as file_object:
                    file_object.write('The loss for epoch {} is {}'.format(epoch,loss_epochs_val[epoch]))
                    file_object.write("\n")
                    file_object.write('----------------------------------------------------------')
                    file_object.write("\n")
                    
    #==========================================================================================================
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Block {} to Block {} + lpp is done.'.format(0,self.nb_blocks-1))
            print('-----------------------------------------------------------------')
                           

#===========================================================================================================

    def save_OneBlock(self,block=0): 
        print('Saving block')
        print('This is training set')
        """
        Saves the outputs of the current layer.
        Parameters
        ----------
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)    
        """
        self.model.eval() #evaluation mode   
        folder    = os.path.join(self.path_save,'ImagesLastBlock')
        folder_save = os.path.join(self.path_save,'results')
           
        for minibatch in self.train_loader:
            if self.mode=='first_layer':
                [names, x_true, x_blurred, h] = minibatch     # gets the minibatch
                if names =='.ipynb_checkpoints': continue
                print('The name is ',names)     
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                h            = Variable(h.type(self.dtype),requires_grad=False)
                batch = x_true.shape[0]
                sizex = x_true.shape[1]
                sizeh = h.shape[1]  
                init = Initialization(batch,sizex,sizeh,self.dtype)
                __,__,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec     = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode)                    
            else:
                [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch
                h_vec = h_vec.type(self.dtype)
                Ch_vec = Ch_vec.type(self.dtype)
                # gets the minibatch
                if names =='.ipynb_checkpoints': continue
                print('The name is {} '.format(names))      
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                h            = Variable(h.type(self.dtype),requires_grad=False)
                batch = x_true.shape[0]
                sizex = x_true.shape[1]
                sizeh = h.shape[1]
                SNR_init = 0
                SNR_temp = 0
                mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,estimatedimage_vec,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode,block=block)   
            hhat_vec=self.T_vec@newmh_vec+self.t
            hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
            print('One block for one image for trainig set is saved')
            for j in range(batch):
                print('This is batch {}'.format(j))
                SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
                SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
                print('The initial SNR is {}'.format(SNR_init))
                print('The current SNR is {}'.format(SNR_temp))
                loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
                print('The RMSE is {}'.format(loss_))
                folder_results = os.path.join(folder,'train','block_'+str(block),'trueimage')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                     'image':x_true[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'blurredimage')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':x_blurred[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'trueblur')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':h[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'noise_std')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':self.noise_std_range[0]})
                folder_results = os.path.join(folder,'train','block_'+str(block),'mk_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':mk_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'diagSigma_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':diagSigma_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'newmh_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':newmh_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'newSigmah_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':newSigmah_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'Gammap_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':Gammap_vec[j].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'train','block_'+str(block),'LAMBDAk_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':LAMBDAk_vec[j,:,:].detach().cpu().numpy().astype('float32')})  

        # validation set
        print('This is validation set')
        for minibatch in self.val_loader:
            if self.mode=='first_layer':
                [names, x_true, x_blurred, h] = minibatch            # gets the minibatch
                if names =='.ipynb_checkpoints': continue
                print('The name is ',names)   
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                h            = Variable(h.type(self.dtype),requires_grad=False)
                batch = x_true.shape[0]
                sizex = x_true.shape[1]
                sizeh = h.shape[1]                
                init = Initialization(batch,sizex,sizeh,self.dtype)
                __,__,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
                mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec     = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode)    
            else:
                [names, x_true, x_blurred, h, noise_std, estimatedimage_vec, diagSigma_vec, h_vec, Ch_vec, gamma_vec, lambda_vec] = minibatch
                h_vec = h_vec.type(self.dtype)
                Ch_vec = Ch_vec.type(self.dtype)
                # gets the minibatch
                if names =='.ipynb_checkpoints': continue
                print('The name is {} '.format(names))      
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                h            = Variable(h.type(self.dtype),requires_grad=False)
                batch = x_true.shape[0]
                sizex = x_true.shape[1]
                sizeh = h.shape[1]
                SNR_init = 0
                SNR_temp = 0
                mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec      = self.model(x_blurred,x_true,estimatedimage_vec,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode,block=block)                 
            hhat_vec=self.T_vec@newmh_vec+self.t
            hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
            print('One block for one image for trainig set is saved')
            for j in range(batch):
                print('This is batch {}'.format(j))
                SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
                SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
                print('The initial SNR is {}'.format(SNR_init))
                print('The current SNR is {}'.format(SNR_temp))
                loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
                print('The RMSE is {}'.format(loss_))
                folder_results = os.path.join(folder,'val','block_'+str(block),'trueimage')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                     'image':x_true[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'blurredimage')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':x_blurred[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'trueblur')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':h[j,:,:].cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'noise_std')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':self.noise_std_range[0]})
                folder_results = os.path.join(folder,'val','block_'+str(block),'mk_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':mk_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'diagSigma_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':diagSigma_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'newmh_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':newmh_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'newSigmah_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':newSigmah_vec[j,:,:].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'Gammap_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':Gammap_vec[j].detach().cpu().numpy().astype('float32')})
                folder_results = os.path.join(folder,'val','block_'+str(block),'LAMBDAk_vec')
                if not os.path.exists(folder_results):
                    os.makedirs(folder_results)
                sio.savemat(os.path.join(folder_results,names[j]),{
                 'image':LAMBDAk_vec[j,:,:].detach().cpu().numpy().astype('float32')})  
    
    def test(self, dataset):  
        """
        Parameters
        ----------
            dataset        (str): name of the test set
        """       
        path_savetest_mk                   = os.path.join(self.path_save,'Results_on_Testsets','Flickr30','mk_vec_N_N')
        path_savetest_newmh                = os.path.join(self.path_save,'Results_on_Testsets','Flickr30','newmh_vec_N_N')
        path_dataset                       = os.path.join(self.path_test, self.name_kernel, dataset)
        

        # creates directory for saving results
        if not os.path.exists(path_savetest_mk):
            os.makedirs(path_savetest_mk)
        if not os.path.exists(path_savetest_newmh):
            os.makedirs(path_savetest_newmh)    
        data          = MyTestset(folder=path_dataset)
        loader        = DataLoader(data, batch_size=self.batch_size[2], shuffle=False)
        loss_epochs_test = 0
        SSIM_epochs_test = 0
        # evaluation mode
        self.model.eval() 
        self.last_layer.eval()
        torch.manual_seed(0)
        for minibatch in tqdm(loader,file=sys.stdout):
            [names, x_true, x_blurred, h] = minibatch # gets the minibatch
            if names =='.ipynb_checkpoints': continue
            print('The name is {} '.format(names))      
            x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
            x_true    = Variable(x_true.type(self.dtype),requires_grad=False)
            h    = Variable(h.type(self.dtype),requires_grad=False)
            batch = x_blurred.shape[0]
            sizex = x_blurred.shape[1]
            sizeh = h.shape[1] 
            x_blurred_ = x_blurred.reshape(batch,1,sizex,sizex)
            x_true_ = x_true.reshape(batch,1,sizex,sizex)
            init = Initialization(batch,sizex,sizeh,self.dtype)
            T_vec,t,h_vec,Ch_vec,gamma_vec,lambda_vec = init.f(x_blurred,h)
            mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec     = self.model(x_blurred,x_true,x_blurred,h,self.noise_std_range[0]**2,h_vec,Ch_vec,gamma_vec,lambda_vec,self.mode) 
            mk_vec_ = mk_vec.reshape(batch,1,sizex,sizex)
            hhat_vec = T_vec@newmh_vec+t
            hhat_vec = torch.reshape(hhat_vec,(batch,sizeh,sizeh))
            loss_mh = self.loss_fun_mh(hhat_vec,h)
            print('The loss over all batches are {}'.format(loss_mh))
            loss_epochs_test += torch.Tensor.item(loss_mh)

            mk_vec_ = self.last_layer(mk_vec_)
            print('PP done')
            mk_vec = mk_vec_.reshape(batch,256,256)
            for j in range(batch):
                print('This is batch {}'.format(j))
                SNR_temp = ComputeSNR(x_true[j,:,:],mk_vec[j,:,:])
                SNR_init = ComputeSNR(x_true[j,:,:],x_blurred[j,:,:])
                print('The initial SNR is {}'.format(SNR_init))
                print('The current SNR is {}'.format(SNR_temp))
                loss_ = self.loss_fun_mh(hhat_vec[j,:,:],h[j,:,:])
                print('The RMSE is {}'.format(loss_))
            for j in range(batch):
                sio.savemat(os.path.join(path_savetest_mk,names[j]),{'image':
            mk_vec[j,:,:].detach().cpu().numpy().astype('float64')})
                print('mk for test set is saved')
                sio.savemat(os.path.join(path_savetest_newmh,names[j]),{'image':
            newmh_vec[j,:,:].detach().cpu().numpy().astype('float64')})
                print('newmh for test set is saved')


            loss_mk  = self.loss_fun_mk(mk_vec_, x_true_)
            print('SSIM is ',loss_mk)
            SSIM_epochs_test += torch.Tensor.item(loss_mk)     
        print('The SSIM over all batches are ', SSIM_epochs_test)
        print('The MSE over all batches are ', loss_epochs_test)
