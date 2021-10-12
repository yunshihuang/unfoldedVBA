"""
Functions and classes used by the unfoldedVBA model.
"""


from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy.io as sio
#import cv2
from numpy import linalg as LA  
import math
import PyTorch_ssim as compute_SSIM

def ComputeSNR(OriginalImage, CurrentImage): 
    OriginalImage = OriginalImage.cpu().detach().numpy()
    CurrentImage = CurrentImage.cpu().detach().numpy()
    diff_arrays=  OriginalImage-CurrentImage  
    numérateur= LA.norm(OriginalImage)
    dénominateur= LA.norm(diff_arrays)
    
    SNRi= math.log ( numérateur / dénominateur )
    SNR= np.mean(20*SNRi/math.log(10))

    return (SNR)


def OpenMat(x):
    """
    Converts a numpy array loaded from a .mat file into a properly ordered tensor.
    """
    return torch.from_numpy(x).type(torch.FloatTensor)
    
def Gaussian2Dkernel(N,s,theta):
    x = np.linspace(0.0, 1.0, N)
    X1,X2 = np.meshgrid(x, x)
    X = np.zeros((N,N,2))
    X[:,:,0] = X1
    X[:,:,1] = X2
    m = np.array(([0.5],[0.5])) # centered blur
    R = np.array(([np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]))
    Sigma = R@np.diag(s**2)@R.T
    #generate gaussian
    C = np.linalg.inv(Sigma)
    Xvm = X.reshape((N**2,2)).T-np.tile(m,(1,N**2))
    G = np.exp(-1/2*np.sum(Xvm*(C@Xvm),axis=0).T)
    h = G.reshape((N,N))/np.sum(G) # normalized kernel
    return h    

def InsidePoint(s,theta,x,y):
    # see if the point is inside the ellipse or not
    # the ellipse equation
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    value = np.power((x-0.5)*costheta+(y-0.5)*sintheta,2)/s[0]**2+np.power((x-0.5)*sintheta-(y-0.5)*costheta,2)/s[1]**2
    inside = 1-(value>=1)*1
    return inside

def DefocusBlur(N,s,theta):
    x = np.linspace(0.0, 1.0, N)
    X1,X2 = np.meshgrid(x, x)
    X1_vec = X1.reshape(N**2,1)
    X2_vec = X2.reshape(N**2,1)
    center = int((N+1)/2)
    h_vec = InsidePoint(s,theta,X1_vec,X2_vec)
    h = h_vec.reshape(N,N)
    # if h is zero, it should be replaced by dirac
    if np.sum(h_vec) == 0:
        h[center-1,center-1] = 1
    else:
        h_vec = h_vec/np.sum(h_vec)
        h = h_vec.reshape(N,N)
    return h

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img        

class OpenMat_transf(object):
    """
    Transforms an array into an ordered tensor.
    """
    def __init__(self):
        super(OpenMat_transf,self).__init__()
    def __call__(self,x):
        return OpenMat(x)
    


class MyDataset_OneBlock(torch.utils.data.Dataset):
    """
    Loads an image before feeding it to layer L_k, k = 1,...,K.
    Attributes
    ----------
        file_names         (list): list of strings, image names, length is n
        file_list          (list): list of strings, paths to: 
        the groundtruth image, (x)
        the blurred image, (y)
        the groundtruth blur, (h)
        the noise standard deviation, (\sigma)
        the estimated x from last layer, (x^{k-1})
        the estimated C_x of image from last layer, (C_x^{k-1})
        the estimated z from last layer, (z^{k-1})
        the estimated C_z from last layer, (C_z^{k-1})
        the estimage gamma from last layer, (\gamma^{k-1})
        the estimated lambda from last layer, (\lambda^{k-1})
        dtype              (str): the type of tensor
    """
    def __init__(self, folder_trueimage, folder_blurredimage, folder_trueblur, folder_noise_std, folder_mk, folder_diagSigma, folder_newmh, folder_newSigmah, folder_Gammap, folder_LAMBDAk, dtype):
        super(MyDataset_OneBlock, self).__init__()
        self.file_names = os.listdir(folder_trueimage)
        self.file_list  = [[os.path.join(folder_trueimage,i),
                            os.path.join(folder_blurredimage,i),
                            os.path.join(folder_trueblur,i),
                            os.path.join(folder_noise_std,i),
                            os.path.join(folder_mk,i),
                            os.path.join(folder_diagSigma,i),
                            os.path.join(folder_newmh,i),
                            os.path.join(folder_newSigmah,i),
                            os.path.join(folder_Gammap,i),
                            os.path.join(folder_LAMBDAk,i)] for i in self.file_names]
        self.dtype = dtype
    def __getitem__(self, index):
        return os.path.splitext(self.file_names[index])[0], torch.tensor(sio.loadmat(self.file_list[index][0])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][1])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][2])['image']), torch.tensor(sio.loadmat(self.file_list[index][3])['image'].item()), torch.tensor(sio.loadmat(self.file_list[index][4])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][5])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][6])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][7])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][8])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][9])['image']).type(self.dtype)
    def __len__(self):
        return len(self.file_list)

    
class MyDataset_OneBlock_RGB(torch.utils.data.Dataset):
    """
    Loads an image before feeding it to layer L_k, k=1,...,K.
    Attributes
    ----------
        file_names         (list): list of strings, image names, length is n
        file_list          (list): list of strings, paths to: 
        the groundtruth image, (x)
        the blurred image, (y)
        the groundtruth blur, (h)
        the noise standard deviation, (\sigma)
        the estimated x from last layer, (x^{k-1})
        the estimated C_x of image from last layer, (C_x^{k-1})
        the estimated z from last layer, (z^{k-1})
        the estimated C_z from last layer, (C_z^{k-1})
        the estimage gamma from last layer, (\gamma^{k-1})
        the estimated lambda from last layer, (\lambda^{k-1})
        dtype              (str): the type of tensor
    """
    def __init__(self, folder_trueimage, folder_blurredimage, folder_trueblur, folder_noise_std, folder_mk, folder_diagSigma, folder_newmh, folder_newSigmah, folder_Gammap, folder_LAMBDAk, dtype):
        super(MyDataset_OneBlock_RGB, self).__init__()
        self.file_names = os.listdir(folder_trueblur)
        self.file_list  = [[os.path.join(folder_trueimage,i),
                            os.path.join(folder_blurredimage,i),
                            os.path.join(folder_trueblur,i),
                            os.path.join(folder_noise_std,i),
                            os.path.join(folder_mk,i),
                            os.path.join(folder_diagSigma,i),
                            os.path.join(folder_newmh,i),
                            os.path.join(folder_newSigmah,i),
                            os.path.join(folder_Gammap,i),
                            os.path.join(folder_LAMBDAk,i)] for i in self.file_names]
        self.dtype = dtype
    def __getitem__(self, index):
        return os.path.splitext(self.file_names[index])[0], torch.tensor(sio.loadmat(self.file_list[index][0])['trueimage']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][1])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][2])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][3])['image'].item()).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][4])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][5])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][6])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][7])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][8])['image']).type(self.dtype), torch.tensor(sio.loadmat(self.file_list[index][9])['image']).type(self.dtype)
    def __len__(self):
        return len(self.file_list) 
    
    
class CircularPadding(nn.Module):
    """
    Performs circular padding on a batch of images for cyclic convolution.
    Attributes
    ----------
        pad_size (int): padding size, same on all sides
    """
    def __init__(self, pad_size):
        super(CircularPadding, self).__init__()
        self.pad_size = pad_size
    
    def forward(self, batch):
        """
        Performs a circular padding on a batch of images (for circular convolution).
        Parameters
        ----------
            batch (torch.FloatTensor): batch of images, size n*c*h*w
        Returns
        -------
            (Variable): data type is torch.FloatTensor, size n*c*(h+2*pad_size)*(w+2*pad_size), padded images
        """
        h = batch.size()[1]#2
        w = batch.size()[2]#3
        z    = torch.cat((batch[:,:,:,w-self.pad_size:w],batch,batch[:,:,:,0:self.pad_size]),3)
        z    = torch.cat((z[:,:,h-self.pad_size:h,:],z,z[:,:,0:self.pad_size,:]),2)
        return Variable(z) 

class Conv2D:
    def __init__(self,H):
        self.H = H
    def __call__(self,x):
        return torch.from_numpy(np.fft.ifft2(self.H*np.fft.fft2(x)).real)     
    
class MyConv2d(nn.Module):
    """
    Performs circular convolution on images with a constant filter.
    Attributes
    ----------
        kernel (torch.cuda.FloatTensor): size c*c*h*w filter
        mode                      (str): 'single' or 'batch'
        stride                    (int): dilation factor
        padding                        : instance of CircularPadding or torch.nn.ReplicationPad2d
    """
    def __init__(self, kernel, mode, pad_type = 'circular', padding=0, stride=1):
        """
        Parameters
        ----------
            gpu                  (str): gpu id
            kernel (torch.FloatTensor): convolution filter
            mode                 (str): indicates if the input is a single image of a batch of images
            pad_type             (str): padding type (default is 'circular')
            padding              (int): padding size (default is 0)
            stride               (int): dilation factor (default is 1)
        """
        super(MyConv2d, self).__init__()
        self.gpu      = 'cuda:0'
        self.kernel   = nn.Parameter(kernel,requires_grad=False)   
        self.mode     = mode #'single' or 'batch'
        self.stride   = stride
        if padding==0:
            size_padding = int((kernel[0,0].size(0)-1)/2)
        else:
            size_padding = padding
        if pad_type == 'replicate':
            self.padding = nn.ReplicationPad2d(size_padding)
        if pad_type == 'circular':
            self.padding = CircularPadding(size_padding) 
            
    def forward(self, x): 
        """
        Performs a 2-D circular convolution.
        Parameters
        ----------
            x (torch.FloatTensor): image(s), size n*c*h*w 
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*h*w if mode='single', size c*h*w if mode='batch'
        """
        if self.mode == 'single':
            return F.conv2d(self.padding(x.unsqueeze(0).cuda()), self.kernel, stride=self.stride).data[0]
        if self.mode == 'batch':
            return F.conv2d(self.padding(x.data), self.kernel, stride=self.stride)



class add_Gaussian_noise(object):
    """
    Adds Gaussian noise to images with a noise standard deviation randomly selected within a range.
    Parameters
    ----------
        std_min (double): minimal value for the noise standard deviation
        std_max (double): maximal value for the noise standard deviation
    """
    def __init__(self,std_range):
        super(add_Gaussian_noise,self).__init__()
        self.std_min = std_range[0]
        self.std_max = std_range[1]
    def __call__(self,x):
        """
        Adds Gaussian noise to images.
        Parameters
        ----------
            x (torch.FloatTensor): images, size n*c*h*w 
        Returns
        -------
            (torch.FloatTensor): noisy images, size n*c*h*w 
        """
        std = np.random.uniform(low=self.std_min,high=self.std_max)
        return x + torch.FloatTensor(x.size()).normal_(0,std).double()    
     
class MyTestset(torch.utils.data.Dataset):
    print('Loading Test set')
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTestset, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        image_test = OpenMat(sio.loadmat(self.file_list[index])['image'])#degraded image
        trueimage = OpenMat(sio.loadmat(self.file_list[index])['trueimage'])#groundtruth image
        blur = OpenMat(sio.loadmat(self.file_list[index])['h'])#groundtruth blur
        return self.file_names[index],trueimage,image_test,blur
    def __len__(self):
        return len(self.file_list)    
    
    
class MyTrainset(torch.utils.data.Dataset):
    print('Loading Train set')
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTrainset, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        image_train = OpenMat(sio.loadmat(self.file_list[index])['image'])# degraded image
        trueimage = OpenMat(sio.loadmat(self.file_list[index])['trueimage'])# groundtruth image
        blur = OpenMat(sio.loadmat(self.file_list[index])['h'])# groundtruth blur
        return self.file_names[index],trueimage,image_train,blur
    def __len__(self):
        return len(self.file_list)       
    
class MyTrainset_RGB1(torch.utils.data.Dataset):
    print('Loading Train RGB set')
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTrainset_RGB1, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        """
        Loads a train image
        Parameters
        ----------
            index (int): index of the image in the list of files
        Returns
        -------
                          (str): image name without the extension
            (torch.FloatTensor): test image, size c*h*w 
        """
        image_train_rgb = OpenMat(sio.loadmat(self.file_list[index])['image'])
        rgb_weights = [0.2989, 0.5870, 0.1140]
        image_train = np.around(np.dot(image_train_rgb[...,:3], rgb_weights),decimals=4)
        blur = OpenMat(sio.loadmat(self.file_list[index])['h'])
        trueimage_rgb = OpenMat(sio.loadmat(self.file_list[index])['trueimage'])
        trueimage = np.around(np.dot(trueimage_rgb[...,:3], rgb_weights),decimals=4)
        return self.file_names[index],trueimage,image_train,blur
    def __len__(self):
        return len(self.file_list) 
    
class MyTrainset_RGB(torch.utils.data.Dataset):
    print('Loading Train RGB set')
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTrainset_RGB, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        """
        Loads a train image
        Parameters
        ----------
            index (int): index of the image in the list of files
        Returns
        -------
                          (str): image name without the extension
            (torch.FloatTensor): test image, size c*h*w 
        """
        image_train_rgb = OpenMat(sio.loadmat(self.file_list[index])['image'])
        rgb_weights = [0.2989, 0.5870, 0.1140]
        image_train = np.around(np.dot(image_train_rgb[...,:3], rgb_weights),decimals=4)
        blur = OpenMat(sio.loadmat(self.file_list[index])['h'])
        trueimage_rgb = OpenMat(sio.loadmat(self.file_list[index])['trueimage'])
        trueimage = np.around(np.dot(trueimage_rgb[...,:3], rgb_weights),decimals=4)
        return self.file_names[index],trueimage_rgb,trueimage,image_train_rgb,image_train,blur
    def __len__(self):
        return len(self.file_list)      
    

class MyTestset_RGB(torch.utils.data.Dataset):
    print('Loading Test RGB set')
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTestset_RGB, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        """
        Loads a test image, if the image is gray, the channel is repeated to create an RGB image.
        Parameters
        ----------
            index (int): index of the image in the list of files
        Returns
        -------
                          (str): image name without the extension
            (torch.FloatTensor): test image, size c*h*w 
        """
        image_test_rgb = OpenMat(sio.loadmat(self.file_list[index])['image'])#degraded image
        rgb_weights = [0.2989, 0.5870, 0.1140]
        image_test = np.around(np.dot(image_test_rgb[...,:3], rgb_weights),decimals=4)
        trueimage_rgb = OpenMat(sio.loadmat(self.file_list[index])['trueimage'])#true image
        trueimage = np.around(np.dot(trueimage_rgb[...,:3], rgb_weights),decimals=4)
        blur = OpenMat(sio.loadmat(self.file_list[index])['h'])#true blur
        return self.file_names[index],trueimage_rgb,trueimage,image_test_rgb,image_test,blur
    def __len__(self):
        return len(self.file_list)        
           

class MyDataset(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer L_0 of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        transf1    (list): list of Transform objects, classic transformation (cropp, to tensor, ...)
        transf2    (list): list of Transform objects, specific for each channel (blurr, noise, ...)
        need_names  (str): 'yes' for outputting image names, 'no' else
    """
    def __init__(self, folder='/path/to/folder/', transf1=None, need_names='no',blur_name=None,blur_type = 'Gaussian', noise_std_range = None):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        transf1   (list): list of Transform objects (default is None)
        transf2   (list): list of Transform objects (default is None)
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset, self).__init__()
        self.folder     = folder
        self.file_name  = os.listdir(self.folder)
        self.file_names = [i for i in self.file_name if i != '.ipynb_checkpoints']
        self.file_list  = [os.path.join(self.folder, i) for i in self.file_names]
        self.transf1    = transf1  
        self.need_names = need_names
        self.blur_name  = blur_name
        self.blur_type  = blur_type
        self.noise_std_range  = noise_std_range
    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
                         If the image has just one channel the function will convert it to an RGB format by 
                         repeating the channel.
       Returns
       -------
                          (str): optional, image name without the extension
            (torch.FloatTensor): image before transformation, size h*w
            (torch.FloatTensor): image after transformation, size h*w
        """
        i = self.transf1(sio.loadmat(self.file_list[index])['image']) # the clean RGB images  
        j = np.zeros_like(i)# the blurred RGB images
        im_size = (256,256)
        N = 9
        if self.blur_type == 'Gaussian': # one random anisotropic Gaussian blur kernel
            s_min = 0.15
            s_max = 0.4
            theta = np.random.uniform(0,2*np.pi)
            s = np.random.uniform(s_min,s_max,2)
            if s[0] != s[1]:
                theta = np.pi/4+np.random.choice([True, False])*1*(np.pi/2) # ensure the blur is symmetric
            kernel = Gaussian2Dkernel(N,s,theta)
        elif self.blur_type == 'Gaussian_isotropic':# one random isotropic Gaussian blur kernel
            kernel = np.zeros((9,9))
            s = np.zeros(2)
            s_min = 0.20
            s_max = 0.40
            s1 = np.random.uniform(s_min,s_max,1)
            s[0] = s1
            s[1] = s1
            theta = 0
            kernel = Gaussian2Dkernel(N,s,theta)
        H = psf2otf(kernel,im_size)
        transf2 = transforms.Compose([Conv2D(H),add_Gaussian_noise(self.noise_std_range)])
        j = transf2(i)
        if self.need_names=='no':
            return i, j
        elif self.need_names=='yes':
            return self.blur_name, kernel, os.path.splitext(self.file_names[index])[0], i, j
    def __len__(self):
        return len(self.file_list) 
    
    
    
class MyDataset_RGB(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer L_0 of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        transf1    (list): list of Transform objects, classic transformation (cropp, to tensor, ...)
        transf2    (list): list of Transform objects, specific for each channel (blurr, noise, ...)
        need_names  (str): 'yes' for outputting image names, 'no' else
    """
    def __init__(self, folder='/path/to/folder/', transf1=None, need_names='no',blur_name=None,blur_type = 'Gaussian', noise_std_range = None):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        transf1   (list): list of Transform objects (default is None)
        transf2   (list): list of Transform objects (default is None)
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset_RGB, self).__init__()
        self.folder     = folder
        self.file_name  = os.listdir(self.folder)
        self.file_names = [i for i in self.file_name if i != '.ipynb_checkpoints']
        self.file_list  = [os.path.join(self.folder, i) for i in self.file_names]
        self.transf1    = transf1   
        self.need_names = need_names
        self.blur_name  = blur_name
        self.blur_type  = blur_type
        self.noise_std_range  = noise_std_range
    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
                         If the image has just one channel the function will convert it to an RGB format by 
                         repeating the channel.
       Returns
       -------
                          (str): optional, image name without the extension
            (torch.FloatTensor): image before transformation, size c*h*w
            (torch.FloatTensor): image after transformation, size c*h*w
        """
        i = self.transf1(sio.loadmat(self.file_list[index])['image']) # the clean RGB images  
        j = np.zeros_like(i)# the blurred RGB images
        im_size = (256,256)
        N = 9
        std_min = self.noise_std_range[0]
        std_max = self.noise_std_range[1]
        noise_std = np.around(np.random.uniform(std_min,std_max),decimals = 4)
        std_range = [noise_std,noise_std]# one random noise level
        if self.blur_type == 'Gaussian': # one random Gaussian blur kernel
            s_min = 0.15
            s_max = 0.4
            theta = np.random.uniform(0,2*np.pi)
            s = np.random.uniform(s_min,s_max,2)
            if s[0] != s[1]:
                theta = np.pi/4+np.random.choice([True, False])*1*(np.pi/2) # ensure the blur is symmetric
            kernel = Gaussian2Dkernel(N,s,theta)
        elif self.blur_type == 'uniform_1':# 5*5 uniform kernel
            kernel = np.zeros((9,9))
            kernel[2:7,2:7] = np.ones((5,5))/(5*5)  
        elif self.blur_type == 'uniform_2':# 7*7 uniform kernel
            kernel = np.zeros((9,9))
            kernel[1:8,1:8] = np.ones((7,7))/(7*7)
        elif self.blur_type == 'defocus': # one random defocus blur kernel
            s_min = 0.2
            s_max = 0.5
            theta = np.random.uniform(0,2*np.pi)
            s = np.random.uniform(s_min,s_max,2)
            if s[0] != s[1]:
                theta = np.pi/4+np.random.choice([True, False])*1*(np.pi/2) # ensure the blur is symmetric
            kernel = DefocusBlur(N,s,theta)
        H = psf2otf(kernel,im_size)
        transf2 = transforms.Compose([Conv2D(H),add_Gaussian_noise(std_range)])
        for channel in range(3):
            j[:,:,channel] = transf2(i[:,:,channel])
        if self.need_names=='no':
            return i, j
        elif self.need_names=='yes':
            return self.blur_name, kernel, os.path.splitext(self.file_names[index])[0], i, j, noise_std
    def __len__(self):
        return len(self.file_list)         