"""
functions that used for unfoldedVBA
"""

import torch
import numpy as np
from numpy.linalg import inv
from scipy.sparse import spdiags
from tqdm import tqdm
import time
#import numba
import sys
from scipy.sparse import csr_matrix
from torchvision import transforms, utils
import scipy.io as sio
import os
from scipy import sparse
from scipy.sparse import find
import math
from numpy import linalg as LA
from torch.utils.data import DataLoader
from Model_files.modules import *
import logging
import scipy
from scipy import ndimage

def flatten(t):
    t = torch.transpose(t,0,1)
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

def convert(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU 
    """
    M = M.tocoo().astype(np.float64)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().cuda()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cuda'))
    return Ms

def matlab_style_gauss2D(shape=(7,7),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def TensorFilter(filt_list, c=3, dtype=torch.FloatTensor):
    """
    Creates a list of tensors from a list of filters, such that the result of the 2-D 
    convolution with one of a tensor and an image of size c*h*w is the convolution of 
    each input channel with the filter (no combination of the channels).
    Parameters
    ----------
        filt_list (list): list of numpy arrays, square filters of size h*h
        c          (int): number of channels
        dtype     (type): (default is torch.FloatTensor)
    Returns
    -------
        tens_filt_list (list): list of torch.FloatTensors, each element is of size c*c*h*h
    """
    tens_filt_list=[]
    for filt in filt_list:
        tens_filt = torch.zeros(c,c,filt.shape[0],filt.shape[0])
        for i in range(c):
            tens_filt[i,i,:,:] = torch.from_numpy(filt).type(dtype)
        tens_filt_list.append(tens_filt.clone())
    return tens_filt_list


def circshift_array (A, b, c):
    if b==0:
        Void= np.zeros_like(A)
        for i in range(len(Void)):
            Void[i]= np.roll(A[i], c)
        return (Void)
    
    else: 
        return (circshift_array (A.T, 0, b)).T

def circshift_tensor_vec (A, b, c):
    if b==0:
        Void= torch.zeros_like(A)
        for i in range(Void.shape[1]):
            Void[:,i,:]= torch.roll(A[:,i,:],c,1)
        return (Void)
    
    else: 
        return torch.transpose(circshift_tensor_vec (torch.transpose(A,1,2), 0, b),1,2)  
    
def difference_matrices (Nx, Ny):
    
    b= [1 for i in range (Nx*Ny)]
    minusb= [-1 for i in range (Nx*Ny)]

    dataDiags= np.array([minusb, b, minusb])
    
    diags1=np.array([-1, 0, Nx*Ny-1])
    diags2=np.array([-Nx, 0, Nx*Ny-Nx])
    
    Dv = spdiags (dataDiags, diags1, Nx*Ny, Nx*Ny).toarray()
    Dh = spdiags (dataDiags, diags2, Nx*Ny, Nx*Ny).toarray()
    
    G = np.concatenate((Dh, Dv), axis=0)
    return (G)

def reset_parameters(name,param):# to be deleted?
    if 'bias' in name :
        param.data = torch.zeros_like(param)
    if 'weight' in name:     
        dimension = len(param.shape)
        if dimension == 4:
            size = param.shape[dimension-1]
            param.data = torch.eye(size).repeat(param.shape[0],param.shape[1],1,1)
        elif dimension == 3:
            param.shape[dimension-1]
            param.data = torch.eye(size).repeat(param.shape[0])  

def RGBtoYUV(x,window_size,dtype):
    # x: the input array batch*256*256*3
    # median filter on the U,V components
    x_new = x.detach().cpu().numpy()
    batch = x.shape[0]
    channel = x.shape[1]
    size = x.shape[2]
    new_U = np.zeros((batch,size,size))
    new_V = np.zeros((batch,size,size))
    # the weights of RGB to YUV 
    U_weights = [-0.14713, -0.28886, 0.436]
    V_weights = [0.615, -0.51499, -0.10001]
    U = np.around(np.dot(x_new[...,:3], U_weights),decimals=4) 
    V = np.around(np.dot(x_new[...,:3], V_weights),decimals=4) 
    for j in range(batch): 
        new_U[j,:,:] = ndimage.median_filter(U[j,:,:], 3, mode = 'constant')
        new_V[j,:,:] = ndimage.median_filter(V[j,:,:], 3, mode = 'constant')
    U_tensor = torch.tensor(new_U).type(dtype)
    V_tensor = torch.tensor(new_V).type(dtype)
    return U_tensor, V_tensor


def YUVtoRGB(Y_new,U_new,V_new,dtype):
    # Y_new,U_new,V_new: the Y,U,V components
    batch = Y_new.shape[0]
    size = Y_new.shape[2]
    Y_new = Y_new.reshape(batch,size,size)
    R_new = torch.zeros((batch,size,size)).type(dtype)
    G_new = torch.zeros((batch,size,size)).type(dtype)
    B_new = torch.zeros((batch,size,size)).type(dtype)
    R_new = Y_new + 1.139834576 * V_new
    G_new = Y_new - 0.3946460533 * U_new - 0.58060 * V_new
    B_new = Y_new + 2.032111938 * U_new
    RGB_new = torch.zeros((batch,3,size,size)).type(dtype)
    RGB_new[:,0,:,:] = R_new
    RGB_new[:,1,:,:] = G_new
    RGB_new[:,2,:,:] = B_new
    return RGB_new
    
    
def CreateKm (sizeh, sizex, symmetry):
    
    center = int((sizeh-1)/2)
    j=0  
    k = []
    kT = []
    r_vec = []
    rt_vec = []
    
    for cc in range (-center, center +1):
        shift_cc= circshift_array(np.eye(sizex), 0, -cc)
        
        for rr in range (-center, center +1):
            shift_rr = circshift_array(np.eye(sizex), 0, -rr)
            k.append( sparse.kron(shift_cc, shift_rr) )
            r_vec.append(find(k[j].T)[0])
            rt_vec.append(find(k[j])[0])
            
            j += 1
            
    r_vec=np.asarray(r_vec).T
    rt_vec=np.asarray(rt_vec).T
    k= np.asarray(k)
    
    if symmetry == 0: # if no symmetry constraint
        
        newMh = int(sizeh*sizeh-1)
        
        newk= np.zeros(newMh, dtype=object)
        diag_quadratick= np.zeros((newMh, newMh))
        
        for m in range(newMh):
            newk[m] = k[m]-k[sizeh*sizeh-1]
        newk0 = k[sizeh*sizeh-1]
        
        #K'_m*k'_n
        #the diagonal entry of K'_m*k'_n
        
        for m in range (newMh):
            diag_quadratick [m, m] = 2
            for n in range (newMh):
                if m!=n:
                    diag_quadratick [m, n] = 1
                    diag_quadratick [n, m] = 1
        
        
        T = np.eye(sizeh*sizeh-1)
        last_row = -np.ones((1,sizeh*sizeh-1))
        T = np.vstack((T,last_row))
        
        t = np.zeros((sizeh**2 -1, 1))
        lastDigit = np.array([1])
        t = np.vstack((t,lastDigit))
        dict_val = 0
            
    
    elif symmetry == 2: # # if 2-part symmetry constraint
        
        new_tempk= np.zeros((sizeh, sizeh), dtype=object)
        newMh= int ( ( (1+ sizeh)* sizeh)/2 -1 )

        matrixk = k.reshape((sizeh, sizeh), order='F')
        for i in range (sizeh):
            for j in range (sizeh):
                new_tempk[i, j] = matrixk[i, j] - k[sizeh**2-1]
                
        newk= np.zeros(int(((sizeh+1)*sizeh)/2 ), dtype=object)
        diag_quadratick= np.zeros((int(((sizeh+1)*sizeh)/2 ), int(((sizeh+1)*sizeh)/2 ) ))
        dict_val= np.zeros((newMh+1,1))
        
        m=0
                
        for i in range (sizeh):
            for j in range (i+1):
                
                if i ==j:   
                    newk[m] = new_tempk[i, j]
                    diag_quadratick[m, m] =2
                    dict_val[m] = 1
                else:    
                    newk[m]= new_tempk[i, j] + new_tempk[j, i]
                    diag_quadratick[m, m] = 6
                    dict_val [m] = 2 
                m += 1
                       
        newk = newk[0 : newMh] #K'_m
        newk0 = k[sizeh**2-1] #K'_0
        
        #K'_m*k'_n
        #the diagonal entry of K'_m*k'_n
        
        for m in range (newMh):
            for n in range (newMh):
                if m!=n:
                    diag_quadratick [m, n] = dict_val[m]*dict_val[n]
                    diag_quadratick [n, m] = dict_val[m]*dict_val[n]
        
        
        T = np.zeros ((sizeh**2, newMh))      
        
        #the first M-1 row
        for i in range (1, sizeh**2):
            
            if i % sizeh != 0 : 
                col = i % sizeh
                row = math.floor(i/sizeh) + 1
            else:
                col = sizeh
                row = i/sizeh
            if row < col :
                ind = int((col*(col-1) + 2*row)/2)
                T[i-1, ind-1] = 1
            elif row >= col:
                ind= int((row*(row-1) + 2*col)/2)
                T[i-1, ind-1] = 1
        
        #the last row
        for j in range (1, newMh+1):
            T[sizeh**2-1, j-1] = -2
            for i in range (sizeh ):
                if j == (1+i)*i/2 :
                    T[sizeh**2-1, j-1] = -1
        
        t = np.zeros((sizeh**2 -1, 1))
        lastDigit = np.array([1])
        t = np.vstack((t,lastDigit))

    return (newMh, newk, newk0, dict_val, diag_quadratick, T, t, r_vec, rt_vec)  


def computeDiagSigma_vec (beta, diagXtX_vec, diagnewterm_vec, P_vec, mk_vec, B_vec, Babs_vec, Btabs_vec):
    
    
    #CalculDiagSigma -- Compute the diag element of the covariance matrix
    Bmksquare_vec = B_vec(mk_vec)**2#||Dm||**2
    Diagnewterm_vec=beta * diagnewterm_vec + beta * diagXtX_vec
    u_vec = P_vec
    diagBtuB_vec= Btabs_vec(u_vec) #DtuD
    diagInvSigma_vec = Diagnewterm_vec + diagBtuB_vec
    diagSigma_vec = torch.pow(diagInvSigma_vec, -1)
    DtDSigma_vec = Bmksquare_vec+ Babs_vec(diagSigma_vec) #DSigmaDt #optimumlambda

    return (DtDSigma_vec, diagSigma_vec)

def computeDiagSigma_vec_RGB (beta, diagXtX_vec, diagnewterm_vec, P_vec, mk_vec, B_vec, Babs_vec, Btabs_vec):
    
    
    #CalculDiagSigma -- Compute the diag element of the covariance matrix
    #Bmksquare = np.square(B(mk)) #||Dm||**2
    Bmksquare_vec = B_vec(mk_vec)**2
    Diagnewterm_vec=diagnewterm_vec*beta.unsqueeze(-1).unsqueeze(-1) + diagXtX_vec*beta.unsqueeze(-1).unsqueeze(-1)
    u_vec = P_vec
    diagBtuB_vec= Btabs_vec(u_vec) #DtuD
    diagInvSigma_vec = Diagnewterm_vec + diagBtuB_vec
    diagSigma_vec = torch.pow(diagInvSigma_vec, -1)
    DtDSigma_vec = Bmksquare_vec+ Babs_vec(diagSigma_vec) #DSigmaDt #optimumlambda

    return (DtDSigma_vec, diagSigma_vec)
    
def compute_diagnewterm_gaussian_vec (newSigmah_vec,diag_qudratick,sizex,dtype):
    batch = newSigmah_vec.shape[0]
    newMh = newSigmah_vec.shape[1]
    N= sizex**2
    
    diagnewterm1 = 0
    diagnewterm2 = 0
    
    for m in range (newMh):
        diagnewterm1_temp = newSigmah_vec[:,m,m]*diag_qudratick[m,m]
        diagnewterm1 = diagnewterm1+diagnewterm1_temp
        for n in range (m+1):
            diagnewterm2_temp = newSigmah_vec[:,m,n]*diag_qudratick[m,n]
            diagnewterm2 = diagnewterm2_temp + diagnewterm2
    
    diagnewterm_vec = (2*diagnewterm2- diagnewterm1).reshape(batch,1,1)*torch.ones((batch,sizex,sizex)).type(dtype)
    
    return  diagnewterm_vec.type(dtype)



def compute_prodnewterm_gaussian_vec (sizeh,newSigmah_vec,newk,r_vec,rt_vec,vector,symmetry,dtype):
    #vector: batch*sizex^2*1
    newMh = newSigmah_vec.shape[1]
    batch = newSigmah_vec.shape[0]
    N= newk[0].shape[0]
    prodnewterm_vec = torch.zeros ((batch, N, )).type(dtype)
    batch_size = vector.shape[0]
    newmk = torch.zeros((batch_size,newk[0].shape[0],1)).type(dtype)
    newMh = int ( ( (1+ sizeh)* sizeh)/2 -1 )

    if symmetry ==2: #only case coded yet
        m=0
        newkmvector=[]
        for i in range (1, sizeh+1):
            for j in range (1, i+1):
                if i == j:
                    if i != sizeh : #except newk_0
                        ind = (i-1)*sizeh + i 
                        newmk = torch.cat((newmk,vector[:,r_vec[:,ind-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)
                else:
                    ind1 = (i-1)*sizeh+j
                    ind2 = (j-1)*sizeh+i 
                    newmk = torch.cat((newmk,vector[:,r_vec[:,ind1-1]]-vector[:,r_vec[:,sizeh*sizeh-1]] 
                                        +vector[:,r_vec[:,ind2-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)
                m += 1
                        
        newkmvector = newmk[:,:,1:].type(dtype) #batch*sizex^2*M'
  
        m = 0
        for i in range (1, sizeh+1):
            for j in range (1, i+1):
                if i == j:
                    if i != sizeh : #except newk_0
                        ind = (i-1)*sizeh + i 
                        prodnewterm_vec += torch.sum(newSigmah_vec[:,m,:].reshape(batch,1,newMh)*(newkmvector[:,rt_vec[:,ind-1],:]-newkmvector[:,rt_vec[:,sizeh*sizeh-1],:]),2)
                else:
                    ind1 = (i-1)*sizeh+j
                    ind2 = (j-1)*sizeh+i 
                    prodnewterm_vec += torch.sum(newSigmah_vec[:,m,:].reshape(batch,1,newMh)*(newkmvector[:,rt_vec[:,ind1-1],:]-newkmvector[:,rt_vec[:,sizeh*sizeh-1],:] 
                                +newkmvector[:,rt_vec[:,ind2-1],:]-newkmvector[:,rt_vec[:,sizeh*sizeh-1],:]),2)
                m = m+1   
    return prodnewterm_vec


def linear_solver_gaussian_vec(beta, sizex ,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,b_vec,kmax,c_vec,tol,newSigmah_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype) :
    #conjugate gradient algorithm to solve system
    #A xout= b
    #c:init
    #tol: tolerance
    batch = newSigmah_vec.shape[0]
    
    gradhtx_vec= lambda u: u- circshift_tensor_vec(u, 0, -1)
    gradvtx_vec= lambda u: u-circshift_tensor_vec(u, -1, 0)
    B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
    Bt_vec= lambda u: gradhtx_vec(u[:,:sizex, :sizex]) + gradvtx_vec(u[:,sizex:, :sizex])

        
        
    xout_vec=c_vec 
    diagnewterm_vec = compute_diagnewterm_gaussian_vec (newSigmah_vec,diag_qudratick,sizex,dtype)
    prodnewterm_vec= lambda vector: compute_prodnewterm_gaussian_vec(sizeh,newSigmah_vec,newk,r_vec,rt_vec,vector,symmetry,dtype)

    def mvmA1_vec (beta,prodnewterm_vec,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,w_vec,sizex): 
        batch = w_vec.shape[0]
        Xw_vec= X_vec(w_vec)
        RXw_vec = Xw_vec
        gradhx_vec= lambda u: u-circshift_tensor_vec(u, 0, 1)
        gradvx_vec= lambda u: u-circshift_tensor_vec(u, 1, 0)
        B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
        Bw_vec= B_vec(w_vec) 
        PBw_vec= P_vec * Bw_vec
        wvector= torch.transpose(w_vec,1,2).reshape(batch,sizex**2,1)
        v_vec= beta*Xt_vec(RXw_vec) + Bt_vec(PBw_vec) + beta * torch.transpose(prodnewterm_vec(wvector).reshape((batch,sizex,sizex)),1,2)
        return (v_vec)  
    
    mvmAfun_vec= lambda u: mvmA1_vec(beta,prodnewterm_vec,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,u,sizex)

    
    Ax2_vec= mvmAfun_vec (xout_vec)

    
    for k in range (kmax): #Checked the use of k countless times
        
        gx2_vec=Ax2_vec-b_vec 
        Agx2_vec=mvmAfun_vec(gx2_vec) 
        ng2_vec= torch.norm(gx2_vec,dim=(1,2))**2
              
        if k==0: 
            Dk2_vec=-gx2_vec 
            ADk2_vec=-Agx2_vec
        
        else:
            alpha_vec= ng2_vec/ng1_vec
            Dk2_vec=-gx2_vec +alpha_vec.reshape(batch,1,1)*Dk1_vec
            ADk2_vec= -Agx2_vec+ alpha_vec.reshape(batch,1,1)*ADk1_vec
        
        u_vec= torch.sum(-Dk2_vec* gx2_vec,dim=(1,2)) / torch.sum(Dk2_vec* ADk2_vec,dim=(1,2))
        xout_vec = xout_vec + Dk2_vec * u_vec.reshape(batch,1,1)
        Ax2_vec = Ax2_vec + ADk2_vec * u_vec.reshape(batch,1,1)
        ng1_vec= ng2_vec
        Dk1_vec=Dk2_vec
        ADk1_vec= ADk2_vec

    return (xout_vec, diagnewterm_vec) 

def linear_solver_gaussian_vec_RGB(beta, sizex ,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,b_vec,kmax,c_vec,tol,newSigmah_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype) :
    #conjugate gradient algorithm to solve system
    #A xout= b
    #c:init
    #tol: tolerance
    batch = newSigmah_vec.shape[0]
    
    gradhtx_vec= lambda u: u- circshift_tensor_vec(u, 0, -1)
    gradvtx_vec= lambda u: u-circshift_tensor_vec(u, -1, 0)
    B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
    Bt_vec= lambda u: gradhtx_vec(u[:,:sizex, :sizex]) + gradvtx_vec(u[:,sizex:, :sizex])

        
        
    xout_vec=c_vec 

    diagnewterm_vec = compute_diagnewterm_gaussian_vec(newSigmah_vec,diag_qudratick,sizex,dtype)
    prodnewterm_vec= lambda vector: compute_prodnewterm_gaussian_vec(sizeh,newSigmah_vec,newk,r_vec,rt_vec,vector,symmetry,dtype)

    def mvmA1_vec (beta,prodnewterm_vec,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,w_vec,sizex): 
        batch = w_vec.shape[0]
        Xw_vec= X_vec(w_vec)
        RXw_vec = Xw_vec
        gradhx_vec= lambda u: u-circshift_tensor_vec(u, 0, 1)
        gradvx_vec= lambda u: u-circshift_tensor_vec(u, 1, 0)
        B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
        Bw_vec= B_vec(w_vec) 
        PBw_vec= P_vec * Bw_vec
        wvector= torch.transpose(w_vec,1,2).reshape(batch,sizex**2,1)
        v_vec= Xt_vec(RXw_vec)*beta.unsqueeze(-1).unsqueeze(-1) + Bt_vec(PBw_vec) +torch.transpose(prodnewterm_vec(wvector).reshape((batch,sizex,sizex)),1,2)*beta.unsqueeze(-1).unsqueeze(-1)
        return (v_vec)  
    
    mvmAfun_vec= lambda u: mvmA1_vec(beta,prodnewterm_vec,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,u,sizex)

    
    Ax2_vec= mvmAfun_vec (xout_vec)

    
    for k in range (kmax): #Checked the use of k countless times
        
        gx2_vec=Ax2_vec-b_vec
        Agx2_vec=mvmAfun_vec(gx2_vec) 
        ng2_vec= torch.norm(gx2_vec,dim=(1,2))**2
              
        if k==0: 
            Dk2_vec=-gx2_vec 
            ADk2_vec=-Agx2_vec
        
        else:
            alpha_vec= ng2_vec/ng1_vec
            Dk2_vec=-gx2_vec +alpha_vec.reshape(batch,1,1)*Dk1_vec
            ADk2_vec= -Agx2_vec+ alpha_vec.reshape(batch,1,1)*ADk1_vec
        
        u_vec= torch.sum(-Dk2_vec* gx2_vec,dim=(1,2)) / torch.sum(Dk2_vec* ADk2_vec,dim=(1,2))
        xout_vec = xout_vec + Dk2_vec * u_vec.reshape(batch,1,1)
        Ax2_vec = Ax2_vec + ADk2_vec * u_vec.reshape(batch,1,1) 
        ng1_vec= ng2_vec
        Dk1_vec=Dk2_vec
        ADk1_vec= ADk2_vec

    return (xout_vec, diagnewterm_vec) 

    
    
def compute_newkmvector_vec (sizeh,newk,r_vec,vector,symmetry,dtype):
    #k'm*vector
    batch = vector.shape[0]
    newmk = torch.zeros((batch,newk[0].shape[0],1)).type(dtype)
    newMh = int ( ( (1+ sizeh)* sizeh)/2 -1 )
    sizexsquare = vector.shape[1]
    

    if symmetry ==2: #only case coded yet
        m=0
        for i in range (1, sizeh+1):
            for j in range (1, i+1):
                if i == j:
                    if i != sizeh : #except newk_0
                        ind = (i-1)*sizeh + i  
                        newmk = torch.cat((newmk,vector[:,r_vec[:,ind-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)
                else:
                    ind1 = (i-1)*sizeh+j
                    ind2 = (j-1)*sizeh+i 
                    newmk = torch.cat((newmk,vector[:,r_vec[:,ind1-1]]-vector[:,r_vec[:,sizeh*sizeh-1]] 
                                        +vector[:,r_vec[:,ind2-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)

                m += 1
                        
        newkmvector = newmk[:,:,1:]
        newk0vector = vector[:,r_vec[:,sizeh*sizeh-1]].reshape(batch,sizexsquare,1)
        
    return newkmvector,newk0vector[:,:,-1]
    
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-cropx//2
    starty = y//2-cropy//2  
    return img[starty:starty+cropy,startx:startx+cropx]    

class Cropcenter:
    def __init__(self,size):
        self.cropx = size[0]
        self.cropy = size[1]
    def __call__(self,img):
        return crop_center(img,self.cropx,self.cropy)

def crop_center_RGB(img,cropx,cropy):
    y,x,ch = img.shape
    crop_img = np.zeros((cropx,cropy,ch))
    startx = x//2-cropx//2
    starty = y//2-cropy//2  
    for channel in range(ch):
        crop_img[:,:,channel] = img[starty:starty+cropy,startx:startx+cropx,channel]    
    return crop_img

class Cropcenter_RGB:
    def __init__(self,size):
        self.cropx = size[0]
        self.cropy = size[1]
    def __call__(self,img):
        return crop_center_RGB(img,self.cropx,self.cropy)
    
class Initialization():
    def __init__(self,batch,sizex,sizeh,dtype):
        if sizex == 256 and sizeh == 9:# use the saved varaibles
            #prior operator
            self.newMh = int(sio.loadmat('Model_files/useful_tools.mat')['newMh'])
            T = sio.loadmat('Model_files/useful_tools.mat')['T']
            t = sio.loadmat('Model_files/useful_tools.mat')['t']
        else:  # create new variables if using other sizes
            (newMh, newk, newk0, dict_val, diag_quadratick, T, t, r_vec, rt_vec) = CreateKm(sizeh, sizex, 2)
            self.newMh = newMh    
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        self.T_vec = T_.repeat(batch,1,1).type(dtype)
        self.t = t.type(dtype)    
        gradhx_vec= lambda u: u-circshift_tensor_vec(u, 0, 1)
        gradvx_vec= lambda u: u-circshift_tensor_vec(u, 1, 0)
        self.B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
        gradhxabs_vec= lambda u: u+ circshift_tensor_vec(u, 0, 1)
        gradvxabs_vec= lambda u: u+ circshift_tensor_vec(u, 1, 0)
        self.Babs_vec= lambda u: torch.cat((gradhxabs_vec(u), gradvxabs_vec(u)), dim=1)#DuDt
        # initialize lambda and gamma
        self.agamma = 0
        self.bgamma = 0
        self.kappa = 0.5
        self.J = 2
        self.batch = batch
        self.sizex = sizex
        self.sizeh = sizeh
        self.aGammapos = self.agamma+self.sizex*self.sizex/(2*self.kappa)
        self.dtype = dtype
    def f(self,x_blurred,h):
        h_vec,Ch_vec = Initialization_vec(self.batch,self.sizex,self.sizeh,self.newMh,self.T_vec,self.t,h,self.dtype)
        LAMBDAk1_vec = init_LAMBDAk1_vec(self.batch, x_blurred, self.B_vec, self.Babs_vec, self.J, self.sizex, self.sizex, 1, self.dtype)
        bGammapos_vec = self.bgamma+torch.sum(torch.pow(LAMBDAk1_vec, self.kappa),(1,2))
        gamma_vec = self.aGammapos/bGammapos_vec
        gamma_vec = gamma_vec.reshape(self.batch,1,1)
        lambda_vec= 2*self.kappa*(torch.pow(LAMBDAk1_vec, self.kappa-1))
        return self.T_vec,self.t,h_vec,Ch_vec,gamma_vec,lambda_vec 



def Initialization_vec(batch,sizex,sizeh,newMh,T_vec,t,h0_vec,dtype):
    """
    The initialization of z and C_z
    """
    h1 = 1/(5**5)*np.ones((5,5))
    h2 = zero_pad(h1, (sizeh,sizeh), position='center')
    h = h2.reshape(1,sizeh**2,1)
    h_vec_ = torch.tensor(h).repeat(batch,1,1).float().type(dtype)
    h_vec = torch.pinverse(T_vec) @ (h_vec_-t)
    Ch_vec = 1/600*torch.eye(newMh).repeat(batch,1,1).type(dtype)
    return h_vec,Ch_vec

def init_LAMBDAk1_vec (batch, mk_vec, B_vec, Babs_vec, J, M, N, xmax, dtype):
    """
    The initialization of \lambda
    """
    diagSigma_vec = xmax * torch.ones_like(mk_vec) 
    sizex = mk_vec.shape[1]
    Bmksquare_vec = torch.pow(B_vec(mk_vec), 2)
    DdiagSigma_vec = Bmksquare_vec + Babs_vec(diagSigma_vec)

    LAMBDAk1_vec= torch.transpose(DdiagSigma_vec,1,2)
    LAMBDAk1_vec= LAMBDAk1_vec.reshape((batch,M, N*J))
    LAMBDAk1_vec= LAMBDAk1_vec[:,:,:sizex] + LAMBDAk1_vec[:,:, sizex:]
    LAMBDAk1_vec= torch.transpose(LAMBDAk1_vec,1,2)
    LAMBDAk1_vec = LAMBDAk1_vec.type(dtype)

    return LAMBDAk1_vec



# FFT2 and IFFT2
def my_fft2_vec(x):
    #x: batch*sizex*sizex
    batch = x.shape[0]
    size1 = x.shape[1]
    size2 = x.shape[2]
    Z = torch.zeros((batch,size1,size2,2))
    Z[:,:,:,0] = x # the real part
    Y = torch.fft(Z,2)
    Y1 = torch.view_as_complex(Y)
    return Y1

def my_ifft2_vec(x):
    #x: batch*sizex*sizex
    Y = torch.view_as_real(x)
    X = torch.ifft(Y,2)
    X1 = X[:,:,:,0]
    return X1 


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
     
        
def create_testset(name_set,path_groundtruth,path_testset,name_kernel,noise_std_range,im_size):
    """
    Creates blurred test grayscale images using the chosen blur kernel and Gaussian noise standard deviation.
    Parameters
    ----------
        name_set          (str): name of the test set
        path_groundtruth  (str): path to groundtruth images
        path_testset      (str): path to the test set
        name_kernel       (str): name of the blur kernel
        noise_std_range  (list): list of two elements, minimal and maximal pixel values
        im_size   (numpy array): number of rows and columns in the images
    """
    np.random.seed(0)
    torch.manual_seed(1)
    
    
    path_kernel = os.path.join(path_testset,name_kernel,'kernel.mat')# a matlab file
    path_save   = os.path.join(path_testset,name_kernel,name_set)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    # first check if the groundtruth is center cropped or not
    if os.path.exists(os.path.join(path_groundtruth,'cropped1',name_set)):
        print('cropped already')
        transf1         = OpenMat_transf()
        already_cropped = 'yes'
        # 8 random anisotropic Gaussian blur kernels
        for j in range(8):
            data            = MyDataset(folder=os.path.join(path_groundtruth,'cropped1',name_set),transf1=transf1, need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded] = minibatch
                img = np.zeros(im_size)
                img_degraded = np.zeros(im_size)
                blur = np.zeros((9,9))
                img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
                img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
                blur[0:9,0:9] = h[0,0:9,0:9]
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image and blur kernel
        print('Gaussian blur done')        
        # 2 random isotropic Gaussian blur kernels    
        for j in range(2):
            data            = MyDataset(folder=os.path.join(path_groundtruth,'cropped1',name_set),transf1=transf1, need_names='yes',blur_name = j+8,blur_type = 'Gaussian_isotropic',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded] = minibatch
                img = np.zeros(im_size)
                img_degraded = np.zeros(im_size)
                blur = np.zeros((9,9))
                img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
                img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
                blur[0:9,0:9] = h[0,0:9,0:9]
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image and blur kernel    
        print('Isotropic Gaussian blur done')        
       
    else:
        # center-crops the test images to match the input size
        print('not cropped')
        transf1          = transforms.Compose([Cropcenter(im_size),lambda x: torch.from_numpy(x)]) 
        path_save_true   = os.path.join(path_groundtruth,'cropped1',name_set)
        if not os.path.exists(path_save_true):
            os.makedirs(path_save_true)
        already_cropped  = 'no'
        # 8 random anisotropic Gaussian blur kernels
        for j in range(8):
            data  = MyDataset(folder=os.path.join(path_groundtruth,'full1',name_set),transf1=transf1,need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)    
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded] = minibatch
                img = np.zeros(im_size)
                img_degraded = np.zeros(im_size)
                blur = np.zeros((9,9))
                img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
                img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
                blur[0:9,0:9] = h[0,0:9,0:9]
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image and blur
                file_name_true = os.path.join(path_save_true,str(image_name[0])+'.mat')
                sio.savemat(file_name_true,{'image':img})# center-cropped image
        print('Gaussian blur done')             
        # 2 random isotropic Gaussian blur kernels
        for j in range(2):
            data            = MyDataset(folder=os.path.join(path_groundtruth,'full1',name_set),transf1=transf1, need_names='yes',blur_name = j+8,blur_type = 'Gaussian_isotropic',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded] = minibatch
                img = np.zeros(im_size)
                img_degraded = np.zeros(im_size)
                blur = np.zeros((9,9))
                img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
                img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
                blur[0:9,0:9] = h[0,0:9,0:9]
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image and blur kernel
                file_name_true = os.path.join(path_save_true,str(image_name[0])+'.mat')
                sio.savemat(file_name_true,{'image':img})# center-cropped image
        print('Isotropic Gaussian blur done')         
     
            
def create_trainset(path_groundtruth,path_trainset,noise_std_range,im_size):
    """
    Creates blurred training graysclae images using the chosen blur kernel and Gaussian noise standard deviation.
    Parameters
    ----------
        path_groundtruth  (str): path to groundtruth images
        path_trainset     (str): path to the train set
        noise_std_range  (list): list of two elements, minimal and maximal pixel values
        im_size   (numpy array): number of rows and columns in the images
    """
    np.random.seed(0)
    torch.manual_seed(1)
    

    path_save   = os.path.join(path_trainset)
    if not os.path.exists(path_save):
        os.makedirs(path_save) 
       
    # center-crops the test images to match the input size
    print('not cropped')
    transf1          = transforms.Compose([Cropcenter(im_size),lambda x: torch.from_numpy(x)]) 

    already_cropped  = 'no'
    # 8 random anisotropic Gaussian blur kernels
    for j in range(8):
        data  = MyDataset(folder=os.path.join(path_groundtruth),transf1=transf1,need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
        loader  = DataLoader(data, batch_size=1, shuffle=False)

        # tqdm shows the progress
        for minibatch in tqdm(loader,file=sys.stdout):
            [blur_name,h,image_name,x,x_degraded] = minibatch
            if image_name =='.ipynb_checkpoints': continue
            img = np.zeros(im_size)
            img_degraded = np.zeros(im_size)
            blur = np.zeros((9,9))
            img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
            img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
            blur[0:9,0:9] = h[0,0:9,0:9]
            file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
            sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image blur and true image 
    print('Gaussian blur done')             
    # 2 random isotropic Gaussian blur kernel     
    for j in range(2):
        data            = MyDataset(folder=os.path.join(path_groundtruth),transf1=transf1, need_names='yes',blur_name = j+8,blur_type = 'Gaussian_isotropic',noise_std_range = noise_std_range)
        loader  = DataLoader(data, batch_size=1, shuffle=False)
        # tqdm shows the progress
        for minibatch in tqdm(loader,file=sys.stdout):
            [blur_name,h,image_name,x,x_degraded] = minibatch
            img = np.zeros(im_size)
            img_degraded = np.zeros(im_size)
            blur = np.zeros((9,9))
            img[0:im_size[0],0:im_size[1]] = x[0,0:im_size[0],0:im_size[1]]
            img_degraded[0:im_size[0],0:im_size[1]] = x_degraded[0,0:im_size[0],0:im_size[1]]
            blur[0:9,0:9] = h[0,0:9,0:9]
            file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
            sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img}) # degraded image and blur kernel
    print('Isotropic Gaussian blur done')         


    
def create_testset_RGB(name_set,path_groundtruth,path_testset,name_kernel,noise_std_range,im_size):
    """
    Creates blurred test RGB images using the chosen blur kernel and Gaussian noise standard deviation.
    Parameters
    ----------
        name_set          (str): name of the test set
        path_groundtruth  (str): path to groundtruth images
        path_testset      (str): path to the test set
        name_kernel       (str): name of the blur kernel
        noise_std_range  (list): list of two elements, minimal and maximal pixel values
        im_size   (numpy array): number of rows and columns in the images, size 1*2
    """
    np.random.seed(0)
    torch.manual_seed(1)
    
    path_kernel = os.path.join(path_testset,name_kernel,'kernel.mat')# a matlab file
    path_save   = os.path.join(path_testset,name_kernel,name_set)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    if os.path.exists(os.path.join(path_groundtruth,'cropped1_RGB',name_set)):
        print('cropped already')
        transf1         = OpenMat_transf()
        already_cropped = 'yes'
        # 10 random Gaussian blurs
        for j in range(10):
            data            = MyDataset_RGB(folder=os.path.join(path_groundtruth,'cropped1_RGB',name_set),transf1=transf1, need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur kernel
        print('Gaussian blur done')        
        # 2 pre-defined uniform blurs       
        for j in range(2):
            data            = MyDataset_RGB(folder=os.path.join(path_groundtruth,'cropped1_RGB',name_set),transf1=transf1, need_names='yes',blur_name = j+10,blur_type = 'uniform_'+str(j+1),noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur kernel    
        print('Uniform blur done')        
         # 3 random defocus blurs       
        for j in range(3):
            data            = MyDataset_RGB(folder=os.path.join(path_groundtruth,'cropped1_RGB',name_set),transf1=transf1, need_names='yes',blur_name = j+12,blur_type = 'defocus',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                print(file_name_degraded)
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur kernel 
        print('Defocus blur done')         
    else:
        # center-crops the test images to match the input size
        print('not cropped')
        transf1          = transforms.Compose([Cropcenter_RGB(im_size),lambda x: torch.from_numpy(x)]) 
        path_save_true   = os.path.join(path_groundtruth,'cropped1_RGB',name_set)
        if not os.path.exists(path_save_true):
            os.makedirs(path_save_true)
        already_cropped  = 'no'
        for j in range(10):
            data  = MyDataset_RGB(folder=os.path.join(path_groundtruth,'full1_RGB',name_set),transf1=transf1,need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
    
    
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur
                file_name_true = os.path.join(path_save_true,str(image_name[0])+'.mat')
                sio.savemat(file_name_true,{'image':img})# center-cropped image
        print('Gaussian blur done')             
        # 2 pre-defined uniform blurs       
        for j in range(2):
            data            = MyDataset_RGB(folder=os.path.join(path_groundtruth,'full1_RGB',name_set),transf1=transf1, need_names='yes',blur_name = j+10,blur_type = 'uniform_'+str(j+1),noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur kernel
                file_name_true = os.path.join(path_save_true,str(image_name[0])+'.mat')
                sio.savemat(file_name_true,{'image':img})# center-cropped image
        print('Uniform blur done')         
        # 3 random defocus blurs       
        for j in range(3):
            data            = MyDataset_RGB(folder=os.path.join(path_groundtruth,'full1_RGB',name_set),transf1=transf1, need_names='yes',blur_name = j+12,blur_type = 'defocus',noise_std_range = noise_std_range)
            loader  = DataLoader(data, batch_size=1, shuffle=False)
            # tqdm shows the progress
            for minibatch in tqdm(loader,file=sys.stdout):
                [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
                img = np.zeros((im_size[0],im_size[1],3))
                img_degraded = np.zeros((im_size[0],im_size[1],3))
                blur = np.zeros((9,9))
                noise = np.zeros((1))
                img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
                img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
                blur[0:9,0:9] = h[0,0:9,0:9]
                noise[0] = noise_std
                file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
                print(file_name_degraded)
                sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img,'noise_std':noise}) # degraded image and blur kernel
                file_name_true = os.path.join(path_save_true,str(image_name[0])+'.mat')
                sio.savemat(file_name_true,{'image':img})# center-cropped image
        print('Defocus blur done')       
            
def create_trainset_RGB(path_groundtruth,path_trainset,noise_std_range,im_size):
    """
    Creates blurred training color images using the chosen blur kernel and Gaussian noise standard deviation.
    Parameters
    ----------
        path_groundtruth  (str): path to groundtruth images
        path_testset      (str): path to the test set
        noise_std_range  (list): list of two elements, minimal and maximal pixel values
        im_size   (numpy array): number of rows and columns in the images, size 1*2
    """
    np.random.seed(0)
    torch.manual_seed(1)

    path_save   = os.path.join(path_trainset)
    if not os.path.exists(path_save):
        os.makedirs(path_save) 
       
    # center-crops the test images to match the input size
    print('not cropped')
    transf1          = transforms.Compose([Cropcenter_RGB(im_size),lambda x: torch.from_numpy(x)]) 

    already_cropped  = 'no'
    for j in range(10):
        data  = MyDataset_RGB(folder=os.path.join(path_groundtruth),transf1=transf1,need_names='yes',blur_name = j,blur_type = 'Gaussian',noise_std_range = noise_std_range)
        loader  = DataLoader(data, batch_size=1, shuffle=False)


        # tqdm shows the progress
        for minibatch in tqdm(loader,file=sys.stdout):
            [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
            if image_name =='.ipynb_checkpoints': continue
            img = np.zeros((im_size[0],im_size[1],3))
            img_degraded = np.zeros((im_size[0],im_size[1],3))
            blur = np.zeros((9,9))
            noise = np.zeros((1))
            img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
            img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
            blur[0:9,0:9] = h[0,0:9,0:9]
            noise[0] = noise_std
            file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
            sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img, 'noise_std':noise}) # degraded image blur and true image 
    print('Gaussian blur done')             
    # 2 pre-defined uniform blurs       
    for j in range(2):
        data            = MyDataset_RGB(folder=os.path.join(path_groundtruth),transf1=transf1, need_names='yes',blur_name = j+10,blur_type = 'uniform_'+str(j+1),noise_std_range = noise_std_range)
        loader  = DataLoader(data, batch_size=1, shuffle=False)
        # tqdm shows the progress
        for minibatch in tqdm(loader,file=sys.stdout):
            [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
            img = np.zeros((im_size[0],im_size[1],3))
            img_degraded = np.zeros((im_size[0],im_size[1],3))
            blur = np.zeros((9,9))
            noise = np.zeros((1))
            img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
            img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
            blur[0:9,0:9] = h[0,0:9,0:9]
            noise[0] = noise_std
            file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
            sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img, 'noise_std':noise}) # degraded image and blur kernel
    print('Uniform blur done')         
    # 3 random defocus blurs       
    for j in range(3):
        data            = MyDataset_RGB(folder=os.path.join(path_groundtruth),transf1=transf1, need_names='yes',blur_name = j+12,blur_type = 'defocus',noise_std_range = noise_std_range)
        loader  = DataLoader(data, batch_size=1, shuffle=False)
        # tqdm shows the progress
        for minibatch in tqdm(loader,file=sys.stdout):
            [blur_name,h,image_name,x,x_degraded,noise_std] = minibatch
            img = np.zeros((im_size[0],im_size[1],3))
            img_degraded = np.zeros((im_size[0],im_size[1],3))
            blur = np.zeros((9,9))
            noise = np.zeros((1))
            img[0:im_size[0],0:im_size[1],0:3] = x[0,0:im_size[0],0:im_size[1],0:3]
            img_degraded[0:im_size[0],0:im_size[1],0:3] = x_degraded[0,0:im_size[0],0:im_size[1],0:3]
            blur[0:9,0:9] = h[0,0:9,0:9]
            noise[0] = noise_std
            file_name_degraded        = os.path.join(path_save,str(image_name[0])+'_blur'+str(blur_name.numpy())[1:-1]+'.mat')
            print(file_name_degraded)
            sio.savemat(file_name_degraded,{'image':img_degraded, 'h':blur, 'trueimage':img, 'noise_std':noise}) # degraded image and blur kernel
    print('Defocus blur done')    

    