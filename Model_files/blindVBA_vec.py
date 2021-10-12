import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys
from numpy.linalg import inv
from scipy.sparse import spdiags
import time
import pickle
from scipy.sparse import csr_matrix
from Model_files.tools import *
from Model_files.modules import *
from Model_files.compute_grad import *

class cardan(torch.autograd.Function):  

    @staticmethod
    def forward(ctx,y_vec,mk0_vec,h0_vec,gvar,xi,h_vec,Ch_vec,gamma_vec,lambda_vec,KmtK0_dict,KmtKn_dict,dtype,mode_training=True):
        print('gvar is ',gvar)   

        """
	    This is one iteration of blind VBA
        Parameters
        ----------
           y_vec (torch.FloatTensor): the degraded image of size batch*sizex*sizex
           mk0_vec (torch.FloatTensor): the estimtaed image of last iteration of size batch*sizex*sizex
           h0_vec (torch.FloatTensor): the original blur of size batch*sizeh*sizeh
           gvar (torch.FloatTensor): the parameter 
           xi (torch.FloatTensor): the estimated xi of size batch
           h_vec (torch.FloatTensor): the estimated z of last iteration of size batch*M'*1
           Ch_vec (torch.FloatTensor): the estimated cov of z of last iteration of size batch*M'*M'
           gamma_vec (torch.FloatTensor): the estimated gamma of last iteration of size batch
           lambda_vec (torch.FloatTensor): the estimated lambda of last iteration of size batch*sizex*sizex
           dtype: the type of device
           mode_training (bool): indicates if the model is in training (True) or testing (False) (default is True)
        Returns
        -------
            mk_vec (torch.FloatTensor): the estimtaed image of current iteration of size batch*sizex*sizex
            diagSigma_vec (torch.FloatTensor): the estimtaed cov of image of current iteration of size batch*sizex*sizex
            newmh_vec (torch.FloatTensor): the estimated z of current iteration of size batch*M'*1
            newSigmah_vec (torch.FloatTensor): the estimated cov of z of current iteration of size batch*M'*M'
            Gammap_vec (torch.FloatTensor): the estimated gamma of current iteration of size batch
            LAMBDAk_vec (torch.FloatTensor): the estimated lambda of current iteration of size batch*sizex*sizex
        """
        
        #initialize variables
        sizex             = mk0_vec.shape[1]
        sizeh             = h0_vec.shape[1] 
        symmetry          = 2
        Ncg               = 10
        tolcg             = 1e-5
        tol               = 1e-6
        kappa             = 0.5
        agamma            = 0
        bgamma            = 0
        batch             = y_vec.shape[0]
        newMh = int(sio.loadmat('Model_files/useful_tools.mat')['newMh'])
        newk_ = sio.loadmat('Model_files/useful_tools.mat')['newk']
        newk = newk_[0]
        newk0 = sio.loadmat('Model_files/useful_tools.mat')['newk0']
        diag_qudratick = sio.loadmat('Model_files/useful_tools.mat')['diag_qudratick']
        diag_qudratick_gpu = torch.tensor(diag_qudratick).float()
        dict_val = sio.loadmat('Model_files/useful_tools.mat')['dict_val']
        dict_val = torch.tensor(dict_val[:newMh,:],dtype=torch.float32).type(dtype)
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        r_vec = sio.loadmat('Model_files/useful_tools.mat')['r_vec']
        rt_vec = sio.loadmat('Model_files/useful_tools.mat')['rt_vec']
        L = sio.loadmat('Model_files/useful_tools.mat')['L']
        L = torch.tensor(L,dtype = torch.float32)
        

        M = y_vec.shape[1]
        N = y_vec.shape[2]
    
        
        gradhx_vec= lambda u: u-circshift_tensor_vec(u, 0, 1)
        gradvx_vec= lambda u: u-circshift_tensor_vec(u, 1, 0)
        gradhtx_vec= lambda u: u- circshift_tensor_vec(u, 0, -1)
        gradvtx_vec= lambda u: u-circshift_tensor_vec(u, -1, 0)
        B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
        Bt_vec= lambda u: gradhtx_vec(u[:,:M, :N]) + gradvtx_vec(u[:,M:, :N])

        gradhxabs_vec= lambda u: u+ circshift_tensor_vec(u, 0, 1)
        gradvxabs_vec= lambda u: u+ circshift_tensor_vec(u, 1, 0)
        gradhtxabs_vec= lambda u: u+ circshift_tensor_vec(u, 0, -1)
        gradvtxabs_vec= lambda u: u+ circshift_tensor_vec(u, -1, 0)
        Babs_vec= lambda u: torch.cat((gradhxabs_vec(u), gradvxabs_vec(u)), dim=1)#DuDt
        Btabs_vec= lambda u: gradhtxabs_vec(u[:,:M, :N]) +gradvtxabs_vec(u[:,M:, :N]) #DtuD
        
        
        # for individual image
        gradhx= lambda u: u- circshift_tensor(u, 0, 1)
        gradvx= lambda u: u- circshift_tensor(u, 1, 0)
        gradhtx= lambda u: u- circshift_tensor(u, 0, -1)
        gradvtx= lambda u: u- circshift_tensor(u, -1, 0)
        B = lambda u: torch.cat((gradhx(u), gradvx(u)), dim=0) #D
        Bt= lambda u: gradhtx(u[:M, :N]) + gradvtx (u[M:, :N])
        

        mk0_vec = mk0_vec
        Bmk_vec=B_vec(mk0_vec)
        J=int (Bmk_vec.shape[1]/M)
        beta = torch.tensor(1/gvar).type(dtype)
        
        newmuh_vec = (1/(sizeh*sizeh))*torch.ones(batch,newMh).type(dtype)
        
        L = L.type(dtype)
        aGammapos = agamma+M*N/(2*kappa)

        
        newmh_vec = h_vec
        newSigmah_vec = Ch_vec
        Gammap_vec = gamma_vec
        LAMBDAk_vec = lambda_vec
        Xip = xi
        
        # main algorithm loop
        # useful items for updating x 
        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(dtype) 
        t = t.type(dtype)
        mh_vec = T_vec @ newmh_vec+t 
        mhmatrix_vec = mh_vec.reshape((batch,sizeh,sizeh)).cpu().numpy() 
        H_vec = np.zeros((batch,sizex,sizex),dtype=np.complex64) # for saving complex number
        for j in range(batch):
            H_vec[j,:,:] = psf2otf(mhmatrix_vec[j,:,:].T, (sizex,sizex))

        if t.device.type == 'cuda':
            X_vec = lambda u: my_ifft2_vec ( torch.mul(H_vec, my_fft2_vec(u).cuda())) 
            Xt_vec = lambda u: my_ifft2_vec (torch.mul(torch.conj(H_vec), my_fft2_vec(u).cuda()))
            H_vec = torch.tensor(H_vec).cuda()
        else:
            X_vec = lambda u: my_ifft2_vec ( torch.mul(H_vec, my_fft2_vec(u))) 
            Xt_vec = lambda u: my_ifft2_vec (torch.mul(torch.conj(H_vec), my_fft2_vec(u)))   
            H_vec = torch.tensor(H_vec)                                          
        Hsquare_vec = np.zeros((batch,sizex,sizex),dtype=np.complex64)
        for j in range(batch):
            Hsquare_vec[j,:,:] = psf2otf(mhmatrix_vec[j,:,:]**2, (sizex,sizex))
        Hsquare_vec = torch.tensor(Hsquare_vec)
        diagXtX_vec = my_ifft2_vec(torch.conj(Hsquare_vec)*my_fft2_vec(torch.ones(batch,sizex, sizex))).type(dtype)
        P_vec = Gammap_vec * LAMBDAk_vec.repeat(1,J,1) 
        newbx_vec= beta*Xt_vec(y_vec)

        # Update X

        print ('--- update x')

        (mk_vec,diagnewterm_vec)= linear_solver_gaussian_vec(beta,sizex,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,newbx_vec,Ncg,mk0_vec,tolcg,newSigmah_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype)
 
        (DdiagSigma_vec,diagSigma_vec)= computeDiagSigma_vec(beta,diagXtX_vec,diagnewterm_vec,P_vec,mk_vec,B_vec,Babs_vec,Btabs_vec)

        diagSigma_vec_np = diagSigma_vec.cpu().numpy()


        # Update newh
    
        print ('--- update h')
        
        mkvector_vec = torch.transpose(mk_vec,1,2).reshape((batch,sizex**2, 1))

        avector_vec= torch.zeros((batch,newMh, 1)).type(dtype)

        Bmatrix_vec= torch.zeros((batch,newMh, newMh)).type(dtype)
        
        yvector_vec = torch.transpose(y_vec,1,2).reshape(batch,sizex**2)
        Km_mkvec,a_vec = compute_newkmvector_vec(sizeh,newk,r_vec,mkvector_vec,2,dtype)
        trace_tempvec = torch.zeros((batch,1)).type(dtype)
        for j in range(batch):
            A = np.diag(diagSigma_vec_np[j,:,:].flatten('F'))
            trace_tempvec[j] = torch.tensor(np.sum(A[r_vec[:,0],r_vec[:,0]]))
        avector_vec = torch.sum(Km_mkvec.T*(yvector_vec-a_vec).T,1).T+trace_tempvec@dict_val.T
        for m in range(newMh):
            for n in range (m+1):
                quadratic_temp =  torch.sum(Km_mkvec[:,:,m] * Km_mkvec[:,:,n],1)
                Bmatrix_vec[:,m,n]= quadratic_temp + diag_qudratick[m,n]*trace_tempvec.T
                Bmatrix_vec[:,n,m]= Bmatrix_vec[:,m,n]
    

        A3_vec = beta * avector_vec
        L_vec = L.repeat(batch,1,1)
        A4_vec = L_vec@newmuh_vec.unsqueeze(-1)
        bhvector_vec= A3_vec.unsqueeze(-1) + A4_vec*Xip.unsqueeze(-1).unsqueeze(-1)
        A1_vec = beta* Bmatrix_vec
        A2_vec = L_vec 
        invsigmah_vec= A1_vec + A2_vec*Xip.unsqueeze(-1).unsqueeze(-1)
        invsigmah_vec = invsigmah_vec
        newSigmah_vec= torch.inverse(invsigmah_vec) 
        newmh_vec= newSigmah_vec @ bhvector_vec

        
        # Update LAMBDAk

        print('--- update lambda')

        
        LAMBDAk1_vec = torch.sum(DdiagSigma_vec.reshape(batch,J,M,N),1)
        LAMBDAk_vec= 2*kappa*(torch.pow(LAMBDAk1_vec, kappa-1))
 
     
        # Update gamma
        print ('--- update gamma')

        bGammapos_vec = bgamma + torch.sum (torch.pow(LAMBDAk1_vec, kappa),(1,2))
        Gammap_vec = aGammapos/ bGammapos_vec
        Gammap_vec = Gammap_vec.reshape(batch,1,1)

        
        if mode_training==True:
            print('It is saved')
            ctx.save_for_backward(Xip,beta,A1_vec,A2_vec,A3_vec,A4_vec,bGammapos_vec,h0_vec,h_vec,newmh_vec,Ch_vec,newSigmah_vec,diagSigma_vec,mk0_vec,mkvector_vec,yvector_vec,lambda_vec,LAMBDAk1_vec,gamma_vec,Gammap_vec,P_vec)
            # no gradients for these outputs
            ctx.mark_non_differentiable(mk_vec)
            ctx.mark_non_differentiable(diagSigma_vec)
            ctx.KmtKn_dict = KmtKn_dict
            ctx.KmtK0_dict = KmtK0_dict

        return mk_vec,diagSigma_vec,newmh_vec,newSigmah_vec,Gammap_vec,LAMBDAk_vec
        
    
    @staticmethod
    def backward(ctx, grad_outputx, grad_outputCx, grad_outputh, grad_outputCh, grad_outputgamma, grad_outputlambda):#grad for outputs

        print('This backward is used!')
        grad_outputx = grad_outputx.data
        grad_outputCx = grad_outputCx.data
        grad_outputh    = grad_outputh.data
        grad_outputCh    = grad_outputCh.data
        grad_outputlambda = grad_outputlambda.data
        grad_outputgamma = grad_outputgamma.data
        
        KmtKn_dict = ctx.KmtKn_dict
        KmtK0_dict = ctx.KmtK0_dict
        
        Xip,beta,A1_vec,A2_vec,A3_vec,A4_vec,bGammapos_vec,h0_vec,h_vec,newmh_vec,Ch_vec,newSigmah_vec,diagSigma_vec,mk0_vec,mkvector_vec,yvector_vec,lambda_vec,LAMBDAk1_vec,gamma_vec,Gammap_vec,P_vec    = ctx.saved_tensors
        Xip = Xip
        batch             = grad_outputh.shape[0]
        sizex             = grad_outputx.shape[1]
        sizeh             = h0_vec.shape[1]
        symmetry          = 2
        Ncg               = 10
        tolcg             = 1e-5
        tol               = 1e-6
        kappa             = 0.5
        agamma            = 0
        bgamma            = 0
        
        if h_vec.device.type == 'cuda':
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
            
        
        newMh = int(sio.loadmat('Model_files/useful_tools.mat')['newMh'])
        newk_ = sio.loadmat('Model_files/useful_tools.mat')['newk']
        newk = newk_[0]
        newk0 = sio.loadmat('Model_files/useful_tools.mat')['newk0']
        diag_qudratick = sio.loadmat('Model_files/useful_tools.mat')['diag_qudratick']
        diag_qudratick_gpu = torch.tensor(diag_qudratick).type(dtype)
        dict_val = sio.loadmat('Model_files/useful_tools.mat')['dict_val']
        dict_val = torch.tensor(dict_val[:newMh,:],dtype=torch.float32).type(dtype)
        T = sio.loadmat('Model_files/useful_tools.mat')['T']
        t = sio.loadmat('Model_files/useful_tools.mat')['t']
        T = torch.tensor(T,dtype = torch.float32)
        t = torch.tensor(t,dtype = torch.float32)
        r_vec = sio.loadmat('Model_files/useful_tools.mat')['r_vec']
        rt_vec = sio.loadmat('Model_files/useful_tools.mat')['rt_vec']

        yvector_vec = yvector_vec.reshape(batch,sizex**2,1)

        gradhx_vec= lambda u: u-circshift_tensor_vec(u, 0, 1)
        gradvx_vec= lambda u: u-circshift_tensor_vec(u, 1, 0)
        gradhtx_vec= lambda u: u- circshift_tensor_vec(u, 0, -1)
        gradvtx_vec= lambda u: u-circshift_tensor_vec(u, -1, 0)
        B_vec = lambda u: torch.cat((gradhx_vec(u), gradvx_vec(u)), dim=1) #D
        Bt_vec= lambda u: gradhtx_vec(u[:,:M, :N]) + gradvtx_vec(u[:,M:, :N])

        gradhxabs_vec= lambda u: u+ circshift_tensor_vec(u, 0, 1)
        gradvxabs_vec= lambda u: u+ circshift_tensor_vec(u, 1, 0)
        gradhtxabs_vec= lambda u: u+ circshift_tensor_vec(u, 0, -1)
        gradvtxabs_vec= lambda u: u+ circshift_tensor_vec(u, -1, 0)
        Babs_vec= lambda u: torch.cat((gradhxabs_vec(u), gradvxabs_vec(u)), dim=1)#DuDt
        Btabs_vec= lambda u: gradhtxabs_vec(u[:,:M, :N]) +gradvtxabs_vec(u[:,M:, :N]) #DtuD
        

        # load the dictionary
        with open('KmtK0_dict.txt', 'rb') as handle:
            KmtK0_dict = pickle.loads(handle.read())
        with open('KmtKn_dict.txt', 'rb') as handle:
            KmtKn_dict = pickle.loads(handle.read())

        # load the dictionary
        with open('sDv_dict.txt', 'rb') as handle:
            sDv_dict_ = pickle.loads(handle.read())
        with open('sDh_dict.txt', 'rb') as handle:
            sDh_dict_ = pickle.loads(handle.read())

        row_v = sDv_dict_['row']
        col_v = sDv_dict_['col']
        value_v = sDv_dict_['value']
        sDv_ = csr_matrix((value_v, (row_v, col_v)), shape=(sizex**2, sizex**2))
        row_h = sDh_dict_['row']
        col_h = sDh_dict_['col']
        value_h = sDh_dict_['value']
        sDh_ = csr_matrix((value_h, (row_h, col_h)), shape=(sizex**2, sizex**2))

        T_ = torch.zeros(1,T.shape[0],T.shape[1])
        T_[0,:,:] = T
        T_vec = T_.repeat(batch,1,1).type(dtype)
        t = t.type(dtype)
        mh_vec = T_vec @ h_vec+t 
        mhmatrix_vec = mh_vec.reshape((batch,sizeh,sizeh)).cpu().numpy() 

        H_vec = np.zeros((batch,sizex,sizex),dtype=np.complex128) # for saving complex number
        for j in range(batch):
            H_vec[j,:,:] = psf2otf(mhmatrix_vec[j,:,:].T, (sizex,sizex)) 

        if h_vec.device.type == 'cuda':
            X_vec = lambda u: my_ifft2_vec ( torch.mul(H_vec, my_fft2_vec(u).cuda())) 
            Xt_vec = lambda u: my_ifft2_vec (torch.mul(torch.conj(H_vec), my_fft2_vec(u).cuda()))
            H_vec = torch.tensor(H_vec).cuda()
        else:
            X_vec = lambda u: my_ifft2_vec ( torch.mul(H_vec, my_fft2_vec(u))) 
            Xt_vec = lambda u: my_ifft2_vec (torch.mul(torch.conj(H_vec), my_fft2_vec(u)))
            H_vec = torch.tensor(H_vec)


        invsigmah_vec= torch.inverse(A1_vec + A2_vec*Xip.unsqueeze(-1).unsqueeze(-1))
        bhvector_vec= A3_vec.unsqueeze(-1) + A4_vec*Xip.unsqueeze(-1).unsqueeze(-1)
        grad_h_xi        = invsigmah_vec@(-A2_vec@invsigmah_vec@bhvector_vec+A4_vec)
        grad_Ch_xi        = -invsigmah_vec@A2_vec@invsigmah_vec 
        
        grad_outputh = grad_outputh.type(dtype)
        grad_outputCh = grad_outputCh.type(dtype)
        grad_outputlambda = grad_outputlambda.type(dtype)
        grad_outputgamma = grad_outputgamma.type(dtype)
        grad_outputgamma = grad_outputgamma.reshape(batch,1,1)
        # grad wrt inputs
        # grad of outputs wrt xi
        grad_C_xi_vec        = torch.sum(grad_outputh*grad_h_xi,dim=(1,2)) + torch.sum(grad_outputCh*grad_Ch_xi,dim=(1,2))#1
        print('BP for xi finished')
        
        # grad of outputs wrt h,Ch,lambda,gamma
        aGammapos = agamma+sizex**2/(2*kappa)
        # saving for BP
        settings = [sizeh,sizex,newMh,batch,kappa,aGammapos,bGammapos_vec,beta,Ncg,tolcg,symmetry]
        constant = [diag_qudratick,diag_qudratick_gpu,newk,r_vec,rt_vec,KmtK0_dict,KmtKn_dict,sDh_,sDv_,X_vec,Xt_vec,B_vec,Bt_vec,Babs_vec,row_v,col_v,value_v,row_h,col_h,value_h]
        variables = [h_vec,newmh_vec,Ch_vec,newSigmah_vec,diagSigma_vec,mk0_vec,mkvector_vec,yvector_vec,lambda_vec,LAMBDAk1_vec,gamma_vec,Gammap_vec,P_vec]


        grad = [grad_outputx,grad_outputCx,grad_outputh,grad_outputCh,grad_outputlambda,grad_outputgamma]

        grad_C_h_vec,grad_C_Ch_vec,grad_C_lambda_vec,grad_C_gamma_vec = compute_grad(dtype, settings, constant, grad, variables)

        grad_input_h        = Variable(grad_C_h_vec.type(dtype),requires_grad=True)
        grad_input_Ch       = Variable(grad_C_Ch_vec.type(dtype),requires_grad=True)
        grad_input_lambda   = Variable(grad_C_lambda_vec.type(dtype),requires_grad=True)
        grad_input_gamma    = Variable(grad_C_gamma_vec.type(dtype),requires_grad=True)
        grad_input_xi       = Variable(grad_C_xi_vec.type(dtype),requires_grad=True)

        print('BP finished')


        return None,None,None,None,grad_input_xi, grad_input_h, grad_input_Ch, grad_input_gamma, grad_input_lambda,None,None,None,None
