import torch
import time
import numpy as np
from Model_files.tools import *
from Model_files.grad import *
from Model_files.modules import *
from torch.nn.modules.loss import _Loss
import gc
import scipy

def compute_grad(dtype, settings, constant, grad, variables):
    # settings
    sizeh = settings[0]
    sizex = settings[1]
    newMh = settings[2]
    batch = settings[3]
    kappa = settings[4]
    aGammapos = settings[5]
    bGammapos_vec = settings[6]
    beta = settings[7]
    Ncg = settings[8]
    tolcg = settings[9]
    symmetry = settings[10]
    # constant
    diag_qudratick = constant[0]
    diag_qudratick_gpu = constant[1]
    newk = constant[2]
    r_vec = constant[3]
    rt_vec = constant[4]
    KmtK0_dict = constant[5]
    KmtKn_dict = constant[6]
    sDh_ = constant[7]
    sDv_ = constant[8]
    X_vec = constant[9]
    Xt_vec = constant[10]
    B_vec = constant[11]
    Bt_vec = constant[12]
    Babs_vec = constant[13]
    row_v = constant[14]
    col_v = constant[15]
    value_v = constant[16]
    row_h = constant[17]
    col_h = constant[18]
    value_h = constant[19]
    # variables
    h_vec = variables[0]#the old newmh
    newmh_vec = variables[1]#the new newmh
    Ch_vec = variables[2]#the old sigmah
    newSigmah_vec = variables[3]#the new sigmah
    diagSigma_vec = variables[4]#the new Cx
    mk0_vec = variables[5]#the old x
    mkvector_vec = variables[6]#the new x
    yvector_vec = variables[7]
    lambda_vec = variables[8]#the old lambda
    LAMBDAk1_vec = variables[9]#the new lambda
    gamma_vec = variables[10]#the old gamma
    Gammap_vec  = variables[11]#the new gamma
    P_vec = variables[12]#the old P_vec
    # grad
    grad_outputx = grad[0]
    grad_outputCx = grad[1]
    grad_outputh = grad[2]
    grad_outputCh = grad[3]
    grad_outputlambda = grad[4]
    grad_outputgamma = grad[5]
    
    print('Data loaded successfully')
    
    if (torch.norm(grad_outputCh)+torch.norm(grad_outputlambda)+torch.norm(grad_outputgamma)).cpu().numpy() == 0:
        if torch.norm(grad_outputx).cpu().numpy() == 0:#last layer of the greedy approach:
            print('last layer of greedy approach')
            # grad_h_h
            ind = np.array((0,2,5,9,14,20,27,35)) # the ind for diagonal entry of h'
            I1 = torch.ones(batch,newMh,1).type(dtype)
            I2 = torch.ones(batch,newMh,1).type(dtype)
            I2[:,ind] = 0
            Isum1_vec = I1+I2

            I1 = -2*torch.ones(batch,newMh,1).type(dtype)
            I2 = -2*torch.ones(batch,newMh,1).type(dtype)
            I2[:,ind] = 0
            Isum2_vec = I1+I2

            vec = torch.sum(diag_qudratick_gpu[:newMh,:newMh]*newmh_vec,dim=1).unsqueeze(-1)

            A1_vec = newSigmah_vec@vec

            A2_vec = newSigmah_vec@Isum1_vec
            grad_g4_f0_vec = -beta*A1_vec+beta*A2_vec
            

            Kmkn_vec = 2*torch.sum(h_vec*diag_qudratick_gpu[:newMh,:newMh],dim=1).unsqueeze(-1)*h_vec
            
            grad_f0_h_vec = -beta*(Kmkn_vec+Isum2_vec)

            grad_g4_h_vec_p1 = (grad_g4_f0_vec@grad_f0_h_vec.reshape(batch,1,newMh))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)
           
            grad_g4_f2_vec,grad_g3_f2_vec = compute_grad_h_x(beta,sizeh,mkvector_vec,yvector_vec,newmh_vec,newSigmah_vec,newk,r_vec,rt_vec,symmetry,dtype)
          
            grad_C_x_vec_h = torch.transpose((torch.transpose(grad_g4_f2_vec,1,2)@grad_outputh).reshape(batch,sizex,sizex),1,2)
          
            grad_C_h_vec_p1_h = torch.transpose(grad_g4_h_vec_p1,1,2)@grad_outputh 
            
            
            # grad_h_Ch
            grad_f0_Ch_vec = -beta*diag_qudratick_gpu[:newMh,:newMh]
            grad_g4_Ch_vec = (grad_g4_f0_vec@grad_f0_Ch_vec.reshape(1,newMh**2))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)
            grad_C_Ch_vec_p1_h = torch.transpose((torch.transpose(grad_g4_Ch_vec,1,2)@grad_outputh).reshape(batch,newMh,newMh),1,2)

      
            # grad_h_lambda
            diagBtuB_vec = Babs_vec(diagSigma_vec**2)#DSigmaDt
            grad_g5_f0_vec = torch.sum(diagBtuB_vec.reshape(batch,2,sizex,sizex),1)
            grad_f0_lambda_vec = -2*kappa*(kappa-1)*gamma_vec*torch.pow(lambda_vec,kappa-2)
            grad_C_lambda_vec_p1_h = torch.sum(grad_outputh*grad_g4_f0_vec)*grad_g5_f0_vec*grad_f0_lambda_vec 
      
            # grad_h_gamma
            lambda_vec1 = 2*kappa*torch.pow(lambda_vec,kappa-1)
            grad_f0_gamma_vec = -torch.sum(lambda_vec1*torch.sum(diagBtuB_vec.reshape(batch,2,sizex,sizex),1),dim=(1,2))
            grad_C_gamma_vec_p1_h =  torch.sum(grad_outputh*grad_g4_f0_vec)*grad_f0_gamma_vec 

            # grad_C_h_vec_p2

         

            grad_C_x_vec = grad_C_x_vec_h

            grad_C_A_vec,none= linear_solver_gaussian_vec(beta,sizex,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,grad_C_x_vec,Ncg,mk0_vec,tolcg,Ch_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype)

            grad_C_h_vec_p2,grad_C_Ch_vec_p2 = compute_grad_C_h_vec(beta,newMh,mkvector_vec,h_vec,grad_C_A_vec,KmtK0_dict,KmtKn_dict,dtype)

            grad_C_h_vec_p2 = grad_C_h_vec_p2.reshape(batch,newMh,1)

            # grad_C_h

            grad_C_h_vec_p1 = grad_C_h_vec_p1_h

            grad_C_h_vec = grad_C_h_vec_p1 + grad_C_h_vec_p2
            ############################################################
 
            # grad_C_Ch
     
            grad_C_Ch_vec_p1 = grad_C_Ch_vec_p1_h

            grad_C_Ch_vec = grad_C_Ch_vec_p1 + grad_C_Ch_vec_p2

            #grad_C_lambda
  
            grad_C_lambda_vec_p2,grad_C_gamma_vec_p2 = compute_grad_C_lambda_vec(kappa,mkvector_vec,grad_C_A_vec,lambda_vec,sDh_,sDv_,gamma_vec,dtype)

            grad_C_lambda_vec_p1 = grad_C_lambda_vec_p1_h

            grad_C_lambda_vec = grad_C_lambda_vec_p1 + grad_C_lambda_vec_p2
            #################################################################################

            # grad_C_gamma
    
            grad_C_gamma_vec_p1 = grad_C_gamma_vec_p1_h

            grad_C_gamma_vec = grad_C_gamma_vec_p1 + grad_C_gamma_vec_p2

            return grad_C_h_vec,grad_C_Ch_vec,grad_C_lambda_vec,grad_C_gamma_vec.reshape(batch,1,1)
        
        elif torch.norm(grad_outputh).cpu().numpy() == 0:#last layer of N-N training
            print('the last layer of N-N training')
            grad_C_x_vec = grad_outputx

            grad_C_A_vec,none= linear_solver_gaussian_vec(beta,sizex,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,grad_C_x_vec,Ncg,mk0_vec,tolcg,Ch_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype) 

            grad_C_h_vec_p2,grad_C_Ch_vec_p2 = compute_grad_C_h_vec(beta,newMh,mkvector_vec,h_vec,grad_C_A_vec,KmtK0_dict,KmtKn_dict,dtype) 

            grad_C_h_vec_p2 = grad_C_h_vec_p2.reshape(batch,newMh,1)

            # grad_C_h

            grad_C_h_vec_p1 = 0

            grad_C_h_vec = grad_C_h_vec_p1 + grad_C_h_vec_p2
            ############################################################

            # grad_C_Ch

            grad_C_Ch_vec_p1 = 0

            grad_C_Ch_vec = grad_C_Ch_vec_p1 + grad_C_Ch_vec_p2

            #grad_C_lambda

            grad_C_lambda_vec_p2,grad_C_gamma_vec_p2 = compute_grad_C_lambda_vec(kappa,mkvector_vec,grad_C_A_vec,lambda_vec,sDh_,sDv_,gamma_vec,dtype)

            grad_C_lambda_vec_p1 = 0

            grad_C_lambda_vec = grad_C_lambda_vec_p1 + grad_C_lambda_vec_p2
            #################################################################################

            # grad_C_gamma
            grad_C_gamma_vec_p1 = 0

            grad_C_gamma_vec = grad_C_gamma_vec_p1 + grad_C_gamma_vec_p2

            return grad_C_h_vec,grad_C_Ch_vec,grad_C_lambda_vec,grad_C_gamma_vec.reshape(batch,1,1)
    else: # the other layers applied to both greedy approach and N-N training 
        
        # grad_h_h
        ind = np.array((0,2,5,9,14,20,27,35)) # the ind for diagonal entry of h'
        I1 = torch.ones(batch,newMh,1).type(dtype)
        I2 = torch.ones(batch,newMh,1).type(dtype)
        I2[:,ind] = 0
        Isum1_vec = I1+I2

        I1 = -2*torch.ones(batch,newMh,1).type(dtype)
        I2 = -2*torch.ones(batch,newMh,1).type(dtype)
        I2[:,ind] = 0
        Isum2_vec = I1+I2

        vec = torch.sum(diag_qudratick_gpu[:newMh,:newMh]*newmh_vec,dim=1).unsqueeze(-1)

        A1_vec = newSigmah_vec@vec

        A2_vec = newSigmah_vec@Isum1_vec

        grad_g4_f0_vec = -beta*A1_vec+beta*A2_vec

        Kmkn_vec = 2*torch.sum(h_vec*diag_qudratick_gpu[:newMh,:newMh],dim=1).unsqueeze(-1)*h_vec

        grad_f0_h_vec = -beta*(Kmkn_vec+Isum2_vec)
        
        #grad_h_h Eq.(139)

        grad_g4_h_vec_p1 = (grad_g4_f0_vec@grad_f0_h_vec.reshape(batch,1,newMh))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)
     
        grad_g4_f2_vec,grad_g3_f2_vec = compute_grad_h_x(beta,sizeh,mkvector_vec,yvector_vec,newmh_vec,newSigmah_vec,newk,r_vec,rt_vec,symmetry,dtype)
   
        grad_C_x_vec_h = torch.transpose((torch.transpose(grad_g4_f2_vec,1,2)@grad_outputh).reshape(batch,sizex,sizex),1,2)
       
        grad_C_x_vec_Ch = ((grad_outputCh.type(dtype).reshape(batch,1,newMh**2))@(grad_g3_f2_vec.reshape(batch,newMh**2,sizex**2))).reshape(batch,sizex,sizex) 
       
        grad_C_h_vec_p1_h = torch.transpose(grad_g4_h_vec_p1,1,2)@grad_outputh 
        
        # grad_Ch_h
        grad_g3_f0_vec = -beta*(newSigmah_vec@diag_qudratick_gpu[:newMh,:newMh]@newSigmah_vec)
        
        grad_g3_h_vec =(grad_g3_f0_vec.reshape(batch,newMh**2,1)@grad_f0_h_vec.reshape(batch,1,newMh))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)

        grad_C_h_vec_p1_Ch = torch.transpose(grad_outputCh.reshape(batch,1,newMh**2)@grad_g3_h_vec,1,2)
        
        # grad_lambda_h
        diagBtuB_vec = Babs_vec(diagSigma_vec**2)#DSigmaDt
        grad_g5_f0_vec = torch.sum(diagBtuB_vec.reshape(batch,2,sizex,sizex),1)
        grad_C_h_vec_p1_lambda =  torch.sum(grad_outputlambda*grad_g5_f0_vec)*grad_f0_h_vec 
   
        #grad_gamma_h
        grad_f6_g5_vec = -(aGammapos*kappa*torch.pow(LAMBDAk1_vec, kappa-1))/((bGammapos_vec**2).reshape(batch,1,1))
        grad_f6_f0_vec = grad_f6_g5_vec*grad_g5_f0_vec
        grad_C_h_vec_p1_gamma =  torch.sum(grad_outputgamma.reshape(batch,1,1)*grad_f6_f0_vec)*grad_f0_h_vec   
    
        grad_C_gamma_lambda_vec = grad_outputgamma*(torch.transpose(grad_f6_g5_vec,1,2).reshape(batch,1,sizex**2))
        grad_C_lambda_vec = grad_C_gamma_lambda_vec+torch.transpose(grad_outputlambda,1,2).reshape(batch,1,sizex**2)
        a_np_vec = mkvector_vec.reshape(batch,1,sizex**2).cpu().numpy()
        grad_C_x_vec_lambda = torch.zeros(batch,sizex,sizex).type(dtype)
        for j in range(batch):
            a_np = a_np_vec[j,:,:]
            f2DhtDh = (a_np*sDh_)*(grad_C_lambda_vec[j,:,:].cpu().numpy())
            f2DvtDv = (a_np*sDv_)*(grad_C_lambda_vec[j,:,:].cpu().numpy())
            grad_C_x_vec_lambda[j,:,:] = torch.tensor(2*(f2DhtDh*sDh_+f2DvtDv*sDv_)).type(dtype).reshape(sizex,sizex).T#contain lambda and gamma 
  
        # grad_C_h_vec_p2

        grad_C_x_vec = grad_C_x_vec_h + grad_C_x_vec_Ch.type(dtype) + grad_C_x_vec_lambda 

        grad_C_A_vec,none= linear_solver_gaussian_vec(beta,sizex,sizeh,X_vec,Xt_vec,B_vec,Bt_vec,P_vec,grad_C_x_vec,Ncg,mk0_vec,tolcg,Ch_vec,newk,diag_qudratick,r_vec,rt_vec,symmetry,dtype) 

        grad_C_h_vec_p2,grad_C_Ch_vec_p2 = compute_grad_C_h_vec(beta,newMh,mkvector_vec,h_vec,grad_C_A_vec,KmtK0_dict,KmtKn_dict,dtype)

        grad_C_h_vec_p2 = grad_C_h_vec_p2.reshape(batch,newMh,1)
      
        # grad_C_h

        grad_C_h_vec_p1 = grad_C_h_vec_p1_h + grad_C_h_vec_p1_Ch + grad_C_h_vec_p1_lambda + grad_C_h_vec_p1_gamma

        grad_C_h_vec = grad_C_h_vec_p1 + grad_C_h_vec_p2
        ############################################################
        
        # grad_h_Ch

        grad_f0_Ch_vec = -beta*diag_qudratick_gpu[:newMh,:newMh]

        grad_g4_Ch_vec = (grad_g4_f0_vec@grad_f0_Ch_vec.reshape(1,newMh**2))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)

        grad_C_Ch_vec_p1_h = torch.transpose((torch.transpose(grad_g4_Ch_vec,1,2)@grad_outputh).reshape(batch,newMh,newMh),1,2)

      
        # grad_Ch_Ch

        grad_g3_Ch_vec_p1 = (grad_g3_f0_vec.reshape(batch,newMh**2,1)@grad_f0_Ch_vec.reshape(1,newMh**2))*torch.sum(diagSigma_vec**2,dim=(1,2)).reshape(batch,1,1)

        grad_C_Ch_vec_p1_Ch = torch.transpose((torch.transpose(grad_outputCh,1,2).reshape(batch,1,newMh**2)@grad_g3_Ch_vec_p1).reshape(batch,newMh,newMh),1,2)
        
        # grad_lambda_Ch
        grad_C_Ch_vec_p1_lambda =  torch.sum(grad_outputlambda*grad_g5_f0_vec)*(grad_f0_Ch_vec.repeat(batch,1,1)) 
    
        # grad_gamma_Ch
        grad_C_Ch_vec_p1_gamma =  torch.sum(grad_outputgamma.reshape(batch,1,1)*grad_f6_f0_vec)*(grad_f0_Ch_vec.repeat(batch,1,1)) 
       
        # grad_C_Ch

        grad_C_Ch_vec_p1 = grad_C_Ch_vec_p1_h + grad_C_Ch_vec_p1_Ch + grad_C_Ch_vec_p1_lambda + grad_C_Ch_vec_p1_gamma 
   
        grad_C_Ch_vec = grad_C_Ch_vec_p1 + grad_C_Ch_vec_p2
        #######################################################################

        # grad_h_lambda
        grad_f0_lambda_vec = -2*kappa*(kappa-1)*gamma_vec*torch.pow(lambda_vec,kappa-2)
        grad_C_lambda_vec_p1_h = torch.sum(grad_outputh*grad_g4_f0_vec)*grad_g5_f0_vec*grad_f0_lambda_vec 
      
        # grad_Ch_lambda
        grad_C_lambda_vec_p1_Ch = torch.sum(grad_outputCh*grad_g3_f0_vec)*grad_g5_f0_vec*grad_f0_lambda_vec 
       
        grad_outputlambda_sqrt = torch.transpose(torch.pow(grad_outputlambda+grad_outputgamma.reshape(batch,1,1)*grad_f6_g5_vec,0.5),1,2).reshape(batch,1,sizex**2) #sqrt(lambda) contain lambda and gamma
        grad_lambda_sqrt = torch.transpose(torch.pow(lambda_vec,0.5*(kappa-1)),1,2).reshape(batch,1,sizex**2) #sqrt(lambda)
        grad_C_lambda_f0 = torch.zeros(batch,sizex,sizex).type(dtype)
        grad_C_gamma_f0 = torch.zeros(batch,sizex,sizex).type(dtype)
        for j in range(batch):
            temp_lambda_sqrt_v = grad_outputlambda_sqrt[j,:,row_v].cpu().numpy().squeeze(0)
            temp_lambda_sqrt_h = grad_outputlambda_sqrt[j,:,row_h].cpu().numpy().squeeze(0)
            value_v_lambda = value_v*temp_lambda_sqrt_v
            value_h_lambda = value_h*temp_lambda_sqrt_h
            sDv_lambda = csr_matrix((value_v_lambda, (row_v, col_v)), shape=(sizex**2, sizex**2))
            sDh_lambda = csr_matrix((value_h_lambda, (row_h, col_h)), shape=(sizex**2, sizex**2))
            temp_matrix = sDv_.T@sDv_+sDh_.T@sDh_
            grad_C_lambda_f0[j,:,:] = torch.from_numpy(np.diag(temp_matrix.todense())).type(dtype).reshape(sizex,sizex).T
            temp_lambda_sqrt_v = grad_lambda_sqrt[j,:,row_v].cpu().numpy().squeeze(0)
            temp_lambda_sqrt_h = grad_lambda_sqrt[j,:,row_h].cpu().numpy().squeeze(0)
            value_v_lambda = value_v*temp_lambda_sqrt_v
            value_h_lambda = value_h*temp_lambda_sqrt_h
            sDv_lambda = csr_matrix((value_v_lambda, (row_v, col_v)), shape=(sizex**2, sizex**2))
            sDh_lambda = csr_matrix((value_h_lambda, (row_h, col_h)), shape=(sizex**2, sizex**2))
            temp_matrix = sDv_.T@sDv_+sDh_.T@sDh_
            grad_C_gamma_f0[j,:,:] = torch.from_numpy(np.diag(temp_matrix.todense())).type(dtype).reshape(sizex,sizex).T

       


        diagBtuB_vec1 = Babs_vec((diagSigma_vec**2)*grad_C_lambda_f0)#DSigmaDt
        diagBtuB_vec2 = Babs_vec((diagSigma_vec**2)*grad_C_gamma_f0)#DSigmaDt
        grad_g5_f0_vec1 = torch.sum(diagBtuB_vec1.reshape(batch,2,sizex,sizex),1)
        grad_g5_f0_vec2 = -2*kappa*torch.sum(diagBtuB_vec2.reshape(batch,2,sizex,sizex),1)
        # grad_lambda_lambda
        grad_C_lambda_vec_p1_lambda_gamma =  grad_g5_f0_vec1*grad_f0_lambda_vec 
       
        # grad_C_lambda

        grad_C_lambda_vec_p2,grad_C_gamma_vec_p2 = compute_grad_C_lambda_vec(kappa,mkvector_vec,grad_C_A_vec,lambda_vec,sDh_,sDv_,gamma_vec,dtype)

        grad_C_lambda_vec_p1 = grad_C_lambda_vec_p1_h + grad_C_lambda_vec_p1_Ch + grad_C_lambda_vec_p1_lambda_gamma 

        grad_C_lambda_vec = grad_C_lambda_vec_p1 + grad_C_lambda_vec_p2
        #################################################################################
    
        # grad_h_gamma
        lambda_vec1 = 2*kappa*torch.pow(lambda_vec,kappa-1)
        grad_f0_gamma_vec = -torch.sum(lambda_vec1*torch.sum(diagBtuB_vec.reshape(batch,2,sizex,sizex),1),dim=(1,2))
        grad_C_gamma_vec_p1_h =  torch.sum(grad_outputh*grad_g4_f0_vec)*grad_f0_gamma_vec
 
        # grad_Ch_gamma
        grad_C_gamma_vec_p1_Ch =  torch.sum(grad_outputCh*grad_g3_f0_vec)*grad_f0_gamma_vec 
       
        # grad_lambda_gamma
        grad_C_gamma_vec_p1_lambda =  torch.sum(grad_outputlambda*grad_g5_f0_vec2) 
    
        # grad_gamma_gamma
        grad_C_gamma_vec_p1_gamma =  torch.sum(grad_outputgamma.reshape(batch,1,1)*grad_f6_g5_vec*grad_g5_f0_vec2) 
     
        # grad_C_gamma
        grad_C_gamma_vec_p1 = grad_C_gamma_vec_p1_h + grad_C_gamma_vec_p1_Ch + grad_C_gamma_vec_p1_lambda + grad_C_gamma_vec_p1_gamma 

        grad_C_gamma_vec = grad_C_gamma_vec_p1 + grad_C_gamma_vec_p2

        grad_C_gamma_vec = grad_C_gamma_vec.reshape(batch,1,1) 


        return grad_C_h_vec,grad_C_Ch_vec,grad_C_lambda_vec,grad_C_gamma_vec