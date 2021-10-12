import torch

def compute_grad_h_x(beta,sizeh,mkvector_vec,yvector_vec,newmh_vec,newSigmah_vec,newk,r_vec,rt_vec,symmetry,dtype):
    vector = mkvector_vec

    newMh = newSigmah_vec.shape[1]
    batch = newSigmah_vec.shape[0]
    N= newk[0].shape[0]

    newmk = torch.zeros((batch,newk[0].shape[0],1)).type(dtype)
    newmk1 = torch.zeros((batch,newk[0].shape[0],1)).type(dtype)
    if symmetry ==2: #only case coded yet
        m=0
        newkmvector=[]
        for i in range (1, sizeh+1):
            for j in range (1, i+1):
                if i == j:
                    if i != sizeh : #except newk_0
                        ind = (i-1)*sizeh + i 
                        newmk = torch.cat((newmk,vector[:,r_vec[:,ind-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)
                        newmk1 = torch.cat((newmk1,yvector_vec[:,rt_vec[:,ind-1]]-yvector_vec[:,rt_vec[:,sizeh*sizeh-1]]),dim=2)

                else:
                    ind1 = (i-1)*sizeh+j
                    ind2 = (j-1)*sizeh+i 
                    newmk = torch.cat((newmk,vector[:,r_vec[:,ind1-1]]-vector[:,r_vec[:,sizeh*sizeh-1]] 
                                        +vector[:,r_vec[:,ind2-1]]-vector[:,r_vec[:,sizeh*sizeh-1]]),dim=2)
                    newmk1 = torch.cat((newmk1,yvector_vec[:,rt_vec[:,ind1-1]]-yvector_vec[:,rt_vec[:,sizeh*sizeh-1]] 
                                        +yvector_vec[:,rt_vec[:,ind2-1]]-yvector_vec[:,rt_vec[:,sizeh*sizeh-1]]),dim=2)
                m += 1

    newkmvector = newmk[:,:,1:].type(dtype)
    Kmty_vec = newmk1[:,:,1:].type(dtype)
    Kmf2 = newkmvector
    CmnKmf2 = Kmf2@newSigmah_vec
    hmKmf2 = newkmvector@newmh_vec
    
    
    CmnKmf2 = torch.transpose(CmnKmf2,1,2).type(dtype)
    KntKmf2_vec = torch.zeros ((batch, N, newMh)).type(dtype)
    CmnKntKmf2_vec = torch.zeros ((batch, newMh, newMh, N)).type(dtype)
    m = 0
    for i in range (1, sizeh+1):
        for j in range (1, i+1):
            if i == j:
                if i != sizeh : #except newk_0
                    ind = (i-1)*sizeh + i 
                    KntKmf2_vec += newSigmah_vec[:,:,m].reshape(batch,1,newMh)*(hmKmf2[:,rt_vec[:,ind-1],:]-hmKmf2[:,rt_vec[:,sizeh*sizeh-1],:])
                    CmnKntKmf2_vec[:,m,:,:] = CmnKmf2[:,:,rt_vec[:,ind-1]]-CmnKmf2[:,:,rt_vec[:,sizeh*sizeh-1]]
            else:
                ind1 = (i-1)*sizeh+j
                ind2 = (j-1)*sizeh+i 
                KntKmf2_vec += newSigmah_vec[:,:,m].reshape(batch,1,newMh)*(hmKmf2[:,rt_vec[:,ind1-1],:]-hmKmf2[:,rt_vec[:,sizeh*sizeh-1],:] 
                            +hmKmf2[:,rt_vec[:,ind2-1],:]-hmKmf2[:,rt_vec[:,sizeh*sizeh-1],:])
                CmnKntKmf2_vec[:,m,:,:] = CmnKmf2[:,:,rt_vec[:,ind1-1]]-CmnKmf2[:,:,rt_vec[:,sizeh*sizeh-1]] +CmnKmf2[:,:,rt_vec[:,ind2-1]]-CmnKmf2[:,:,rt_vec[:,sizeh*sizeh-1]]
            m = m+1      

    K0tKmf2_vec = Kmf2[:,rt_vec[:,sizeh*sizeh-1],:]
    grad_g4_f2_vec = -2*torch.transpose(KntKmf2_vec,1,2)*beta.unsqueeze(-1).unsqueeze(-1)+newSigmah_vec@torch.transpose(Kmty_vec-2*K0tKmf2_vec,1,2)*beta.unsqueeze(-1).unsqueeze(-1)

    newSigmah_vec_gpu = newSigmah_vec.reshape(batch,1,newMh,newMh).type(dtype)
    grad_g3_f2_vec = -2*torch.transpose(newSigmah_vec_gpu@torch.transpose(CmnKntKmf2_vec,1,2),1,2)*beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type(dtype)
  
    return grad_g4_f2_vec,grad_g3_f2_vec


def compute_grad_C_h_vec(beta,newMh,mkvector_vec,h_vec,grad_C_A_vec,KmtK0_dict,KmtKn_dict,dtype):
    batch = mkvector_vec.shape[0]
    sizex = grad_C_A_vec.shape[1]
    a_vec = mkvector_vec.reshape(batch,sizex**2)
    b_vec = torch.transpose(grad_C_A_vec,1,2).reshape((batch,sizex**2))
    KmtK0_vec = torch.zeros(batch,newMh).type(dtype)
    KmtKn_vec = torch.zeros(batch,newMh,newMh).type(dtype)
    for m in range(newMh):
        name = str(m)
        row = KmtK0_dict[name]['row']
        col = KmtK0_dict[name]['col']
        value = KmtK0_dict[name]['value']
        KmtK0_vec[:,m] = torch.sum((a_vec[:,row]*b_vec[:,col]+a_vec[:,col]*b_vec[:,row])*torch.tensor(value).type(dtype),1)
        for n in range(m+1):
            name = str(m)+','+str(n)
            row = KmtKn_dict[name]['row']
            col = KmtKn_dict[name]['col']
            value = KmtKn_dict[name]['value']
            KmtKn_vec[:,m,n] = torch.sum(a_vec[:,row]*b_vec[:,col]*torch.tensor(value).type(dtype),1)
            KmtKn_vec[:,n,m] = torch.sum(a_vec[:,col]*b_vec[:,row]*torch.tensor(value).type(dtype),1)
    grad_C_h_vec = -(2*torch.sum(h_vec*KmtKn_vec,1)*h_vec.reshape(batch,newMh)+KmtK0_vec)*beta.unsqueeze(-1)
    grad_C_Ch_vec = -KmtKn_vec*beta.unsqueeze(-1).unsqueeze(-1)
    
    return grad_C_h_vec,grad_C_Ch_vec

def compute_grad_C_lambda_vec(kappa,mkvector_vec,grad_C_A_vec,lambda_vec,sDh_,sDv_,gamma_vec,dtype): 
    batch = mkvector_vec.shape[0]
    sizex = grad_C_A_vec.shape[1]
    a_np_vec=torch.transpose(mkvector_vec,1,2).cpu().numpy()
    bvec = torch.transpose(grad_C_A_vec,1,2).reshape((batch,sizex**2))
    b_np_vec=bvec.reshape(batch,1,sizex**2).cpu().numpy()   
    A_vec = torch.zeros(batch,sizex,sizex).type(dtype)
    grad_C_lambda_vec = torch.zeros(batch,sizex,sizex).type(dtype)
    for j in range(batch):
        a_np = a_np_vec[j,:,:]
        b_np = b_np_vec[j,:,:]
        DhtDh = (a_np*sDh_)*(b_np*sDh_)
        DvtDv = (a_np*sDv_)*(b_np*sDv_)
        DtD = torch.tensor(DhtDh+DvtDv).type(dtype)
        A_vec[j,:,:] = DtD.reshape(sizex,sizex).T
    grad_C_lambda_vec = -2*gamma_vec.reshape(batch,1,1)*kappa*(kappa-1)*(lambda_vec**(kappa-2))*A_vec
    grad_C_gamma_vec = torch.sum(-2*kappa*(lambda_vec**(kappa-1))*A_vec,dim=(1,2))

    return grad_C_lambda_vec,grad_C_gamma_vec
# CHECK ALL!
