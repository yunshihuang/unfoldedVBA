import pickle
from Model_files.tools import *

def init_settings0(mode):
    if mode != 'test':
        sizeh = 9
        sizex = 256
        symmetry = 2
        (newMh,newk,newk0,dict_val,diag_qudratick,T,t,r_vec,rt_vec) = CreateKm(sizeh,sizex,2)

        with open('KmtK0_dict.txt', 'rb') as handle:
            KmtK0_dict = pickle.loads(handle.read())

        KmtKn_dict = {}
        for m in range(newMh):
            Km = newk[m]
            for n in range(m+1):
                name = str(m)+','+str(n)        
                KmtKn_dict[name] = {}
                Kn = newk[n]
                row,col,value = find(Km.T@Kn)
                KmtKn_dict[name]['row'] = row
                KmtKn_dict[name]['col'] = col
                KmtKn_dict[name]['value'] = value 
    return KmtK0_dict, KmtKn_dict        

    
          