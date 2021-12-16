#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp
import math
import sys
import matplotlib.pyplot as plt


# In[2]:


class Reaction:
    def __init__(self,g,parameters=None):
        if not parameters:
            parameters={}
        self.g=g
        self.data=pp.initialize_data(g, {}, 'reaction', parameters)
        self.const_rate=None
        
    def set_const_rate(self):
        data=self.data[pp.PARAMETERS]["reaction"]
        A=data["A"]
        const=data["rate_const"]
        E=data["E"]
        R=data["R"]
        temperature=data["temperature"]
        
        self.const_rate=A*const*math.exp(-E/(R*temperature))
        
    
    def compute_rd(self,past_sol):
        data=self.data[pp.PARAMETERS]["reaction"]
        ph=data["ph"]
        phi=data["mass_weight"]
        K_eq=data["K_eq"]
        p=np.power(past_sol,2)/(K_eq*math.pow(10,-2*ph))
        #rhs=np.zeros(Nx)
        rhs=np.zeros(self.g.num_cells)
        for i in range(self.g.num_cells):
            #rhs[i]=h*phi*max(self.const_rate*(1.0-p[i]),0.0)
            rhs[i]=self.g.cell_volumes[i]*max(self.const_rate*(1.0-p[i]),0.0)
        return rhs
    
    def compute_rd_6_reagents(self,Ca,SiO2,H_piu,CaSiO3,rd):
        data=self.data[pp.PARAMETERS]["reaction"]
        porosity=data["porosity"]
        kd=data["kd"]
        K_sol=data["K_sol"]
        omega=np.zeros(rd.size)
        for i in range(rd.size):
            omega[i]=Ca[i]*SiO2[i]/(H_piu[i]*H_piu[i])
            omega/=K_sol
            rd[i]=self.g.cell_volumes[i]*porosity[i]*kd*max((1-omega[i]),0.0)*CaSiO3[i]
        return rd
    
    
            
                    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




