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


Nx=100
#Ny=10
phys_dims=[1]
g=pp.CartGrid([Nx],phys_dims)
g.compute_geometry()    


# In[3]:


data={"temperature":423,
       "A":1.9e-2,
       "rate_const":0.004466,
       "E":4700,
       "R":8.314,
       "K_eq":10e9,
       "ph":3.5}


# In[4]:


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
        
    
    def compute_rd(self,past_sol,h,Nx):
        data=self.data[pp.PARAMETERS]["reaction"]
        ph=data["ph"]
        phi=data["mass_weight"]
        K_eq=data["K_eq"]
        p=np.power(past_sol,2)/(K_eq*math.pow(10,-2*ph))
        rhs=np.zeros(Nx)
        for i in range(Nx):
            #rhs[i]=h*phi*max(self.const_rate*(1.0-p[i]),0.0)
            rhs[i]=h*max(self.const_rate*(1.0-p[i]),0.0)
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
            rd[i]=porosity[i]*kd*max((1-omega[i]),0.0)*CaSiO3[i]
        return rd
    
    
            
            
            
        
        
        
        


# In[5]:


reaction=Reaction(g,data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




