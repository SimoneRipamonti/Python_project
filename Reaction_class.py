#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
        
    def get_const_rate(self):
        data=self.data[pp.PARAMETERS]["reagent"]
        A=data["A"]
        const=data["rate_const"]
        E=data["E"]
        R=data["R"]
        temperature=data["temperature"]
        
        self.const_rate=A*const*math.exp(-E/(R*temperature))
        
    
    def compute_rd(self,past_sol):
        data=self.data[pp.PARAMETERS]["reagent"]
        ph=data["ph"]
        phi=data["mass_weight"]
        K_eq=data["K_eq"]
        p=np.power(past_sol,2)/(K_eq*math.pow(10,-2*ph))
        for i in range(Nx):
            rhs[i]=h*phi[i]*max(self.const_rate*(1.0-p[i]),0.0)
            return rhs
    
        
        


# In[5]:





# In[ ]:




