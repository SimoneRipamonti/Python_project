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
    #def __init__(self,parameters=None):
        #if not parameters:
            #parameters={}
        #self.parameters=parameters
        #self.const_rate=None
        #self.ph=parameters["ph"]
        #self.phi=parameters["mass_weight"]
        #self.K_eq=parameters["K_eq"]
    def __init__(self,parameters=None):
        if not parameters:
            parameters={}
        
        self.data=parameters
        self.const_rate=None
        
    def compute_rd(self,tracer,rd):
        data=self.data
        ph=data["ph"]
        #phi=data["mass_weight"]
        K_eq=data["K_eq"]
        p=np.power(tracer,2)/(K_eq*math.pow(10,-2*ph))
        for i in range(tracer.size):
            rd[i]=self.const_rate*max((1.0-p[i]),0.0)
        return rd
    
    def set_const_rate(self):
        
        data=self.data
        A=data["A"]
        const=data["rate_const"]
        E=data["E"]
        R=data["R"]
        temperature=data["temperature"]
        
        self.const_rate=A*const*math.exp(-E/(R*temperature))
            
    
    def compute_rd_6_reagents(self,Ca,SiO2,H_piu,CaSiO3,rd,rhs_mass_psi1):
        data=self.data
        porosity=data["porosity"]
        kd=data["kd"]
        K_sol=data["K_sol"]
        omega=np.zeros(rd.size)
        for i in range(rd.size):
            omega[i]=Ca[i]*SiO2[i]/(H_piu[i]*H_piu[i])
            omega/=K_sol
            rd[i]=kd*max((1-omega[i]),0.0)*CaSiO3[i]
        rd=rhs_mass_psi1*rd
        return rd
    
    
            
                    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




