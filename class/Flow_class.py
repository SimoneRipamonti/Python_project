#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp
import math


# In[4]:


class Flow:
    def __init__(self,g,parameters=None,method="Tpfa"):
        
        if not parameters:
            parameters={}
        if not method:
            method={}
        self.g=g
        self.data=pp.initialize_data(g, {}, 'flow', parameters)
        self.method=method
        self.Nx=g.num_cells
    
    def set_source(self):
        f=np.zeros(self.Nx)
        f_lambda=self.data[pp.PARAMETERS]["flow"]["f_lambda"]
        for i in range(self.g.num_cells):
            f[i]=0.01*f_lambda(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
        self.data[pp.PARAMETERS]['flow']["source"]=f
        
        
    
    def solve(self):
        if(self.method=="Tpfa"):
            print("Tpfa")
            flow_discretization = pp.Tpfa("flow")
        elif(self.method=="Mpfa"):
            flow_discretization = pp.Mpfa("flow")
        elif(self.method=="MVEM"):
            print(self.method)
            flow_discretization = pp.MVEM("flow")
    
        flow_discretization.discretize(self.g, self.data)
        A, b_flow = flow_discretization.assemble_matrix_rhs(self.g, self.data)
        
        
        if(self.method=="MVEM"):
            rhs_discretization = pp.DualScalarSource("flow")
        else:
            rhs_discretization = pp.ScalarSource("flow")
        
        rhs_discretization.discretize(self.g, self.data)
        _, b_rhs = rhs_discretization.assemble_matrix_rhs(self.g, self.data)
        
        if(self.method=="MVEM"):
            up = sps.linalg.spsolve(A, b_flow+b_rhs)
            p=flow_discretization.extract_pressure(self.g, up, self.data)
        else:
            p = sps.linalg.spsolve(A, b_flow+b_rhs)
        return p
    
    def print_pressure(self,p):
        pp.plot_grid(self.g,p,figsize=(15,12))
 
            
        

    
         
        
        


# In[ ]:





# In[ ]:





# In[ ]:




