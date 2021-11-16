#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import scipy.sparse as sps
import porepy as pp
import math
import matplotlib.pyplot as plt


# In[14]:


Nx=100
#phys_dims=[1,1]
phys_dims=[1]
#g=pp.CartGrid([Nx,Ny],phys_dims)
g=pp.CartGrid([Nx],phys_dims)
g.compute_geometry()
#pp.plot_grid(g,figsize=(15,12))
#p.plot_grid(g)


# In[15]:


# Permeability
perm = pp.SecondOrderTensor(1e-7*np.ones(g.num_cells))                     
f_lambda= lambda x,y,z: math.sin(4*math.pi*x)
# Boundary conditions
b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
bc = pp.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
bc_val = np.zeros(g.num_faces)
bc_val[0]=1e6
bc_val[bc_val.size-1]=-800000.0

# Collect all parameters in a dictionary
parameters = {"second_order_tensor": perm, "f_lambda": f_lambda, "bc": bc, "bc_values": bc_val}


# In[16]:


class Flow:
    def __init__(self,g,parameters=None,method="Tpfa"):
        
        if not parameters:
            parameters={}
        if not method:
            method={}
        self.g=g
        self.data=pp.initialize_data(g, {}, 'flow', parameters)
        self.method=method
    
    def set_source(self):
        f=np.zeros(Nx)
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
 
            
        

    
         
        
        


# In[17]:


flow=Flow(g,parameters,"Mpfa")
flow.set_source()
p=flow.solve()


# In[18]:


h=1/Nx
x=np.linspace(1/(2*Nx),1-1/(2*Nx),Nx)
plt.plot(x,p)


# In[ ]:




