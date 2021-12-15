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
        
    def set_bc(self):
        tol=1e-4
        b_faces=self.g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val=np.zeros(self.g.num_faces)
        bc_lambda=self.data[pp.PARAMETERS]["flow"]["bc_lambda"]
        b_face_centers=self.g.face_centers[:,b_faces]
        
        for i in range(b_faces.size):
            bc_val[b_faces[i]]=bc_lambda(b_face_centers[0,i],b_face_centers[1,i],b_face_centers[2,i])
        
        bc = pp.BoundaryCondition(self.g, b_faces, ['dir']*b_faces.size)
        
        self.data[pp.PARAMETERS]["flow"]["bc"] = bc
        self.data[pp.PARAMETERS]["flow"]["bc_values"]=bc_val
        self.data[pp.PARAMETERS]["flow"].pop("bc_lambda")
        
        
        
    def set_source(self):
        f=np.zeros(self.Nx)
        f_lambda=self.data[pp.PARAMETERS]["flow"]["f_lambda"]
        for i in range(self.g.num_cells):
            f[i]=0.01*f_lambda(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
        self.data[pp.PARAMETERS]['flow']["source"]=f
        self.data[pp.PARAMETERS]["flow"].pop("f_lambda")
        
    
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
        
        self.data[pp.PARAMETERS]["flow"] 
        if(self.method=="MVEM"):
            rhs_discretization = pp.DualScalarSource("flow")
        else:
            rhs_discretization = pp.ScalarSource("flow")
        
        rhs_discretization.discretize(self.g, self.data)
        _, b_rhs = rhs_discretization.assemble_matrix_rhs(self.g, self.data)
        
        if(self.method=="MVEM"):
            up = sps.linalg.spsolve(A, b_flow+b_rhs)
            p=flow_discretization.extract_pressure(self.g, up, self.data)
            self.data[pp.PARAMETERS]["flow"]["darcy_flux"]=flow_discretization.extract_flux(self.g, up, self.data)
        else:
            p = sps.linalg.spsolve(A, b_flow+b_rhs)
        return p
    
    def plot_pressure(self,p):
        pp.plot_grid(self.g,p,figsize=(15,12))
    
    def get_flux(self,p):
        if(self.method!="MVEM"):
            self.data[pp.STATE] = {"pressure": p}
            pp.fvutils.compute_darcy_flux(self.g, data=self.data)
        darcy_flux = self.data[pp.PARAMETERS]["flow"]["darcy_flux"]
        return darcy_flux
    
    def plot_pressure_flux(self,flux,p):
        if(self.method=="MVEM"):
            flow_discretization = pp.MVEM("flow")
            P0u = flow_discretization.project_flux(self.g,flux,self.data)
            pp.plot_grid(self.g, p, P0u * 0.2, figsize=(15, 12))
        else:
            print("For plotting vel and pressure togheter with this function, it is mandatory to use the MVEM method")

        
            
        

    
         
        
        


# In[ ]:





# In[ ]:





# In[ ]:




