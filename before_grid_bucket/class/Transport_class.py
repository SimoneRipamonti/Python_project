#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp


# In[2]:


class Transport:
    def __init__(self,g,parameters=None,method="Explicit"):
        
        if not parameters:
            parameters={}
        if not method:
            method={}
        self.g=g
        self.data=pp.initialize_data(g, {}, 'transport', parameters)
        self.method=method
        
    def set_bc(self):
        
        tol=1e-4
        
        b_faces = self.g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(self.g.num_faces)
        
        if b_faces.size != 0:
            
            b_face_centers = self.g.face_centers[:, b_faces]
            b_inflow = b_face_centers[0, :] < tol
            b_outflow = b_face_centers[0, :] > self.g.nodes[0][-1]-tol
            
            labels = np.array(["neu"] * b_faces.size)
            labels[b_inflow] = self.data[pp.PARAMETERS]["transport"]["bc_type"][0]
            labels[b_outflow] = self.data[pp.PARAMETERS]["transport"]["bc_type"][1]
            
            bc = pp.BoundaryCondition(self.g, b_faces, labels)
            
            bc_val[b_faces[b_inflow]] = self.data[pp.PARAMETERS]["transport"]["bc_value"][0]
            bc_val[b_faces[b_outflow]] = self.data[pp.PARAMETERS]["transport"]["bc_value"][1]
            
            #tracer[b_faces[b_inflow]]= self.data[pp.PARAMETERS]["transport"]["bc_value"][0]
            #tracer[b_faces[b_outflow]] = self.data[pp.PARAMETERS]["transport"]["bc_value"][1]
        
        else:
            bc = pp.BoundaryCondition(self.g) #, empty, empty)
        
        self.data[pp.PARAMETERS]["transport"]["bc"] = bc
        self.data[pp.PARAMETERS]["transport"]["bc_values"]=bc_val
        self.data[pp.PARAMETERS]["transport"].pop("bc_type")
        self.data[pp.PARAMETERS]["transport"].pop("bc_value")
        
        #return tracer
    
    def set_initial_cond(self,tracer):
        tracer_t0=self.data[pp.PARAMETERS]["transport"]["initial_cond"]
        for i in range(self.g.num_cells):
            tracer[i]=tracer_t0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
     
    #def set_matrices():
    def get_transport_lhs_rhs(self):
        
        data=self.data[pp.PARAMETERS]["transport"]
        kw_t="transport"
        node_discretization = pp.Upwind(kw_t)
        source_discretization = pp.ScalarSource(kw_t)
        mass_discretization = pp.MassMatrix(kw_t)
        
        node_discretization.discretize(self.g,self.data)
        source_discretization.discretize(self.g,self.data)
        mass_discretization.discretize(self.g,self.data)
        
        A_upwind,b_upwind=node_discretization.assemble_matrix_rhs(self.g,self.data)
        #_,b_source=source_discretization.assemble_matrix_rhs(self.g,self.data)
        A_mass,b_mass=mass_discretization.assemble_matrix_rhs(self.g,self.data)
        
        dt=data["time_step"]
        decay=data["lambda_lin_decay"]
        
        if data["method"]=="Explicit":
            lhs =1/dt*A_mass
            rhs_matrix=1/dt*A_mass-A_upwind-decay*A_mass
        else:
            lhs=1/dt*A_mass+A_upwind
            rhs_matrix=1/dt*A_mass-decay*A_mass
            
        rhs_b=b_upwind+b_mass        
        
        
        return lhs,rhs_b,rhs_matrix
    
    def set_and_get_matrices(self,tracer):
        self.set_bc()
        self.set_initial_cond(tracer)
        tracer_lhs,tracer_rhs_b,tracer_rhs_matrix=self.get_transport_lhs_rhs()
        return tracer_lhs,tracer_rhs_b,tracer_rhs_matrix

