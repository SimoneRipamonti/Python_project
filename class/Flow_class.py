#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp
import math
import data.flow_benchmark_2d_geiger_setup as setup


# In[1]:


class Flow:
    def __init__(self,gb,domain,parameter):
        self.gb=gb
        self.domain=domain
        self.param=parameter
        self.aperture=parameter["aperture"]
        self.k=parameter["perm"]
        self.k_frac=parameter["fracture_perm"]
    
    def set_data(self):
        
        aperture=self.aperture
        fracture_perm=self.param["fracture_perm"]
        
        #fracture_perm_1=self.param["fracture_perm_1"]
        #fracture_perm_2=self.param["fracture_perm_2"]
        
        kx=self.param["perm"]        
        
        j=0
        for g, d in self.gb:
            specific_volumes = np.power(aperture, self.gb.dim_max()-g.dim)
            # Permeability
            k = kx*np.ones(g.num_cells) * specific_volumes#è la kx e basta per la frattura
            
            
            if g.dim < self.gb.dim_max():#la g è quella della frattura?
                #k *= fracture_perm
                k=fracture_perm*np.ones(g.num_cells) * specific_volumes

            #if g.dim < self.gb.dim_max():#la g è quella della frattura?
                #if j==0:
                    #k *= fracture_perm_1
                #else:
                    #k*=fracture_perm_2
                    
            #print("k")
            #print(k)
            perm = pp.SecondOrderTensor(k)                   
            
            f=self.set_source(g,specific_volumes)
            
            #BOUNDARIES
            bc,bc_val=self.set_bc(g)
            
            ##PER ESEMPIO PHASE FLOW
            #b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            #bc = pp.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
            #bc_val = np.zeros(g.num_faces)
            #bc_val[b_faces] = g.face_centers[1, b_faces]
            
            parameters = {"second_order_tensor": perm, "source": f, "bc": bc, "bc_values": bc_val}
            pp.initialize_data(g, d, "flow", parameters)
        
        j=0
        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            kn = fracture_perm/ (aperture/2)
            kn*=1e-3
            pp.initialize_data(mg, d, "flow", {"normal_diffusivity": kn})
            print("KN")
            print(kn)
            # Division through aperture/2 may be thought of as taking the gradient, i.e.
            # dividing by the distance from the matrix to the center of the fracture.
            
            
            #if j==0:
                #kn = fracture_perm_1 / (aperture/2)
                #pp.initialize_data(mg, d, "flow", {"normal_diffusivity": kn})
            #else:
                #kn = fracture_perm_2 / (aperture/2)
                #pp.initialize_data(mg, d, "flow", {"normal_diffusivity": kn})
            j+=1
                

    def change_perm(self,por,por_frac):
        a=0
        for g,d in self.gb:
            specific_volumes = np.power(self.aperture, self.gb.dim_max()-g.dim)
            # Permeability
            k = np.ones(g.num_cells) * specific_volumes#è la kx e basta per la frattura
            if g.dim < self.gb.dim_max():
                for i in range(k.size):
                    k[i]=k[i]*self.k_frac*np.power(1/(d[pp.STATE]["CaSiO3"][i]*3.98e-2+1)/por_frac,3)
                a=k[0]
                print("k")
                print(k)
            else:
                for i in range(k.size):
                    k[i]=k[i]*self.k*np.power(1/(d[pp.STATE]["CaSiO3"][i]*3.98e-2+1)/por,3)
                print("k")
                print(k)
            
            perm = pp.SecondOrderTensor(k)
            d[pp.PARAMETERS]["flow"]["second_order_tensor"]=perm
                        
        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            kn=a/(self.aperture/2)
            pp.initialize_data(mg, d, "flow", {"normal_diffusivity": kn})
            
            print("KN")
            print(kn)
        
        
    
    def add_data(self):
        setup.add_data(self.gb, self.domain,self.param["fracture_perm"])
        
    
    def set_bc(self,g):
        bc_value=self.param["bc_value"]
        bc_type=self.param["bc_type"]
        #bc_lambda=self.param["bc_lambda"]
        
        b_faces=g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val=np.zeros(g.num_faces)
        
        if b_faces.size != 0:
            b_face_centers=g.face_centers[:,b_faces]
            tol=1e-5
            left= b_face_centers[0, :] < self.domain["xmin"]+tol
            right= b_face_centers[0, :] > self.domain["xmax"]-tol
            
            #for i in range(b_faces.size):
            #bc_val[b_faces[i]]=bc_lambda(b_face_centers[0,i],b_face_centers[1,i],b_face_centers[2,i])
            labels = np.array(["neu"] * b_faces.size)
            if(bc_type[0]=="dir"):
                labels[left] = "dir"
            if(bc_type[1]=="dir"):
                labels[right]="dir"
            
            bc_val[b_faces[left]] =bc_value[0]
            bc_val[b_faces[right]]=bc_value[1]
            
            bc= pp.BoundaryCondition(g, b_faces,labels)
        
        else:
            bc= pp.BoundaryCondition(g, np.empty(0), np.empty(0))
        
        return bc,bc_val
        
        
        
    def set_source(self,g,specific_volumes):
        f_lambda=self.param["f_lambda"]
        f=np.zeros(g.num_cells)
        for i in range(g.num_cells):
            f[i]=specific_volumes*g.cell_volumes[i]*f_lambda(g.cell_centers[0,i],g.cell_centers[1,i],g.cell_centers[2,i])
        return f
        
    
    def discretize(self):
        method=self.param["method"]
        if(method=="Tpfa"):
            flow_discretization = pp.Tpfa("flow")
        else:
            flow_discretization = pp.Mpfa("flow")
        #elif(method=="MVEM"):
            #flow_discretization = pp.MVEM("flow")
        
        source_discretization = pp.ScalarSource("flow")
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}#gradi libertà, per misti "faces":1
            d[pp.DISCRETIZATION] = {"pressure": {"diffusive": flow_discretization,
                                             "source": source_discretization}}
        
        flow_coupling_discretization = pp.RobinCoupling("flow", flow_discretization)
        for e, d in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    g1: ("pressure", "diffusive"),
                    g2: ("pressure", "diffusive"),
                    e: ("mortar_flux", flow_coupling_discretization),
                }
            }
    
    def solve(self):
        assembler = pp.Assembler(self.gb)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        solution = sps.linalg.spsolve(A, b)
        assembler.distribute_variable(solution)
    
    def get_flux(self):
        pp.fvutils.compute_darcy_flux(self.gb,keyword_store="transport",lam_name="mortar_flux")
    
    def plot_pressure(self):
        pp.plot_grid(self.gb,"pressure",figsize=(15,12))
        
    #def plot_flux(self):
        #pp.plot_grid(self.gb,"darcy_flux",figsize=(15,12))
        


# In[ ]:





# In[ ]:





# In[ ]:




