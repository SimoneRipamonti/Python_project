#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp
import math


# In[ ]:


class Transport:
    def __init__(self,gb,domain,parameter,grid_variable="tracer",mortar_variable="mortar_tracer",kw="transport"):
        
        self.gb=gb
        self.domain=domain
        self.param=parameter
        self.grid_variable=grid_variable
        self.mortar_variable=mortar_variable
        self.parameter_keyword=kw
        
    def set_data(self):
        aperture_=self.param["aperture"]
        por=self.param["por"]
        por_frac=self.param["por_frac"] 
        time_step=self.param["time_step"]
        t_max=self.param["t_max"]
        
        for g, d in self.gb:
            # Boundary conditions: Dirichlet for left and right side of the domain
            unity = np.ones(g.num_cells)
            empty = np.empty(0)
            bc,bc_val=self.set_bc(g)
            if g.dim == self.gb.dim_max():#se sono nella matrice porosa
                porosity = por* unity
                aperture = 1
            else:#se sono nella frattura
                porosity = por_frac*unity
                aperture = np.power(aperture_, self.gb.dim_max() - g.dim)
            # Inherit the aperture assigned for the flow problem
            
            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "time_step": time_step,
                "mass_weight": porosity * aperture,
                "t_max": t_max,
            }
            pp.initialize_default_data(g, d,self.parameter_keyword, specified_parameters)
            # Store the dimension in the dictionary for visualization purposes
            d[pp.STATE].update({"dimension": g.dim * np.ones(g.num_cells)})
        
        for e, d in self.gb.edges():#edges del grafo Gridbucket
            d[pp.PARAMETERS].update_dictionaries(self.parameter_keyword, {})
            d[pp.DISCRETIZATION_MATRICES][self.parameter_keyword] = {}
            
    def set_bc(self,g):
        tol = 1e-4
        
        bc_value=self.param["bc_value"]
        bc_type=self.param["bc_type"]
        
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        
        unity = np.ones(g.num_cells)
        empty = np.empty(0)
        
        if b_faces.size != 0:
            b_face_centers = g.face_centers[:, b_faces]
            b_inflow = b_face_centers[0, :] < tol
            b_outflow = b_face_centers[0, :] > 1-tol
            
            labels = np.array(["neu"] * b_faces.size)
            labels[np.logical_or(b_inflow, b_outflow)] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)
                
            bc_val[b_faces[b_inflow]] = bc_value[0]
            bc_val[b_faces[b_outflow]]= bc_value[1]
        else:
            bc = pp.BoundaryCondition(g) #, empty, empty)
        return bc,bc_val
    
    def discretize(self):
        
        # Identifier of the discretization operator on each grid
        advection_term = "advection"
        source_term = "source"
        mass_term = "mass"
        
        # Identifier of the discretization operator between grids
        advection_coupling_term = "advection_coupling"
        
        # Discretization objects
        node_discretization = pp.Upwind("transport")
        source_discretization = pp.ScalarSource("transport")
        mass_discretization = pp.MassMatrix("transport")
        edge_discretization = pp.UpwindCoupling("transport")
        
        # Loop over the nodes in the GridBucket, define primary variables and discretization schemes
        for g, d in self.gb:
            # Assign primary variables on this grid. It has one degree of freedom per cell.
            d[pp.PRIMARY_VARIABLES] = {self.grid_variable: {"cells": 1, "faces": 0}}
            # Assign discretization operator for the variable.
            d[pp.DISCRETIZATION] = {
                self.grid_variable: {
                    advection_term: node_discretization,
                    source_term: source_discretization,
                    mass_term: mass_discretization,
                }
            }
            if g.dim == 2:
                data = d[pp.PARAMETERS][self.parameter_keyword]
            
            # Loop over the edges in the GridBucket, define primary variables and discretizations
            for e, d in self.gb.edges():
                g1, g2 = self.gb.nodes_of_edge(e)
                # The mortar variable has one degree of freedom per cell in the mortar grid
                d[pp.PRIMARY_VARIABLES] = {self.mortar_variable: {"cells": 1}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    advection_coupling_term: {
                        g1: (self.grid_variable, advection_term),
                        g2: (self.grid_variable, advection_term),
                        e: (self.mortar_variable, edge_discretization),
                    }
                }
    
    def get_transport_lhs_rhs(self):
        # Use a filter to let the assembler consider grid and mortar variable only
        filt = pp.assembler_filters.ListFilter(variable_list=[self.grid_variable, self.mortar_variable])
        assembler = pp.Assembler(self.gb)
        
        assembler.discretize(filt=filt)
        A, b = assembler.assemble_matrix_rhs(filt=filt, add_matrices=False)
        #A=assembler.assemble_matrix_rhs(filt=filt, add_matrices=False,only_matrix=True)
        #b=assembler.assemble_matrix_rhs(filt=filt, add_matrices=False,only_rhs=True)
        
        # Identifier of the discretization operator on each grid
        advection_term = "advection"
        source_term = "source"
        mass_term = "mass"
        
        # Identifier of the discretization operator between grids
        advection_coupling_term = "advection_coupling"
        
        advection_coupling_term += ("_" + self.mortar_variable + "_" + self.grid_variable + "_" + self.grid_variable)
        mass_term += "_" + self.grid_variable
        advection_term += "_" + self.grid_variable
        source_term += "_" + self.grid_variable
        
        print("mass")
        print(A[mass_term])
        print("Advection")
        print(A[advection_term])
        print("A_coupling")
        print(A[advection_coupling_term])
        
        print("b_source")
        print(b[source_term])
        print("b_advection")
        print(b[advection_term])
        print("b_coupling")
        print(b[advection_coupling_term])
        
        lhs = A[mass_term] + self.param["time_step"] * (A[advection_term] + A[advection_coupling_term])
        rhs_source_adv = b[source_term] + self.param["time_step"] * (b[advection_term] + b[advection_coupling_term])
        rhs_mass=A[mass_term]
        
        return lhs,rhs_source_adv,rhs_mass,assembler
    
    #def set_initial_cond(self,tracer,assembler):
        #tracer_t0=self.param["initial_cond"]
        #for i in range(tracer.size):
            #for g,d in self.gb:
                #tracer[i]=tracer_t0(g.cell_centers[0,i],g.cell_centers[1,i],g.cell_centers[2,i])
            
        #assembler.distribute_variable(tracer, variable_names=[self.grid_variable, self.mortar_variable])
        #return assembler
    
    def get_flux(self):
        pp.fvutils.compute_darcy_flux(self.gb,keyword_store="transport",lam_name="mortar_flux")
    
    def plot_tracer(self):
        pp.plot_grid(self.gb, self.grid_variable, figsize=(15, 12))
     
    #def set_and_get_matrices(self,tracer):
        #self.set_bc()
        #self.set_initial_cond(tracer)
        #tracer_lhs,tracer_rhs_b,tracer_rhs_matrix=self.get_transport_lhs_rhs()
        #return tracer_lhs,tracer_rhs_b,tracer_rhs_matrix

