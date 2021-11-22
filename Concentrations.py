#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp


# In[2]:


class Concentrations:
    def __init__(self,Nx,Nt,parameters=None):
        
        if not parameters:
            parameters={}
        self.Nx=Nx
        self.Nt=Nt
        self.Ca=np.zeros((Nx,Nt))
        self.CaSiO3=np.zeros((Nx,Nt))
        self.CO2=np.zeros((Nx,Nt))
        self.H_piu=zeors((Nx,Nt))
        self.SiO2=zeros((Nx,Nt))
        self.HCO3=zeros((Nx,Nt))
        self.data=pp.initialize_data(g, {}, "6reagents", parameters)
    
    def compute_psi(self,step,psi1,psi2,psi3,psi4,psi5):
        psi1=self.Ca[:,step]
        psi2=self.H_piu[:,step]-self.HCO3[:,step]
        psi3=self.CO2[:,step]+self.HCO3[:,step]
        psi4=self.CaSiO3[:,step]
        psi5=self.SiO2[:,step]
        return psi1,psi2,psi3,psi4,psi5
    
    def set_bc_psi(self):
        data=self.data[pp.PARAMETERS]["6reagents"]
        bc_psi1=data["bc_value_Ca"]
        bc_psi2=data["bc_value_H_piu"]-data["bc_value_HCO3"]
        bc_psi3=data["bc_value_CO2"]+data["bc_value_HCO3"]
        bc_psi4=data["bc_value_CaSiO3"]
        bc_psi5=data["bc_value_SiO2"]
        return bc_psi1,bc_psi2,bc_psi3,bc_psi4,bc_psi5
    
    def set_initial_cond(self):
        data_t0=self.data[pp.PARAMETERS]["6reagents"]
        Ca_0=data_t0["init_cond_Ca"]
        CO2_0=data_t0["init_cond_CO2"]
        HCO3_0=data_t0["init_cond_HCO3"]
        H_piu_0=data_t0["init_cond_H_piu"]
        SiO2_0=data_t0["init_cond_SiO2"]
        CaSiO3_0=data_t0["init_cond_CaSiO3"]
        for i in range(self.g.num_cells):
            self.Ca[:,0]=Ca_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
            self.CaSiO3[:,0]=CaSiO3_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
            self.CO2[:,0]=CO2_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
            self.H_piu[:,0]=H_piu_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
            self.SiO2[:,0]=SiO2_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
            self.HCO3[:,0]=HCO3_0(self.g.cell_centers[0,i],self.g.cell_centers[1,i],self.g.cell_centers[2,i])
    
    def set_solver(psi_lhs):
        IEsolver = sps.linalg.factorized(psi_lhs)
        return IEsolver
    
    def transport_and_reaction(psi,psi_rhs_matrix,psi_rhs_b,rd,solver,h):
        psi=solver(psi_rhs_matrix*psi+psi_rhs_b+rd*h)
        return psi
    
    def Esplicit_Euler(psi,lhs,rhs_b,rhs_matrix,rd):
        IEsolver = sps.linalg.factorized(lhs)
        psi = IEsolver(rhs_matrix*psi+rhs_b+rd*h)
        return psi
    
    def one_step_transport_reaction(psi1,psi2,psi3,psi4,psi5,lhs_psi1,rhs_b_psi1,rhs_matrix_psi1,
                                    lhs_psi2,rhs_b_psi2,rhs_matrix_psi2,lhs_psi3,rhs_b_psi3,
                                    rhs_matrix_psi3,lhs_psi4,rhs_b_psi4,rhs_matrix_psi4,
                                    lhs_psi5,rhs_b_psi5,rhs_matrix_psi5,rd,h):
        
        Esplicit_Euler(psi1,lhs_psi1,rhs_b_psi1,rhs_matrix_psi1,rd)
        Esplicit_Euler(psi2,lhs_psi2,rhs_b_psi2,rhs_matrix_psi2,-2*rd)
        Esplicit_Euler(psi3,lhs_psi3,rhs_b_psi3,rhs_matrix_psi3,np.zeros(Nx))
        Esplicit_Euler(psi4,lhs_psi4,rhs_b_psi4,rhs_matrix_psi4,-rd)
        Esplicit_Euler(psi5,lhs_psi5,rhs_b_psi5,rhs_matrix_psi5,rd)
    
    def compute_concentration(self,psi1,psi2,psi3,psi4,psi5,step,K_eq):
        old_it=np.zeros(6)
        rhs=np.zeros(6)
        Jacob=np.zeros([6,6])
        Jacob[0,0]=1.0
        Jacob[1,1]=1.0
        Jacob[1,2]=-1.0
        Jacob[2,2]=1.0
        Jacob[2,3]=1.0;
        Jacob[3,4]=1.0;
        Jacob[4,5]=1.0;
        Jacob[5,3]=-K_eq
        
        max_iter=500
        tol=1e-15
        dx=zeros(6)
        for i in range(Nx):
            old_it=[Ca[i,step-1],H_piu[i,step-1],HCO3[i,step-1],CO2[i,step-1],CaSiO3[i,step-1],SiO2[i,step-1]]
            itera=0
            err=1
            while itera<max_iter and err>tol:
                rhs=compute_rhs(rhs,old_it,psi1(i),psi2(i),psi3(i),psi4(i),psi5(i),K_eq)
                Jacob=compute_Jacob(Jacob,old_it)
                Jac=sps.linalg.factorized(Jacob)
                dx=Jac.solve(-rhs)
                err=np.linalg.norm(dx)/np.linalg.norm(old_it)
                old_it+=dx
                itera+=1
            self.Ca[i,step]=old_it[0]
            self.H_piu[i,step]=old_it[1]
            self.HCO3[i,step]=old_it[2]
            self.CO2[i,step]=old_it[3]
            self.CaSiO3[i,step]=old_it[4]
            self.SiO2[i,step]=old_it[5]
    
    def compute_rhs(rhs,old_it,psi1,psi2,psi3,psi4,psi5,K_eq):
        Ca=old_it[0]
        H_piu=old_it[1]
        HCO3=old_it[2]
        CO2=old_it[3]
        CaSiO3=old_it[4]
        SiO2=old_it[5]
        rhs=[Ca-psi1,H_piu-HCO3-psi2,CO2+HCO3-psi3,CaSiO3-psi4,SiO2-psi5,H_piu*HCO3-K_eq*CO2]
        return rhs
    
    def compute_Jacob(Jacob,old_it):
        HCO3=old_it[2]
        H_piu=old_it[1]
        Jacob[5,1]=HCO3
        Jacob[5,2]=H_piu
        return Jacob               


# In[ ]:




