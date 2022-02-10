#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import porepy as pp


# In[2]:


class Concentrations:
    def __init__(self,gb,Nt,parameters=None):
        
        if not parameters:
            parameters={}
        self.gb=gb
        self.Nt=Nt
        self.data=pp.initialize_data(gb, {}, "6reagents", parameters)
        
    def set_bc_psi(self):
        data=self.data[pp.PARAMETERS]["6reagents"]
        bc_psi1=data["bc_value_Ca"]
        bc_psi2=np.array(data["bc_value_H_piu"])-np.array(data["bc_value_HCO3"])
        bc_psi3=np.array(data["bc_value_CO2"])+np.array(data["bc_value_HCO3"])
        bc_psi4=data["bc_value_CaSiO3"]
        bc_psi5=data["bc_value_SiO2"]
        
        return bc_psi1,bc_psi2,bc_psi3,bc_psi4,bc_psi5
    
    def Explicit_Euler(self,psi,lhs,rhs_b,rhs_matrix,rd,dt):
        IEsolver = sps.linalg.factorized(lhs)
        psi = IEsolver(rhs_matrix*psi+rhs_b+rd*dt)
        return psi
    
    def one_step_transport_reaction(self,psi1,psi2,psi3,psi4,psi5,lhs_psi1,rhs_b_psi1,rhs_mass_psi1,
                                                                        lhs_psi2,rhs_b_psi2,rhs_mass_psi2,
                                                                        lhs_psi3,rhs_b_psi3,rhs_mass_psi3,
                                                                        lhs_psi4,rhs_b_psi4,rhs_mass_psi4,
                                                                        lhs_psi5,rhs_b_psi5,rhs_mass_psi5,rd,dt):
        psi1=self.Explicit_Euler(psi1,lhs_psi1,rhs_b_psi1,rhs_mass_psi1,rd,dt)
        psi2=self.Explicit_Euler(psi2,lhs_psi2,rhs_b_psi2,rhs_mass_psi2,-2*rd,dt)
        psi3=self.Explicit_Euler(psi3,lhs_psi3,rhs_b_psi3,rhs_mass_psi3,np.zeros(psi1.size),dt)
        psi4=self.Explicit_Euler(psi4,lhs_psi4,rhs_b_psi4,rhs_mass_psi4,-rd,dt)
        psi5=self.Explicit_Euler(psi5,lhs_psi5,rhs_b_psi5,rhs_mass_psi5,rd,dt)
        return psi1,psi2,psi3,psi4,psi5
    
    def compute_concentration(self,psi1,psi2,psi3,psi4,psi5,K_eq,Ca,H_piu,HCO3,CO2,CaSiO3,SiO2):
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
        
        old_it=np.zeros(6)
        
        max_iter=500
        tol=1e-15
        dx=np.zeros(6)
        for i in range(psi1.size):
            old_it=[Ca[i],H_piu[i],HCO3[i],CO2[i],CaSiO3[i],SiO2[i]]
            itera=0
            err=1
            while itera<max_iter and err>tol:
                rhs=self.compute_rhs(rhs,old_it,psi1[i],psi2[i],psi3[i],psi4[i],psi5[i],K_eq)
                Jacob=self.compute_Jacob(Jacob,old_it)
                Jac=sps.linalg.factorized(Jacob)
                dx=Jac(np.zeros(6)-rhs)
                err=np.linalg.norm(dx)/np.linalg.norm(old_it)
                old_it+=dx
                itera+=1
            Ca[i]=old_it[0]
            H_piu[i]=old_it[1]
            HCO3[i]=old_it[2]
            CO2[i]=old_it[3]
            CaSiO3[i]=old_it[4]
            SiO2[i]=old_it[5]
        return Ca,H_piu,HCO3,CO2,CaSiO3,SiO2
            
    def compute_rhs(self,rhs,old_it,psi1,psi2,psi3,psi4,psi5,K_eq):
        Ca=old_it[0]
        H_piu=old_it[1]
        HCO3=old_it[2]
        CO2=old_it[3]
        CaSiO3=old_it[4]
        SiO2=old_it[5]
        rhs=[Ca-psi1,H_piu-HCO3-psi2,CO2+HCO3-psi3,CaSiO3-psi4,SiO2-psi5,H_piu*HCO3-K_eq*CO2]
        return rhs
    
    def compute_Jacob(self,Jacob,old_it):
        HCO3=old_it[2]
        H_piu=old_it[1]
        Jacob[5,1]=HCO3
        Jacob[5,2]=H_piu
        return Jacob 
    
    def compute_psi(self,psi1,psi2,psi3,psi4,psi5,Ca,CO2,SiO2,H_piu,HCO3,CaSiO3):
        psi1=Ca
        psi2=H_piu-HCO3
        psi3=CO2+HCO3
        psi4=CaSiO3
        psi5=SiO2
        return psi1,psi2,psi3,psi4,psi5


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




