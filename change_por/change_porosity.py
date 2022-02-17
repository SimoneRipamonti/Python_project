#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import porepy as pp
import scipy.sparse as sps
import math 
import sys
sys.path.append("../class")
from Flow_class import Flow
from Transport_class import Transport
from Reaction_class import Reaction


# In[1]:


def change_por(por,por_frac,flow1,flow2,transport_Ca,transport_CaSiO3):
    
    flow1.change_perm(por,por_frac)
    flow1.discretize()
    flow1.solve()
    
    flow2.change_perm(por,por_frac)
    flow2.discretize()
    flow2.solve()
    
    transport_Ca.get_flux("Ca")
    transport_Ca.set_porosity("Ca")
    transport_Ca.discretize("Ca")
    lhs_Ca,rhs_source_adv_Ca,rhs_mass_Ca,assembler_Ca=transport_Ca.get_transport_lhs_rhs("Ca")
    
    transport_CaSiO3.get_flux("CaSiO3")
    transport_CaSiO3.set_porosity("CaSiO3")
    transport_CaSiO3.discretize("CaSiO3")
    lhs_CaSiO3,rhs_source_adv_CaSiO3,rhs_mass_CaSiO3,assembler_CaSiO3=transport_CaSiO3.get_transport_lhs_rhs("CaSiO3")
    
    IEsolver_Ca = sps.linalg.factorized(lhs_Ca)
    IEsolver_CaSiO3 = sps.linalg.factorized(lhs_CaSiO3)
    
    return rhs_source_adv_Ca,rhs_mass_Ca,rhs_source_adv_CaSiO3,rhs_mass_CaSiO3,IEsolver_Ca,IEsolver_CaSiO3
    


# In[ ]:


def compute_new_porosity(gb1):
    
    for g,d in gb1:
        if g.dim < self.gb.dim_max():
            por_frac=1-d[pp.STATE]["CaSiO3"]*3.98e-2
        else:
            por=1-d[pp.STATE]["CaSiO3"]*3.98e-2
    
    return por,por_frac

