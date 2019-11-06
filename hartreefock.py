import numpy as np
import scipy.linalg as la
import sys
import os
from utils import *

#####################################################################

def rhf_calc_hubbard(Nelec,Hcore):

    #simplified subroutine to perform a mean-field (ie U=0) calculation for Hubbard model

    #Diagonalize hopping-hamiltonian
    evals,orbs = diagonalize(Hcore)

    #Form the 1RDM
    P = rdm_1el(orbs,int(Nelec/2))

    return P

#####################################################################

def rdm_1el(C,Ne):
    #subroutine that calculates and returns the one-electron density matrix in original site basis

    Cocc = C[:,:Ne]
    P = 2*np.dot( Cocc,np.transpose(np.conjugate(Cocc)) )

    return P

#####################################################################

