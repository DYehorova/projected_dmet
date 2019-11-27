#Mod that contatins subroutines necessary to calculate the analytical form
#of the first time-derivative of the mean-field 1RDM
#this is necessary when integrating the MF 1RDM and CI coefficients explicitly
#while diagonalizing the MF 1RDM at each time-step to obtain embedding orbitals

import numpy as np
import multiprocessing as multproc

#####################################################################

def get_ddt_mf1rdm_serial( system, Nocc ):

    #Subroutine to solve the inversion equation for the first
    #time-derivative of the mf 1RDM

    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    chi = np.zeros( [system.Nsites,system.Nsites] )
    for i in range(Nocc):
        for j in range(Nocc,system.Nsites):
            chi[i,j] = 1.0/(system.NOevals[i]-system.NOevals[j])
            chi[j,i] = 1.0/(system.NOevals[j]-system.NOevals[i])

    Umat = 2

#####################################################################

