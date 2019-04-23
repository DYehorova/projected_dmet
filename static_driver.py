#Routines to run a static projected-DMET calculation

import numpy as np
from scipy.optimize import brentq
import system_mod
import hartreefock as hf
import sys
import os
import utils

import pyscf.fci

#####################################################################

def static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0, mubool=True, Maxitr=100 ):

    #Nsites  - total number of sites (or basis functions) in total system
    #Nele    - total number of electrons
    #Nfrag   - total number of fragments for DMET calculation
    #impindx - a list of numpy arrays containing the impurity indices for each fragment
    #h_site  - 1 e- hamiltonian in site-basis for total system
    #V_site  - 2 e- hamiltonian in site-basis for total system
    #hamtype - integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
    #Maxitr  - max number of DMET iterations

    print
    print '********************************************'
    print '     STARTING STATIC DMET CALCULATION       '
    print '********************************************'
    print

    #Check for input errors
    check_for_error( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype )

    #Begin by calculating initial mean-field Hamiltonian
    print 'Calculating initial mean-field 1RDM for total system'
    if( hamtype == 0 ):
        mf1RDM = hf.rhf_calc_hubbard( Nele, h_site ) #PING need to change this to general HF call
    elif( hamtype == 1 ):
        mf1RDM = hf.rhf_calc_hubbard( Nele, h_site )

    #Initialize the total system including the mf 1RDM and fragment information
    print 'Initialize fragment information'
    tot_system = system_mod.system( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mf1RDM )


    #DMET outer-loop
    convg = False
    for itr in range(Maxitr):

        #Perform correlated embedding calculations
        if( mubool ):
            #PING need to change this to the algorithm used by boxiao/zhihao
            #Self-consistently update global chemical potential for embedding calculations
            #to obtain correct DMET number of electrons
            lint = -1.0
            rint = 1.0
            brentq( Nele_cost_function, lint, rint, tot_system )
        else:
            #Single impurity calculation not using a global chemical potential for embedding calculations
            tot_system.corr_emb_calc()

        #Form the global density matrix from all impurities
        tot_system.get_glob1RDM()

        #Form new mean-field 1RDM from the first N-occupied natural orbitals of global 1RDM
        tot_system.get_nat_orbs()
        tot_system.get_new_mf1RDM( tot_system.Nele/2 )

        convg = True

        if(convg): break

    if( convg ):
        print 'DMET calculation succesfully converged'
        print
    else:
        print 'WARNING: DMET calculation finished, but did not converge in', Maxitr, 'iterations'
        print

    tot_system.get_DMET_E()
    print tot_system.DMET_E

#####################################################################

def Nele_cost_function( mu, tot_system ):

    tot_system.mu = mu
    tot_system.corr_emb_calc()
    tot_system.get_DMET_Nele()

    return tot_system.Nele - tot_system.DMET_Nele

#####################################################################

def check_for_error( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0 ):
    #Subroutine that takes all inputs and checks for any input errors

    #Check number of indices
    if( sum([len(arr) for arr in impindx]) > Nsites ):
        print 'ERROR: List of impurity indices (impindx) has more indices than sites'
        print
        exit()
    elif( sum([len(arr) for arr in impindx]) < Nsites ):
        print 'ERROR: List of impurity indices (impindx) has fewer indices than sites'
        print
        exit()

    #Check number of fragments
    if( len(impindx) != Nfrag ):
        print 'ERROR: Number of fragments specified by Nfrag does not match'
        print '       number of fragments in list of impurity indices (impindx)'
        print
        exit()

    #Check that impurities defined using unique indices
    chk = impindx[0]
    for count, arr in enumerate(impindx):
        if( count != 0 ):
            chk = np.concatenate((chk,arr))

    unqchk, cnt = np.unique(chk, return_counts=True)

    if( len(chk) != len(unqchk) ):
        print 'ERROR: The following indices were repeated in the definition of the impurities:'
        print unqchk[cnt>1]
        print
        exit()       

    #Check that for each fragment, impurities are assigned in ascending order (does not have to be sequential)
    for count, arr in enumerate(impindx):
        if( not np.all( np.diff(arr) > 0 ) ):
            print 'ERROR: Fragment number',count,'does not have impurity indices in ascending order'
            print
            exit()

#####################################################################

