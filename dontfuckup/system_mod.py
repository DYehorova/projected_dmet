#Define a class for the total system

import numpy as np
import fragment_mod
import sys
import os
import utils

######## TOTAL SYSTEM CLASS #######

class system():

    #####################################################################

    def __init__( self, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0, mf1RDM = None, mu=0 ):

        #initialize total system variables
        self.Nsites  = Nsites #total number of sites (or basis functions) in total system
        self.Nele    = Nele #total number of electrons
        self.Nfrag   = Nfrag #total number of fragments
        self.mu      = mu #global chemical potential added to only impurity sites in each embedding hamiltonian
        self.hamtype = hamtype #integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity

        #initialize fragment information
        #note that here impindx should be a list of numpy arrays containing the impurity indices for each fragment
        self.frag_list = []
        for i in range(Nfrag):
            self.frag_list.append( fragment_mod.fragment( impindx[i], Nsites, Nele ) )

        #initialize list that takes site index and outputs fragment index corresponding to that site
        self.site_to_frag_list = []
        for i in range(Nsites):
            for ifrag, arr in enumerate(impindx):
                if( i in arr ):
                    #self.site_to_frag_list.append( ( ifrag, np.argwhere(arr==i)[0][0] ) ) PING, this is saved if also need to know index of where that impurity appears in the list of impurities for that fragment
                    self.site_to_frag_list.append(ifrag)

        #initialize total system hamiltonian and mean-field 1RDM
        self.h_site  = h_site
        self.V_site  = V_site
        self.mf1RDM  = mf1RDM

    #####################################################################

    def get_glob1RDM( self ):
        #Subroutine to obtain global 1RDM formed from all fragments

        #initialize global 1RDM to be complex if rotation matrix is seen to be complex
        if( np.iscomplexobj( self.frag_list[0].rotmat ) ):
            self.glob1RDM = np.zeros( [ self.Nsites, self.Nsites ], dtype=complex)
        else:
            self.glob1RDM = np.zeros( [ self.Nsites, self.Nsites ] )

        #form global 1RDM
        for p in range(self.Nsites):
            for q in range(self.Nsites):

                #fragment associated with site p & q
                pfrag = self.frag_list[ self.site_to_frag_list[p] ]
                qfrag = self.frag_list[ self.site_to_frag_list[q] ]

                #index corresponding to the impurity and bath range in the rotation matrix for each fragment
                #rotation matrix ordered as (sites) x (impurity,virtual,bath,core)
                pidx = np.r_[ :pfrag.Nimp, pfrag.Nimp + pfrag.Nvirt : 2*pfrag.Nimp + pfrag.Nvirt ]
                qidx = np.r_[ :qfrag.Nimp, qfrag.Nimp + qfrag.Nvirt : 2*qfrag.Nimp + qfrag.Nvirt ]

                #form p,q part of global 1RDM using democrating partitioning
                #note that contraction only needs impurity & bath parts of rotation matrix
                self.glob1RDM[p,q]  = 0.5 * utils.matprod( pfrag.rotmat[p,pidx], pfrag.corr1RDM, pfrag.rotmat[q,pidx].conj().T )
                self.glob1RDM[p,q] += 0.5 * utils.matprod( qfrag.rotmat[p,qidx], qfrag.corr1RDM, qfrag.rotmat[q,qidx].conj().T )

    #####################################################################

    def get_nat_orbs( self ):
        #Subroutine to obtain natural orbitals of global 1RDM

        self.NOevals, self.NOevecs = np.linalg.eigh( self.glob1RDM )

    #####################################################################

    def get_new_mf1RDM( self, Nocc ):
        #Subroutine to obtain a new idempotent (mean-field) 1RDM from the
        #First Nocc natural orbitals of the global 1RDM

        NOocc = self.NOevecs[ :, :Nocc ]
        self.mf1RDM = 2.0 * np.dot( NOocc, NOocc.T.conj() )

    #####################################################################

    def corr_emb_calc( self ):
        #Subroutine to perform full correlated calculation on each fragment
        #including transformations to embedding basis
        for frag in self.frag_list:
            frag.static_corr_calc( self.mf1RDM, self.mu, self.h_site, self.V_site, self.hamtype )

    #####################################################################

    def get_DMET_Nele( self ):
        #Subroutine to calculate the number of electrons summed over all impurities
        self.DMET_Nele = 0.0
        for frag in self.frag_list:
            self.DMET_Nele += np.real( np.trace( frag.corr1RDM[:frag.Nimp,:frag.Nimp] ) )

    #####################################################################

    def get_DMET_E( self ):
        #Subroutine to calculate the DMET energy
        self.DMET_E = 0.0
        for frag in self.frag_list:
            frag.get_frag_E()
            self.DMET_E += frag.Efrag

    #####################################################################
