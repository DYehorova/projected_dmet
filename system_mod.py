#Define a class for the total system

import numpy as np
import fragment_mod
import xmat_mod
import sys
import os
import utils
import multiprocessing as multproc

import time

######## TOTAL SYSTEM CLASS #######

class system():

    #####################################################################

    def __init__( self, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0, mf1RDM = None, periodic=False, mu=0 ):

        #initialize total system variables
        self.Nsites  = Nsites #total number of sites (or basis functions) in total system
        self.Nele    = Nele #total number of electrons
        self.Nfrag   = Nfrag #total number of fragments
        self.mu      = mu #global chemical potential added to only impurity sites in each embedding hamiltonian
        self.hamtype = hamtype #integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
        self.periodic = periodic #boolean stating whether system is periodic and thus all impurities are the same

        #initialize fragment information
        #note that here impindx should be a list of numpy arrays containing the impurity indices for each fragment
        self.frag_list = []
        for i in range(Nfrag):
            self.frag_list.append( fragment_mod.fragment( impindx[i], Nsites, Nele ) )

        #initialize list that takes site index and outputs fragment index corresponding to that site
        #and separate list that outputs the index of where that impurity appears in the list of impurities for that fragment
        self.site_to_frag_list = []
        self.site_to_impindx = []
        for i in range(Nsites):
            for ifrag, arr in enumerate(impindx):
                if( i in arr ):
                    self.site_to_frag_list.append(ifrag)
                    self.site_to_impindx.append( np.argwhere(arr==i)[0][0] )
                    #self.site_to_frag_list.append( ( ifrag, np.argwhere(arr==i)[0][0] ) ) PING combines both lists into one list of tuples

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

    def static_corr_calc_wrapper( self, frag ):
        #Subroutine that simply calls the static_corr_calc subroutine for the given fragment class
        #The wrapper is necessary to parallelize using Pool

        frag.static_corr_calc( self.mf1RDM, self.mu, self.h_site, self.V_site, self.hamtype )
        return frag

    #####################################################################

    def corr_emb_calc( self, nproc ):
        #Subroutine to perform full correlated calculation on each fragment
        #including transformations to embedding basis
        if( not self.periodic ):
            #non-periodic: calculate each fragment separately in parallel
            if( nproc == 1 ):
                for frag in self.frag_list:
                    frag.static_corr_calc( self.mf1RDM, self.mu, self.h_site, self.V_site, self.hamtype )
            else:
                frag_pool = multproc.Pool(nproc)
                self.frag_list = frag_pool.map( self.static_corr_calc_wrapper, self.frag_list )
                frag_pool.close()
        else:
            #periodic: calculate the first fragment only - this is not correct
            frag0 = self.frag_list[0]
            frag0.static_corr_calc( self.mf1RDM, self.mu, self.h_site, self.V_site, self.hamtype )

            #copy first fragment to all other fragments
            for frag in self.frag_list[1:]:
                self.copy_rotmat( frag0, frag )
                frag.env1RDM_evals  = np.copy( frag0.env1RDM_evals )
                frag.h_emb          = np.copy( frag0.h_emb )
                frag.V_emb          = np.copy( frag0.V_emb )
                frag.Ecore          = np.copy( frag0.Ecore )
                frag.h_emb_halfcore = np.copy( frag0.h_emb_halfcore )
                frag.CIcoeffs       = np.copy( frag0.CIcoeffs )
                frag.corr1RDM       = np.copy( frag0.corr1RDM )

    #####################################################################

    def copy_rotmat( self, frag_ref, frag_copy ):
        #Subroutine to copy the rotation matrix from frag_ref to frag_copy
        #Complication is that impurity orbitals correspond to different sites in each fragment

        #obtain just environment block of rotmat to be copied over
        rotmat_env = np.copy( frag_ref.rotmat[:,frag_ref.Nimp:] )
        rotmat_env = np.delete( rotmat_env, frag_ref.impindx, axis=0 )

        #insert unit vectors at appropiate indices for impurity orbitals
        frag_copy.rotmat = np.zeros( [ frag_copy.Nsites, frag_copy.Nimp ] )
        for imp in range(frag_copy.Nimp):
            indx                          = frag_copy.impindx[imp]
            frag_copy.rotmat[ indx, imp ] = 1.0
            rotmat_env                    = np.insert( rotmat_env, indx, 0.0, axis=0 )

        #form full rotation matrix for new fragment from reference fragment
        #rotation matrix is ordered as impurity, virtual, bath, core
        frag_copy.rotmat = np.concatenate( (frag_copy.rotmat,rotmat_env), axis=1 )

    #####################################################################

    def get_frag_corr1RDM( self ):
        #Subroutine to calculate correlated 1RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr1RDM()

    #####################################################################

    def get_frag_corr12RDM( self ):
        #Subroutine to calculate correlated 1RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr12RDM()

    #####################################################################

    def get_frag_Hemb( self ):
        #Subroutine to calculate embedding Hamiltonian for each fragment
        for frag in self.frag_list:
            frag.get_Hemb( self.h_site, self.V_site, self.hamtype )

    #####################################################################

    def get_DMET_Nele( self ):
        #Subroutine to calculate the number of electrons summed over all impurities
        #Necessary to calculate fragment 1RDMs prior to this routine
        self.DMET_Nele = 0.0
        for frag in self.frag_list:
            self.DMET_Nele += np.real( np.trace( frag.corr1RDM[:frag.Nimp,:frag.Nimp] ) )

    #####################################################################

    def get_DMET_E( self ):
        #Subroutine to calculate the DMET energy

        self.get_frag_Hemb()
        self.get_frag_corr12RDM()

        self.DMET_E = 0.0
        for frag in self.frag_list:
            frag.get_frag_E()
            self.DMET_E += np.real( frag.Efrag ) #discard what should be numerical error of imaginary part

    #####################################################################

    def get_natorb_rotmat_contraction( self ):
        #Subroutine to contract the rotation matrix for each fragment
        #with the natural orbitals of the global 1RDM
        #NOTE: make sure natural orbitals have been previously calculated

        for frag in self.frag_list:
            frag.contract_natorb_rotmat( self.NOevecs )

    #####################################################################

    def get_frag_ddt_corr1RDM( self ):
        #Subroutine to calculate first time-derivative of correlated 1RDMS for each fragment

        for frag in self.frag_list:
            frag.get_ddt_corr1RDM()

    #####################################################################

    def get_Xmats( self, Nocc, nproc ):

        #Solve for super vector containing non-redundant terms of the X-matrices for each fragment
        if( nproc == 1 ):
            Xvec = xmat_mod.solve_Xvec_serial( self, Nocc )
        else:
            Xvec = xmat_mod.solve_Xvec_parallel( self, Nocc, nproc )

        #Unpack super vector into X-matrices for each fragment
        #The X super-vector is indexed by A, emb1, and emb2, where A runs over all fragments
        #emb1 runs over the core, bath, and virtual orbitals for each fragment A
        #and emb2 runs over all non-redundant terms given emb1
        Xidx = 0
        for ifragA, fragA in enumerate(self.frag_list):

            fragA.init_Xmat()

            for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

                if emb1 in fragA.virtrange:
                    emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
                elif emb1 in fragA.bathrange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
                elif emb1 in fragA.corerange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

                for emb2 in emb2range:

                    fragA.Xmat[emb1,emb2] = Xvec[Xidx] #PING in principle can make X-matrix to not have impurity orbitals since those terms are zero
                    Xidx += 1

    ######################################################################

