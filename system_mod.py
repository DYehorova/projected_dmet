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

    def corr_emb_calc( self ):
        #Subroutine to perform full correlated calculation on each fragment
        #including transformations to embedding basis
        for frag in self.frag_list:
            frag.static_corr_calc( self.mf1RDM, self.mu, self.h_site, self.V_site, self.hamtype )

    #####################################################################

    def get_frag_corr1RDM( self ):
        #Subroutine to calculate correlated 1RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr1RDM()

    #####################################################################

    def get_frag_Hemb( self ):
        #Subroutine to calculate embedding Hamiltonian for each fragment
        for frag in self.frag_list:
            frag.get_Hemb( self.h_site, self.V_site, self.hamtype )

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

    def get_Xmats( self, Nocc ):

        #Solve for super vector containing non-redundant terms of the X-matrices for each fragment
        Xvec = self.solve_Xvec( Nocc )

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

    #####################################################################

    def solve_Xvec( self, Nocc ):
        #Subroutine to solve the coupled equations for the
        #X-matrices for all of the fragments
        #Returns the super-vector containing the elements of all the non-redundant terms of the X-matrices for each fragment
        #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
        #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

        #Calculate contraction between rotation matrix of each fragment and natural orbitals of global 1RDM
        self.get_natorb_rotmat_contraction()

        #Calculate first time-derivative of correlated 1RDMS for each fragment
        self.get_frag_ddt_corr1RDM()

        #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
        chi = np.zeros( [self.Nsites,self.Nsites] )
        for i in range(self.Nsites):
            for j in range(self.Nsites):
                if( i != j ):
                    chi[i,j] = 1.0/(self.NOevals[i]-self.NOevals[j])


        #Form omega super-matrix indexed by fragment-embedding orbital-embedding orbital, site orbital-embedding orbital
        #Note that number of site orbitals and embedding orbitals is the same - the size of the system
        omega  = np.zeros( [self.Nfrag*self.Nsites**2, self.Nsites**2], dtype=complex )
        for ifragA in range(self.Nfrag):
            for emb1 in range(self.Nsites):
                for emb2 in range(self.Nsites):

                    #Left-index for super-matrix
                    Lidx = emb2+emb1*self.Nsites+ifragA*self.Nsites**2

                    for site1 in range(self.Nsites):
                        for emb3 in range(self.Nsites):

                            #Right Index for super matrix
                            Ridx = emb3+site1*self.Nsites

                            #calculate omega super-matrix - einsum summing over natural orbitals
                            omega[Lidx,Ridx] = np.einsum( 'j,j,ij,i,i', np.conjugate(self.frag_list[ifragA].NO_rot[:,emb1]), self.frag_list[self.site_to_frag_list[site1]].NO_rot[:,emb3], \
                                                          chi[:Nocc,:], self.NOevecs[site1,:Nocc], self.frag_list[ifragA].NO_rot[:Nocc,emb2] ) + \
                                               np.einsum( 'i,i,ij,j,j', np.conjugate(self.frag_list[ifragA].NO_rot[:Nocc,emb1]), self.frag_list[self.site_to_frag_list[site1]].NO_rot[:Nocc,emb3], \
                                                          chi[:Nocc,:], self.NOevecs[site1,:], self.frag_list[ifragA].NO_rot[:,emb2] )


        #Form Y super-vector indexed by A, emb1, and emb2, where A runs over all fragments
        #emb1 runs over the core, bath, and virtual orbitals for each fragment A
        #and emb2 runs over all non-redundant terms given emb1
        sizeY = 0
        for ifragA, fragA in enumerate(self.frag_list):
            for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

                if emb1 in fragA.virtrange:
                    emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
                elif emb1 in fragA.bathrange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
                elif emb1 in fragA.corerange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

                for emb2 in emb2range:
                    sizeY += 1

        Yvec = np.zeros( sizeY, dtype=complex )

        Yidx = 0
        for ifragA, fragA in enumerate(self.frag_list):
            for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

                if emb1 in fragA.virtrange:
                    emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
                elif emb1 in fragA.bathrange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
                elif emb1 in fragA.corerange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

                for emb2 in emb2range:

                    #Left-indices for omega matrix
                    Lidx1 = emb2+emb1*self.Nsites+ifragA*self.Nsites**2
                    Lidx2 = emb1+emb2*self.Nsites+ifragA*self.Nsites**2

                    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
                    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
                    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
                    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

                    for site1 in range(self.Nsites):
                        frag_site1 = self.frag_list[self.site_to_frag_list[site1]]
                        for emb3 in np.concatenate( ( frag_site1.imprange, frag_site1.bathrange ) ):

                            #Right index for omega matrix
                            Ridx = emb3+site1*self.Nsites

                            #Values of d/dt correlated 1RDM
                            if emb3 in frag_site1.bathrange:
                                emb3 -= frag_site1.Nvirt #necessary to subtract off Nvirt only for bath orbitals b/c embedding orbitals go as imp,virt,bath,core
                            val1  = 1j*frag_site1.ddt_corr1RDM[ emb3, self.site_to_impindx[site1] ]
                            val2  = 1j*frag_site1.ddt_corr1RDM[ self.site_to_impindx[site1], emb3 ]

                            #Sum terms into Y super-vector
                            Yvec[Yidx] += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx] * val1 + val2 * np.conjugate(omega[Lidx2,Ridx]) )

                    Yidx += 1


        #Form phi super-matrix where first index is same as the Y super-vector above
        #and second index is indexed by B, emb3, and emb4
        #where B runs over all fragments
        #emb3 runs over the core, bath, and virtual orbitals for each fragment corresponding to the fragment where site1 is an impurity
        #and emb4 runs over all non-redundant terms given emb3
        #note that phi is a square matrix with a single index having the same dimensions as the Y super-vector above

        phi = np.zeros( [sizeY,sizeY], dtype=complex )

        Lphiidx = 0
        for ifragA, fragA in enumerate(self.frag_list):
            for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

                if emb1 in fragA.virtrange:
                    emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
                elif emb1 in fragA.bathrange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
                elif emb1 in fragA.corerange:
                    emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

                for emb2 in emb2range:

                    #Left-indices for omega matrix
                    Lidx1 = emb2+emb1*self.Nsites+ifragA*self.Nsites**2
                    Lidx2 = emb1+emb2*self.Nsites+ifragA*self.Nsites**2

                    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
                    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
                    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
                    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

                    Rphiidx = 0
                    for ifragB, fragB in enumerate(self.frag_list):
                        for emb3 in np.concatenate( ( fragB.virtrange, fragB.bathrange, fragB.corerange ) ):
            
                            if emb3 in fragB.virtrange:
                                emb4range = np.concatenate( (fragB.bathrange,fragB.corerange) )
                            elif emb3 in fragB.bathrange:
                                emb4range = np.concatenate( (fragB.virtrange,fragB.corerange) )
                            elif emb3 in fragB.corerange:
                                emb4range = np.concatenate( (fragB.virtrange,fragB.bathrange) )
            
                            for emb4 in emb4range:

                                for site1 in fragB.impindx: #site1 corresponds to the index in the site basis of the impurity orbitals of fragment B

                                    #Right indeces for omega matrix
                                    Ridx1 = emb3+site1*self.Nsites
                                    Ridx2 = emb4+site1*self.Nsites

                                    #Values of correlated 1RDM
                                    #Note that if embedding orbital (ie emb3 or emb4) is not a bath orbital, then the correlated 1RDM is zero
                                    #because site1 always corresponds to an impurity orbital
                                    if emb4 in fragB.bathrange:
                                        val1 = fragB.corr1RDM[ emb4-fragB.Nvirt, self.site_to_impindx[site1] ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
                                    else:
                                        val1 = 0.0
                                    if emb3 in fragB.bathrange:
                                        val2 = fragB.corr1RDM[ self.site_to_impindx[site1], emb3-fragB.Nvirt ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
                                    else:
                                        val2 = 0.0

                                    #sum terms into phi super-matrix
                                    phi[Lphiidx,Rphiidx] += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx1]*val1 - np.conjugate(omega[Lidx2,Ridx2])*val2 )

                                Rphiidx += 1

                    Lphiidx += 1

        #Solve inversion equation for X super vector
        return np.linalg.solve( np.eye(sizeY)-phi, Yvec )

    #####################################################################



