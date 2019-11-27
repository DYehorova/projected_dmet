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
            #Xvec = self.solve_Xvec_serial( Nocc )
            Xvec = xmat_mod.solve_Xvec_serial( self, Nocc )
        else:
            #Xvec = self.solve_Xvec_parallel( Nocc, nproc )
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

    #def solve_Xvec_serial( self, Nocc ):
    #    #Subroutine to solve the coupled equations for the
    #    #X-matrices for all of the fragments in serial
    #    #Returns the super-vector containing the elements of all the non-redundant terms of the X-matrices for each fragment
    #    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #    #Calculate contraction between rotation matrix of each fragment and natural orbitals of global 1RDM
    #    self.get_natorb_rotmat_contraction()

    #    #Calculate first time-derivative of correlated 1RDMS for each fragment
    #    self.get_frag_ddt_corr1RDM()

    #    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    #    chi = np.zeros( [self.Nsites,self.Nsites] )
    #    for i in range(Nocc):
    #        for j in range(Nocc,self.Nsites):
    #            chi[i,j] = 1.0/(self.NOevals[i]-self.NOevals[j])
    #            chi[j,i] = 1.0/(self.NOevals[j]-self.NOevals[i])

    #    #Calculate size of X-vec and the matrices needed to calculate it
    #    #Given by A*emb1*emb2, where A runs over all fragments
    #    #emb1 runs over the core, bath, and virtual orbitals for each fragment A
    #    #and emb2 runs over all non-redundant terms given emb1
    #    #also define a dictionary that takes the tuple (A,emb1,emb2) and outputs the index in the X-vec
    #    sizeX = 0
    #    indxdict = {}
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:
    #                indxdict[ (ifragA, emb1, emb2) ] = sizeX
    #                sizeX += 1

    #    #Form omega super-matrix indexed by same as X-vec, site orbital-embedding orbital
    #    #Note that number of site orbitals and embedding orbitals is the same - the size of the system
    #    omega = np.zeros( [sizeX, self.Nsites**2], dtype=complex )
    #    Lidx  = 0 #Left-index for super-matrix
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:

    #                for site1 in range(self.Nsites):
    #                    for emb3 in range(self.Nsites):

    #                        #Right Index for super matrix
    #                        Ridx = emb3+site1*self.Nsites

    #                        #calculate omega super-matrix
    #                        omega[Lidx,Ridx] = self.calc_omega_term( fragA, emb1, emb2, site1, emb3, Nocc, chi )

    #                Lidx += 1

    #    #Form Y super-vector
    #    Yidx = 0
    #    Yvec = np.zeros( sizeX, dtype=complex )
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:

    #                #Calculate element of Y vector
    #                Yvec[Yidx] = self.calc_Yvec_term( ifragA, fragA, emb1, emb2, indxdict, omega )

    #                Yidx += 1


    #    #Form phi super-matrix where first index is same as the Y super-vector above
    #    #and second index is indexed by B, emb3, and emb4
    #    #where B runs over all fragments
    #    #emb3 runs over the core, bath, and virtual orbitals for each fragment corresponding to the fragment where site1 is an impurity
    #    #and emb4 runs over all non-redundant terms given emb3
    #    #note that phi is a square matrix with a single index having the same dimensions as the Y super-vector above
    #    phi = np.zeros( [sizeX,sizeX], dtype=complex )
    #    Lphiidx = 0
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:

    #                Rphiidx = 0
    #                for ifragB, fragB in enumerate(self.frag_list):
    #                    for emb3 in np.concatenate( ( fragB.virtrange, fragB.bathrange, fragB.corerange ) ):
    #        
    #                        if emb3 in fragB.virtrange:
    #                            emb4range = np.concatenate( (fragB.bathrange,fragB.corerange) )
    #                        elif emb3 in fragB.bathrange:
    #                            emb4range = np.concatenate( (fragB.virtrange,fragB.corerange) )
    #                        elif emb3 in fragB.corerange:
    #                            emb4range = np.concatenate( (fragB.virtrange,fragB.bathrange) )
    #        
    #                        for emb4 in emb4range:
    #                            phi[Lphiidx,Rphiidx] = self.calc_phi_term( ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega )
    #                            Rphiidx += 1

    #                Lphiidx += 1

    #    #Solve inversion equation for X super vector
    #    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

    ######################################################################

    #def solve_Xvec_parallel( self, Nocc, nproc ):
    #    #Subroutine to solve the coupled equations for the
    #    #X-matrices for all of the fragments in parallel
    #    #Returns the super-vector containing the elements of all the non-redundant terms of the X-matrices for each fragment
    #    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #    #Calculate contraction between rotation matrix of each fragment and natural orbitals of global 1RDM
    #    self.get_natorb_rotmat_contraction()

    #    #Calculate first time-derivative of correlated 1RDMS for each fragment
    #    self.get_frag_ddt_corr1RDM()

    #    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    #    chi = np.zeros( [self.Nsites,self.Nsites] )
    #    for i in range(Nocc):
    #        for j in range(Nocc,self.Nsites):
    #            chi[i,j] = 1.0/(self.NOevals[i]-self.NOevals[j])
    #            chi[j,i] = 1.0/(self.NOevals[j]-self.NOevals[i])

    #    #Calculate size of X-vec and the matrices needed to calculate it
    #    #Given by A*emb1*emb2, where A runs over all fragments
    #    #emb1 runs over the core, bath, and virtual orbitals for each fragment A
    #    #and emb2 runs over all non-redundant terms given emb1
    #    #also define a dictionary that takes the tuple (A,emb1,emb2) and outputs the index in the X-vec
    #    sizeX = 0
    #    indxdict = {}
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:
    #                indxdict[ (ifragA, emb1, emb2) ] = sizeX
    #                sizeX += 1

    #    ###### Form omega super-matrix in parallel indexed by same as X-vec, site orbital-embedding orbital ######
    #    #Note that number of site orbitals and embedding orbitals is the same - the size of the system

    #    #Form list of all necessary indices to send to Pool
    #    omega_indx_list = []
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:
    #                for site1 in range(self.Nsites):
    #                    for emb3 in range(self.Nsites):

    #                        omega_indx_list.append( ( fragA, emb1, emb2, site1, emb3, Nocc, chi ) )

    #    #Calculate terms in omega in parallel
    #    omega_pool = multproc.Pool(nproc)
    #    omega = omega_pool.starmap( self.calc_omega_term, omega_indx_list )
    #    omega_pool.close()

    #    #Unpack results into omega super-matrix
    #    omega = np.asarray(omega)
    #    omega = omega.reshape((sizeX, self.Nsites**2))

    #    ###### Finished forming omega super-matrix in parallel ######


    #    ###### Form Y super-vector in parallel ######

    #    #Form list of all indices to send to Pool
    #    Yvec_indx_list = []
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:

    #                Yvec_indx_list.append( ( ifragA, fragA, emb1, emb2, indxdict, omega ) )

    #    #Calculate terms in Y in parallel
    #    Yvec_pool = multproc.Pool(nproc)
    #    Yvec = Yvec_pool.starmap( self.calc_Yvec_term, Yvec_indx_list )
    #    Yvec_pool.close()
    #    Yvec = np.asarray(Yvec)

    #    ###### Finished forming Y super-vector in parallel ######


    #    ###### Form phi super-matrix in parallel ######

    #    #the first index is same as the Y super-vector above
    #    #and second index is indexed by B, emb3, and emb4
    #    #where B runs over all fragments
    #    #emb3 runs over the core, bath, and virtual orbitals for each fragment corresponding to the fragment where site1 is an impurity
    #    #and emb4 runs over all non-redundant terms given emb3
    #    #note that phi is a square matrix with a single index having the same dimensions as the Y super-vector above

    #    #Form list of all necessary indices to send to Pool
    #    phi_indx_list = []
    #    for ifragA, fragA in enumerate(self.frag_list):
    #        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):

    #            if emb1 in fragA.virtrange:
    #                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
    #            elif emb1 in fragA.bathrange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
    #            elif emb1 in fragA.corerange:
    #                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )

    #            for emb2 in emb2range:
    #                for ifragB, fragB in enumerate(self.frag_list):
    #                    for emb3 in np.concatenate( ( fragB.virtrange, fragB.bathrange, fragB.corerange ) ):
    #        
    #                        if emb3 in fragB.virtrange:
    #                            emb4range = np.concatenate( (fragB.bathrange,fragB.corerange) )
    #                        elif emb3 in fragB.bathrange:
    #                            emb4range = np.concatenate( (fragB.virtrange,fragB.corerange) )
    #                        elif emb3 in fragB.corerange:
    #                            emb4range = np.concatenate( (fragB.virtrange,fragB.bathrange) )
    #        
    #                        for emb4 in emb4range:

    #                            phi_indx_list.append( (ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega) )

    #    #Calculate terms in phi in parallel
    #    phi_pool = multproc.Pool(nproc)
    #    phi = phi_pool.starmap( self.calc_phi_term, phi_indx_list, 1 )
    #    phi_pool.close()

    #    #Unpack results into phi super-matrix
    #    phi = np.asarray(phi)
    #    phi = phi.reshape((sizeX, sizeX))

    #    ###### End forming phi super-matrix in parallel ######

    #    #Solve inversion equation for X super vector
    #    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

    ######################################################################

    #def calc_omega_term( self, fragA, emb1, emb2, site1, emb3, Nocc, chi ):

    #    #calculate a term in the omega super-matrix given the appropriate indices
    #    #einsum summing over natural orbitals

    #    return np.einsum( 'j,j,ij,i,i', np.conjugate(fragA.NO_rot[Nocc:,emb1]), self.frag_list[self.site_to_frag_list[site1]].NO_rot[Nocc:,emb3], \
    #                      chi[:Nocc,Nocc:], self.NOevecs[site1,:Nocc], fragA.NO_rot[:Nocc,emb2] ) + \
    #           np.einsum( 'i,i,ij,j,j', np.conjugate(fragA.NO_rot[:Nocc,emb1]), self.frag_list[self.site_to_frag_list[site1]].NO_rot[:Nocc,emb3], \
    #                      chi[:Nocc,Nocc:], self.NOevecs[site1,Nocc:], fragA.NO_rot[Nocc:,emb2] )

    ######################################################################

    #def calc_Yvec_term( self, ifragA, fragA, emb1, emb2, indxdict, omega ):

    #    #Calculate a term in the Y super vector given the appropriate indices

    #    #Left-indices for omega matrix
    #    Lidx1 = indxdict[ (ifragA, emb1, emb2) ]
    #    Lidx2 = indxdict[ (ifragA, emb2, emb1) ]

    #    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
    #    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    #    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
    #    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

    #    Yval = 0.0
    #    for site1 in range(self.Nsites):
    #        frag_site1 = self.frag_list[self.site_to_frag_list[site1]]
    #        for emb3 in np.concatenate( ( frag_site1.imprange, frag_site1.bathrange ) ):

    #            #Right index for omega matrix
    #            Ridx = emb3+site1*self.Nsites

    #            #Values of d/dt correlated 1RDM
    #            if emb3 in frag_site1.bathrange:
    #                emb3 -= frag_site1.Nvirt #necessary to subtract off Nvirt only for bath orbitals b/c embedding orbitals go as imp,virt,bath,core
    #            val1  = 1j*frag_site1.ddt_corr1RDM[ emb3, self.site_to_impindx[site1] ]
    #            val2  = 1j*frag_site1.ddt_corr1RDM[ self.site_to_impindx[site1], emb3 ]

    #            #Sum terms into element of Y super-vector
    #            Yval += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx] * val1 + val2 * np.conjugate(omega[Lidx2,Ridx]) )

    #    return Yval

    ######################################################################

    #def calc_phi_term( self, ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega ):

    #    #Calculate a term in the phi super matrix

    #    #Left-indices for omega matrix
    #    Lidx1 = indxdict[ (ifragA, emb1, emb2) ]
    #    Lidx2 = indxdict[ (ifragA, emb2, emb1) ]

    #    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
    #    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    #    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
    #    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

    #    phi_val = 0.0
    #    for site1 in fragB.impindx: #site1 corresponds to the index in the site basis of the impurity orbitals of fragment B
    #    
    #        #Right indices for omega matrix
    #        Ridx1 = emb3+site1*self.Nsites
    #        Ridx2 = emb4+site1*self.Nsites
    #    
    #        #Values of correlated 1RDM
    #        #Note that if embedding orbital (ie emb3 or emb4) is not a bath orbital, then the correlated 1RDM is zero
    #        #because site1 always corresponds to an impurity orbital
    #        if emb4 in fragB.bathrange:
    #            val1 = fragB.corr1RDM[ emb4-fragB.Nvirt, self.site_to_impindx[site1] ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
    #        else:
    #            val1 = 0.0
    #        if emb3 in fragB.bathrange:
    #            val2 = fragB.corr1RDM[ self.site_to_impindx[site1], emb3-fragB.Nvirt ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
    #        else:
    #            val2 = 0.0
    #    
    #        #sum terms into phi super-matrix
    #        phi_val += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx1]*val1 - np.conjugate(omega[Lidx2,Ridx2])*val2 )

    #    return phi_val

    ######################################################################
