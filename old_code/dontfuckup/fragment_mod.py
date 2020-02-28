#Define a class for a fragment, including all quantities specific to a given fragment

import numpy as np
import fci_mod
import sys
import os
import utils
import applyham_pyscf

import pyscf.fci #mrar

######## FRAGMENT CLASS #######

class fragment():

    #####################################################################

    def __init__( self, impindx, Nsites, Nele ):
        self.impindx = impindx #array defining index of impurity orbitals in site basis
        self.Nimp    = impindx.shape[0] #number of impurity orbitals in fragment
        self.Nsites  = Nsites #total number of sites (or basis functions) in total system
        self.Nele    = Nele #total number of electrons in total system

        self.Ncore = int(Nele/2) - self.Nimp #Number of core orbitals in fragment
        self.Nvirt = Nsites - 2*self.Nimp - self.Ncore #Number of virtual orbitals in fragment

        #range of orbitals in embedding basis, embedding basis always indexed as impurity, virtual, bath, core
        self.imprange  = range(0, self.Nimp)
        self.virtrange = range(self.Nimp, self.Nimp+self.Nvirt)
        self.bathrange = range(self.Nimp+self.Nvirt, 2*self.Nimp+self.Nvirt)
        self.corerange = range(2*self.Nimp+self.Nvirt, self.Nsites)

    #####################################################################

    def get_rotmat( self, mf1RDM ):
        #Subroutine to generate rotation matrix from site to embedding basis
        #PING currently impurities have to be listed in ascending order (though dont have to be sequential)

        #remove rows/columns corresponding to impurity sites from mf 1RDM
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 0 )
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 1 )

        #print(mf1RDM)
        #print()#msh

        #diagonalize environment part of 1RDM to obtain embedding (virtual, bath, core) orbitals
        evals, evecs = np.linalg.eigh( mf1RDM )

        #print(evals)
        #print()
        #print(evecs)#msh
        #print()
        #print('*********************************')
        #print()

        #form rotation matrix consisting of unit vectors for impurity and the evecs for embedding
        #rotation matrix is ordered as impurity, virtual, bath, core
        self.rotmat = np.zeros( [ self.Nsites, self.Nimp ] )
        for imp in range(self.Nimp):
            indx                     = self.impindx[imp]
            self.rotmat[ indx, imp ] = 1.0
            evecs                    = np.insert( evecs, indx, 0.0, axis=0 )

        self.rotmat = np.concatenate( (self.rotmat,evecs), axis=1 )
        self.env1RDM_evals = evals

    #####################################################################
        
    def get_Hemb( self, h_site, V_site, hamtype=0 ):
        #Subroutine to the get the 1 and 2 e- terms of the Hamiltonian in the embedding basis
        #Transformation accounts for interaction with the core
        #Also calculates 1 e- term with only 1/2 interaction with the core - this is used in calculation of DMET energy

        #remove the virtual states from the rotation matrix
        #the rotation matrix is of form ( site basis fcns ) x ( impurities, virtual, bath, core )
        rotmat_small = np.delete( self.rotmat, np.s_[self.Nimp:self.Nimp+self.Nvirt], 1 )

        #rotate the 1 e- terms, h_emb currently ( impurities, bath, core ) x ( impurities, bath, core )
        h_emb = utils.rot1el( h_site, rotmat_small )

        #define 1 e- term of size ( impurities, bath ) x ( impurities, bath ) that will only have 1/2 interaction with the core
        self.h_emb_halfcore = np.copy( h_emb[ :2*self.Nimp, :2*self.Nimp ] )

        #rotate the 2 e- terms
        if( hamtype == 0 ):
            #General hamiltonian, V_emb currently ( impurities, bath, core ) ^ 4
            V_emb = utils.rot2el_chem( V_site, rotmat_small )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            rotmat_vsmall = rotmat_small[:,:2*self.Nimp] #remove core states from rotation matrix
            self.V_emb = V_site*np.einsum( 'ap,cp,pb,pd->abcd', utils.adjoint( rotmat_vsmall ), utils.adjoint( rotmat_vsmall ), rotmat_vsmall, rotmat_vsmall )


        #augment the impurity/bath 1e- terms from contribution of coulomb and exchange terms btwn impurity/bath and core
        #and augment the 1 e- term with only half the contribution from the core to be used in DMET energy calculation
        if( hamtype == 0 ):
            #General hamiltonian
            for core in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):
                h_emb[ :2*self.Nimp, :2*self.Nimp ] = h_emb[ :2*self.Nimp, :2*self.Nimp ] + 2*V_emb[ :2*self.Nimp, :2*self.Nimp, core, core ] - V_emb[ :2*self.Nimp, core, core, :2*self.Nimp ]
                self.h_emb_halfcore += V_emb[ :2*self.Nimp, :2*self.Nimp, core, core ] - 0.5*V_emb[ :2*self.Nimp, core, core, :2*self.Nimp ]
        elif( hamtype == 1):
            #Hubbard hamiltonian
            core_int = V_site * np.einsum( 'ap,pb,p->ab', utils.adjoint( rotmat_vsmall ), rotmat_vsmall, np.einsum( 'pe,ep->p',rotmat_small[:,2*self.Nimp:], utils.adjoint( rotmat_small[:,2*self.Nimp:] ) ) )
            h_emb[ :2*self.Nimp, :2*self.Nimp ] += core_int
            self.h_emb_halfcore += 0.5*core_int


        #calculate the energy associated with core-core interactions, setting it numerically to a real number since it always will be
        Ecore = 0
        for core1 in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):

            Ecore += 2*h_emb[ core1, core1 ]

            if( hamtype == 0 ):
                #General hamiltonian
                for core2 in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):
                    Ecore += 2*V_emb[ core1, core1, core2, core2 ] - V_emb[ core1, core2, core2, core1 ]
            elif( hamtype == 1):
                #Hubbard hamiltonian
                vec = np.einsum( 'pe,ep->p',rotmat_small[:,2*self.Nimp:],utils.adjoint( rotmat_small[:,2*self.Nimp:] ) )
                Ecore += V_site * np.einsum( 'p,p', vec, vec )


        self.Ecore = Ecore.real

        #shrink h_emb and V_emb arrays to only include the impurity and bath
        self.h_emb = h_emb[ :2*self.Nimp, :2*self.Nimp ]
        if( hamtype == 0 ):
            #General hamiltonian
            self.V_emb = V_emb[ :2*self.Nimp, :2*self.Nimp, :2*self.Nimp, :2*self.Nimp ]

    #####################################################################

    def add_mu_Hemb( self, mu ):

        #Subroutine to add a chemical potential, mu, to only the impurity sites of embedding Hamiltonian
        for i in range( self.Nimp ):
            self.h_emb[i,i] += mu

    #####################################################################

    def solve_GS( self ):
        #Use the embedding hamiltonian to solve for the FCI ground-state

        self.CIcoeffs = fci_mod.FCI_GS( self.h_emb, self.V_emb, self.Ecore, 2*self.Nimp, (self.Nimp,self.Nimp) )

    #####################################################################

    def get_corr1RDM( self ):
        #Subroutine to get the FCI 1RDM
    
        self.corr1RDM = fci_mod.get_corr1RDM( self.CIcoeffs, 2*self.Nimp, (self.Nimp,self.Nimp) )

    #####################################################################

    def get_corr12RDM( self ):
        #Subroutine to get the FCI 1RDM and 2RDM
    
        self.corr1RDM, self.corr2RDM = fci_mod.get_corr12RDM( self.CIcoeffs, 2*self.Nimp, (self.Nimp,self.Nimp) )

    #####################################################################

    def static_corr_calc( self, mf1RDM, mu, h_site, V_site, hamtype=0 ):
        #Subroutine to perform all steps of the static correlated calculation

        self.get_rotmat( mf1RDM ) #1) get rotation matrix to embedding basis
        self.get_Hemb( h_site, V_site, hamtype ) #2) use rotation matrix to compute embedding hamiltonian
        self.add_mu_Hemb( mu ) #3) add chemical potential to only impurity sites of embedding hamiltonian
        self.solve_GS() #4) perform corrleated calculation using embedding hamiltonian
        self.get_corr1RDM() #5) calculate correlated 1RDM

    #####################################################################

    def get_frag_E( self ):
        #Subroutine to calculate contribution to DMET energy from fragment
        #Need to calculate embedding hamiltonian and 1/2 rdms prior to calling this routine
        #Using democratic partitioning using Eq. 28 from  Wouters JCTC 2016
        #This equation uses 1 e- part that only includes half the interaction with the core
        #Notation for 1RDM is rho_pq = < c_q^dag c_p >
        #Notation for 2RDM is gamma_pqrs = < c_p^dag c_r^dag c_s c_q >
        #Notation for 1 body terms h1[p,q] = <p|h|q>
        #Notation for 2 body terms V[p,q,r,s] = (pq|rs)

        #Calculate fragment energy using democratic partitioning
        self.Efrag = 0.0
        for orb1 in range(self.Nimp):
            for orb2 in range(2*self.Nimp):
                self.Efrag += self.h_emb_halfcore[ orb1, orb2 ] * self.corr1RDM[ orb2, orb1 ]
                for orb3 in range(2*self.Nimp):
                    for orb4 in range(2*self.Nimp):
                        self.Efrag += 0.5 * self.V_emb[ orb1, orb2, orb3, orb4 ] * self.corr2RDM[ orb1, orb2, orb3, orb4 ]

    #####################################################################

    def get_ddt_corr1RDM( self ):
        #Subroutine to calculate first time-derivative of correlated 1RDM
        #Only calculated in the necessary impurity-bath space

        Ctil = applyham_pyscf.apply_ham_pyscf_fully_complex( self.CIcoeffs, self.h_emb, self.V_emb, self.Nimp, self.Nimp, 2*self.Nimp, self.Ecore )

        rdmtil = fci_mod.get_trans1RDM( self.CIcoeffs, Ctil, 2*self.Nimp, 2*self.Nimp )

        self.ddt_corr1RDM = -1j*( rdmtil - utils.adjoint( rdmtil ) )

    #####################################################################

    def contract_natorb_rotmat( self, NOevecs ):
        #Subroutine to contract the rotation matrix of the given fragment with
        #the natural orbitals of the total system global 1RDM

        self.NO_rot = utils.matprod( utils.adjoint( NOevecs), self.rotmat )

    #####################################################################

    def init_Xmat( self ):
        #Subroutine to initialize the X-matrix to zero

        self.Xmat = np.zeros( [self.Nsites, self.Nsites], dtype=complex )

    #####################################################################
