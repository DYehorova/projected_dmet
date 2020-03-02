#Mod that contatins subroutines necessary to calculate the analytical form
#of the first time-derivative of the mean-field 1RDM
#this is necessary when integrating the MF 1RDM and CI coefficients explicitly
#while diagonalizing the MF 1RDM at each time-step to obtain embedding orbitals

import utils
import numpy as np
import multiprocessing as multproc

import time

#####################################################################

def get_ddt_mf1rdm_serial( system, Nocc, nproc ):

    #Subroutine to solve the inversion equation for the first
    #time-derivative of the mf 1RDM

    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    t1 = time.time() #grr

    #Calculate 4-index U tensor
    Umat = calc_Umat( system.NOevals, system.NOevecs, system.Nsites, Nocc )

    t2 = time.time() #grr

    #Calculate 4-index R tensor
    Rmat = calc_Rmat( system )

    t3 = time.time() #grr

    #Calculate 2-index theta super-matrix
    thetamat = calc_thetamat( Umat, Rmat, system.Nsites )

    t4 = time.time() #grr

    #Calculate 2-index phi matrix
    phimat = calc_phimat( system, nproc )

    t5 = time.time() #grr

    #Calculate Y super-vector
    Yvec = calc_Yvec( Umat, phimat, system.Nsites )

    t6 = time.time() #grr

    #Solve inversion equation for time derivative of mean-field 1RDM as super-vector
    ddt_mf1rdm = np.linalg.solve( thetamat, Yvec)

    t7 = time.time() #grr

    #grr
    #print()
    #print('U = ',t2-t1)
    #print('R = ',t3-t2)
    #print('theta = ',t4-t3)
    #print('phi = ',t5-t4)
    #print('Y = ',t6-t5)
    #print('inv = ',t7-t6)

    #print()

    #Unpack super-vector to normal matrix form of time derivative of mean-field 1RDM
    return ddt_mf1rdm.reshape( [ system.Nsites, system.Nsites ] )

#####################################################################

def calc_Umat( natevals, natorbs, Nsites, Nocc ):

    #Subroutine to calculate the summed U tensor

    #First form matrix given by one over the difference in the global 1RDM natural orbital evals 
    #(ie just the evals of the global 1RDM)
    #This only has to be defined for the occupied - virtual block b/c MF rdm invariant to intra-space rotations
    chi = np.zeros( [ Nocc, (Nsites-Nocc) ] )
    for mu in range(Nocc):
        for nu in range(Nocc,Nsites):
            chi[ mu, nu-Nocc ] = 1.0/(natevals[mu]-natevals[nu])

    #Adjoint of natural orbitals
    adj_natorbs = utils.adjoint( natorbs )

    #Form U tensor by summing over natural orbitals
    #PING can calculate U in this form more quickly using its hermiticity
    U = np.einsum( 'ij,ri,iq -> jqr', chi, natorbs[:,:Nocc], adj_natorbs[:Nocc,:] )
    U = np.einsum( 'pj,js,jqr -> sqpr', natorbs[:,Nocc:], adj_natorbs[Nocc:,:], U )

    #Return the total summed U matrix
    return U + np.einsum( 'sqpr -> qsrp', U )

#####################################################################

def calc_Rmat( system ):

    #Subroutine to calculate the 4-index R tensor
    #Currently assume that all fragments have same number of impurities

    Ncore = system.frag_list[0].Ncore
    Nvirt = system.frag_list[0].Nvirt
    Nbath = system.frag_list[0].Nimp
    Nsites = system.Nsites
    bathrange = system.frag_list[0].bathrange

    #Concatenated list of virtual and core orbitals
    virtcore = np.concatenate( ( system.frag_list[0].virtrange, system.frag_list[0].corerange ) )

    #Unpack necessary components for each fragment
    rotmat_unpck   = np.zeros( [ Nsites, Nsites, Nsites ], dtype=complex )
    corr1RDM_unpck = np.zeros( [ Nbath, Nsites ], dtype=complex )
    chi            = np.zeros( [ Nbath, Ncore+Nvirt, Nsites ] )
    for r in range( Nsites ):

        #Fragment associated with site r
        frag = system.frag_list[ system.site_to_frag_list[r] ]

        #unpacked embedding orbitals
        rotmat_unpck[:,:,r] = np.copy( frag.rotmat )

        #unpacked correlated 1RDM
        corr1RDM_unpck[:,r] = frag.corr1RDM[ frag.Nimp:, system.site_to_impindx[r] ]

        #Unpack the calculated one over the difference in the environment orbital eigenvalues
        #This only has to be defined for the bath - core/virtual space
        #Subtract off terms b/c indexing of embedding orbitals goes as imp,virt,bath,core
        i = 0
        for a in frag.bathrange:
            j = 0
            for c in virtcore:
                chi[i,j,r] = 1.0/( frag.env1RDM_evals[a-frag.Nimp] - frag.env1RDM_evals[c-frag.Nimp] )
                j += 1
            i += 1

    #Form R tensor by summing over embedding orbitals
    R = np.einsum( 'acr,uar,ar -> cur', chi, rotmat_unpck[ :, bathrange, : ], corr1RDM_unpck )
    R = np.einsum( 'scr,tcr,cur -> stur', rotmat_unpck[ :, virtcore, :], np.conjugate( rotmat_unpck[ :, virtcore, : ] ), R )

    return R

#####################################################################

def calc_thetamat( Umat, Rmat, Nsites ):

    #Subroutine to calculate the 4-index theta matrix

    #Contract Umat and Rmat
    thetamat = np.dot( np.einsum( 'sqpr -> pqsr', Umat ).reshape(Nsites**2,Nsites**2), np.einsum( 'stur -> srtu', Rmat ).reshape(Nsites**2,Nsites**2)  )

    #Sum up theta matrix
    thetamat = thetamat + np.conjugate( np.einsum( 'pqtu -> qput', thetamat.reshape(Nsites,Nsites,Nsites,Nsites) ).reshape(Nsites**2,Nsites**2) )

    #Return theta matrix subtracted from identity matrix
    return np.eye(Nsites**2)-thetamat

#####################################################################

def calc_phimat( system, nproc ):

    #Subroutine to calculate the 2-index phi matrix

    #Calculate first time-derivative of correlated 1RDMS for each fragment
    system.get_frag_ddt_corr1RDM( nproc )

    phimat = np.zeros( [ system.Nsites, system.Nsites ], dtype=complex )
    for r in range( system.Nsites ):

        #Fragment corresponding to site r
        frag = system.frag_list[system.site_to_frag_list[r]]

        #Impurity index within fragment corresponding to site r
        rimp = system.site_to_impindx[r]

        #Impurity and bath orbital range
        impbath = np.concatenate( ( frag.imprange, frag.bathrange ) )

        #Calculate one column of phi-matrix
        phimat[:,r] = np.dot( frag.rotmat[:,impbath], 1j * frag.ddt_corr1RDM[:,rimp] )

    return phimat

#####################################################################

def calc_Yvec( Umat, phimat, Nsites ):

    #Subroutine to calculate the Y super-vector

    #Contract Umat and phimat
    Yvec = np.einsum( 'sqpr,sr -> pq', Umat, phimat )

    #Sum up Y vector
    Yvec = Yvec - utils.adjoint(Yvec)

    #Reshape Y vector to be a super-vector
    return -1j*Yvec.reshape(Nsites**2)

#####################################################################


