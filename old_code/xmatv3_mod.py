#Mod that contains subroutines associated with calculating the different
#X-matrices associated with each fragment when using the formulation
#to directly propagate the orbitals for each fragment and not propagating the MF density matrix

import numpy as np
import multiprocessing as multproc
import time

#####################################################################

def solve_Xvec_serial( system, Nocc ):

    #Subroutine to solve the coupled equations for the
    #X-matrices for all of the fragments in serial
    #Returns the super-vector containing the elements of all the non-redundant terms of the X-matrices for each fragment
    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #Calculate contraction between rotation matrix of each fragment and natural orbitals of global 1RDM
    system.get_natorb_rotmat_contraction()
    
    #Calculate first time-derivative of correlated 1RDMS for each fragment
    system.get_frag_ddt_corr1RDM()

    #Calculate size of X-vec and the matrices needed to calculate it
    #Given by A*emb1*emb2, where A runs over all fragments
    #emb1 runs over the core, bath, and virtual orbitals for each fragment A
    #and emb2 runs over all non-redundant terms given emb1
    #also define a dictionary that takes the tuple (A,emb1,emb2) and outputs the index in the X-vec
    sizeX = 0
    indxdict  = {}
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
                indxdict[ (ifragA, emb1, emb2) ] = sizeX
                sizeX += 1

    #Make an array that re-orders index of X-vec to be A,emb1,emb2 given convention above
    reorder_indx = np.zeros(sizeX,dtype=int)
    cnt = 0
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
                reorder_indx[cnt] = indxdict[ (ifragA,emb2,emb1) ]
                cnt += 1

    #Calculate 2-index omega super-matrix
    omega = calc_omega_mat( system, indxdict, sizeX, Nocc )

    #Calculate Y super-vector
    Yvec = calc_Yvec( system, indxdict, reorder_indx, sizeX, omega )

    #Calculate phi super-matrix, phi is a square matrix with same dimension and indexing as X-vec
    phi = calc_phi_mat( system, indxdict, reorder_indx, sizeX, omega )

    #Solve inversion equation for X super vector
    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

#####################################################################

def calc_omega_mat( system, indxdict, sizeX, Nocc ):

    #Form omega super-matrix indexed by same as X-vec, site orbital-embedding orbital
    #such that second index of omega matrix is given by indx = emb + Nsites * site
    #Note that number of site orbitals and embedding orbitals is the same - the size of the system

    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    chi = np.zeros( [ Nocc, system.Nsites ] )
    for mu in range(Nocc):
        for nu in range(system.Nsites):
            if( mu != nu ):
                chi[ mu, nu ] = 1.0/(system.NOevals[mu]-system.NOevals[nu])

    #Unpack natural orbital - rotation matrix contraction into a 3-index tensor containing all fragments
    #Unpack product of natural orbital - rotation matrix contraction into a 3-index tensor containing all fragments
    #Indexed by X-vec index, natural orbital, natural orbital
    unpck_NO_rot = np.zeros( [ sizeX, system.Nsites, system.Nsites ], dtype=complex )
    for indx, iX in indxdict.items():
        ifrag, emb1, emb2 = indx
        NO_rot = system.frag_list[ ifrag ].NO_rot
        unpck_NO_rot[iX,:,:] = np.einsum( 'i,j->ji', np.conjugate(NO_rot[:,emb1]), NO_rot[:,emb2] )

    #Form four-index tensor corresponding to product of natural orbital - rotation matrix contraction and the natural orbitals
    #where the 4 indices are site index, embedding orbital, natural orbital, natural orbital
    #ie calculating the term big_r,c,mu,nu = phi^F(r)_nu,c * U_r,mu, where r is a site orbital, F(r) is the fragment corresponding to that site
    #c is an embedding orbital, mu and nu are natural orbitals, U are the natural orbitals
    #and phi is the contraction of the natural orbital and the rotation matrix for fragment F(r)
    big = np.zeros( [ system.Nsites, system.Nsites, system.Nsites, system.Nsites ], dtype=complex  )
    for r in range(system.Nsites):
        ifrag = system.site_to_frag_list[r]
        frag  = system.frag_list[ifrag]
        big[ r, :, :, : ] = np.einsum( 'ic,j->cji', frag.NO_rot, system.NOevecs[r,:] )

    #reshape to a 3-index tensor where r and c are combined into a single index
    big = big.reshape( [ system.Nsites**2, system.Nsites, system.Nsites ] )

    #Form omega matrix by summing over natural orbitals
    omega  = np.einsum( 'ij,pij,qij->pq', chi, unpck_NO_rot[:,:Nocc,:], big[:,:Nocc,:] )
    omega += np.einsum( 'ij,pji,qji->pq', chi, unpck_NO_rot[:,:,:Nocc], big[:,:,:Nocc] )

    return omega

#####################################################################

def calc_Yvec( system, indxdict, reorder_indx, sizeX, omega ):

    #Calculate the Y super-vector with the same linear dimension as the X-matrix

    #Create array of combined indices of r and c, where r is over all sites, but c is only over impurity and bath orbitals
    #for the fragment corresponding to site r
    lst = []
    for r in range(system.Nsites):
        frag = system.frag_list[system.site_to_frag_list[r]]
        lst.extend( np.concatenate( ( frag.imprange, frag.bathrange) ) + r*system.Nsites )

    #Unpack d/dt of correlated 1RDM
    unpck_ddt_corr1RDM  = []
    for r in range(system.Nsites):
        ifrag   = system.site_to_frag_list[r]
        frag    = system.frag_list[ifrag]
        impindx = system.site_to_impindx[r]
        unpck_ddt_corr1RDM.extend( 1j * frag.ddt_corr1RDM[:,impindx] )

    #Contract omega matrix and d/dt of correlated 1RDM
    Yvec =  np.dot( omega[:,lst], unpck_ddt_corr1RDM )
    Yvec += np.dot( np.conjugate( omega[ reorder_indx[:,None],lst ] ), -np.conjugate( unpck_ddt_corr1RDM ) )

    #Multiply by 1 over difference of embedding orbital eigenvalues for each fragment
    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    for indx, iX in indxdict.items():
        ifrag, emb1, emb2 = indx
        frag  = system.frag_list[ifrag]
        eval1 = frag.env1RDM_evals[emb1-frag.Nimp]
        eval2 = frag.env1RDM_evals[emb2-frag.Nimp]

        Yvec[iX] *= 1.0/(eval2-eval1)

    return Yvec

#####################################################################

def calc_phi_mat( system, indxdict, reorder_indx, sizeX, omega ):

    #Calculate the phi super matrix, which is a square matrix with each dimension
    #given by the same dimension as the X-matrix

    #Contract omega matrix and correlated 1RDM
    phi = np.zeros( [ sizeX, sizeX ], dtype=complex )
    for indx, iX in indxdict.items():

        #indices associated with right index of phi matrix
        ifrag, emb1, emb2 = indx
        frag = system.frag_list[ ifrag ]

        #Note that emb1 and emb2 are never both in the bath

        if emb2 in frag.bathrange:

            #Appropriate right indices for omega matrix for the given fragment and embedding orbital
            omega_indx = emb1 + frag.impindx * system.Nsites

            #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
            emb2 = emb2 - frag.Nvirt

            #if statement only calculates this contribution to phi if emb2 is a bath orbital, otw this term is zero
            phi[:,iX] = np.dot( omega[:,omega_indx], frag.corr1RDM[emb2,:frag.Nimp] )

        if emb1 in frag.bathrange:

            #Appropriate right indices for omega matrix for the given fragment and embedding orbital
            omega_indx = emb2 + frag.impindx * system.Nsites

            #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
            emb1 = emb1 - frag.Nvirt

            #if statement only calculates this contribution to phi if emb2 is a bath orbital, otw this term is zero
            phi[:,iX] = -np.dot( np.conjugate( omega[reorder_indx[:,None],omega_indx] ), frag.corr1RDM[:frag.Nimp,emb1] )

    #Multiply by 1 over difference of embedding orbital eigenvalues for each fragment associated with left-index of phi matrix
    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    for indx, iX in indxdict.items():
        ifrag, emb1, emb2 = indx
        frag  = system.frag_list[ifrag]
        eval1 = frag.env1RDM_evals[emb1-frag.Nimp]
        eval2 = frag.env1RDM_evals[emb2-frag.Nimp]

        phi[iX,:] *= 1.0/(eval2-eval1)

    return phi

#####################################################################
