#Mod that contains subroutines associated with calculating the different
#X-matrices associated with each fragment when using the formulation
#to directly propagate the orbitals for each fragment and not propagating the MF density matrix

import numpy as np
import multiprocessing as multproc

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
    
    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    chi = np.zeros( [system.Nsites,system.Nsites] )
    for i in range(Nocc):
        for j in range(Nocc,system.Nsites):
            chi[i,j] = 1.0/(system.NOevals[i]-system.NOevals[j])
            chi[j,i] = 1.0/(system.NOevals[j]-system.NOevals[i])

    #Calculate size of X-vec and the matrices needed to calculate it
    #Given by A*emb1*emb2, where A runs over all fragments
    #emb1 runs over the core, bath, and virtual orbitals for each fragment A
    #and emb2 runs over all non-redundant terms given emb1
    #also define a dictionary that takes the tuple (A,emb1,emb2) and outputs the index in the X-vec
    sizeX = 0
    indxdict = {}
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
    
    #Form omega super-matrix indexed by same as X-vec, site orbital-embedding orbital
    #Note that number of site orbitals and embedding orbitals is the same - the size of the system
    omega = np.zeros( [sizeX, system.Nsites**2], dtype=complex )
    Lidx  = 0 #Left-index for super-matrix
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
    
                for site1 in range(system.Nsites):
                    for emb3 in range(system.Nsites):
    
                        #Right Index for super matrix
                        Ridx = emb3+site1*system.Nsites
    
                        #calculate omega super-matrix
                        omega[Lidx,Ridx] = calc_omega_term( system, fragA, emb1, emb2, site1, emb3, Nocc, chi )
    
                Lidx += 1
    
    #Form Y super-vector
    Yidx = 0
    Yvec = np.zeros( sizeX, dtype=complex )
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
    
                #Calculate element of Y vector
                Yvec[Yidx] = calc_Yvec_term( system, ifragA, fragA, emb1, emb2, indxdict, omega )
    
                Yidx += 1
    
    #Form phi super-matrix where first index is same as the Y super-vector above
    #and second index is indexed by B, emb3, and emb4
    #where B runs over all fragments
    #emb3 runs over the core, bath, and virtual orbitals for each fragment corresponding to the fragment where site1 is an impurity
    #and emb4 runs over all non-redundant terms given emb3
    #note that phi is a square matrix with a single index having the same dimensions as the Y super-vector above
    phi = np.zeros( [sizeX,sizeX], dtype=complex )
    Lphiidx = 0
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
    
                Rphiidx = 0
                for ifragB, fragB in enumerate(system.frag_list):
                    for emb3 in np.concatenate( ( fragB.virtrange, fragB.bathrange, fragB.corerange ) ):
    
                        if emb3 in fragB.virtrange:
                            emb4range = np.concatenate( (fragB.bathrange,fragB.corerange) )
                        elif emb3 in fragB.bathrange:
                            emb4range = np.concatenate( (fragB.virtrange,fragB.corerange) )
                        elif emb3 in fragB.corerange:
                            emb4range = np.concatenate( (fragB.virtrange,fragB.bathrange) )
    
                        for emb4 in emb4range:
                            phi[Lphiidx,Rphiidx] = calc_phi_term( system, ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega )
                            Rphiidx += 1
    
                Lphiidx += 1
    
    #Solve inversion equation for X super vector
    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

#####################################################################

def solve_Xvec_parallel( system, Nocc, nproc ):
    #Subroutine to solve the coupled equations for the
    #X-matrices for all of the fragments in parallel
    #Returns the super-vector containing the elements of all the non-redundant terms of the X-matrices for each fragment
    #NOTE: prior to this routine being called, necessary to have the rotation matrices and 1RDM for each fragment
    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #Calculate contraction between rotation matrix of each fragment and natural orbitals of global 1RDM
    system.get_natorb_rotmat_contraction()

    #Calculate first time-derivative of correlated 1RDMS for each fragment
    system.get_frag_ddt_corr1RDM()

    #Form matrix given by one over the difference in the global 1RDM natural orbital evals (ie just the evals of the global 1RDM)
    chi = np.zeros( [system.Nsites,system.Nsites] )
    for i in range(Nocc):
        for j in range(Nocc,system.Nsites):
            chi[i,j] = 1.0/(system.NOevals[i]-system.NOevals[j])
            chi[j,i] = 1.0/(system.NOevals[j]-system.NOevals[i])

    #Calculate size of X-vec and the matrices needed to calculate it
    #Given by A*emb1*emb2, where A runs over all fragments
    #emb1 runs over the core, bath, and virtual orbitals for each fragment A
    #and emb2 runs over all non-redundant terms given emb1
    #also define a dictionary that takes the tuple (A,emb1,emb2) and outputs the index in the X-vec
    sizeX = 0
    indxdict = {}
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

    ###### Form omega super-matrix in parallel indexed by same as X-vec, site orbital-embedding orbital ######
    #Note that number of site orbitals and embedding orbitals is the same - the size of the system
    
    #Form list of all necessary indices to send to Pool
    omega_indx_list = []
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
                for site1 in range(system.Nsites):
                    for emb3 in range(system.Nsites):
    
                        omega_indx_list.append( ( system, fragA, emb1, emb2, site1, emb3, Nocc, chi ) )
    
    #Calculate terms in omega in parallel
    omega_pool = multproc.Pool(nproc)
    omega = omega_pool.starmap( calc_omega_term, omega_indx_list )
    omega_pool.close()
    
    #Unpack results into omega super-matrix
    omega = np.asarray(omega)
    omega = omega.reshape((sizeX, system.Nsites**2))
    
    ###### Finished forming omega super-matrix in parallel ######


    ###### Form Y super-vector in parallel ######
    
    #Form list of all indices to send to Pool
    Yvec_indx_list = []
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
    
                Yvec_indx_list.append( ( system, ifragA, fragA, emb1, emb2, indxdict, omega ) )
    
    #Calculate terms in Y in parallel
    Yvec_pool = multproc.Pool(nproc)
    Yvec = Yvec_pool.starmap( calc_Yvec_term, Yvec_indx_list )
    Yvec_pool.close()
    Yvec = np.asarray(Yvec)
    
    ###### Finished forming Y super-vector in parallel ######


    ###### Form phi super-matrix in parallel ######
    
    #the first index is same as the Y super-vector above
    #and second index is indexed by B, emb3, and emb4
    #where B runs over all fragments
    #emb3 runs over the core, bath, and virtual orbitals for each fragment corresponding to the fragment where site1 is an impurity
    #and emb4 runs over all non-redundant terms given emb3
    #note that phi is a square matrix with a single index having the same dimensions as the Y super-vector above
    
    #Form list of all necessary indices to send to Pool
    phi_indx_list = []
    for ifragA, fragA in enumerate(system.frag_list):
        for emb1 in np.concatenate( ( fragA.virtrange, fragA.bathrange, fragA.corerange ) ):
    
            if emb1 in fragA.virtrange:
                emb2range = np.concatenate( (fragA.bathrange,fragA.corerange) )
            elif emb1 in fragA.bathrange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.corerange) )
            elif emb1 in fragA.corerange:
                emb2range = np.concatenate( (fragA.virtrange,fragA.bathrange) )
    
            for emb2 in emb2range:
                for ifragB, fragB in enumerate(system.frag_list):
                    for emb3 in np.concatenate( ( fragB.virtrange, fragB.bathrange, fragB.corerange ) ):
    
                        if emb3 in fragB.virtrange:
                            emb4range = np.concatenate( (fragB.bathrange,fragB.corerange) )
                        elif emb3 in fragB.bathrange:
                            emb4range = np.concatenate( (fragB.virtrange,fragB.corerange) )
                        elif emb3 in fragB.corerange:
                            emb4range = np.concatenate( (fragB.virtrange,fragB.bathrange) )
    
                        for emb4 in emb4range:
    
                            phi_indx_list.append( (system, ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega) )
    
    #Calculate terms in phi in parallel
    phi_pool = multproc.Pool(nproc)
    phi = phi_pool.starmap( calc_phi_term, phi_indx_list, 1 )
    phi_pool.close()
    
    #Unpack results into phi super-matrix
    phi = np.asarray(phi)
    phi = phi.reshape((sizeX, sizeX))
    
    ###### End forming phi super-matrix in parallel ######

    #Solve inversion equation for X super vector
    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

#####################################################################

def calc_omega_term( system, fragA, emb1, emb2, site1, emb3, Nocc, chi ):

    #calculate a term in the omega super-matrix given the appropriate indices
    #einsum summing over natural orbitals

    return np.einsum( 'j,j,ij,i,i', np.conjugate(fragA.NO_rot[Nocc:,emb1]), system.frag_list[system.site_to_frag_list[site1]].NO_rot[Nocc:,emb3], \
                      chi[:Nocc,Nocc:], system.NOevecs[site1,:Nocc], fragA.NO_rot[:Nocc,emb2] ) + \
           np.einsum( 'i,i,ij,j,j', np.conjugate(fragA.NO_rot[:Nocc,emb1]), system.frag_list[system.site_to_frag_list[site1]].NO_rot[:Nocc,emb3], \
                      chi[:Nocc,Nocc:], system.NOevecs[site1,Nocc:], fragA.NO_rot[Nocc:,emb2] )

#####################################################################

def calc_Yvec_term( system, ifragA, fragA, emb1, emb2, indxdict, omega ):

    #Calculate a term in the Y super vector given the appropriate indices

    #Left-indices for omega matrix
    Lidx1 = indxdict[ (ifragA, emb1, emb2) ]
    Lidx2 = indxdict[ (ifragA, emb2, emb1) ]

    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

    Yval = 0.0
    for site1 in range(system.Nsites):
        frag_site1 = system.frag_list[system.site_to_frag_list[site1]]
        for emb3 in np.concatenate( ( frag_site1.imprange, frag_site1.bathrange ) ):

            #Right index for omega matrix
            Ridx = emb3+site1*system.Nsites

            #Values of d/dt correlated 1RDM
            if emb3 in frag_site1.bathrange:
                emb3 -= frag_site1.Nvirt #necessary to subtract off Nvirt only for bath orbitals b/c embedding orbitals go as imp,virt,bath,core
            val1  = 1j*frag_site1.ddt_corr1RDM[ emb3, system.site_to_impindx[site1] ]
            val2  = 1j*frag_site1.ddt_corr1RDM[ system.site_to_impindx[site1], emb3 ]

            #Sum terms into element of Y super-vector
            Yval += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx] * val1 + val2 * np.conjugate(omega[Lidx2,Ridx]) )

    return Yval

#####################################################################

def calc_phi_term( system, ifragA, fragA, emb1, emb2, fragB, emb3, emb4, indxdict, omega ):

    #Calculate a term in the phi super matrix

    #Left-indices for omega matrix
    Lidx1 = indxdict[ (ifragA, emb1, emb2) ]
    Lidx2 = indxdict[ (ifragA, emb2, emb1) ]

    #Eigenvalues associated with orbitals emb1 and emb2 of environment part of 1RDM for fragA
    #Subtract off Nimp because indexing of embedding orbitals goes as imp,virt,bath,core
    eval1 = fragA.env1RDM_evals[emb1-fragA.Nimp]
    eval2 = fragA.env1RDM_evals[emb2-fragA.Nimp]

    phi_val = 0.0
    for site1 in fragB.impindx: #site1 corresponds to the index in the site basis of the impurity orbitals of fragment B

        #Right indices for omega matrix
        Ridx1 = emb3+site1*system.Nsites
        Ridx2 = emb4+site1*system.Nsites

        #Values of correlated 1RDM
        #Note that if embedding orbital (ie emb3 or emb4) is not a bath orbital, then the correlated 1RDM is zero
        #because site1 always corresponds to an impurity orbital
        if emb4 in fragB.bathrange:
            val1 = fragB.corr1RDM[ emb4-fragB.Nvirt, system.site_to_impindx[site1] ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
        else:
            val1 = 0.0
        if emb3 in fragB.bathrange:
            val2 = fragB.corr1RDM[ system.site_to_impindx[site1], emb3-fragB.Nvirt ] #necessary to subtract off Nvirt b/c embedding orbitals go as imp,virt,bath,core
        else:
            val2 = 0.0

        #sum terms into phi super-matrix
        phi_val += 1.0/(eval2-eval1) * ( omega[Lidx1,Ridx1]*val1 - np.conjugate(omega[Lidx2,Ridx2])*val2 )

    return phi_val

    #Solve inversion equation for X super vector
    return np.linalg.solve( np.eye(sizeX)-phi, Yvec )

#####################################################################
