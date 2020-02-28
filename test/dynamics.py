import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
import utils
import fci_mod
import pyscf.fci
import integrators
import applyham_pyscf

#################################################

def kernel( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nsteps, Nimp ):

    file_corrdens = open( 'corr_density.dat', 'w' )
    file_mfdens = open( 'mfdensity.dat', 'w' )

    mf1RDM   = mf1RDM.astype(complex)
    CIcoeffs = CIcoeffs.astype(complex)

    rotmat, evals = get_rotmat( mf1RDM, Nsites, Nimp )

    current_time = 0.0
    for step in range(Nsteps):

        print('Writing data at step ', step, 'and time', current_time, 'for test calculation')
        print_data( current_time, CIcoeffs, mf1RDM, Nimp, file_corrdens, file_mfdens )
        sys.stdout.flush()

        mf1RDM, CIcoeffs, rotmat = integrate( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nimp, rotmat )
        #mf1RDM, CIcoeffs, rotmat = integrate2( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nimp, rotmat )

        current_time = (step+1)*delt

    print('Writing data at step ', step+1, 'and time', current_time, 'for test calculation')
    print_data( current_time, CIcoeffs, mf1RDM, Nimp, file_corrdens, file_mfdens )
    sys.stdout.flush()

    file_corrdens.close()
    file_mfdens.close()

#################################################

def integrate( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nimp, rotmat ):

    l1, k1, rotmat = one_rk_step( mf1RDM, CIcoeffs, Nsites, Nele, h_site, Nimp, delt, rotmat )

    l2, k2, rotmat = one_rk_step( mf1RDM + 0.5*l1, CIcoeffs + 0.5*k1, Nsites, Nele, h_site, Nimp, delt, rotmat )

    l3, k3, rotmat = one_rk_step( mf1RDM + 0.5*l2, CIcoeffs + 0.5*k2, Nsites, Nele, h_site, Nimp, delt, rotmat )
 
    l4, k4, rotmat = one_rk_step( mf1RDM + 1.0*l3, CIcoeffs + 1.0*k3, Nsites, Nele, h_site, Nimp, delt, rotmat )

    mf1RDM   = mf1RDM   + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
    CIcoeffs = CIcoeffs + 1.0/6.0 * ( k1 + 2.0*k2 + 2.0*k3 + k4 )

    rotmat_old = np.copy(rotmat)
    rotmat, evals = get_rotmat( mf1RDM, Nsites, Nimp )
    phase  = np.diag( np.round( np.diag( np.real( np.dot( utils.adjoint( rotmat_old ), rotmat ) ) ) ) )
    rotmat = np.dot( rotmat, phase )

    return mf1RDM, CIcoeffs, rotmat

#################################################

def one_rk_step( mf1RDM, CIcoeffs, Nsites, Nele, h_site, Nimp, delt, rotmat_old ):

    iddt_mf1RDM   = utils.commutator( h_site, mf1RDM )
    change_mf1RDM = -1j * delt * iddt_mf1RDM

    rotmat, evals = get_rotmat( mf1RDM, Nsites, Nimp )

    phase  = np.diag( np.round( np.diag( np.real( np.dot( utils.adjoint( rotmat_old ), rotmat ) ) ) ) )
    rotmat = np.dot( rotmat, phase )

    h_emb, Ecore = get_Hemb( h_site, rotmat, Nimp, Nsites, Nele )

    V_emb        = np.zeros( [2*Nimp,2*Nimp,2*Nimp,2*Nimp], dtype=complex )

    Xmat, Xmat_sml = get_Xmat( mf1RDM, iddt_mf1RDM, rotmat, evals, Nsites, Nele, Nimp )

    Ecore = 0.0
    change_CIcoeffs = -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex( CIcoeffs, h_emb-Xmat_sml, V_emb, Nimp, Nimp, 2*Nimp, Ecore )

    return change_mf1RDM, change_CIcoeffs, rotmat

#################################################

def integrate2( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nimp, rotmat ):

    l1, k1, m1 = one_rk_step2( mf1RDM, CIcoeffs, Nsites, Nele, h_site, Nimp, delt, rotmat )

    l2, k2, m2 = one_rk_step2( mf1RDM + 0.5*l1, CIcoeffs + 0.5*k1, Nsites, Nele, h_site, Nimp, delt, rotmat + 0.5*m1 )

    l3, k3, m3 = one_rk_step2( mf1RDM + 0.5*l2, CIcoeffs + 0.5*k2, Nsites, Nele, h_site, Nimp, delt, rotmat + 0.5*m2 )
 
    l4, k4, m4 = one_rk_step2( mf1RDM + 1.0*l3, CIcoeffs + 1.0*k3, Nsites, Nele, h_site, Nimp, delt, rotmat + 1.0*m3 )

    mf1RDM   = mf1RDM   + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
    CIcoeffs = CIcoeffs + 1.0/6.0 * ( k1 + 2.0*k2 + 2.0*k3 + k4 )
    rotmat   = rotmat   + 1.0/6.0 * ( m1 + 2.0*m2 + 2.0*m3 + m4 )

    return mf1RDM, CIcoeffs, rotmat

#################################################

def one_rk_step2( mf1RDM, CIcoeffs, Nsites, Nele, h_site, Nimp, delt, rotmat ):

    iddt_mf1RDM   = utils.commutator( h_site, mf1RDM )
    change_mf1RDM = -1j * delt * iddt_mf1RDM

    evals = np.copy( np.real( np.diag( utils.rot1el( mf1RDM[Nimp:,Nimp:], rotmat[Nimp:,Nimp:] ) ) ) )

    #####grr####
    #chk1, chk2 = get_rotmat( mf1RDM, Nsites, Nimp )
    #print(rotmat[Nimp:,Nimp:])
    #print()
    #print(chk1[Nimp:,Nimp:])
    #print()
    #print(evals)
    #print()
    #print(chk2)
    #print()
    #print('------------------')
    #print()
    ############

    Xmat, Xmat_sml = get_Xmat( mf1RDM, iddt_mf1RDM, rotmat, evals, Nsites, Nele, Nimp )

    change_rotmat = -1j * delt * np.dot( rotmat, Xmat )

    h_emb, Ecore = get_Hemb( h_site, rotmat, Nimp, Nsites, Nele )

    V_emb        = np.zeros( [2*Nimp,2*Nimp,2*Nimp,2*Nimp], dtype=complex )

    Ecore = 0.0
    change_CIcoeffs = -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex( CIcoeffs, h_emb-Xmat_sml, V_emb, Nimp, Nimp, 2*Nimp, Ecore )

    return change_mf1RDM, change_CIcoeffs, change_rotmat

#################################################

def get_Xmat( mf1RDM, iddt_mf1RDM, rotmat, evals, Nsites, Nele, Nimp ):

    Ncore = round(Nele/2) - Nimp
    Nvirt = Nsites - 2*Nimp - Ncore

    bathrange = np.arange(Nimp+Nvirt, 2*Nimp+Nvirt)
    virtrange = np.arange(Nimp, Nimp+Nvirt)
    corerange = np.arange(2*Nimp+Nvirt, Nsites)

    X = np.zeros( [Nsites,Nsites], dtype=complex )

    #bath-bath
    for b in bathrange:
        for a in bathrange:
            if( b != a ):

                for p in range(Nimp,Nsites):
                    for q in range(Nimp,Nsites):
                        X[b,a] += 1.0/(evals[a-Nimp]-evals[b-Nimp]) * np.conjugate( rotmat[p,b] ) * iddt_mf1RDM[p,q] * rotmat[q,a]

    #bath-core
    for b in bathrange:
        for a in corerange:

            for p in range(Nimp,Nsites):
               for q in range(Nimp,Nsites):
                   X[b,a] += 1.0/(evals[a-Nimp]-evals[b-Nimp]) * np.conjugate( rotmat[p,b] ) * iddt_mf1RDM[p,q] * rotmat[q,a]

            X[a,b] = np.conjugate(X[b,a])

    #bath-virtual
    for b in bathrange:
        for a in virtrange:

            for p in range(Nimp,Nsites):
               for q in range(Nimp,Nsites):
                   X[b,a] += 1.0/(evals[a-Nimp]-evals[b-Nimp]) * np.conjugate( rotmat[p,b] ) * iddt_mf1RDM[p,q] * rotmat[q,a]

            X[a,b] = np.conjugate(X[b,a])

    #core-virtual
    for b in corerange:
        for a in virtrange:

            for p in range(Nimp,Nsites):
               for q in range(Nimp,Nsites):
                   X[b,a] += 1.0/(evals[a-Nimp]-evals[b-Nimp]) * np.conjugate( rotmat[p,b] ) * iddt_mf1RDM[p,q] * rotmat[q,a]

            X[a,b] = np.conjugate(X[b,a])

    phase = - np.real( np.dot( rotmat[Nimp,Nimp:], X[Nimp:,Nimp:] ) ) / np.real( rotmat[Nimp,Nimp:] )
    X[Nimp:,Nimp:] += np.diag(phase)

    X_sml = np.zeros([2*Nimp,2*Nimp],dtype=complex)
    X_sml[Nimp:,Nimp:] = np.copy( X[ bathrange[:,None], bathrange ] )

    return X, X_sml

#################################################

def get_rotmat( mf1RDM, Nsites, Nimp ):

    #remove rows/columns corresponding to impurity sites from mf 1RDM
    for imp in range(Nimp):
        mf1RDM  = np.delete( mf1RDM, 0, axis=0 )
        mf1RDM  = np.delete( mf1RDM, 0, axis=1 )

    #diagonalize environment part of 1RDM to obtain embedding (virtual, bath, core) orbitals
    evals, evecs = np.linalg.eigh( mf1RDM )

    #form rotation matrix consisting of unit vectors for impurity and the evecs for embedding
    #rotation matrix is ordered as impurity, virtual, bath, core
    rotmat = np.zeros( [ Nsites, Nimp ] )
    for imp in range(Nimp):
        rotmat[ imp, imp ] = 1.0
        evecs              = np.insert( evecs, imp, 0.0, axis=0 )

    rotmat = np.concatenate( (rotmat,evecs), axis=1 )

    return rotmat, evals

#################################################

def get_Hemb( h_site, rotmat, Nimp, Nsites, Nele ):

    Ncore = round(Nele/2) - Nimp
    Nvirt = Nsites - 2*Nimp - Ncore 

    rotmat_small = np.delete( rotmat, np.s_[Nimp:Nimp+Nvirt], 1 )

    h_emb = utils.rot1el( h_site, rotmat_small )

    Ecore = 0.0
    for core in range( 2*Nimp, 2*Nimp+Ncore ):
        Ecore += h_emb[core,core]

    h_emb = h_emb[ :2*Nimp, :2*Nimp ]

    return h_emb, core

#################################################

def print_data( current_time, CIcoeffs, mf1RDM, Nimp, file_corrdens, file_mfdens ):

    fmt_str = '%20.8e'

    corrdens = np.copy( np.real( np.diag( fci_mod.get_corr1RDM( CIcoeffs, 2*Nimp, (Nimp, Nimp) ) ) ) )
    corrdens = np.insert( corrdens, 0, current_time )
    np.savetxt( file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )

    mfdens = np.copy( np.real( np.diag( mf1RDM ) ) )
    mfdens = np.insert( mfdens, 0, current_time )
    np.savetxt( file_mfdens, mfdens.reshape(1, mfdens.shape[0]), fmt_str )

#################################################

