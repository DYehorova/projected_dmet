#Mod that contains subroutines associated with pyscf FCI calculations

import numpy as np
import sys
import os
import utils
import pyscf.fci

#####################################################################

def FCI_GS( h, V, Ecore, Norbs, Nele ):

    #Subroutine to perform groundstate FCI calculation using pyscf

    cisolver = pyscf.fci.direct_spin1.FCI()

    cisolver.conv_tol = 1e-16

    E_FCI, CIcoeffs = cisolver.kernel( h, V, Norbs, Nele )

    return CIcoeffs

#####################################################################

def get_corr1RDM( CIcoeffs, Norbs, Nele ):

    #Subroutine to get the FCI 1RDM, notation is rho_pq = < c_q^dag c_p >

    if( np.iscomplexobj(CIcoeffs) ):

        Re_CIcoeffs = np.copy( CIcoeffs.real )
        Im_CIcoeffs = np.copy( CIcoeffs.imag )

        corr1RDM  = 1j * pyscf.fci.direct_spin1.trans_rdm1( Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele )

        corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1( Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele )

        corr1RDM += pyscf.fci.direct_spin1.make_rdm1( Re_CIcoeffs, Norbs, Nele )

        corr1RDM += pyscf.fci.direct_spin1.make_rdm1( Im_CIcoeffs, Norbs, Nele )

    else:

        corr1RDM  = pyscf.fci.direct_spin1.make_rdm1( CIcoeffs, Norbs, Nele )

    return corr1RDM

#####################################################################

def get_corr12RDM( CIcoeffs, Norbs, Nele ):

    #Subroutine to get the FCI 1 & 2 RDMs together
    #Notation for 1RDM is rho_pq = < c_q^dag c_p >
    #Notation for 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if( np.iscomplexobj(CIcoeffs) ):

        Re_CIcoeffs = np.copy( CIcoeffs.real )
        Im_CIcoeffs = np.copy( CIcoeffs.imag )

        corr1RDM, corr2RDM  = pyscf.fci.direct_spin1.trans_rdm12( Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele )

        corr1RDM = corr1RDM*1j
        corr2RDM = corr2RDM*1j

        tmp1, tmp2 = pyscf.fci.direct_spin1.trans_rdm12( Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele )

        corr1RDM -= 1j * tmp1
        corr2RDM -= 1j * tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12( Re_CIcoeffs, Norbs, Nele )

        corr1RDM += tmp1
        corr2RDM += tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12( Im_CIcoeffs, Norbs, Nele )

        corr1RDM += tmp1
        corr2RDM += tmp2

    else:

        corr1RDM, corr2RDM  = pyscf.fci.direct_spin1.make_rdm12( CIcoeffs, Norbs, Nele )

    return corr1RDM, corr2RDM

#####################################################################



