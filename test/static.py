import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
import utils
import fci_mod
import pyscf.fci
import dynamics

#################################################

def kernel( mf1RDM, Nsites, Nele, h_site, Nimp ):

    rotmat, evals = dynamics.get_rotmat( mf1RDM, Nsites, Nimp )

    h_emb, Ecore = dynamics.get_Hemb( h_site, rotmat, Nimp, Nsites, Nele )

    CIcoeffs = fci_mod.FCI_GS( h_emb, np.zeros( [2*Nimp,2*Nimp,2*Nimp,2*Nimp] ), 0.0, 2*Nimp, (Nimp,Nimp) )

    return CIcoeffs

#################################################


