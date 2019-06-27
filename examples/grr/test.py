import numpy as np
import sys
import os
sys.path.append('/home/jkretchm/projected_dmet')
import static_driver
import pyscf.fci
import utils
import make_hams
import fci_mod

N     = 6
Nele  = N

U = 3.0 
h_site, V_site = make_hams.make_1D_hubbard( N, U, 1.0, True )

#FCI Check
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.conv_tol = 1e-16
cisolver.verbose = 3
E_FCI, CIcoeffs = cisolver.kernel( h_site, V_site, N, Nele )
print 'E_FCI = ',E_FCI

print fci_mod.get_FCI_E( h_site, V_site, 0.0, CIcoeffs, N, Nele/2, Nele/2 )
