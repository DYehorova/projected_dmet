import numpy as np
import sys
import os
sys.path.append('/home/jkretchm/projected_dmet')
import static_driver
import pyscf.fci
import utils
import make_hams

N     = 6
Nele  = N
Nfrag = 6

mubool  = True
hamtype = 0

#N=4 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]) ]
#impindx = [ np.array([0,1]), np.array([2,3]) ]

#N=6 tilings
impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]) ]
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]) ]
#impindx = [ np.array([0,1,2]), np.array([3,4,5]) ]
#impindx = [ np.array([5,4,1]), np.array([0,3,2]) ]
#impindx = [ np.array([1,4,5]), np.array([0,2,3]) ]

#N=8 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]), np.array([6]), np.array([7]) ]
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]) ]
#impindx = [ np.array([0,1,2,3]), np.array([4,5,6,7]) ]

#N=10 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]) ]
#impindx = [ np.array([0,1,2,3,4]), np.array([5,6,7,8,9]) ]
#impindx = [ np.array([7,8,9,3,4]), np.array([0,1,2,5,6])]
#impindx  = [ np.array([0,1]), np.array([2,3,7]), np.array([4,5,9]), np.array([6,8]) ]

U = 0.0 
h_site, V_site = make_hams.make_1D_hubbard( N, U, 1.0, True )

#V_site = U * np.ones([N,N,N,N])

#V_site = np.random.random((N,N,N,N))
## Restore permutation symmetry
#V_site = V_site + V_site.transpose(1,0,2,3)
#V_site = V_site + V_site.transpose(0,1,3,2)
#V_site = V_site + V_site.transpose(2,3,0,1)

the_dmet = static_driver.static_driver( N, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool )
the_dmet.kernel()

#FCI Check
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.conv_tol = 1e-16
cisolver.verbose = 3
E_FCI, CIcoeffs = cisolver.kernel( h_site, V_site, N, Nele )
print 'E_FCI = ',E_FCI
