import numpy as np
import sys
import os
sys.path.append('/home/jkretchm/projected_dmet')
import static_driver
import dynamics_driver
import pyscf.fci
import utils
import make_hams

NL     = 3
NR     = 2
Nsites = NL+NR+1
Nele   = Nsites
Nfrag  = 6

t  = 0.4
Vg = 0.0



tleads  = 1.0
Full    = True

mubool  = False
hamtype = 0

delt   = 0.0001
Nstep  = 100000
Nprint = 1000

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


#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

the_dmet = static_driver.static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool )
the_dmet.kernel()

#FCI Check for static calculation
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.conv_tol = 1e-16
cisolver.verbose = 3
E_FCI, CIcoeffs = cisolver.kernel( h_site, V_site, Nsites, Nele )
print 'E_FCI = ',E_FCI

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
#Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

rt_dmet = dynamics_driver.dynamics_driver( h_site, V_site, hamtype, the_dmet.tot_system, delt, Nstep, Nprint )
rt_dmet.kernel()

