import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
import static_driver
import dynamics_driver
import pyscf.fci
import utils
import make_hams
import time
import pickle

NL     = 3
NR     = 2
Nsites = NL+NR+1
Nele   = Nsites
Nfrag  = 3

t  = 0.4
Vg = 0.0



tleads  = 1.0
Full    = True

mubool  = False
hamtype = 0

nproc  = 1

delt   = 0.001
Nstep  = 500
Nprint = 10
integ  = 'rk4_orb'

#N=4 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]) ]
#impindx = [ np.array([0,1]), np.array([2,3]) ]

#N=6 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]) ]
impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]) ]
#impindx = [ np.array([0,1,2]), np.array([3,4,5]) ]
#impindx = [ np.array([5,4,1]), np.array([0,3,2]) ]
#impindx = [ np.array([1,4,5]), np.array([0,2,3]) ]
#impindx = [ np.array([0]), np.array([1,2,3]), np.array([4,5]) ]

#N=8 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]), np.array([6]), np.array([7]) ]
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]) ]
#impindx = [ np.array([0,1,2,3]), np.array([4,5,6,7]) ]

#N=10 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]) ]
#impindx = [ np.array([0,1,2,3,4]), np.array([5,6,7,8,9]) ]
#impindx = [ np.array([7,8,9,3,4]), np.array([0,1,2,5,6])]
#impindx  = [ np.array([0,1]), np.array([2,3,7]), np.array([4,5,9]), np.array([6,8]) ]


#Dynamics Calculation
U     = 0.0
Vbias = -0.001
#Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

restart_file = open( '../restart_system.dat', 'rb' )
tot_system = pickle.load( restart_file )
restart_file.close()

rt_dmet = dynamics_driver.dynamics_driver( h_site, V_site, hamtype, tot_system, delt, Nstep, Nprint, integ, nproc )
rt_dmet.kernel()

