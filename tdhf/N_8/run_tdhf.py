import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/tdhf')
import tdhf
import hartreefock
import make_hams

NL     = 4
NR     = 3
Nsites = NL+NR+1
Nele   = Nsites

t  = 0.4
Vg = 0.0

tleads  = 1.0
Full    = False

delt   = 0.001
Nstep  = 5000
Nprint = 1

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

mf1RDM = hartreefock.rhf_calc_hubbard( Nele, h_site )

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

tdhf = tdhf.tdhf( Nsites, Nele, h_site, mf1RDM, delt, Nstep, Nprint )
tdhf.kernel()
