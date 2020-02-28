import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/tdhf')
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/test')
import dynamics
import static
import hartreefock
import make_hams
import pyscf.fci

NL     = 4
NR     = 3
Nsites = NL+NR+1
Nele   = Nsites
Nimp   = 2

t  = 0.4
Vg = 0.0

tleads  = 1.0
Full    = True

delt   = 0.001
Nstep  = 1000
Nprint = 1

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

mf1RDM   = hartreefock.rhf_calc_hubbard( Nele, h_site )
CIcoeffs = static.kernel( mf1RDM, Nsites, Nele, h_site, Nimp )

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

dynamics.kernel( mf1RDM, CIcoeffs, Nsites, Nele, h_site, delt, Nstep, Nimp )

