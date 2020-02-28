import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/tdfci')
import tdfci
import make_hams
import fci_mod

NL     = 3
NR     = 2
Nsites = NL+NR+1
Nele   = Nsites

t  = 0.4
Vg = 0.0

tleads  = 1.0
Full    = True

delt   = 0.001
Nstep  = 5000
Nprint = 100

#Initital Static Calculation
U     = 1.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

CIcoeffs = fci_mod.FCI_GS( h_site, V_site, 0.0, Nsites, Nele )

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

tdfci = tdfci.tdfci( Nsites, Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint )
tdfci.kernel()

