import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/tdhf')
import tdhf
import hartreefock
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
Nprint = 1

#Initital Static Calculation
U     = 1.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

CIcoeffs = fci_mod.FCI_GS( h_site, V_site, 0.0, Nsites, Nele )
mf1RDM   = fci_mod.get_corr1RDM( CIcoeffs, Nsites, Nele )

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

tdhf = tdhf.tdhf( Nsites, Nele, h_site, mf1RDM, delt, Nstep, Nprint )
tdhf.kernel()
