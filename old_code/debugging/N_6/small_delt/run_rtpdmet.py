import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
import static_driver
import dynamics_driver
import make_hams
import utils

NL = 2
NR = 2
Ndots = 2
Nsites = NL+NR+Ndots
Nele   = Nsites

Nimp   = 2
Nfrag  = int(Nsites/Nimp)

timp     = 1.0
timplead = 1.0
tleads   = 1.0
Vg       = 0.0
Full     = True

mubool  = False
hamtype = 0

nproc  = 1

delt   = 0.0001
Nstep  = 44000
Nprint = 1
integ  = 'rk4_orb'

#Define Fragment Indices
impindx = []
for i in range(Nfrag):
    impindx.append( np.arange(i*Nimp,(i+1)*Nimp) )

print('impurity indices are:')
print(impindx)
print()

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full )

the_dmet = static_driver.static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool, nproc )
the_dmet.kernel()

#Check hartree-fock energy
evals,orbs = utils.diagonalize(h_site)
Ehf = 2.0*np.sum(evals[:round(Nele/2)])

print('The Hartree-Fock energy is: ',Ehf)
print()

#Dynamics Calculation
U     = 1.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full )

rt_dmet = dynamics_driver.dynamics_driver( h_site, V_site, hamtype, the_dmet.tot_system, delt, Nstep, Nprint, integ, nproc )
rt_dmet.kernel()

