import numpy as np
import sys
import os
sys.path.append('/Users/Joshua/Documents/Chan_group/projected_dmet/')
import static_driver
import pyscf.fci
import utils
import make_hams
import applyham_pyscf

N     = 2
Nele  = N

mubool  = True

h_site = np.zeros([N,N],dtype=complex)
V_site = np.zeros([N,N,N,N],dtype=complex)

U = 2.0+0.3j 

a = -0.5
b = 0.1+0.2j
c = 0.1-0.2j
d = 0.5

h_site[0,0] = a
h_site[0,1] = b
h_site[1,0] = c
h_site[1,1] = d

V_site[0,0,0,0] = U
V_site[1,1,1,1] = U


Ham = np.zeros([4,4],dtype=complex)

Ham[0,0] = 2*a+U
Ham[1,0] = c
Ham[2,0] = c
Ham[3,0] = 0.0

Ham[0,1] = b
Ham[1,1] = a+d
Ham[2,1] = 0.0
Ham[3,1] = c

Ham[0,2] = b
Ham[1,2] = 0.0
Ham[2,2] = a+d
Ham[3,2] = c

Ham[0,3] = 0.0
Ham[1,3] = b
Ham[2,3] = b
Ham[3,3] = 2*d+U


CIcoeffs = np.zeros([2,2],dtype=complex)
CIcoeffs[0,0] = 0.8 + 0.1j
CIcoeffs[1,0] = -0.7 + 0.3j
CIcoeffs[0,1] = 0.1 - 0.5j
CIcoeffs[1,1] = 0.3 - 0.2j

CIvec = np.zeros(4,dtype=complex)
CIvec[0] = CIcoeffs[0,0]
CIvec[1] = CIcoeffs[1,0]
CIvec[2] = CIcoeffs[0,1]
CIvec[3] = CIcoeffs[1,1]

print np.dot(Ham,CIvec)
print
print applyham_pyscf.apply_ham_pyscf_fully_complex( CIcoeffs, h_site, V_site, Nele/2, Nele/2, N, 0.0 )

