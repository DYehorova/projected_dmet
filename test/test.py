import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Chan_group/projected_dmet/')
import utils

A = np.arange(9).reshape(3,3)

print(A)

indx1 = np.array([1,0,2])
indx2 = np.array([0,2])

print()
print( A[ indx1[:,None], indx2 ] )

