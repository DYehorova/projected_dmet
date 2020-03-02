import numpy as np

A = np.arange(27).reshape(3,3,3)

B = np.arange(27).reshape(3,3,3)

C = np.multiply( A,B )

print(A)
print()
print(B)
print()
print(C)

