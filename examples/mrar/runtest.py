import test

N = 6
nproc = 3
Nsteps = 200

myclass = test.my_class(N,Nsteps)

myclass.loop(nproc)
#myclass.parallelized_code(nproc)

