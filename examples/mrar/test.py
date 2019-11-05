import numpy as np
import multiprocessing as multproc

class my_class():

    def __init__(self, N, Nsteps):

        self.Nsteps = Nsteps

        self.my_list = []
        for i in range(N):
            self.my_list.append(i)

    def loop( self, nproc ):

        for i in range(self.Nsteps):
            print('Step ',i)
            self.parallelized_code(nproc)
            print()

    def parallelized_code( self, nproc ):

        mypool = multproc.Pool( nproc )
        mypool.map( wrapper, self.my_list )


def wrapper( item ):

    print(item)
