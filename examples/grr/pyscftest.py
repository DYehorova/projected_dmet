from pyscf import gto, scf, lib

print('num_threads=',lib.num_threads())

mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
mf  = scf.RHF(mol)
mf.kernel()

