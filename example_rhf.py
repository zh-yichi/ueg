'''
Restricted Hartree-Fock (and CCSD) for the uniform electron gas.

The plane-wave Slater determinant is the exact RHF solution for a closed-shell
UEG, so the SCF converges in one iteration from the default (aufbau) guess.
'''

import my_ueg
from pyscf import cc

nocc = 7        # electrons per spin, closed shells: 7, 19, 27, 33, 57, 81, 93
rs = 5.0
gamma = 2.0     # k_cut = gamma * k_fermi

mol = my_ueg.M(rs=rs, nelec=(nocc, nocc), gamma=gamma, verbose=4)
print(f"number of k-points (orbitals): {mol.nao}")

mf = my_ueg.RHF(mol)
escf = mf.kernel()
print(f"E(RHF) = {escf:.10f}   per electron = {escf/mol.nelectron:.10f}")

# RHF -> UHF (triplet) stability: a negative eigenvalue signals spin-symmetry breaking
print("lowest triplet orbital-Hessian eigenvalues:", my_ueg.rhf_triplet_instability(mf))

mycc = cc.RCCSD(mf)
mycc.kernel()
print(f"E(CCSD corr) per electron = {mycc.e_corr/mol.nelectron:.10f}")
