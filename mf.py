import numpy as np
from pyscf import gto, scf, ao2mo, cc
from ueg import ueg_qc

system = ueg_qc(1.0, (7, 7), e_cut_red=4.0)
k_points = system.get_k_points()
n_kpts = k_points.shape[0]
print(f"number of k-points: {n_kpts}")
h0 = system.madelung() / 2
h1 = system.get_h1_real(k_points)
eri = system.get_eri_tensor_real(k_points)

mol = gto.M(verbose=0)
mol.nelectron = system.n_particles
mol.nao = n_kpts
mol.incore_anyway = True
mol.energy_nuc = lambda *args: h0
mol.verbose = 0
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n_kpts)
# mf.get_jk = get_jk
mf.verbose = 4
mf._eri = ao2mo.restore(8, eri, n_kpts)
mf.init_guess = "1e"
escf = mf.kernel()
print(f"escf: {escf}")

mycc = cc.RCCSD(mf)
# mycc.verbose = 0
mycc.max_cycle = 100
mycc.conv_tol = 1e-7
mycc.kernel()
print("CCSD energy", mycc.e_tot)
print("CCSD energy per electron", mycc.e_tot / system.n_particles)
et_correction = mycc.ccsd_t()
print("CCSD(T) energy", mycc.e_tot + et_correction)
print("CCSD(T) energy per electron", (mycc.e_tot + et_correction) / system.n_particles)

