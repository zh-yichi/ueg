import numpy as np
from pyscf import gto, scf, cc
from ueg import ueg_qc, my_ueg

nocc = 7 # 7, 19, 27, 33, 57, 81, 93
system = ueg_qc(1.0, (nocc, nocc), e_cut_red=2)
my_sys = my_ueg(rs=1.0, nelec=(nocc, nocc))
k_points = my_sys.get_kpts(np.sqrt(2))
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
mol.verbose = 4

dm = np.zeros((n_kpts,n_kpts))
dm[:nocc,:nocc] = np.eye(nocc) * 2.0

mf = scf.RHF(mol)
mf.energy_nuc = lambda *args: h0
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n_kpts)
# mf.get_jk = get_jk
mf._eri = eri
mf.init_guess = "1e"
mf.max_cycle = -1
escf = mf.kernel(dm0=dm)

print(f"escf: {escf}")
print(f"escf per electron: {escf/system.n_particles}")

mf.mo_coeff = np.eye(n_kpts)

mycc = cc.RCCSD(mf)
mycc.max_cycle = 100
# mycc1.conv_tol = 1e-7
mycc.kernel()
print("CCSD energy", mycc.e_tot)
print("CCSD Corr per electron", mycc.e_corr / system.n_particles)
print("CCSD energy per electron", mycc.e_tot / system.n_particles)

my_sys.prep_afqmc(mycc,chol_cut=1e-6)

options = {'n_eql': 3,
           'n_prop_steps': 50,
           'n_ene_blocks': 2,
           'n_sr_blocks': 5,
           'n_blocks': 40,
           'n_walkers': 200,
           'n_batch': 1,
           'seed': 2,
           'walker_type': 'rhf',
           'trial': 'ccsd_pt',
           'dt':0.005,
           'free_projection':False,
           'ad_mode':None,
           'use_gpu': True,
           }
from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.run_afqmc(options,nproc=1)
