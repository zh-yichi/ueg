import numpy as np
from pyscf import gto, scf, cc
from my_ueg import my_ueg

nocc = 7 # 7, 19, 27, 33, 57, 81, 93
my_sys = my_ueg(rs=1.0, nelec=(nocc, nocc), gamma=np.sqrt(2))
nkpts = my_sys.nkpts
print(f"number of k-points: {nkpts}")
h0 = my_sys.madelung() / 2
h1 = my_sys.get_h1_real()
cderi = my_sys.get_cderi_real()

mol = gto.M()
mol.nelectron = my_sys.nparticle
mol.nao = nkpts
mol.incore_anyway = True
mol.verbose = 4
mol.max_memory = 20000

dm = np.zeros((nkpts,nkpts))
dm[:nocc,:nocc] = np.eye(nocc) * 2.0

mf = scf.RHF(mol).density_fit()
mf.energy_nuc = lambda *args: h0
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(nkpts)
mf._cderi = cderi
#mf.init_guess = "1e"
mf.max_cycle = -1
escf = mf.kernel(dm0=dm)

print(f"escf: {escf}")
print(f"escf per electron: {escf/my_sys.nparticle}")

mf.mo_coeff = np.eye(nkpts)

mycc = cc.RCCSD(mf)
mycc.max_cycle = 100
mycc.conv_tol = 1e-7
mycc.kernel()
et = mycc.ccsd_t()
print("CCSD Energy", mycc.e_corr)
print("CCSD Energy per electron", mycc.e_corr / my_sys.nparticle)
print("CCSD(T) Energy electron", et)
print("CCSD(T) Energy per electron", et / my_sys.nparticle)

my_sys.prep_afqmc(mycc)

options = {'n_eql': 3,
           'n_prop_steps': 50,
           'n_ene_blocks': 1,
           'n_sr_blocks': 5,
           'n_blocks': 40,
           'n_walkers': 20,
           'n_batch': 1,
           'seed': 2,
           'walker_type': 'rhf',
           'trial': 'ccsd_pt',
           'dt':0.005,
           'free_projection':False,
           'ad_mode':None,
           'use_gpu': False,
           }
from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.run_afqmc(options, nproc=1)

options = {'n_eql': 3,
           'n_prop_steps': 50,
           'n_ene_blocks': 1,
           'n_sr_blocks': 5,
           'n_blocks': 40,
           'n_walkers': 20,
           'n_batch': 1,
           'seed': 2,
           'walker_type': 'rhf',
           'trial': 'cisd',
           'dt':0.005,
           'free_projection':False,
           'ad_mode':None,
           'use_gpu': False,
           }
from ad_afqmc.prop_unrestricted import prop_unrestricted
prop_unrestricted.run_afqmc(options, nproc=1)
