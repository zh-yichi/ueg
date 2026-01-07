options = {'n_eql': 3,
           'n_prop_steps': 50,
            'n_ene_blocks': 2,
            'n_sr_blocks': 5,
            'n_blocks': 40,
            'n_walkers': 200,
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

