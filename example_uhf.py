'''
Unrestricted Hartree-Fock for the uniform electron gas with spin-symmetry breaking.

Starting from the plane-wave (RHF) determinant, plain SCF never leaves the
symmetric solution: it is a stationary point of the UHF energy. run_uhf therefore
alternates SCF with an internal stability analysis and rotates the orbitals along
any negative Hessian mode until a stable (local minimum) UHF solution is reached.
'''

import my_ueg

nocc = 7
rs = 5.0
gamma = 2.0

mol = my_ueg.M(rs=rs, nelec=(nocc, nocc), gamma=gamma, verbose=4)

# restricted reference
mf_r = my_ueg.RHF(mol)
e_rhf = mf_r.kernel()

# 1) plain UHF from the aufbau guess: stays on the spin-symmetric solution
mf_u = my_ueg.UHF(mol)
mf_u.kernel()

# 2) UHF with stability following (optionally start from a random spin kick,
#    e.g. dm0=my_ueg.spin_kick_dm(mol, amp=0.05, seed=1))
mf_u = my_ueg.run_uhf(mol)
s2, mult = mf_u.spin_square()

print(f"E(RHF)             = {e_rhf:.10f}")
print(f"E(UHF, stable)     = {mf_u.e_tot:.10f}   stable = {mf_u.stable}")
print(f"(E_UHF - E_RHF)/N  = {(mf_u.e_tot - e_rhf)/mol.nelectron:.3e}")
print(f"<S^2>              = {s2:.4f}")
print(f"(1/N) int |m(r)|   = {my_ueg.local_moment(mol, mf_u.make_rdm1()):.4f}")
