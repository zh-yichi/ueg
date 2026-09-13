'''
Uniform electron gas (UEG) in a finite plane-wave basis, set up for PySCF.

The N-electron UEG is placed in a cubic box of volume V = N 4pi/3 rs^3 with
periodic boundary conditions. The one-particle basis is the set of plane
waves |k| <= k_cut = gamma * k_fermi, rotated to real cos(k.r)/sin(k.r)
orbitals so that every integral is real. The Coulomb interaction is supplied
as a 3-index (density-fitting like) tensor, one auxiliary index per momentum
transfer q != 0, and the Madelung self-interaction is folded into E_nuc.

Usage:

    import my_ueg
    mol = my_ueg.M(rs=5.0, nelec=(7, 7), gamma=2.0, verbose=4)
    mf = my_ueg.RHF(mol)          # or my_ueg.UHF(mol)
    mf.kernel()

Closed shells (electrons per spin): 1, 7, 19, 27, 33, 57, 81, 93, ...
'''

import numpy as np
import scipy.linalg
from pyscf import gto, scf, lib
from pyscf.scf import stability
from pyscf.soscf import newton_ah
einsum = lib.einsum

# Madelung constant of the simple cubic lattice, in units of 1/L
MADELUNG_SC = -2.837297


class my_ueg:

    def __init__(self, rs, nelec, gamma):
        self.rs = rs
        self.nelec = tuple(nelec)
        self.gamma = gamma
        self.nparticle = sum(nelec)
        self.volume = self.nparticle * 4*np.pi/3 * rs**3
        self.length = self.volume**(1/3)
        self.N_cut = gamma * (3*self.nparticle/(8*np.pi))**(1/3)
        self.kpts = self.get_kpts()
        self.nkpts = self.kpts.shape[0]
        self._ints = None

    def canonical_sign(self, npts):
        # define the sign by first non-zero element
        nz = np.argmax(npts != 0, axis=1)
        signs = np.ones(len(npts), dtype=int)
        mask = npts[np.arange(len(npts)), nz] < 0
        signs[mask] = -1

        return npts * signs[:, None]

    def get_npts(self, N_cut=None, with_zero=True):
        '''
        generate a 3D-spherical grid points of integers
        {(n1,n2,n3)| n1^2+n2^2+n3^2<N_cut^2; n1,n2,n3 in Z}
        sorted by incresing length and n next to -n
        '''
        if N_cut is None:
            N_cut = self.N_cut

        n_max = int(np.ceil(N_cut))

        n = np.arange(-n_max, n_max + 1)
        nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")

        n2 = nx**2 + ny**2 + nz**2
        mask = n2 <= N_cut**2 + 1e-10
        npts = np.stack((nx[mask], ny[mask], nz[mask]), axis=1)

        return self._sort_pairs(npts, with_zero)

    def _sort_pairs(self, pts, with_zero):
        '''unique integer vectors sorted by length, ordered as 0, n, -n, ...'''
        can_pts = self.canonical_sign(pts)
        unq_pts = np.unique(can_pts, axis=0)
        unq_pts = unq_pts[np.argsort(np.sum(unq_pts**2, axis=1))]
        # unq_pts[0] is the zero vector
        sort_pts = np.stack((unq_pts[1:], -unq_pts[1:]), axis=1).reshape(-1, 3)

        if with_zero:
            sort_pts = np.vstack([[0, 0, 0], sort_pts])

        return sort_pts

    def get_mpts(self, npts=None):
        '''
        calculate the q = k1-k3 on integer grid points.
        m should effectively lives in a sphere of Mcut = 2*Ncut
        But since the discrete nature of lattice points, its
        safer to calculate {m} by {n} directly than using 2*Ncut.
        #mpts ~ 4pi/3*8 #npts
        return: {m = n1-n3|n1, n3 in npts, m != 0}
        '''
        if npts is None:
            npts = self.get_npts()

        mpts = (npts[:, None, :] - npts[None, :, :]).reshape(-1, 3)
        return self._sort_pairs(mpts, with_zero=False)

    def get_kpts(self, gamma=None, with_zero=True):
        '''
        get the k-points in a sphere by k_cut = gamma * k_fermi
        ordered by 0,...,k,-k,... with the length increasing
        '''
        if gamma is None:
            gamma = self.gamma

        Nf = (3*self.nparticle/(8*np.pi))**(1/3)
        npts = self.get_npts(gamma * Nf, with_zero)

        return npts * (2*np.pi/self.length)

    def get_qpts(self, mpts=None):
        '''
        q = k1-k3
        '''
        if mpts is None:
            mpts = self.get_mpts()

        return mpts * (2*np.pi/self.length)

    def get_vq(self, qpts=None):
        '''
        V(q) = 4pi / q^2 / V_cell
        '''
        if qpts is None:
            qpts = self.get_qpts()

        q2 = np.sum(qpts**2, axis=1)
        return 4*np.pi / q2 / self.volume

    def pw2real(self, nkpts=None, with_zero=True):
        '''
        get the unitary transformation that
        transforms plane-wave basis to cos, sin basis.
        kpts are ordered in +k, -k pairs, s.t. each block:
        [coskx]  =  1/sqrt(2)[[ 1, 1]]  [exp(+ikx)]
        [sinkx]              [[-i, i]]  [exp(-ikx)]
        with_zero: add gamma point
        '''
        if nkpts is None:
            nkpts = self.nkpts

        blk = np.array([[1.0, 1.0], [-1.0j, 1.0j]], dtype=np.complex128) / np.sqrt(2)

        if with_zero:
            nblks = (nkpts - 1) // 2
            u = np.kron(np.eye(nblks), blk)
            u = np.block([
                [np.array([[1.0]]), np.zeros((1, 2*nblks))],
                [np.zeros((2*nblks, 1)), u]
                ])
        else:
            nblks = nkpts // 2
            u = np.kron(np.eye(nblks), blk)

        return u

    def madelung(self):
        '''
        Madelung term
        interaction of each charge with its image in other cells
        '''
        return MADELUNG_SC * self.nparticle / self.length

    def energy_nuc(self):
        '''constant energy shift: N * v_M / 2'''
        return self.madelung() / 2

    def get_h1(self, kpts=None):
        if kpts is None:
            kpts = self.kpts
        return np.diag(np.sum(kpts**2, axis=1)/2)

    def get_h1_real(self, kpts=None):
        if kpts is None:
            kpts = self.kpts
        h1_pw = self.get_h1(kpts)
        uk = self.pw2real(len(kpts))
        h1 = uk.conj() @ h1_pw @ uk.T
        return h1.real

    def _int_keys(self, vecs, nmax):
        '''map integer 3-vectors with |components| <= nmax to unique integers'''
        base = 2*nmax + 1
        v = vecs + nmax
        return (v[..., 0]*base + v[..., 1])*base + v[..., 2]

    def get_eris_hard(self, kpts=None):
        '''
        Reference 4-index ERIs in the plane-wave basis (for testing, O(nk^4)).
        <k1k2|V|k3k4> = 4pi/V 1/(k1-k3)^2 delta(k1+k2,k3+k4)
        returned in (11|22) notation, i.e. (k1 k3|k2 k4)
        '''
        if kpts is None:
            kpts = self.kpts

        dk = kpts[:, None, :] - kpts[None, :, :]
        q2 = np.sum(dk**2, axis=-1)
        vq = np.zeros_like(q2)
        vq[q2 > 1e-10] = 4*np.pi / q2[q2 > 1e-10] / self.volume

        dn = np.rint(dk * (self.length/(2*np.pi))).astype(int)
        nmax = np.abs(dn).max()
        key13 = self._int_keys(dn, nmax)
        key42 = self._int_keys(-dn, nmax)    # (k4 - k2) indexed as [k2, k4]
        consv = key13[:, :, None, None] == key42[None, None, :, :]

        return vq[:, :, None, None] * consv

    def get_cderi_pw(self, npts=None, mpts=None, qpts=None):
        '''L_{q(m),k1(n1),k3(n3)} = delta(n1-n3,m)*V(q)^1/2'''

        if npts is None:
            npts = self.get_npts()
        if mpts is None:
            mpts = self.get_mpts(npts)
        if qpts is None:
            qpts = self.get_qpts(mpts)

        nk, nq = len(npts), len(mpts)
        nmax = np.abs(mpts).max()
        lookup = np.full((2*nmax + 1)**3, -1, dtype=int)
        lookup[self._int_keys(mpts, nmax)] = np.arange(nq)

        g = lookup[self._int_keys(npts[:, None, :] - npts[None, :, :], nmax)]
        p, q = np.nonzero(g >= 0)
        g = g[p, q]

        cderi = np.zeros((nq, nk, nk))
        cderi[g, p, q] = np.sqrt(self.get_vq(qpts))[g]

        return cderi

    @staticmethod
    def _rotate_pairs(x, axis, conj=False, with_zero=True):
        '''
        apply the block-diagonal pw -> real unitary (see pw2real) along `axis`
        without building the dense matrix
        '''
        x = np.moveaxis(x, axis, 0)
        out = np.empty(x.shape, dtype=np.complex128)
        s = 1 if with_zero else 0
        if with_zero:
            out[0] = x[0]
        a, b = x[s::2], x[s+1::2]
        phase = -1j if conj else 1j
        out[s::2] = (a + b) / np.sqrt(2)
        out[s+1::2] = phase * (b - a) / np.sqrt(2)
        return np.moveaxis(out, 0, axis)

    def get_cderi_real(self, cderi=None):
        '''
        transform the 3-index integral from pw to cos and sin basis
        equivalent to: u_q L_{q,rs} u_k^*[p,r] u_k[q,s]
        '''

        if cderi is None:
            cderi = self.get_cderi_pw()

        cderi = self._rotate_pairs(cderi, 0, with_zero=False)
        cderi = self._rotate_pairs(cderi, 1, conj=True)
        cderi = self._rotate_pairs(cderi, 2)
        assert abs(cderi.imag).max() < 1e-10
        cderi = np.ascontiguousarray(cderi.real)

        return lib.pack_tril(cderi) # -> (nq,nk*(nk+1)/2) save the lower triangular

    def get_integrals(self):
        '''(E_const, h1, packed cderi) in the real basis, cached'''
        if self._ints is None:
            self._ints = (self.energy_nuc(), self.get_h1_real(), self.get_cderi_real())
        return self._ints

    def eval_ao(self, coords):
        '''
        real basis functions on real-space points, shape (ngrids, nkpts)
        phi_0 = 1/sqrt(V), then sqrt(2/V) cos(k.r), sqrt(2/V) sin(k.r) per +k,-k pair
        '''
        kr = np.asarray(coords) @ self.kpts[1::2].T
        ao = np.empty((len(kr), self.nkpts))
        ao[:, 0] = 1
        ao[:, 1::2] = np.sqrt(2) * np.cos(kr)
        ao[:, 2::2] = np.sqrt(2) * np.sin(kr)
        return ao / np.sqrt(self.volume)

    def prep_afqmc(self, mycc,
                   amp_file = "amplitudes.npz",
                   chol_file = "FCIDUMP_chol"):

        from ad_afqmc import pyscf_interface

        mf = mycc._scf
        mol = mf.mol
        nelec = mol.nelec
        nao = mol.nao

        t1 = np.array(mycc.t1)
        t2 = mycc.t2
        t2 = t2.transpose(0, 2, 1, 3)
        np.savez(amp_file, t1=t1, t2=t2)

        # calculate cholesky integrals
        print("# Preparing AFQMC_PT for Homogeneous Electron Gas")

        h0 = mf.energy_nuc()
        h1 = mf.get_hcore()
        chol = lib.unpack_tril(mf._cderi)
        nchol = chol.shape[0]

        v0 = 0.5 * einsum("gpr,grq->pq", chol, chol, optimize="optimal")
        h1_mod = h1 - v0
        chol = chol.reshape((chol.shape[0], -1))

        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis (k-points): {nao}")
        print(f"# Number of CholVecs (q-points): {nchol}")

        pyscf_interface.write_dqmc(
            h1,
            h1_mod,
            chol,
            sum(nelec),
            nao,
            h0,
            ms=0,
            filename=chol_file,
        )

        return None


# backward compatibility: the fast cderi build is now the default
my_ueg_faster = my_ueg


# --------------------------------------------------------------------------
# PySCF wrappers
# --------------------------------------------------------------------------

def M(rs, nelec, gamma=2.0, **kwargs):
    '''
    Build a pyscf Mole describing the UEG. kwargs (verbose, max_memory,
    output, ...) are passed to gto.M. The UEG model is kept as mol.ueg.
    '''
    ueg = my_ueg(rs=rs, nelec=nelec, gamma=gamma)
    kwargs.setdefault('max_memory', 20000)
    mol = gto.M(**kwargs)
    mol.nelectron = ueg.nparticle
    mol.spin = ueg.nelec[0] - ueg.nelec[1]
    mol.nao = ueg.nkpts
    mol.incore_anyway = True
    mol.ueg = ueg
    return mol

def aufbau_dm(mol):
    '''
    plane-wave (Slater determinant) density matrices, i.e. the lowest-|k|
    orbitals occupied for each spin: (dm_alpha, dm_beta)
    '''
    nk = mol.nao
    dm = np.zeros((2, nk, nk))
    for s, n in enumerate(mol.nelec):
        dm[s, np.arange(n), np.arange(n)] = 1.0
    return dm

def _ueg_scf(mf):
    mol = mf.mol
    h0, h1, cderi = mol.ueg.get_integrals()
    nk = mol.nao
    restricted = not isinstance(mf, scf.uhf.UHF)

    mf = mf.density_fit()
    mf._cderi = cderi
    mf.chkfile = None
    mf.init_guess = 'plane-wave aufbau'
    mf.energy_nuc = lambda *args: h0
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nk)

    def get_init_guess(*args, **kwargs):
        dm = aufbau_dm(mol)
        return dm[0] + dm[1] if restricted else dm
    mf.get_init_guess = get_init_guess
    return mf

def RHF(mol):
    '''density-fitted RHF (ROHF if mol.spin != 0) for the UEG'''
    return _ueg_scf(scf.RHF(mol))

def UHF(mol):
    '''density-fitted UHF for the UEG'''
    return _ueg_scf(scf.UHF(mol))


# --------------------------------------------------------------------------
# Spin-symmetry breaking tools
# --------------------------------------------------------------------------

def _lowest_eigs(hop, hdiag, nroots=3, nguess=6, seed=0, tol=1e-6):
    '''
    Lowest eigenpairs of an orbital Hessian by Davidson. At the plane-wave
    determinant the Hessian is block diagonal in the momentum transfer, so
    random guess vectors are added to make sure every sector is sampled.
    '''
    rng = np.random.default_rng(seed)
    x0 = [rng.standard_normal(hdiag.size) for _ in range(nguess)]
    for i in np.argsort(hdiag)[:nroots]:
        x0.append(np.eye(1, hdiag.size, i).ravel())

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd) < 1e-8] = 1e-8
        return dx / hdiagd

    e, v = lib.davidson(hop, x0, precond, tol=tol, nroots=nroots,
                        max_cycle=200, max_space=40, verbose=0)
    return np.atleast_1d(e), np.atleast_2d(v)

def rhf_triplet_instability(mf, **kwargs):
    '''
    Lowest eigenvalues of the real RHF -> real UHF (triplet) orbital Hessian.
    A negative value means the RHF solution is unstable towards spin-symmetry
    breaking.
    '''
    _, _, hop, hdiag = stability._gen_hop_rhf_external(mf, with_symmetry=False)
    return _lowest_eigs(hop, hdiag, **kwargs)[0]

def uhf_internal_instability(mf, **kwargs):
    '''lowest eigenpairs of the real UHF orbital Hessian (pyscf convention)'''
    _, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ,
                                           with_symmetry=False)
    return _lowest_eigs(lambda x: hop(x).real * 2, hdiag * 2, **kwargs)

def spin_kick_dm(mol, amp=0.1, seed=None, dm0=None):
    '''
    Symmetry-broken UHF guess: rotate the alpha orbitals by exp(K) and the beta
    orbitals by exp(-K), K a random antisymmetric matrix of size ~amp.
    '''
    nk = mol.nao
    rng = np.random.default_rng(seed)
    kappa = amp * rng.standard_normal((nk, nk))
    kappa = kappa - kappa.T
    if dm0 is None:
        dm0 = aufbau_dm(mol)
    dm = []
    for s, sign in enumerate((1, -1)):
        e, c = np.linalg.eigh(dm0[s])
        c = scipy.linalg.expm(sign * kappa) @ c[:, e > 0.5]
        dm.append(c @ c.T)
    return np.array(dm)

def run_uhf(mol, dm0=None, conv_tol=1e-9, max_stab_cycle=20, stab_thresh=-1e-5,
            verbose=None):
    '''
    Self-consistent UHF that follows internal instabilities until it lands in a
    local minimum:
      1. SCF (DIIS) from dm0; fall back to second-order SCF if not converged
      2. lowest eigenpair of the UHF orbital Hessian
      3. if negative, rotate the orbitals along that mode and go to 1
    Returns the converged mf (mf.stable and mf.hessian_eig are set).
    '''
    log = lib.logger.new_logger(mol, verbose)
    mf = UHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = 200
    mf._keys = mf._keys.union(['stable', 'hessian_eig'])
    if verbose is not None:
        mf.verbose = verbose
    if dm0 is None:
        dm0 = mf.get_init_guess()

    for it in range(max_stab_cycle + 1):
        mf.kernel(dm0=dm0)
        if not mf.converged:
            log.note('DIIS SCF not converged, switching to second-order SCF')
            mf_so = mf.newton()
            mf_so.kernel(mf.mo_coeff, mf.mo_occ)
            mf.mo_coeff, mf.mo_occ = mf_so.mo_coeff, mf_so.mo_occ
            mf.mo_energy, mf.e_tot = mf_so.mo_energy, mf_so.e_tot
            mf.converged = mf_so.converged

        e, v = uhf_internal_instability(mf, nroots=1)
        mf.hessian_eig = e[0]
        mf.stable = e[0] > stab_thresh
        log.note('stability cycle %d  E = %.10f  lowest Hessian eig = %.3e',
                 it, mf.e_tot, e[0])
        if mf.stable:
            break

        nocca = np.count_nonzero(mf.mo_occ[0] > 0)
        nvira = mol.nao - nocca
        mo = (stability._rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[0, :nocca*nvira]),
              stability._rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[0, nocca*nvira:]))
        dm0 = mf.make_rdm1(mo, mf.mo_occ)

    return mf

def spin_density(mol, dm, ngrid=24):
    '''
    charge and spin density on a uniform ngrid^3 mesh of the simulation box
    returns (coords, rho, m) with m = rho_alpha - rho_beta
    '''
    ueg = mol.ueg
    x = np.arange(ngrid) * ueg.length / ngrid
    coords = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    ao = ueg.eval_ao(coords)
    dm = np.asarray(dm)
    if dm.ndim == 2:
        dm = np.array((dm/2, dm/2))
    rho_s = einsum('gp,spq,gq->sg', ao, dm, ao)
    return coords, rho_s[0] + rho_s[1], rho_s[0] - rho_s[1]

def local_moment(mol, dm, ngrid=24):
    '''integrated absolute spin density per electron, (1/N) int |m(r)| dr'''
    _, _, m = spin_density(mol, dm, ngrid)
    return np.mean(np.abs(m)) * mol.ueg.volume / mol.nelectron
