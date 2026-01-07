import numpy as np

from jax import (
    config,
    vmap,
    jit,
    random,
    numpy as jnp,
    lax,
    scipy as jsp,
)
import jax
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from functools import partial

import itertools
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ueg:
    r_s: float
    n_elec: Tuple[int, int]
    box_length: float = 0.0
    rec_lattice: Tuple = ()
    dim: int = 3
    volume: float = 0.0
    n_particles: int = 0
    density: float = 0.0
    seed: int = 0

    def __post_init__(self):
        assert self.dim == 3, "Only 3D systems are supported."
        assert (
            self.n_elec[0] == self.n_elec[1]
        ), "Only unpolarized systems are supported."
        self.box_length = (
            4 / 3 * jnp.pi * self.r_s**3 * (self.n_elec[0] + self.n_elec[1])
        ) ** (1 / 3)
        self.rec_lattice = (2 * jnp.pi / self.box_length,) * 3
        self.volume = self.box_length**3
        self.n_particles = self.n_elec[0] + self.n_elec[1]
        self.density = self.n_particles / self.volume

    def get_occ_k_points(self) -> jax.Array:
        """Get the occupied k-points for the system."""
        dk = 1 + 1e-5
        max_k = int(jnp.ceil(self.n_elec[0] * dk) ** (1 / 3.0))
        ordinals = sorted(range(-max_k, max_k + 1), key=abs)
        ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))
        kpoints = ordinals @ (jnp.array(self.rec_lattice) * jnp.eye(3)).T
        kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
        k_norms = jnp.linalg.norm(kpoints, axis=1)
        kpoints = kpoints[k_norms <= k_norms[self.n_elec[0] - 1] * dk]

        kpoints_list = kpoints.tolist()
        result = []
        result.append(kpoints_list[0])
        # remove gamma from consideration
        kpoints_list = [k for i, k in enumerate(kpoints_list) if i != 0]

        pairs = {}
        processed = set()
        for k in kpoints_list:
            k_tuple = tuple(k)
            if k_tuple in processed:
                continue

            neg_k = tuple(-x for x in k)
            processed.add(k_tuple)
            if neg_k in map(tuple, kpoints_list):
                processed.add(neg_k)

            canonical = None
            for i, val in enumerate(k):
                if abs(val) > 1e-10:
                    if val > 0:
                        canonical = k_tuple
                        partner = neg_k
                    else:
                        canonical = neg_k
                        partner = k_tuple
                    break

            if canonical is not None:
                pairs[canonical] = partner

        sorted_canonicals = sorted(pairs.keys(), key=lambda k: sum(x * x for x in k))
        for canonical in sorted_canonicals:
            result.append(canonical)
            result.append(pairs[canonical])
        return jnp.array(result)

    @partial(jit, static_argnums=(0,))
    def _calc_dis(self, pos: jax.Array) -> Tuple:
        box_length = jnp.array([self.box_length, self.box_length, self.box_length])
        pos_up = pos[0]
        pos_dn = pos[1]
        pos_flat = jnp.concatenate([pos_up, pos_dn], axis=0)
        n_particles = pos_flat.shape[0]

        def get_disp(i, j):
            dr = pos_flat[i] - pos_flat[j]
            dr = dr - box_length * jnp.round(dr / box_length)
            return dr

        disp = vmap(
            lambda i: vmap(get_disp, in_axes=(None, 0))(i, jnp.arange(n_particles))
        )(jnp.arange(n_particles))
        dist = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-10)
        mask = ~jnp.eye(n_particles, dtype=bool)
        dist = jnp.where(mask, dist, 0.0)
        return dist, disp

    def init_walker_data(self, n_walkers: int) -> dict:
        def walker_init(subkey):
            subkey, subkey_up = random.split(subkey)
            pos_up = random.uniform(subkey_up, (self.n_elec[0], 3)) * self.box_length
            subkey, subkey_dn = random.split(subkey)
            pos_dn = random.uniform(subkey_dn, (self.n_elec[1], 3)) * self.box_length
            pos = jnp.array([pos_up, pos_dn])
            dist, disp = self._calc_dis(pos)
            return pos, dist, disp

        random_key = random.PRNGKey(self.seed)
        random_key, *subkeys = random.split(random_key, n_walkers + 1)
        pos, dist, disp = vmap(walker_init)(jnp.array(subkeys))
        return {
            "pos": pos,
            "dist": dist,
            "disp": disp,
            "random_key": random_key,
        }

    @partial(jit, static_argnums=(0,))
    def update_walker_data(self, new_pos_batch: jax.Array, walker_data: dict) -> dict:
        assert new_pos_batch.shape == walker_data["pos"].shape

        def update_single_walker(carry, new_pos_i):
            dist, disp = self._calc_dis(new_pos_i)
            return carry, (dist, disp)

        _, (dist, disp) = lax.scan(update_single_walker, None, new_pos_batch)
        walker_data["dist"] = dist
        walker_data["disp"] = disp
        walker_data["pos"] = new_pos_batch
        return walker_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ueg_qc(ueg):
    """Quantum chemistry class for the UEG."""

    e_cut_red: float = 1.0  # reduced cutoff energy in (2pi/L)**2 units

    def get_k_points(self) -> jax.Array:
        """Get the k-point basis for the system based on e_cut."""
        e_cut = self.e_cut_red * (2 * np.pi / self.box_length) ** 2
        max_k = int(jnp.ceil(jnp.sqrt(e_cut * 2)))
        ordinals = sorted(range(-max_k, max_k + 1), key=abs)
        ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))
        kpoints = ordinals @ (jnp.array(self.rec_lattice) * jnp.eye(3)).T
        kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
        k_norms = jnp.linalg.norm(kpoints, axis=1) ** 2 / 2
        kpoints = kpoints[k_norms <= e_cut]

        kpoints_list = kpoints.tolist()
        result = []
        result.append(kpoints_list[0])
        # remove gamma from consideration
        kpoints_list = [k for i, k in enumerate(kpoints_list) if i != 0]

        pairs = {}
        processed = set()
        for k in kpoints_list:
            k_tuple = tuple(k)
            if k_tuple in processed:
                continue

            neg_k = tuple(-x for x in k)
            processed.add(k_tuple)
            if neg_k in map(tuple, kpoints_list):
                processed.add(neg_k)

            canonical = None
            for i, val in enumerate(k):
                if abs(val) > 1e-10:
                    if val > 0:
                        canonical = k_tuple
                        partner = neg_k
                    else:
                        canonical = neg_k
                        partner = k_tuple
                    break

            if canonical is not None:
                pairs[canonical] = partner

        sorted_canonicals = sorted(pairs.keys(), key=lambda k: sum(x * x for x in k))
        for canonical in sorted_canonicals:
            result.append(canonical)
            result.append(pairs[canonical])
        return jnp.array(result)

    def madelung(self):
        return (
            -2.837297
            * (3.0 / 4.0 / jnp.pi) ** (1.0 / 3.0)
            * self.n_particles ** (2.0 / 3.0)
            / self.r_s
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_h1(self, k_points: jax.Array) -> jax.Array:
        """Get the one-body Hamiltonian in plane wave basis.
        Includes the Madelung constant."""
        h1 = jnp.diag(jnp.sum(k_points**2, axis=-1) / 2)
        # madelung = 0.5 * self.madelung() / self.n_particles
        return h1  # + madelung * jnp.eye(k_points.shape[0])

    @partial(jax.jit, static_argnums=(0,))
    def get_h1_real(self, k_points: jax.Array) -> jax.Array:
        """Get the one-body Hamiltonian in real basis."""
        h1_pw = self.get_h1(k_points)
        unitary = self.unitary_pw_to_real(k_points)
        h1 = unitary.conj() @ h1_pw @ unitary.T
        return h1.real

    @partial(jax.jit, static_argnums=(0,))
    def eri_element(
        self,
        k_points: jax.Array,
        p: jax.Array,
        q: jax.Array,
        r: jax.Array,
        s: jax.Array,
    ) -> jax.Array:
        """Periodic Coulomb interaction integral ( pq | rs )."""
        g1 = k_points[q] - k_points[p]
        g2 = k_points[r] - k_points[s]
        momentum_conserved = jnp.all(jnp.isclose(g1, g2), axis=-1)
        g1_squared = jnp.sum(g1 * g1, axis=-1)
        non_zero = g1_squared > 1e-10
        element = 4 * jnp.pi / g1_squared / self.volume
        element = jnp.where(jnp.isinf(element) | jnp.isnan(element), 0.0, element)
        return momentum_conserved * non_zero * element

    @partial(jax.jit, static_argnums=(0,))
    def get_eri_tensor(self, k_points: jax.Array) -> jax.Array:
        """Get the ERI tensor in plane wave basis."""
        n_kpts = k_points.shape[0]
        idx = jnp.arange(n_kpts)
        p_idx, q_idx, r_idx, s_idx = jnp.meshgrid(idx, idx, idx, idx, indexing="ij")
        p_flat = p_idx.flatten()
        q_flat = q_idx.flatten()
        r_flat = r_idx.flatten()
        s_flat = s_idx.flatten()
        eri_flat = self.eri_element(k_points, p_flat, q_flat, r_flat, s_flat)
        eri = eri_flat.reshape(n_kpts, n_kpts, n_kpts, n_kpts)
        return eri

    # @partial(jax.jit, static_argnums=(0,))
    def unitary_pw_to_real(self, k_points: jax.Array) -> jax.Array:
        """Unitary transformation from plane wave basis to real cos, sin basis.
        Assumes k_points arranged so that +k, -k pairs are adjacent.
        """
        n_kpts = k_points.shape[0]
        unitary = jnp.zeros((n_kpts, n_kpts), dtype=jnp.complex128)
        unitary_block = jnp.array([[-1.0j, 1.0j], [1.0, 1.0]]) / jnp.sqrt(2.0)
        n_blocks = (n_kpts - 1) // 2
        unitary = unitary.at[0, 0].set(1.0)
        unitary = unitary.at[1:, 1:].set(
            jsp.linalg.block_diag(*([unitary_block] * n_blocks))
        )
        return unitary

    @partial(jax.jit, static_argnums=(0,))
    def get_eri_tensor_real(self, k_points: jax.Array) -> jax.Array:
        """Calculate the ERI tensor in real basis using the unitary transformation."""
        eri = self.get_eri_tensor(k_points)
        unitary = self.unitary_pw_to_real(k_points)
        eri = jnp.einsum("ip,pqrs->iqrs", unitary.conj(), eri, optimize=True)
        eri = jnp.einsum("jq,iqrs->ijrs", unitary, eri, optimize=True)
        eri = jnp.einsum("kr,ijrs->ijks", unitary.conj(), eri, optimize=True)
        eri = jnp.einsum("ls,ijks->ijkl", unitary, eri, optimize=True).real
        return eri

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class my_ueg:

    def __init__(self, rs, nelec):
        self.rs = rs
        self.nelec = nelec
        self.nparticle = sum(nelec)
        self.volume = self.nparticle * 4*np.pi/3 * rs**3
        self.length = self.volume**(1/3)

    def canonical_sign(self, npts):
        # define the sign by first non-zero element
        nz = np.argmax(npts != 0, axis=1)
        signs = np.ones(len(npts), dtype=int)
        mask = npts[np.arange(len(npts)), nz] < 0
        signs[mask] = -1

        return npts * signs[:, None]

    def get_npts(self, N_cut, with_zero=True):
        '''
        generate a 3D-spherical grid points of integers
        {(n1,n2,n3)| n1^2+n2^2+n3^2<N_cut^2; n1,n2,n3 in Z}
        sorted by incresing length and n next to -n
        '''
        n_max = int(np.ceil(N_cut))

        n = np.arange(-n_max, n_max + 1)
        nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")

        n2 = nx**2 + ny**2 + nz**2
        mask = n2 <= N_cut**2 + 1e-10
        npts = np.stack((nx[mask], ny[mask], nz[mask]), axis=1)    

        can_npts = self.canonical_sign(npts)
        unq_npts = np.unique(can_npts, axis=0)
        unq_npts = unq_npts[np.argsort(np.sum(unq_npts**2, axis=1))]
        sort_npts = [[unq_npts[i],-1*unq_npts[i]] for i in range(1,len(unq_npts))]
        sort_npts = np.vstack(sort_npts)
        
        if with_zero:
            sort_npts = np.vstack([[0,0,0],sort_npts])

        return sort_npts

    def get_kpts(self, gamma=2, with_zero=True):
        '''
        get the k-points in a sphere
        by k_cut = gamma * k_fermi
        '''
        rs = self.rs
        Np = sum(self.nelec)

        Nf = (3*Np/(8*np.pi))**(1/3)
        Nc = gamma * Nf
        npts = self.get_npts(Nc, with_zero)
        
        L = (4*np.pi*Np/3)**(1/3) * rs
        kpts = npts * (2*np.pi/L)

        return kpts
    
    def madelung(self):
        '''
        Madelung term 
        interaction of each charge with its image in other cells
        '''
        rs = self.rs
        Np = sum(self.nelec)
        em = -2.837297 * (3/(4*np.pi))**(1/3) * Np**(2/3) / rs
        
        return em
    
    def get_h1(self, kpts):
        h1 = np.diag(np.sum(kpts**2, axis=1)/2)
        return h1
    
    def get_eris_hard(self, kpts):
        '''
        The naive way of calculating k-space eris 
        <k1k2|V|k3k4> = 4pi/V 1/(k1-k3)^2 delta(k1+k2,k3+k4)
        transpose to (11|22) notation
        '''
        
        Nk = len(kpts)
        eris = np.zeros((Nk, Nk, Nk, Nk), dtype=float)

        for i1, k1 in enumerate(kpts):
            for i2, k2 in enumerate(kpts):
                for i3, k3 in enumerate(kpts):
                    for i4, k4 in enumerate(kpts):
                        q = k1 - k3
                        g = k4 - k2
                        q2 = np.dot(q, q)
                        if q2 < 1e-10:
                            continue
                        consv = np.linalg.norm(g-q) < 1e-12
                        if not consv:
                            continue

                        eris[i1, i2, i3, i4] = 4*np.pi / q2 / self.volume
        
        return eris.transpose(0,2,1,3)
    
    def prep_afqmc(self,
                   mycc,
                   chol_cut=1e-6,
                   amp_file="amplitudes.npz",
                   chol_file="FCIDUMP_chol"):
        
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
        print("# Calculating Cholesky integrals")
        
        h0 = mf.energy_nuc()
        h1 = mf.get_hcore()
        h2 = mf._eri.reshape(nao**2,nao**2)
        chol = pyscf_interface.modified_cholesky(h2,chol_cut)
        chol = chol.reshape((-1, nao, nao))
        nchol = chol.shape[0]

        v0 = 0.5 * jnp.einsum("gpr,grq->pq", chol, chol, optimize="optimal")
        h1_mod = h1 - v0
        chol = chol.reshape((chol.shape[0], -1))

        print("# Finished calculating Cholesky integrals#")
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nao}")
        print(f"# Number of Cholesky vectors: {nchol}")

        
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
