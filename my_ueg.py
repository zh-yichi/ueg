import numpy as np
from pyscf import lib
einsum = lib.einsum

class my_ueg:

    def __init__(self, rs, nelec, gamma):
        self.rs = rs
        self.nelec = nelec
        self.gamma = gamma
        self.nparticle = sum(nelec)
        self.volume = self.nparticle * 4*np.pi/3 * rs**3
        self.length = self.volume**(1/3)
        self.N_cut = gamma * (3*self.nparticle/(8*np.pi))**(1/3)
        self.kpts = self.get_kpts()
        # self.qpts = self.get_qpts()
        self.nkpts = self.kpts.shape[0]
        # self.nqpts = self.qpts.shape[0]

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

        can_npts = self.canonical_sign(npts)
        unq_npts = np.unique(can_npts, axis=0)
        unq_npts = unq_npts[np.argsort(np.sum(unq_npts**2, axis=1))]
        sort_npts = [[unq_npts[i],-1*unq_npts[i]] for i in range(1,len(unq_npts))]
        sort_npts = np.vstack(sort_npts)
        # sort_npts = sort_npts[1:]
        
        if with_zero:
            sort_npts = np.vstack([[0,0,0],sort_npts])

        return sort_npts
    
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
        
        mpts = np.array([[],[],[]]).T
        for n1 in npts:
            for n3 in npts:
                mpts = np.vstack((mpts,n1-n3))

        can_mpts = self.canonical_sign(mpts)
        unq_mpts = np.unique(can_mpts, axis=0)
        unq_mpts = unq_mpts[np.argsort(np.sum(unq_mpts**2, axis=1))]
        sort_mpts = [[unq_mpts[i],-1*unq_mpts[i]] for i in range(1,len(unq_mpts))]
        sort_mpts = np.vstack(sort_mpts)

        return sort_mpts

    def get_kpts(self, gamma=None, with_zero=True):
        '''
        get the k-points in a sphere by k_cut = gamma * k_fermi
        ordered by 0,...,k,-k,... with the length increasing
        '''
        if gamma is None:
            gamma = self.gamma

        rs = self.rs
        Np = self.nparticle

        Nf = (3*Np/(8*np.pi))**(1/3)
        Nc = gamma * Nf

        npts = self.get_npts(Nc, with_zero)
        
        L = (4*np.pi*Np/3)**(1/3) * rs
        kpts = npts * (2*np.pi/L)

        return kpts
    
    def get_qpts(self, mpts=None):
        '''
        q = k1-k3
        '''
        if mpts is None:
            mpts = self.get_mpts()

        rs = self.rs
        Np = sum(self.nelec)
        
        L = (4*np.pi*Np/3)**(1/3) * rs
        qpts = mpts * (2*np.pi/L)
        
        return qpts

    def get_vq(self, qpts=None):
        '''
        V(q) = 4pi / q^2 / V_cell
        '''
        if qpts is None:
            qpts = self.get_qpts()

        q2 = np.sum(qpts**2, axis=1)
        vq = 4*np.pi / q2 / self.volume
        return vq
    
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
        elif not with_zero:
            nblks = nkpts // 2
            u = np.kron(np.eye(nblks), blk)

        return u
    
    def madelung(self):
        '''
        Madelung term 
        interaction of each charge with its image in other cells
        '''
        rs = self.rs
        Np = self.nparticle
        em = -2.837297 * (3/(4*np.pi))**(1/3) * Np**(2/3) / rs
        
        return em
    
    def get_h1(self, kpts=None):
        if kpts is None:
            kpts = self.kpts
        h1 = np.diag(np.sum(kpts**2, axis=1)/2)
        return h1
    
    def get_h1_real(self, kpts=None):
        if kpts is None:
            kpts = self.kpts
        h1_pw = self.get_h1(kpts)
        uk = self.pw2real()
        h1 = uk.conj() @ h1_pw @ uk.T
        return h1.real
    
    def get_eris_hard(self, kpts=None):
        '''
        The naive way of calculating k-space eris 
        <k1k2|V|k3k4> = 4pi/V 1/(k1-k3)^2 delta(k1+k2,k3+k4)
        transpose to (11|22) notation
        '''
        if kpts is None:
            kpts = self.kpts

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
    
    def get_delta_k1k3q(self):
        '''<k1,k3|q> = delta(k1-k3,q)'''
        if npts is None:
            npts = self.get_npts()
        if mpts is None:
            mpts = self.get_mpts()
            
        d = np.empty((len(npts),len(npts),len(mpts)))
        for p,n1 in enumerate(npts):
            for q,n3 in enumerate(npts):
                for g,m in enumerate(mpts):
                    if np.linalg.norm(n1-n3-m) < 1e-12:
                        d[p,q,g] = 1.
                    else:
                        d[p,q,g] = 0.
        return d

    def get_cderi_pw(self, npts=None, mpts=None, qpts=None):
        '''L_{q(m),k1(n1),k3(n3)} = delta(n1-n3,m)*V(q)^1/2'''
        
        if npts is None:
            npts = self.get_npts()
        if mpts is None:
            mpts = self.get_mpts()
        if qpts is None:
            qpts = self.get_qpts()

        cderi = np.empty((len(mpts),len(npts),len(npts)))
        for p,n1 in enumerate(npts):
            for q,n3 in enumerate(npts):
                for g,m in enumerate(mpts):
                    if np.linalg.norm(n1-n3-m) < 1e-12:
                        q2 = np.sum(qpts[g]**2)
                        vq = 4*np.pi / q2 / self.volume
                        cderi[g,p,q] = np.sqrt(vq)
                    else:
                        cderi[g,p,q] = 0.
        return cderi
    
    # def get_cderi_pw_faster(self, npts=None, mpts=None, qpts=None):
    #     '''L_{q(m),k1(n1),k3(n3)} = delta(n1-n3,m)*V(q)^1/2'''
        
    #     if npts is None:
    #         npts = self.get_npts()
    #     if mpts is None:
    #         mpts = self.get_mpts()
    #     if qpts is None:
    #         qpts = self.get_qpts()
        
    #     m_dict = {tuple(m): g for g, m in enumerate(mpts)}
    #     cderi = np.zeros((len(mpts), len(npts), len(npts)))
    #     vq = np.zeros(len(mpts))

    #     for g in range(len(mpts)):
    #         q2 = np.dot(qpts[g], qpts[g])
    #         vq[g] = np.sqrt(4 * np.pi / q2 / self.volume)

    #     for p,n1 in enumerate(npts):
    #         for q,n3 in enumerate(npts):
    #             key = tuple(n1 - n3)
    #             g = m_dict.get(key, None)
    #             if g is not None:
    #                 cderi[g, p, q] = vq[g]
    #     return cderi

    def get_cderi_real(self, cderi=None, uk=None, uq=None):
        '''
        transform the 3-index integral from pw to cos and sin basis
        '''

        if cderi is None:
            cderi = self.get_cderi_pw() #get_cderi_pw()
        
        nqpts = cderi.shape[0]
        
        if uk is None:
            uk = self.pw2real()
        
        if uq is None:
            uq = self.pw2real(nqpts, with_zero=False)
        
        cderi = cderi.transpose(1,2,0)
        cderi = einsum('pr,rsj,sq->pqj', uk.conj(), cderi,uk.T, optimize=True)
        cderi = einsum('pqj,jg->pqg', cderi, uq.T, optimize=True).transpose(2,0,1).real

        return  lib.pack_tril(cderi) # -> (nq,nk*(nk+1)/2) save the lower triangular
    
    def prep_afqmc(self, mycc,
                   amp_file = "amplitudes.npz",
                   chol_file = "FCIDUMP_chol"):
        
        from ad_afqmc import pyscf_interface

        mf = mycc._scf
        mol = mf.mol
        nelec = mol.nelec
        nao = mol.nao
        
        # if pt_or_ci.lower() == 'pt':
        t1 = np.array(mycc.t1)
        t2 = mycc.t2
        t2 = t2.transpose(0, 2, 1, 3)
        np.savez(amp_file, t1=t1, t2=t2)
        # elif pt_or_ci.lower() == 'ci':
        #     t1 = np.array(mycc.t1)
        #     t2 = mycc.t2.transpose(0, 2, 1, 3)
        #     ci2 = t2 + einsum("ia,jb->iajb", t1, t1)
        #     np.savez(amp_file, ci1=t1, ci2=ci2)

        # calculate cholesky integrals
        print("# Preparing AFQMC_PT for Homogeneous Electron Gas")
        
        h0 = mf.energy_nuc()
        h1 = mf.get_hcore()
        chol = lib.unpack_tril(mf._cderi)
        nchol = chol.shape[0]

        v0 = 0.5 * einsum("gpr,grq->pq", chol, chol, optimize="optimal")
        h1_mod = h1 - v0
        chol = chol.reshape((chol.shape[0], -1))

        # print("# Finished calculating Cholesky integrals#")
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


class my_ueg_faster(my_ueg):

    def get_cderi_pw(self, npts=None, mpts=None, qpts=None):
        '''L_{q(m),k1(n1),k3(n3)} = delta(n1-n3,m)*V(q)^1/2'''
        
        if npts is None:
            npts = self.get_npts()
        if mpts is None:
            mpts = self.get_mpts()
        if qpts is None:
            qpts = self.get_qpts()
        
        m_dict = {tuple(m): g for g, m in enumerate(mpts)}
        cderi = np.zeros((len(mpts), len(npts), len(npts)))
        vq = np.zeros(len(mpts))

        for g in range(len(mpts)):
            q2 = np.dot(qpts[g], qpts[g])
            vq[g] = np.sqrt(4 * np.pi / q2 / self.volume)

        for p,n1 in enumerate(npts):
            for q,n3 in enumerate(npts):
                key = tuple(n1 - n3)
                g = m_dict.get(key, None)
                if g is not None:
                    cderi[g, p, q] = vq[g]
                    
        return cderi