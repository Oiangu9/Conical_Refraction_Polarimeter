import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import j0, j1
import cv2
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


@jax.jit
def _gaussian_a( a0, k):
    return a0*jnp.exp(-k**2/4.0)


@jax.jit
def compute_B0_block(n,a0, rho0, rs_block, ks, z, dk, j0_ksrs):
    return jnp.sum( a0*jnp.exp(-ks**2/4.0)*jnp.exp(-1j*ks**2*z**2/(2*n))*jnp.cos(ks*rho0)*j0_ksrs*ks, axis=-1)*dk/(2*jnp.pi)

@jax.jit
def compute_B1_block(n,a0, rho0, rs_block, ks, z, dk, j1_ksrs):
    return jnp.sum( a0*jnp.exp(-ks**2/4.0)*jnp.exp(-1j*ks**2*z**2/(2*n))*jnp.sin(ks*rho0)*j1_ksrs*ks, axis=-1)*dk/(2*jnp.pi)

@jax.jit
def compute_Intensity(B0,B1, in_polrz, sin_phis, cos_phis):
    return jnp.linalg.norm( jnp.stack((
        (B0+B1*cos_phis)*in_polrz[0]+B1*sin_phis*in_polrz[1],
        (B0-B1*cos_phis)*in_polrz[1]+B1*sin_phis*in_polrz[0]
        )), axis=0)**2 # [2, ny, nx, nz]

@jax.jit
def compute_D_matrix_raw(B0, B1, sin_phis, cos_phis):
    return jnp.stack((
        B0+B1*cos_phis, B1*sin_phis,
        B1*sin_phis, B0-B1*cos_phis
        )) # [2, 2, ny, nx, nz]

@jax.jit
def pieces_for_I_LP(B0, B1):
    return jnp.stack((
        B0.real**2+B0.imag**2 + B1.real**2+B1.imag**2, 2*(B1.conjugate()*B0).real
    ))


@jax.jit
def compute_intensity_Todor(phis, w0, rs, R0, pol_angle):
    return jnp.cos((phis-2*pol_angle)/2)**2*(jnp.sqrt(2/(9*w0*jnp.pi))*(2*jnp.exp(-2*(rs-R0-w0)**2/w0**2)+jnp.exp(-2*(rs-R0+w0)**2/w0**2 )))/jnp.pi


class RingSimulator_GPU():
    def __init__(self, n, a0, max_k, num_k, nx, sim_chunk_x, sim_chunk_y):
        self.n=n
        self.a0=a0
        self.max_k=max_k
        self.num_k=num_k
        self.nx=nx
        self.ny=nx
        self.xmin=-(nx-1)/2
        self.xmax=(nx-1)/2
        self.ymin=-(nx-1)/2
        self.ymax=(nx-1)/2
        self.sim_chunk_x=sim_chunk_x
        self.sim_chunk_y=sim_chunk_y
        self.dx=(self.xmax-self.xmin)/(nx-1)
        self.last_R0_pixels=None
        self.last_Z=None
        self.last_w0_pixels=None
        self._prepare_grid()
        
    def compute_D_matrix(self, R0_pixels, Z, w0_pixels): # used for dataset generator!
        self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        return compute_D_matrix_raw(self.B0, self.B1, self.sin_phis, self.cos_phis)
    
    def compute_pieces_for_I_LP_and_CP(self, R0_pixels, Z, w0_pixels): # used for dataset generator!
        self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        return pieces_for_I_LP(self.B0, self.B1) #, self.phis[:,:,0])

    #@jax.partial(jax.jit, static_argnums=(0,))
    def _prepare_grid(self):
        self.xs=jnp.broadcast_to( jnp.linspace(self.xmin, self.xmax, self.nx),
            (self.ny, self.nx)) #[ny,nx]
        self.ys=jnp.broadcast_to( jnp.flip(jnp.linspace(self.ymin, self.ymax, self.ny)),
            (self.nx, self.ny)).transpose() #[ny,nx]

        self.rs=jnp.expand_dims(jnp.sqrt(self.xs**2+self.ys**2), axis=-1) #[ny,nx,1] Broadcastable to k
        self.phis=jnp.expand_dims(jnp.arctan2(self.ys, self.xs),  axis=-1) #[ny,nx,1]

        self.cos_phis = jnp.squeeze(jnp.cos(self.phis)) #[ny,nx]
        self.sin_phis = jnp.squeeze(jnp.sin(self.phis)) #[ny,nx]

        self.chunks_x = jnp.array(list(range(0,self.nx,self.sim_chunk_x))+[self.nx])
        self.chunks_y = jnp.array(list(range(0, self.ny,self.sim_chunk_y))+[self.ny])

        self.B0=jnp.zeros((self.ny, self.nx), dtype=jnp.complex64) #[ny, nx]
        self.B1=jnp.zeros((self.ny, self.nx), dtype=jnp.complex64) #[ny, nx]
        self.ks, self.dk = jnp.linspace(start=0, stop=self.max_k, num=self.num_k, endpoint=True, retstep=True)

    def _compute_B0_B1(self, Z, R0, w0):
        # There is a big inefficiency due to the need to compute the j0, j1 sequentially with numpy. Aaand this is the hard part actually! This part is not GPU parallelized yet!!!
        for ix in range(len(self.chunks_x)-1):
            for iy in range(len(self.chunks_y)-1):
                rs_block=(1/w0)*self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]]
                self.B0=self.B0.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B0_block( self.n, self.a0, R0/w0, rs_block, self.ks, Z, self.dk, jax.device_put(j0(self.ks*rs_block))  ))#[ny, nx, 1]
                self.B1=self.B1.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B1_block( self.n, self.a0, R0/w0, rs_block, self.ks, Z, self.dk, jax.device_put(j1(self.ks*rs_block))  ))  #[ny, nx, 1]

    def _compute_D_and_Intensity_Turpin(self, in_polrz): # This is rather very fast
        return compute_Intensity(self.B0, self.B1, in_polrz, self.sin_phis, self.cos_phis)


    def compute_CR_ring(self, CR_ring_angle, R0_pixels, Z, w0_pixels):
        # If the argument R0, L and Z are the same as the last time B0B1 was computed, then do not recomupte them
        if self.last_R0_pixels!=R0_pixels or self.last_Z!=Z or self.last_w0_pixels!=w0_pixels:
            self.last_R0_pixels=R0_pixels
            self.last_w0_pixels=w0_pixels
            self.last_Z=Z
            self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        I=self._compute_D_and_Intensity_Turpin(jnp.array([jnp.cos(CR_ring_angle/2), jnp.sin(CR_ring_angle/2)]))
        return np.asarray(I/jnp.max(I))

    def compute_and_save_CR_ring_image(self, CR_ring_angle, R0_pixels, Z, w0_pixels, out_path, name):
        I=self.compute_CR_ring(CR_ring_angle, R0_pixels, Z, w0_pixels)
        cv2.imwrite(f"{out_path}/[{name}]__PolAngle_{CR_ring_angle/2:.15f}_CRAngle_{CR_ring_angle:.15f}_Z_{Z}_w0_pix_{w0_pixels}_R0_pix_{R0_pixels}.png", (6553*I).astype(np.uint16))