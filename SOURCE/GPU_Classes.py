import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import j0, j1
import matplotlib.pyplot as plt
import cv2
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


@jax.jit
def _gaussian_a( a0, k):
    return a0*jnp.exp(-k**2/4.0)


@jax.jit
def compute_B0_block(n,a0, rho0, rs_block, ks, z,dk, j0_ksrs):
    #print(type(a0),type(ks),type(z),type(n),type(rho0),type(rs_block),type(dk))
    return jnp.sum( a0*jnp.exp(-ks**2/4.0)*jnp.exp(-1j*ks**2*z**2/(2*n))*jnp.cos(ks*rho0)*j0_ksrs*ks, axis=-1)*dk/(2*jnp.pi)

@jax.jit
def compute_B1_block(n,a0, rho0, rs_block, ks, z, dk, j1_ksrs):
    #print(type(j1(ks*rs_block)))
    return jnp.sum( a0*jnp.exp(-ks**2/4.0)*jnp.exp(-1j*ks**2*z**2/(2*n))*jnp.sin(ks*rho0)*j1_ksrs*ks, axis=-1)*dk/(2*jnp.pi)

@jax.jit
def compute_Intensity(B0,B1, in_polrz, sin_phis, cos_phis):
    return jnp.linalg.norm( jnp.stack((
        (B0+B1*cos_phis)*in_polrz[0]+B1*sin_phis*in_polrz[1],
        (B0-B1*cos_phis)*in_polrz[1]+B1*sin_phis*in_polrz[0]
        )), axis=0)**2 # [2, ny, nx, nz]



@jax.jit
def compute_intensity_Todor(phis, w0, rs, R0, pol_angle):
    return jnp.cos((phis-2*pol_angle)/2)**2*(jnp.sqrt(2/(9*w0*jnp.pi))*(2*jnp.exp(-2*(rs-R0-w0)**2/w0**2)+jnp.exp(-2*(rs-R0+w0)**2/w0**2 )))/jnp.pi


class RingSimulator_GPU():
    def __init__(self, n, w0, R0, a0, max_k, num_k, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, sim_chunk_x, sim_chunk_y):
        self.n=n
        self.w0=w0
        self.R0=R0
        self.a0=a0
        self.max_k=max_k
        self.num_k=num_k
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.zmin=zmin
        self.zmax=zmax
        self.rho0=R0/w0
        self.sim_chunk_x=sim_chunk_x
        self.sim_chunk_y=sim_chunk_y
        self._prepare_grid()
        self._compute_B0_B1()

    #@jax.partial(jax.jit, static_argnums=(0,))
    def _prepare_grid(self):
        self.xs=jnp.broadcast_to( jnp.linspace(self.xmin, self.xmax, self.nx),
            (self.ny, self.nx)) #[ny,nx]
        self.ys=jnp.broadcast_to( jnp.flip(jnp.linspace(self.ymin, self.ymax, self.ny)),
            (self.nx, self.ny)).transpose() #[ny,nx]
        self.zs=jnp.linspace(self.zmin, self.zmax, self.nz) #[nz]

        self.rs=jnp.expand_dims(jnp.sqrt(self.xs**2+self.ys**2), axis=-1) #[ny,nx,1] Broadcastable to k
        self.phis=jnp.expand_dims(jnp.arctan2(self.ys, self.xs),  axis=-1) #[ny,nx,1]

        self.cos_phis = jnp.cos(self.phis) #[ny,nx,1] Broadcastable to z
        self.sin_phis = jnp.sin(self.phis) #[ny,nx,1]

        self.chunks_x = jnp.array(list(range(0,self.nx,self.sim_chunk_x))+[self.nx])
        self.chunks_y = jnp.array(list(range(0, self.ny,self.sim_chunk_y))+[self.ny])

        self.B0=jnp.zeros((self.ny, self.nx, self.nz), dtype=jnp.complex64) #[ny, nx, nz]
        self.B1=jnp.zeros((self.ny, self.nx, self.nz), dtype=jnp.complex64) #[ny, nx, nz]
        self.ks, self.dk = jnp.linspace(start=0, stop=self.max_k, num=self.num_k, endpoint=True, retstep=True)

    def _compute_B0_B1(self):
        # There is a big inefficiency due to the need of computing the j0, j1 sequentially with numpy. Aaand this is the hard part actually!
        for iz in range(len(self.zs)):
            for ix in range(len(self.chunks_x)-1):
                for iy in range(len(self.chunks_y)-1):
                    rs_block=self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]]
                    self.B0=self.B0.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1], iz].set(
                    compute_B0_block( self.n, self.a0, self.rho0, rs_block, self.ks, self.zs[iz], self.dk, jax.device_put(j0(self.ks*rs_block))  ))#[ny, nx, 1]
                    self.B1=self.B1.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1], iz].set(
                    compute_B1_block( self.n, self.a0, self.rho0, rs_block, self.ks, self.zs[iz], self.dk, jax.device_put(j1(self.ks*rs_block))  ))  #[ny, nx, 1]

    def _compute_D_and_Intensity_Turpin(self, in_polrz): # This is rather very fast
        return compute_Intensity(self.B0, self.B1, in_polrz, self.sin_phis, self.cos_phis)

    def _plot_Intensity(self, I, output_path, input_polarization):
        input_angle=input_polarization if type(input_polarization) in [int, float] else np.arctan2(input_polarization.real[1], input_polarization.real[0])
        for iz, z in enumerate(self.zs):
            plt.imshow(np.asarray(I[:,:,iz]), cmap='hot', origin='upper', interpolation="none",
                extent=[self.xmin,self.xmax,self.ymin,self.ymax])
            plt.colorbar()
            plt.title(f"Input Polarization: {input_polarization} \nz={z}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"{output_path}/PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}.png")
            plt.clf()
            cv2.imwrite(f"{output_path}/Raw_PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}.png",
                np.asarray((65535*I[:,:,iz]/jnp.max(I[:,:,iz]))).astype(np.uint16))

    def compute_intensity_Trupin_and_Plot(self, input_polarization_vec, out_path):
        self._plot_Intensity(
            self._compute_D_and_Intensity_Turpin(in_polrz=input_polarization_vec),
             out_path, input_polarization_vec)


    def compute_intensity_Todor_and_Plot(self, input_polarization_angle, out_path):
        self.nz=1
        self._plot_Intensity(
            compute_intensity_Todor( phis=self.phis, w0=self.w0, rs=self.rs, R0=self.R0, pol_angle=input_polarization_angle),
            out_path, input_polarization_angle)



class RingSimulator_Optimizer_GPU():
    # La computacion de B0,B1 es lo mas costoso con diferencia, pero al parecer solo dependen
    # de R0 y de z (No del ángulo, así que en las linear optimizations es interesante saber esto!)
    def __init__(self, n, w0, a0, max_k, num_k, nx, ny, xmin, xmax, ymin, ymax, sim_chunk_x, sim_chunk_y):
        self.n=n
        self.w0=w0
        self.a0=a0
        self.max_k=max_k
        self.num_k=num_k
        self.nx=nx
        self.ny=ny
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.sim_chunk_x=sim_chunk_x
        self.sim_chunk_y=sim_chunk_y
        self.dx=(xmax-xmin)/(nx-1)
        self.last_R0=None
        self.last_Z=None
        self._prepare_grid()

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

    def _compute_B0_B1(self, Z, R0):
        # There is a big inefficiency due to the need of computing the j0, j1 sequentially with numpy. Aaand this is the hard part actually!
        for ix in range(len(self.chunks_x)-1):
            for iy in range(len(self.chunks_y)-1):
                rs_block=self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]]
                self.B0=self.B0.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B0_block( self.n, self.a0, R0/self.w0, rs_block, self.ks, Z, self.dk, jax.device_put(j0(self.ks*rs_block))  ))#[ny, nx, 1]
                self.B1=self.B1.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B1_block( self.n, self.a0, R0/self.w0, rs_block, self.ks, Z, self.dk, jax.device_put(j1(self.ks*rs_block))  ))  #[ny, nx, 1]

    def _compute_D_and_Intensity_Turpin(self, in_polrz): # This is rather very fast
        return compute_Intensity(self.B0, self.B1, in_polrz, self.sin_phis, self.cos_phis)

    def _plot_Intensity(self, I, output_path, input_polarization, Z, R0):
        input_angle=input_polarization if type(input_polarization) in [int, float] else np.arctan2(input_polarization.real[1], input_polarization.real[0])
        cv2.imwrite(f"{output_path}/Raw_PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}_R0_{R0}.png",       np.asarray((65535*I/jnp.max(I))).astype(np.uint16))

    def compute_CR_ring(self, CR_ring_angle, R0_pixels, Z):
        # If the argument R0 and Z are the same as the last time B0B1 was computed, then do not recomupte them
        if self.last_R0!=R0_pixels or self.last_Z!=Z:
            self.last_R0=R0_pixels
            self.last_Z=Z
            self._compute_B0_B1(Z, R0_pixels*self.dx)
        I=self._compute_D_and_Intensity_Turpin(jnp.array([jnp.cos(CR_ring_angle/2), jnp.sin(CR_ring_angle/2)]))
        return np.asarray(I/jnp.max(I))

    def compute_and_plot_CR_ring(self, CR_ring_angle, R0_pixels, Z, out_path, name):
        I=self.compute_CR_ring(CR_ring_angle, R0_pixels, Z)
        cv2.imwrite(f"{out_path}/[{name}]__PolAngle_{CR_ring_angle/2:.15f}_CRAngle_{CR_ring_angle:.15f}_Z_{Z}_R0_{R0_pixels*self.dx}.png",
                (65535*I).astype(np.uint16))



if __name__ == "__main__":

    phi_CRs = [-3, -2, np.pi/2, -1, 0, 1, np.pi/2, 2, 3, np.pi]
    #phi_CRs = [-3]

    '''
    print("\n\n\nTesting General Simulator:")
    simulator=RingSimulator_GPU( n=1.5, w0=1, R0=7, a0=1.0, max_k=50, num_k=300, nx=200, ny=200, nz=1, xmin=-15, xmax=15, ymin=-15, ymax=15, zmin=0, zmax=0, sim_chunk_x=200, sim_chunk_y=200)

    os.makedirs('./Simulated/General/Full/', exist_ok=True)
    os.makedirs('./Simulated/General/Approx/', exist_ok=True)

    for phi_CR in phi_CRs:
        print(f"Computed {phi_CR}")
        simulator.compute_intensity_Trupin_and_Plot( jnp.array([np.cos(phi_CR/2), np.sin(phi_CR/2)]), './Simulated/General/Full/')
        print("Hard one done!\n")
        simulator.compute_intensity_Todor_and_Plot(phi_CR/2, './Simulated/General/Approx/')
    '''

    print("\n\n\nTesting Optimizer Simulator:")
    simulator=RingSimulator_Optimizer_GPU( n=1.5, w0=1, a0=1.0, max_k=50, num_k=500, nx=200, ny=200,  xmin=-15, xmax=15, ymin=-15, ymax=15, sim_chunk_x=200, sim_chunk_y=200)

    os.makedirs('./Simulated/Optimizer/Full/', exist_ok=True)
    os.makedirs('./Simulated/Optimizer/Approx/', exist_ok=True)

    for phi_CR in phi_CRs:
        print(f"Computed {phi_CR}")
        simulator.compute_and_plot_CR_ring( phi_CR, 60, 0, './Simulated/Optimizer/Full/', '')
        print("Hard one done!\n")
        #simulator.compute_intensity_Todor_and_Plot(phi_CR/2, './Simulated/Optimizer/Approx/')