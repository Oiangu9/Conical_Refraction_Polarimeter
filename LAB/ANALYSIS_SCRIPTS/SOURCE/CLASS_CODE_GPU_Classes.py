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
def compute_B0_block(n,a0, rho0, rs_block, ks, z, dk, j0_ksrs):
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
    # pasa que si queremos optimizar tambien el grosor del anillo, hara falta multiplicar por un
    # factor diferente los puntos xs, rs etc
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
        # There is a big inefficiency due to the need of computing the j0, j1 sequentially with numpy. Aaand this is the hard part actually!
        for ix in range(len(self.chunks_x)-1):
            for iy in range(len(self.chunks_y)-1):
                rs_block=(1/w0)*self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]]
                self.B0=self.B0.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B0_block( self.n, self.a0, R0/w0, rs_block, self.ks, Z, self.dk, jax.device_put(j0(self.ks*rs_block))  ))#[ny, nx, 1]
                self.B1=self.B1.at[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]].set(
                compute_B1_block( self.n, self.a0, R0/w0, rs_block, self.ks, Z, self.dk, jax.device_put(j1(self.ks*rs_block))  ))  #[ny, nx, 1]

    def _compute_D_and_Intensity_Turpin(self, in_polrz): # This is rather very fast
        return compute_Intensity(self.B0, self.B1, in_polrz, self.sin_phis, self.cos_phis)

    def _plot_Intensity(self, I, output_path, input_polarization, Z, R0, w0):
        input_angle=input_polarization if type(input_polarization) in [int, float] else np.arctan2(input_polarization.real[1], input_polarization.real[0])
        cv2.imwrite(f"{output_path}/Raw_PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}_R0_{R0}_w0_{w0}.png",       np.asarray((65535*I/jnp.max(I))).astype(np.uint16))

    # the next two functions are just for the generation of datasets in the deep learning approach
    def compute_D_matrix(self, R0_pixels, Z, w0_pixels):
        self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        return compute_D_matrix_raw(self.B0, self.B1, self.sin_phis, self.cos_phis)
    def compute_pieces_for_I_LP(self, R0_pixels, Z, w0_pixels):
        self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        return pieces_for_I_LP(self.B0, self.B1) #, self.phis[:,:,0])

    def compute_CR_ring(self, CR_ring_angle, R0_pixels, Z, w0_pixels):
        # If the argument R0, L and Z are the same as the last time B0B1 was computed, then do not recomupte them
        if self.last_R0_pixels!=R0_pixels or self.last_Z!=Z or self.last_w0_pixels!=w0_pixels:
            self.last_R0_pixels=R0_pixels
            self.last_w0_pixels=w0_pixels
            self.last_Z=Z
            self._compute_B0_B1(Z, R0_pixels, w0_pixels)
        I=self._compute_D_and_Intensity_Turpin(jnp.array([jnp.cos(CR_ring_angle/2), jnp.sin(CR_ring_angle/2)]))
        return np.asarray(I/jnp.max(I))

    def compute_and_plot_CR_ring(self, CR_ring_angle, R0_pixels, Z, w0_pixels, out_path, name):
        I=self.compute_CR_ring(CR_ring_angle, R0_pixels, Z, w0_pixels)
        cv2.imwrite(f"{out_path}/[{name}]__PolAngle_{CR_ring_angle/2:.15f}_CRAngle_{CR_ring_angle:.15f}_Z_{Z}_w0_pix_{w0_pixels}_R0_pix_{R0_pixels}.png",
                (6553*I).astype(np.uint16))



if __name__ == "__main__":








    #phi_CRs = [-3, -2, np.pi/2, -1, 0, 1, np.pi/2, 2, 3, np.pi]
    #phi_CRs = [-2.3932]
    # phi_CRs=[0] # such that 2*26.146deg is their difference


    print("\n\n\nTesting General Simulator:")

    nx=540
    max_k=50
    num_k=1100

    phiCRs = [#None, 0,
        #None, 0, np.pi/2, np.pi/8,
        #np.pi/4, np.pi/4, np.pi/4, np.pi/4,
        #np.pi/4, np.pi/4, np.pi/4, np.pi/4,
        None, None, None, np.pi/4
        ]
    w0s = [#24, 24,
        #24, 24, 24, 24,
        #10, 20, 30, 36,
        #83.5, 41.75, 20.875, 10.44,
        24, 24, 24, 24
        ]
    R0s = [#167, 167,
        #167, 167, 167, 167,
        #60, 120, 180, 220,
         #167, 167, 167, 167,
         167, 167, 167, 167,
         ]
    Zs = [#0, 0,
        #0, 0, 0, 0,
        #0, 0, 0, 0,
        #0, 0, 0, 0,
        #7, 6, 2, 2
        #5, 3, 1.5, 0.5
        1.7, 1.8, 1.2, 0.8
        ]
    names = [#'simul11', 'simul13', 'simul21', 'simul22', 'suml23', 'simul24', 'simul31', 'simul32', 'simul33', 'simul34',
        #'simul41', 'simul42', 'simul43', 'simul44',
        'simul590', 'simul5990', 'simul5999', 'simul599990']
    out = "../../../REPORT/Figures/"
    simulator =RingSimulator_Optimizer_GPU( n=1.5, a0=1.0, max_k=max_k, num_k=num_k, nx=nx,
                                      sim_chunk_x=nx, sim_chunk_y=nx)

    for R0, w0, phiCR, z, name in zip(R0s, w0s, phiCRs, Zs, names):
        if simulator.last_R0_pixels!=R0 or simulator.last_Z!=z or simulator.last_w0_pixels!=w0:
            simulator.last_R0_pixels=R0
            simulator.last_w0_pixels=w0
            simulator.last_Z=z
            simulator._compute_B0_B1(z, R0, w0)
        print(R0, w0, phiCR, z, name)
        if phiCR is None:
            I=simulator._compute_D_and_Intensity_Turpin(in_polrz=jnp.array([1, 1j])/jnp.sqrt(2))
        else:
            I=simulator._compute_D_and_Intensity_Turpin(jnp.array([np.cos(phiCR/2), np.sin(phiCR/2)]))
        cv2.imwrite(f"{out}/{name}.png", np.asarray((255*(I/jnp.max(I)))).astype(np.uint8))
        print(f"Done {name}")

    '''

    simulator=RingSimulator_GPU( n=1.5, w0=1, R0=R0, a0=1.0, max_k=50, num_k=1000, nx=nx, ny=nx, nz=1, xmin=-(nx-1)/2, xmax=(nx-1)/2, ymin=-(nx-1)/2, ymax=(nx-1)/2, zmin=z, zmax=z, sim_chunk_x=nx, sim_chunk_y=nx)

    os.makedirs('./Simulated/General/Full/', exist_ok=True)
    os.makedirs('./Simulated/General/Approx/', exist_ok=True)

    for phi_CR in phi_CRs:
        print(f"Computing {phi_CR}")
        simulator.compute_intensity_Trupin_and_Plot( jnp.array([np.cos(phi_CR/2), np.sin(phi_CR/2)]), './Simulated/General/Full/')
        print("Hard one done!\n")
        simulator.compute_intensity_Todor_and_Plot(phi_CR/2, './Simulated/General/Approx/')

    print("\n\n\nTesting Optimizer Simulator:")
    simulator=RingSimulator_Optimizer_GPU( n=1.5, a0=1.0, max_k=50, num_k=1200, nx=550, sim_chunk_x=550, sim_chunk_y=550)

    os.makedirs('./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution', exist_ok=True)
    os.makedirs('./Simulated/Optimizer/Approx/', exist_ok=True)

    for phi_CR in phi_CRs:
        print(f"Computing {phi_CR}")
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=137.79, Z=0, w0_pixels=30.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Big_Radious_Very_Thin_Width_')
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=137.79, Z=0, w0_pixels=23.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Big_Radious_Thin_Width_')
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=80.7, Z=0, w0_pixels=23.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Small_Radious_Thin_Width_')
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=20.7, Z=0, w0_pixels=23.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Very_Small_Radious_Thin_Width_')
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=137.7, Z=0, w0_pixels=15.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Big_Radious_Thick_Width_')
        simulator.compute_and_plot_CR_ring( phi_CR, R0_pixels=137.7, Z=0, w0_pixels=10.25, out_path='./Simulated/Optimizer/Full/SIMULATED_Kumar_Example/1101_Resolution/', name='Big_Radious_Very_Thick_Width_')
        print("Hard one done!\n")
        #simulator.compute_intensity_Todor_and_Plot(phi_CR/2, './Simulated/Optimizer/Approx/')
    '''
