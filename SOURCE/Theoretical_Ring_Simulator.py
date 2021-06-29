import numpy as np
from scipy.special import j0, j1
import matplotlib.pyplot as plt
import cv2
'''
th_pol=phi_CR/2
phi_CR=th_pol*2

input_pol_vec= (cos(th_pol), sin(th_pol)) con th_pol in [-pi/2, pi/2)
= (cos(phi_CR/2), sin(phi_CR/2)) con phi_CR in [-pi, pi)
'''

class RingSimulator():
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


    def _prepare_grid(self):
        self.xs=np.broadcast_to( np.linspace(self.xmin, self.xmax, self.nx),
            (self.ny, self.nx)) #[ny,nx]
        self.ys=np.broadcast_to( np.flip(np.linspace(self.ymin, self.ymax, self.ny)),
            (self.nx, self.ny)).transpose() #[ny,nx]
        self.zs=np.linspace(self.zmin, self.zmax, self.nz) #[nz]

        self.rs=np.expand_dims(np.sqrt(self.xs**2+self.ys**2), axis=-1) #[ny,nx,1] Broadcastable to k
        self.phis=np.expand_dims(np.arctan2(self.ys, self.xs),  axis=-1) #[ny,nx,1]

        self.cos_phis = np.cos(self.phis) #[ny,nx,1] Broadcastable to z
        self.sin_phis = np.sin(self.phis) #[ny,nx,1]

    def _gaussian_a(self, k):
        return self.a0*np.exp(-k**2/4.0)

    def _compute_B0_B1(self):
        self.B0=np.zeros((self.ny, self.nx, self.nz), dtype=np.complex64) #[ny, nx, nz]
        self.B1=np.zeros((self.ny, self.nx, self.nz), dtype=np.complex64) #[ny, nx, nz]

        ks, dk = np.linspace(start=0, stop=self.max_k, num=self.num_k, endpoint=True, retstep=True)

        chunks_x = list(range(0, self.B0.shape[1],self.sim_chunk_x))+[self.B0.shape[1]]
        chunks_y = list(range(0, self.B0.shape[0],self.sim_chunk_y))+[self.B0.shape[0]]

        for iz, z in enumerate(self.zs):
            for ix in range(len(chunks_x)-1):
                for iy in range(len(chunks_y)-1):
                    self.B0[ chunks_x[ix]:chunks_x[ix+1], chunks_y[iy]:chunks_y[iy+1], iz] = np.sum( self._gaussian_a(ks)*np.exp(-1j*ks**2*z**2/(2*self.n))*np.cos(ks*self.rho0)*j0(ks*self.rs[chunks_x[ix]:chunks_x[ix+1], chunks_y[iy]:chunks_y[iy+1]])*ks, axis=-1)*dk/(2*np.pi) #[ny, nx, 1]
                    self.B1[ chunks_x[ix]:chunks_x[ix+1], chunks_y[iy]:chunks_y[iy+1], iz] = np.sum( self._gaussian_a(ks)*np.exp(-1j*ks**2*z**2/(2*self.n))*np.sin(ks*self.rho0)*j1(ks*self.rs[chunks_x[ix]:chunks_x[ix+1], chunks_y[iy]:chunks_y[iy+1]])*ks, axis=-1 )*dk/(2*np.pi) #[ny, nx, 1]

    def _compute_electric_displacements(self, in_polrz):
        self.D = np.stack((
            (self.B0+self.B1*self.cos_phis)*in_polrz[0]+self.B1*self.sin_phis*in_polrz[1],
            (self.B0-self.B1*self.cos_phis)*in_polrz[1]+self.B1*self.sin_phis*in_polrz[0]
            )) # [2, ny, nx, nz]

    def _compute_intensity_Turpin(self):
        self.I = np.linalg.norm(self.D, axis=0)**2
        #self.I=np.abs(self.B0)**2+np.abs(self.B1)**2

    def _plot_Intensity(self, output_path, input_polarization):
        input_angle=input_polarization if type(input_polarization) in [int, float] else np.arctan2(input_polarization.real[1], input_polarization.real[0])
        for iz, z in enumerate(self.zs):
            plt.imshow(self.I[:,:,iz], cmap='hot', origin='upper', interpolation="none",
                extent=[self.xmin,self.xmax,self.ymin,self.ymax])
            plt.colorbar()
            plt.title(f"Input Polarization: {input_polarization} \nz={z}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"{output_path}/PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}.png")
            plt.clf()
            cv2.imwrite(f"{output_path}/Raw_PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{z}.png",
                (65535*self.I[:,:,iz]/np.max(self.I[:,:,iz])).astype(np.uint16))


    def compute_intensity_Trupin_and_Plot(self, input_polarization_vec, out_path):
        self._compute_electric_displacements(input_polarization_vec)
        self._compute_intensity_Turpin()
        self._plot_Intensity(out_path, input_polarization_vec)

    def _compute_intensity_Todor(self, pol_angle):
        self.nz=1
        self.I = np.cos((self.phis-2*pol_angle)/2)**2*(np.sqrt(2/(9*self.w0*np.pi))*(2*np.exp(-2*(self.rs-self.R0-self.w0)**2/self.w0**2)+np.exp(-2*(self.rs-self.R0+self.w0)**2/self.w0**2 )))/np.pi
    def compute_intensity_Todor_and_Plot(self, input_polarization_angle, out_path):
        self._compute_intensity_Todor( input_polarization_angle)
        self._plot_Intensity(out_path, input_polarization_angle)

def RingSimulator_Optimizer():
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
        self._prepare_grid()

    def _prepare_grid(self):
        self.xs=np.broadcast_to( np.linspace(self.xmin, self.xmax, self.nx),
            (self.ny, self.nx)) #[ny,nx]
        self.ys=np.broadcast_to( np.flip(np.linspace(self.ymin, self.ymax, self.ny)),
            (self.nx, self.ny)).transpose() #[ny,nx]

        self.rs=np.expand_dims(np.sqrt(self.xs**2+self.ys**2), axis=-1) #[ny,nx,1] Broadcastable to k
        self.phis=np.expand_dims(np.arctan2(self.ys, self.xs),  axis=-1) #[ny,nx,1]

        self.cos_phis = np.cos(self.phis) #[ny,nx,1] Broadcastable to z, unused here
        self.sin_phis = np.sin(self.phis) #[ny,nx,1]

        self.ks, self.dk = np.linspace(start=0, stop=self.max_k, num=self.num_k, endpoint=True, retstep=True)

        self.chunks_x = list(range(0, self.nx,self.sim_chunk_x))+[self.nx]
        self.chunks_y = list(range(0, self.ny,self.sim_chunk_y))+[self.ny]

        self.B0=np.zeros((self.ny, self.nx), dtype=np.complex64) #[ny, nx]
        self.B1=np.zeros((self.ny, self.nx), dtype=np.complex64) #[ny, nx]


    def _gaussian_a(self, k):
        return self.a0*np.exp(-k**2/4.0)

    def _compute_B0_B1(self, Z, R0):
        for ix in range(len(chunks_x)-1):
            for iy in range(len(chunks_y)-1):
                self.B0[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]] = np.sum( self._gaussian_a(self.ks)*np.exp(-1j*self.ks**2*Z**2/(2*self.n))*np.cos(self.ks*R0/self.w0)*j0(self.ks*self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]])*self.ks, axis=-1)*self.dk/(2*np.pi) #[ny, nx, 1]
                self.B1[ self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]] = np.sum( self._gaussian_a(ks)*np.exp(-1j*self.ks**2*Z**2/(2*self.n))*np.sin(self.ks*R0/self.w0)*j1(self.ks*self.rs[self.chunks_x[ix]:self.chunks_x[ix+1], self.chunks_y[iy]:self.chunks_y[iy+1]])*self.ks, axis=-1 )*self.dk/(2*np.pi) #[ny, nx, 1]

    def _compute_electric_displacements(self, in_polrz):
        self.D = np.stack((
            (self.B0+self.B1*self.cos_phis)*in_polrz[0]+self.B1*self.sin_phis*in_polrz[1],
            (self.B0-self.B1*self.cos_phis)*in_polrz[1]+self.B1*self.sin_phis*in_polrz[0]
            )) # [2, ny, nx, nz]

    def _compute_intensity(self):
        self.I = np.linalg.norm(self.D, axis=0)**2

    def _compute_intensity_and_electric_displacements(self, in_polarzat):
        self.I = np.linal.norm( np.stack((
            (self.B0+self.B1*self.cos_phis)*in_polarzat[0]+self.B1*self.sin_phis*in_polarzat[1],
            (self.B0-self.B1*self.cos_phis)*in_polarzat[1]+self.B1*self.sin_phis*in_polarzat[0]
            )), axis=0 )**2 # [2, ny, nx, nz]

    def compute_CR_ring(self, CR_ring_angle, R0_pixels, Z):
        self.R0=R0_pixels*self.dx
        self.rho0=R0/self.w0
        self._compute_B0_B1(Z)
        self._compute_intensity_and_electric_displacements(np.array([np.cos(CR_ring_angle/2), np.sin(CR_ring_angle/2)]))
        return self.I

    def _plot_Intensity(self, output_path, input_polarization,Z):
        input_angle=input_polarization if type(input_polarization) in [int, float] else np.arctan2(input_polarization.real[1], input_polarization.real[0])
        cv2.imwrite(f"{output_path}/Raw_PolAngle_{input_angle:.15f}_Vector_{input_polarization}_CRAngle_{2*input_angle:.15f}_Z_{Z}.png",
            (65535*self.I[:,:]/np.max(self.I[:,:])).astype(np.uint16))
    def compute_and_plot_CR_ring(self, CR_ring_angle, R0_pixels, Z, out_path, name):
        self.compute_CR_ring(CR_ring_angle, R0_pixels, Z)
        cv2.imwrite(f"{output_path}/{name}__PolAngle_{CR_ring_angle/2:.15f}_CRAngle_{CR_ring_angle:.15f}_Z_{Z}.png",
            (65535*self.I/np.max(self.I)).astype(np.uint16))




if __name__ == "__main__":

    import os
    phi_CRs = [-3, -2, np.pi/2, -1, 0, 1, np.pi/2, 2, 3, np.pi]
    simulator=RingSimulator( n=1.5, w0=1, R0=10, a0=1.0, max_k=50, num_k=1000, nx=1215, ny=1215, nz=1, xmin=-15, xmax=15, ymin=-15, ymax=15, zmin=0, zmax=0, sim_chunk_x=500, sim_chunk_y=500)

    for phi_CR in phi_CRs:
        os.makedirs('./Simulated/phi_CR/Full/', exist_ok=True)
        os.makedirs('./Simulated/phi_CR/Approx/', exist_ok=True)

        simulator.compute_intensity_Trupin_and_Plot( np.array([np.cos(phi_CR/2), np.sin(phi_CR/2)]), './Simulated/phi_CR/Full/')
        simulator.compute_intensity_Todor_and_Plot(phi_CR/2, './Simulated/phi_CR/Approx/')
