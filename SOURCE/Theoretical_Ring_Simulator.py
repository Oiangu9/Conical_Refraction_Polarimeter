import numpy as np
from scipy.special import j0, j1
import matplotlib.pyplot as plt
import cv2

class RingSimulator():
    def __init__(self, n, w0, R0, a0, max_k, num_k, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
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

        for iz, z in enumerate(self.zs):
            self.B0[ :, :, iz] = np.sum( self._gaussian_a(ks)*np.exp(-1j*ks**2*z**2/(2*self.n))*np.cos(ks*self.rho0)*j0(ks*self.rs)*ks, axis=-1)*dk/(2*np.pi) #[ny, nx, 1]
            self.B1[ :, :, iz] = np.sum( self._gaussian_a(ks)*np.exp(-1j*ks**2*z**2/(2*self.n))*np.sin(ks*self.rho0)*j1(ks*self.rs)*ks, axis=-1 )*dk/(2*np.pi) #[ny, nx, 1]

    def _compute_electric_displacements(self, in_polrz):
        self.D = np.stack((
            (self.B0+self.B1*self.cos_phis)*in_polrz[0]+self.B1*self.sin_phis*in_polrz[1],
            (self.B0-self.B1*self.cos_phis)*in_polrz[1]+self.B1*self.sin_phis*in_polrz[0]
            )) # [2, ny, nx, nz]

    def _compute_intensity_Turpin(self):
        self.I = np.linalg.norm(self.D, axis=0)**2
        #self.I=np.abs(self.B0)**2+np.abs(self.B1)**2

    def _plot_Intensity(self, output_path, input_polarization):
        for iz in range(self.nz):
            plt.imshow(self.I[:,:,iz], cmap='hot', origin='upper', interpolation="none",
                extent=[self.xmin,self.xmax,self.ymin,self.ymax])
            plt.colorbar()
            plt.title(f"Input Polarization: {str(input_polarization)} \nz={self.zs[iz]}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"{output_path}/{str(input_polarization)}_{iz}.png")
            plt.clf()
            cv2.imwrite(f"{output_path}/Raw_{str(input_polarization)}_{iz}.png",
                (65535*self.I[:,:,iz]/np.max(self.I[:,:,iz])).astype(np.uint16))


    def compute_intensity_Trupin_and_Plot(self, input_polarization, out_path):
        self._compute_electric_displacements(input_polarization)
        self._compute_intensity_Turpin()
        self._plot_Intensity(out_path, input_polarization)

    def _compute_intensity_Todor(self, pol):
        self.nz=1
        self.I = np.sin((self.phis-pol)/2)**2*(np.sqrt(2/(9*self.w0*np.pi))*(2*np.exp(-2*(self.rs-self.R0-self.w0)**2/self.w0**2)+np.exp(-2*(self.rs-self.R0+self.w0)**2/self.w0**2 )))/np.pi
    def compute_intensity_Todor_and_Plot(self, input_polarization, out_path):
        self._compute_intensity_Todor( input_polarization)
        self._plot_Intensity(out_path, input_polarization)


if __name__ == "__main__":

    simulator=RingSimulator( n=1.5, w0=1, R0=10, a0=1.0, max_k=50, num_k=1000, nx=300, ny=320, nz=1, xmin=-15, xmax=15, ymin=-15, ymax=15, zmin=0, zmax=0)
    simulator.compute_intensity_Trupin_and_Plot(np.array([1,1j])/np.sqrt(2))
    simulator.compute_intensity_Todor_and_Plot(0)
