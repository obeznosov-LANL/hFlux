#****************************************************************
# ©  2024. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated
#  by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
# All rights in the program are reserved by Triad National Security, LLC,
# and the U.S. Department of Energy/National Nuclear Security Administration.
# The Government is granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works,
# distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
#****************************************************************/

from ctypes import *
import os,sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
# from ..thirdparty.scikit-fem.skfem import *

# locating the 'libsample.so' file in t  ctypes.POINTER(ctypes.c_double),
_file = 'libhflux.so'
_path = os.path.join(*(os.path.split(os.path.abspath(__file__))[:-2] + ("lib64", _file)))
_mod = cdll.LoadLibrary(_path)

class hFieldData(Structure):
    _fields_ = [("R0",c_double),
                ("Z0",c_double),
                ("hR",c_double),
                ("hZ",c_double),
                ("nR", c_int),
                ("nZ", c_int),
                ("total_sz", c_int),
                ("m_data", POINTER(c_double)),
                ("MA_CENTER_R", c_double),
                ("MA_CENTER_Z", c_double),
                ("nModes", c_int)]

class hFieldInterface:

    def plotB(self,BR,BP,BZ,phi):
        fig, ax = plt.subplots(1,3)
        #CS = ax.contour(BR, np.linspace(-2.0,2.0,num=40))
        CS1 = ax[0].contour(BR[:,phi,:])
        CS2 = ax[1].contour(BP[:,phi,:])
        CS3 = ax[2].contour(BZ[:,phi,:])
        #CS = ax.contour(BZ, np.linspace(-2.0,2.0,num=40))
        ax[0].clabel(CS1, inline=True, fontsize=20)
        ax[1].clabel(CS2, inline=True, fontsize=20)
        ax[2].clabel(CS3, inline=True, fontsize=20)
        ax[0].set_title('B_R')
        ax[0].set_title('B_phi')
        ax[0].set_title('B_Z')
        fig.set_figheight(20)
        fig.set_figwidth(35)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        plt.savefig("B.png")

    def projectTo(self,NR,NZ,dR,dZ,BX,BY,BZ,coords,connectivities):
        num_points,num_phi = connectivities.shape
        BR_rpz = np.zeros(connectivities.shape)
        BP_rpz = np.zeros(connectivities.shape)
        BZ_rpz = np.zeros(connectivities.shape)
        rz = np.zeros((BR_rpz.shape[0],2))
        for i,c in enumerate(connectivities):
            rz[i,0] = coords[0][c[0]]
            rz[i,1] = coords[2][c[0]]
            for j,inode in enumerate(c):
                x = coords[0][inode]
                y = coords[1][inode]
                r_hat = np.array([x,y])/np.linalg.norm([x,y])
                p_hat = np.array([-y,x])/np.linalg.norm([x,y])
                BR_rpz[i,j] = np.array([BX[inode],BY[inode]]).dot(r_hat)
                BP_rpz[i,j] = np.array([BX[inode],BY[inode]]).dot(p_hat)
                BZ_rpz[i,j] = BZ[inode]
        tri = Delaunay(rz)
        min_r = np.min(rz[:,0])
        min_z = np.min(rz[:,1])
        max_r = dR*NR + min_r
        max_z = dZ*NZ + min_z
        r_c = np.linspace(min_r,max_r,NR)
        z_c = np.linspace(min_z,max_z,NZ)
        rz_c = np.array([(x_, y_) for y_ in z_c for x_ in r_c ])
        BR_c_rpz = np.zeros((NZ*NR,1))
        BP_c_rpz = np.zeros((NZ*NR,1))
        BZ_c_rpz = np.zeros((NZ*NR,1))
        to_rz = CloughTocher2DInterpolator(tri,BR_rpz[:,0],fill_value=0)
        BR_c_rpz[:,0] = to_rz(rz_c)
        to_rz = CloughTocher2DInterpolator(tri,BP_rpz[:,0],fill_value=0)
        BP_c_rpz[:,0] = to_rz(rz_c)
        to_rz = CloughTocher2DInterpolator(tri,BZ_rpz[:,0],fill_value=0)
        BZ_c_rpz[:,0] = to_rz(rz_c)

        return min_r,min_z,BR_c_rpz, BP_c_rpz, BZ_c_rpz

    def __init__(self,R0,Z0,NR,NP,NZ,dR,dZ,filenames):

        with open(filenames[0], 'r') as f:
          BR_ = [float(x) for x in f.readlines()]
        with open(filenames[1], 'r') as f:
          BP_ = [float(x) for x in f.readlines()]
        with open(filenames[2], 'r') as f:
          BZ_ = [float(x) for x in f.readlines()]
        BR = np.zeros((1 + 2*NP, NZ, NR))
        BP = np.zeros((1 + 2*NP, NZ, NR))
        BZ = np.zeros((1 + 2*NP, NZ, NR))

        for k in range(1+2*NP):
          for j in range(NZ):
            for i in range(NR):
              BR[k,j,i] = BR_[i + j * NR + k * NZ*NR]
              BP[k,j,i] = BP_[i + j * NR + k * NZ*NR]
              BZ[k,j,i] = BZ_[i + j * NR + k * NZ*NR]


        fig, ax = plt.subplots(1,3)
        #CS = ax.contour(BR, np.linspace(-2.0,2.0,num=40))
        CS1 = ax[0].contour(BR[0,:,:])
        CS2 = ax[1].contour(BP[0,:,:])
        CS3 = ax[2].contour(BZ[0,:,:])
        #CS = ax.contour(BZ, np.linspace(-2.0,2.0,num=40))
        ax[0].clabel(CS1, inline=True, fontsize=20)
        ax[1].clabel(CS2, inline=True, fontsize=20)
        ax[2].clabel(CS3, inline=True, fontsize=20)
        ax[0].set_title('B_R')
        ax[1].set_title('B_phi')
        ax[2].set_title('B_Z')
        fig.set_figheight(20)
        fig.set_figwidth(35)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        plt.savefig("B.png")
        plt.close(fig)
        self.BR = BR
        self.BP = BP
        self.BZ = BZ
        self.MA_CENTER_R = c_double( 3.0)
        self.MA_CENTER_Z = c_double(0.0)
        self.gethFieldDataSize = _mod.gethFieldDataSize
        self.gethFieldDataSize.argtypes = (c_int, c_int, POINTER(hFieldData))
        self.gethFieldDataSize.restype = c_int

        self.inithFields = _mod.inithFields
        self.inithFields.argtypes = [c_double, c_double, c_double, c_double,
                                      c_int, c_int, c_int,
                                      POINTER(c_double), POINTER(c_double), POINTER(c_double),
                                      POINTER(hFieldData), POINTER(c_double)]
        self.evaluateFields = _mod.evaluateFields
        self.evaluateFields.argtypes = [POINTER(hFieldData), c_int, c_int,
                                   POINTER(c_double), c_double, POINTER(c_double),
                                   POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        self.computeMagneticAxis = _mod.computeMagneticAxis
        self.computeMagneticAxis.argtypes = [POINTER(hFieldData), POINTER(c_double), POINTER(c_double)]
        self.computePoincarePlotData = _mod.computePoincarePlotData
        self.computePoincarePlotData.argtypes = [POINTER(hFieldData), POINTER(c_double), c_int, c_int]
        self.computeSafetyFactor = _mod.computeSafetyFactor
        self.computeSafetyFactor.argtypes = [POINTER(hFieldData), POINTER(c_double), c_int]

        self.hdata = hFieldData()
        hFieldSize = self.gethFieldDataSize(NR, NZ, byref(self.hdata)) * (1+2*NP)
        self.buffer = np.zeros((hFieldSize))
        self.hdata.m_data = self.buffer.ctypes.data_as(POINTER(c_double))

        self.inithFields(R0,
                               Z0,
                               dR,
                               dZ,
                               NR,
                               NP,
                               NZ,
                               BR.ctypes.data_as(POINTER(c_double)),
                               BP.ctypes.data_as(POINTER(c_double)),
                               BZ.ctypes.data_as(POINTER(c_double)),
                               byref(self.hdata), self.buffer.ctypes.data_as(POINTER(c_double)))

        self.computeMagneticAxis(byref(self.hdata), byref(self.MA_CENTER_R), byref(self.MA_CENTER_Z))
        print('Magnetic axis: ', self.MA_CENTER_R.value, self.MA_CENTER_Z.value)

    def getPoincarePlot(self,num_turns,num_seeds, radius):
        pdata = np.zeros((num_turns+1,num_seeds,2))
        pdata[0,:,0] = np.linspace(self.MA_CENTER_R.value, self.MA_CENTER_R.value + radius, num_seeds)
        pdata[0,:,1] = self.MA_CENTER_Z
        self.computePoincarePlotData(byref(self.hdata), pdata.ctypes.data_as(POINTER(c_double)), num_turns, num_seeds)
        return pdata

    def getSafetyFactor(self,num_points,r0, rend):
        Qdata = np.zeros((3,num_points))
        Qdata[0,:] = np.linspace(r0, rend, num_points)
        self.computeSafetyFactor(byref(self.hdata), Qdata.ctypes.data_as(POINTER(c_double)), num_points)
        Qdata[2,:] /= (2*np.pi)
        return Qdata
    def evalFields(self, rr, zz, NZplot, NRplot, phi):
        Bplot = np.zeros((NZplot, NRplot, 3))
        Psiplot = np.zeros((NZplot, NRplot))
        Chiplot = np.zeros((NZplot, NRplot))
        self.evaluateFields(byref(self.hdata), NRplot, NZplot,
                rr.ctypes.data_as(POINTER(c_double)), phi, zz.ctypes.data_as(POINTER(c_double)),
                Bplot.ctypes.data_as(POINTER(c_double)),
                Psiplot.ctypes.data_as(POINTER(c_double)),
                Chiplot.ctypes.data_as(POINTER(c_double)))
        return Bplot, Psiplot, Chiplot

    @staticmethod
    def getQatPsi(qdata,psi):
        return np.interp(psi,np.flip(qdata[0,:]),np.flip(qdata[3,:]))

    def saveSafetyFactors(self,nx,ny,psi,qdata,ts,filename):

        min_x = 1.979; min_y = -2.205; max_x = 4.3975; max_y = 2.356
        dx = (max_x-min_x)/nx; dy = (max_y - min_y)/ny
        z = 0.0
        xs = ""; ys = ""
        for x in np.linspace(min_x+2*dx,max_x-6*dx,nx): xs += " {:f}".format(x)
        for y in np.linspace(min_y+2*dy,max_y-6*dy,ny): ys += " {:f}".format(y)
        CellData = ""
        Coordinates = """
      <DataArray type="Float32" format="ascii">
        {:s}
      </DataArray>
      <DataArray type="Float32" format="ascii">
        {:s}
      </DataArray>
      <DataArray type="Float32" format="ascii">
        {:f}
      </DataArray>
      """.format(xs,ys,z)
        raw_values = ""
        for j in range(ny):
            for i in range(nx):
                raw_values += "{:f} ".format(self.getQatPsi(qdata,psi[j,i]))
            raw_values += "\n"
        pointData= """
      <DataArray type="Float32" Name="Q" format="ascii">
        {:s}
      </DataArray>
    """.format(raw_values)
        txt ="""<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
  <RectilinearGrid WholeExtent="{:3d} {:3d} {:3d} {:3d} {:3d} {:3d}">
    <Piece Extent="{:3d} {:3d} {:3d} {:3d} {:3d} {:3d}">
      <PointData>{:s}</PointData>
      <CellData>{:s}</CellData>
      <Coordinates>{:s}</Coordinates>
    </Piece>
  </RectilinearGrid>
</VTKFile>
""".format(0,nx-1,0,ny-1,0,0,
           0,nx-1,0,ny-1,0,0,
           pointData,CellData,Coordinates)
        with open(filename+".{:03d}.vtr".format(ts), "w") as f:
            f.write(txt)

