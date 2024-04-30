import os
import numpy as np
import pickle
import math
import scipy as sp
from scipy import interpolate
from numpy.linalg import solve
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from time import time
import spekpy
import skimage
import copy
import matplotlib
import random

from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 

from cil.io import NikonDataReader, RAWFileWriter, TIFFStackReader, TIFFWriter
from cil.utilities.jupyter import islicer
from cil.utilities.display import show_geometry, show2D
from cil.recon import FDK
from cil.plugins.tigre import FBP, ProjectionOperator
from cil.processors import TransmissionAbsorptionConverter, Slicer, CentreOfRotationCorrector
from cil.optimisation.algorithms import CGLS, SIRT

from cil.optimisation.algorithms import GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction, SmoothMixedL21Norm

from cil.io import NikonDataReader
from cil.utilities.jupyter import islicer
from cil.utilities.display import show_geometry, show2D, show1D
from cil.recon import FDK, FBP
from cil.plugins.tigre import ProjectionOperator#, FBP
from cil.processors import TransmissionAbsorptionConverter, AbsorptionTransmissionConverter, Slicer, Normaliser, Padder
from cil.optimisation.algorithms import CGLS, SIRT
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry
from cil.utilities.noise import gaussian, poisson

from sim_main import lin_interp_sino2D

print(os.getcwd())
if os.getcwd() == "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev":
    os.chdir('analysis/s214743_bsc')
    print(os.getcwd())

base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
bfig_dir = os.path.join(base_dir,'bjobs/figs')


from sim_main import fun_attenuation, generate_spectrum, generate_triangle_image, staircase_bhc

from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
# from scipy.signal import convolve
from scipy.ndimage import convolve1d

from cil.optimisation.operators import CompositionOperator, FiniteDifferenceOperator, MatrixOperator
from cil.framework import DataContainer

def TV_2D(im):
    None

def apply_convolution(data, kernel):
    # Apply 1D convolution along the second axis (axis=1) for each projection
    return convolve1d(data, kernel, axis=1, mode='constant', cval=0.0)

def gaussian_kernel(beta, gamma, threshold=1e-3, plot=False):
    beta,gamma = float(beta),float(gamma)
    # Based on setting the gaussians equal to the thresholdf
    cutoff = int(np.ceil( gamma * np.sqrt(-np.log(0.5*threshold)) ))
    
    # Make sure we include the displacement in both directions
    length = round(2*cutoff + 2 * beta)
    
    # Ensure length is odd
    length = int(np.ceil(length))
    if length % 2 == 0:
        length += 1
    
    # Generate the kernel
    extend = length//2
    x = np.linspace(-extend, extend, length)
    kernel = np.exp(-((x - beta) ** 2) / (gamma ** 2)) + np.exp(-((x + beta) ** 2) / (gamma ** 2))
    
    # Normalize the kernel (maybe)
    # kernel /= np.sum(kernel)

    ###
    if plot:
        # print(x)
        # print(kernel)
        plt.plot(x,kernel,'-o')
        plt.show()
    
    return kernel

# gaussian_kernel(30,25)
# gaussian_kernel(15,10)
# gaussian_kernel(0,25)

def compute_scatter_basis(data,basis_params):
        # basis params: list of lists [alpha,beta,gamma]
        S = np.zeros((len(basis_params),*data.shape))
        alpha,beta,gamma = [list(param) for param in zip(*basis_params)]
        for i in range(S.shape[0]):
            S[i] = apply_convolution(alpha[i]*data*np.exp(-data), gaussian_kernel(beta[i],gamma[i]))
        
        return S

def compute_image_basis(data,scatter_basis):
    None

def clip_otsu_segment(recon, ig, clip=True, title=''):
    if clip:
        tau = threshold_otsu(np.clip(recon.as_array(),a_min=0,a_max=None))
    else:
        tau = threshold_otsu(recon.as_array())
    
    segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    show2D(segmented, title=title)
    return segmented

def testing():
    # file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    trans = AbsorptionTransmissionConverter()(data).as_array()
    sino = data.as_array()
    b = data.as_array()
    basis_params = [[1,0,5],[1,10,10],[1,10,20],[1,0,20],[1,20,20],[1,0,40],[1,0,100]]
    I_S = compute_scatter_basis(sino,basis_params)
    proj = 1400
    detector_indices = np.arange(0,sino.shape[1],1)

    markersize=3
    
    basis_idx = 5
    plt.plot(detector_indices,trans[proj,:],'b-',markersize=markersize, label='raw transmission')
    # plt.plot(detector_indices,0.5*1/basis_params[basis_idx][2]*I_S[basis_idx,proj,:],'r-',markersize=markersize)
    plt.plot(detector_indices,0.05*1/basis_params[basis_idx][2]*I_S[basis_idx,proj,:],'r-',markersize=markersize, label='I_S')
    plt.legend()
    plt.show()

    I_S = 0.05*1/basis_params[basis_idx][2]*I_S[basis_idx]
    I_Q = trans-I_S
    plt.plot(detector_indices,I_Q[proj],'b-',markersize=markersize)
    plt.show()
    _ = gaussian_kernel(*basis_params[basis_idx][1:3],plot=True)

    # data_corr = AcquisitionData(array=np.array(-np.log(I_Q), dtype='float32'), geometry=ag)
    # s = np.log(I_S+trans)-np.log(trans)
    # s = -np.log(-I_S+trans)+np.log(trans) # gives good results
    # s = -np.log(I_S+trans)+np.log(trans)
    # s = -np.log(trans) + np.log(trans-I_S)
    # s = I_S/(I_S+trans)
    s = b + np.log(I_Q) # what is stated in the article
    data_corr = AcquisitionData(array=np.array(s, dtype='float32'), geometry=ag)
    recon = FDK(data).run(verbose=0)
    recon_corr = FDK(data_corr).run(verbose=0)
    # show2D(recon)
    # show2D(recon_corr)
    show2D([recon,-recon_corr])

    clip_otsu_segment(recon, ig, clip=False)
    clip_otsu_segment(recon_corr, ig, clip=False)
    
    plt.hist(0.25*data.as_array().flatten(),bins=150)
    plt.hist(data_corr.as_array().flatten(),bins=150,fc=(0, 1, 0, 0.5))

def asd1():
    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    recon = FDK(data).run(verbose=0)
    P = recon.as_array()
    nx,ny = ig.shape

    trans = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]
    nc = len(basis_params)
    I_S = compute_scatter_basis(b,basis_params)
    I_S = 0.05*I_S
    I_Q = trans-I_S
    s = b[None,:,:] + np.log(I_Q)
    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    op1 = MatrixOperator(Mext)
    op2 = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()

    c = DataContainer(np.array([1,0.1,0.1,0.1]))
    w = K.direct(c)
    v = F(w)

    l = -np.inf*np.ones(1+nc)
    l[0] = 0
    u = np.inf*np.ones(1+nc)
    u[0] = 0
    G = IndicatorBox(lower=l,upper=u)
    pdhg = PDHG(f=F, g=G, operator=K, initial=c)
    pdhg.run(iterations=1)

    fista = FISTA(f=F, g=G, operator=K, initial=c)

    show2D(w[0]) # why is this so good?
    show2D(np.sqrt(w[0]**2+w[1]**2))
    show2D( np.sqrt(w[0]**2) + np.sqrt(w[1]**2) )
    show2D(np.sqrt(np.sqrt(w[0]**2)))
    show2D(w[0]>0.01)
    
def asd2():
    def fd_grad(func, x, h=1e-5):
        n = len(x)
        grad = np.zeros(n)
        f_x = func(x)
        for i in range(n):
            x_plus_h = np.array(x)
            x_plus_h[i] += h
            f_x_plus_h = func(x_plus_h)
            grad[i] = (f_x_plus_h - f_x) / h
        return grad
    
    # file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    recon = FDK(data).run(verbose=0)
    P = recon.as_array()
    nx,ny = ig.shape

    trans = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    basis_params = [[1,10,1],[1/2,10,2],[1/4,10,4]]
    # basis_params = [[1,30,1],[1/2,30,2],[1/5,100,5]]
    # basis_params = [[1,10,1],[1/10,10,1],[1/100,10,1]]
    # basis_params = [[1/2,100,2]]
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]
    # basis_params = [[1/40,0,40],[1/100,0,100],[1/200,0,200]]
    # basis_params = [[1/40,0,40],[1/100,0,100],[1/200,0,200]]
    # basis_params = [[1/40,0,40]]
    nc = len(basis_params)
    I_S = compute_scatter_basis(b,basis_params)
    I_S = 0.05*I_S
    I_Q = trans-I_S
    s = b[None,:,:] + np.log(I_Q)
    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        show2D(-S[i])
        # show2D(S[i])

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = - S.reshape(nc, nx*ny).T

    op1 = MatrixOperator(Mext)
    op2 = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = SmoothMixedL21Norm(epsilon=1e-2)

    # c = DataContainer(np.array([1,0.1,0.1,0.1]))
    # w = K.direct(c)
    # v = F(w)
    def obj(x):
        c = np.ones(1+nc)
        c[1:] = x
        data_c = DataContainer(c)
        return F(K.direct(data_c))
    
    x0 = np.array([0.1,0.1,0.1])
    # x0 = np.array([1,1,1])
    # x0 = np.array([0.1])

    l,u = -np.inf*np.ones(nc),np.zeros(nc)
    # l,u = np.zeros(nc),np.ones(nc)*np.inf
    bounds = sp.optimize.Bounds(lb=l, ub=u, keep_feasible=False)

    n_angles,n_panel = s.shape[1:]
    A = - s.reshape(nc, n_angles*n_panel).T
    ubA = trans.flatten()
    con = sp.optimize.LinearConstraint(A, ub=ubA)
    options = {'maxiter': 10, 'disp': True}
    def callback(x):
        print("Current solution:", x)
    
    # res = sp.optimize.minimize(fun=obj, jac='3-point', x0=x0, constraints=con, options=options, callback=callback)
    # res = sp.optimize.minimize(fun=obj, jac='3-point', x0=x0, bounds=bounds, constraints=con, options=options, callback=callback)
    # res = sp.optimize.minimize(fun=obj, jac='3-point', x0=x0, bounds=bounds, options=options, callback=callback)
    res = sp.optimize.minimize(fun=obj, jac='3-point', x0=x0, options=options, callback=callback)
    c = np.ones(1+nc)
    c[1:] = res.x
    Q = Mext.dot(c).reshape((nx,ny))
    show2D(Q)

    q = trans - A.dot(res.x).reshape((n_angles,n_panel))
    print(q.min())

    fd_grad(obj,res.x)
    # d = 1e-4
    # d_vec = np.zeros(1+nc)
    # d_vec[1] = d
    # (obj(x0+d_vec)-obj(x0-d_vec))/(2*d)


def ESC(basis_params, num_iter=50, delta=1/11, c_l=1):
    def compute_k(data,B,I,c,delta):
        n_angle = data.shape[0] # ag.config.angles ?
        I_over_sum_c_B = I / np.sum(c[:,None,None,None]*B, axis=0)
        min_I_over_sum_c_B = np.min(I_over_sum_c_B, axis=(1, 2))
        k = np.where(min_I_over_sum_c_B < 1, min_I_over_sum_c_B*(1-delta), 1)
        return k

    data_downsampled = downsample_and_subset_projections()
    recon_init = FDK(data_downsampled)
    tau_air, air_mask = segment_data(recon_init)
    B_downsampled = compute_basis(data_downsampled, basis_params)
    c = np.ones(B.shape[0]) / B.shape[0] * np.random.rand(B.shape[0])
    for n in range(num_iter):
        k = compute_k(data_downsampled,B_downsampled,I,c,delta)
        p_downsampled = -np.log(data_downsampled.as_array() - k*np.sum(c[:,None,None,None]*B_downsampled, axis=0))
        f = c_l * FDK(p_downsampled, geometry=ig)
        c,c_l = update_cs(f,tau_air,air_mask)

    B = upsample_basis_and_projections(data,B_downsampled)
    data_cor = data.as_array() - k*np.sum(c[:,None,None,None]*B, axis=0)
    return data_cor

if __name__ == '__main__':
    # testing()
    None