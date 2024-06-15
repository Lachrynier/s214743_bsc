from IPython import get_ipython
ipython = get_ipython()
# ipython.run_line_magic(r"%load_ext autoreload")
# ipython.run_line_magic(r"%autoreload 2")
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

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
                                       ZeroFunction, SmoothMixedL21Norm, MixedL11Norm

from cil.io import NikonDataReader
from cil.utilities.jupyter import islicer
from cil.utilities.display import show_geometry, show2D, show1D
from cil.recon import FDK, FBP
from cil.plugins.tigre import ProjectionOperator#, FBP
from cil.processors import TransmissionAbsorptionConverter, AbsorptionTransmissionConverter, Slicer, Normaliser, Padder
from cil.optimisation.algorithms import CGLS, SIRT
from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry, VectorData, VectorGeometry
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

from cil.optimisation.operators import CompositionOperator, FiniteDifferenceOperator, MatrixOperator, LinearOperator
from cil.framework import DataContainer, BlockDataContainer

# from generate_plots import setup_generic_cil_geometry
from bhc import load_centre

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

# def compute_image_basis(data,scatter_basis):
#     None

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


def simulate_scatter():
    # from generate_plots import setup_generic_cil_geometry
    # physical_size = 1
    # voxel_num = 1000
    # ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)
    # ig = ag.get_ImageGeometry()

    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    # plt.plot(b[100,:])
    # plt.plot(trans[0,:])
    # basis_params = [[1/40,0,40]]
    factor = [20,40,100,200][2]
    basis_params = [[1/factor,0,factor]]

    I_S = compute_scatter_basis(b,basis_params)
    # I_S = 0.05*I_S*3
    I_S = 0.05*I_S*3
    I = I_P + I_S[0]

    idx = 0
    plt.plot(I_P[idx,:],label='I_P')
    plt.plot(I_S[0,idx,:],label='I_S')
    plt.plot(I[idx,:],label='I=I_P+I_S')
    plt.legend()
    plt.show()
    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)
    show2D(recon, title='reconstruction of -ln(I_P+I_S)')
    hori_idx = 700
    direction = 'horizontal_x'
    show1D(recon, [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))
    plt.show()

    P = recon.as_array()
    b = data_scatter.as_array()
    nx,ny = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [40,100][1]
    # basis_params = [[1/factor,0,factor]]

    factors = [40,100,200]
    basis_params = [[1/factor,0,factor] for factor in factors]
    # basis_params[0][0]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if True:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[basis_idx,idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = -np.log(I_S+I_P) + np.log(I_P)
        ###

    print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    plt.title(f'basis_idx: {basis_idx}')
    plt.sca(ax[0])
    plt.plot(s[basis_idx,idx,:],label='s')
    plt.legend()

    plt.sca(ax[1])
    plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    plt.legend()
    plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = - S.reshape(nc, nx*ny).T

    op1 = MatrixOperator(Mext)
    op2 = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    def obj(x):
        c = np.ones(1+nc)
        c[1:] = x
        data_c = DataContainer(c)
        return F(K.direct(data_c))
    
    def obj2(x):
        Q = P + np.sum(x[:,None,None]*S, axis=0)
        diff1 = np.diff(Q, axis=0)
        diff2 = np.diff(Q, axis=1)
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
        return tv
    # x0 = np.array([0.1,0.1,0.1])
    # x0 = np.array([0.1])
    x0 = 0.1*np.ones(nc)

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
    # c = np.ones(1+nc)
    # c[1:] = res.x # "optimal"
    # c[1:] = np.zeros(nc)
    # c[1:] = np.array([3])
    # Q = Mext.dot(c).reshape((nx,ny))
    Q = P + np.sum(res.x[:,None,None]*S, axis=0)
    show2D(Q)
    show1D(ImageData(Q,geometry=ig), [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))

    ###
    q = trans - A.dot(res.x).reshape((n_angles,n_panel))
    print(q.min())


    # op = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    # K = CompositionOperator(op)
    # F = MixedL21Norm()
    def TV(im):
        op = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
        return MixedL21Norm()(op.direct(im))
    # TotalVariation().func(BlockDataContainer(DataContainer(Q)))
    print(TV(ImageData(Q.astype('float32'), geometry=ig)))
    print(TV(ImageData(P.astype('float32'), geometry=ig)))

    def total_variation(image):
        # Calculate differences along both axes
        diff1 = np.diff(image, axis=0)
        diff2 = np.diff(image, axis=1)
        
        # Compute the total variation
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
    
        return tv
    
    print(total_variation(Q))
    print(total_variation(P))
    print(total_variation(mu*recon_P.as_array()))

def scatter_const():
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    I = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    recon = FDK(data, image_geometry=ig).run(verbose=0)

    factors = [100]
    basis_params = [[1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    I_S = 0.05*I_S*3*0.5
    
    idx = 0
    basis_idx = 0
    plt.plot(I_S[basis_idx,idx,:],label='I_S')
    plt.plot(I[idx,:],label='I=I_P+I_S')
    plt.legend()
    plt.ylim(0,1)
    plt.show()

    # I_cor = I-np.sum(I_S,axis=0)
    I_cor = I - 0.07
    data_cor = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I_cor, dtype=np.float32), geometry=ag))
    b_cor = data_cor.as_array()

    recon_cor = FDK(data_cor, image_geometry=ig).run(verbose=0)
    show2D([recon,recon_cor])

def simulate_scatter1_2():
    # from generate_plots import setup_generic_cil_geometry
    # physical_size = 1
    # voxel_num = 1000
    # ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)
    # ig = ag.get_ImageGeometry()

    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()

    factors = [40,100,200]
    basis_params = [[1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    I_S = 0.05*I_S*3
    # I = I_P + I_S[0]
    I = I_P + np.sum(I_S, axis=0) / len(basis_params)

    idx = 0
    plt.plot(I_P[idx,:],label='I_P')
    for i in range(len(basis_params)):
        plt.plot(I_S[0,idx,:],label='I_S')
    plt.plot(I[idx,:],label='I=I_P+I_S')
    plt.legend()
    plt.show()
    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)
    show2D(recon, title='reconstruction of -ln(I_P+I_S)')
    hori_idx = 700
    direction = 'horizontal_x'
    show1D(recon, [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))
    plt.show()

    P = recon.as_array()
    b = data_scatter.as_array()
    nx,ny = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [40,100][1]
    # basis_params = [[1/factor,0,factor]]

    # factors = [40,100,200]
    factors = [100]
    basis_params = [[1/factor,0,factor] for factor in factors]
    # basis_params[0][0]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if True:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[basis_idx,idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = -np.log(I_S+I_P) + np.log(I_P)
        ###

    print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    plt.title(f'basis_idx: {basis_idx}')
    plt.sca(ax[0])
    plt.plot(s[basis_idx,idx,:],label='s')
    plt.legend()

    plt.sca(ax[1])
    plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    plt.legend()
    plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    op1 = MatrixOperator(Mext)
    op2 = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    def obj(x):
        c = np.ones(1+nc)
        c[1:] = x
        data_c = DataContainer(c)
        return F(K.direct(data_c))
    
    def obj2(x):
        Q = P + np.sum(x[:,None,None]*S, axis=0)
        diff1 = np.diff(Q, axis=0)
        diff2 = np.diff(Q, axis=1)
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
        return tv
    # x0 = np.array([0.1,0.1,0.1])
    # x0 = np.array([0.1])
    x0 = 0.1*np.ones(nc)

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
    # c = np.ones(1+nc)
    # c[1:] = res.x # "optimal"
    # c[1:] = np.zeros(nc)
    # c[1:] = np.array([3])
    # Q = Mext.dot(c).reshape((nx,ny))
    Q = P + np.sum(res.x[:,None,None]*S, axis=0)
    show2D(Q)
    show1D(ImageData(Q,geometry=ig), [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))

    ###
    q = trans - A.dot(res.x).reshape((n_angles,n_panel))
    print(q.min())


    # op = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
    # K = CompositionOperator(op)
    # F = MixedL21Norm()
    def TV(im):
        op = GradientOperator(ImageGeometry(voxel_num_x=nx,voxel_num_y=ny))
        return MixedL21Norm()(op.direct(im))
    # TotalVariation().func(BlockDataContainer(DataContainer(Q)))
    print(TV(ImageData(Q.astype('float32'), geometry=ig)))
    print(TV(ImageData(P.astype('float32'), geometry=ig)))

    def total_variation(image):
        # Calculate differences along both axes
        diff1 = np.diff(image, axis=0)
        diff2 = np.diff(image, axis=1)
        
        # Compute the total variation
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
    
        return tv
    
    print(total_variation(Q))
    print(total_variation(P))
    print(total_variation(mu*recon_P.as_array()))

def simulate_scatter2():
    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]

    factors = [40,100]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    # I_S = 0.05*I_S*3
    # I_S = 0.05*I_S*3
    I = I_P + np.sum(I_S,axis=0)/len(basis_params)

    basis_idx = 0
    idx = 0
    plt.plot(I_P[idx,:],label=r'I_P')
    plt.plot(I_S[basis_idx,idx,:],label=r'I_S')
    # plt.plot(I[idx,:],label=r'I=I_P+I_S')
    plt.legend()
    plt.show()
    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)
    show2D(recon, title='reconstruction of -ln(I_P+I_S)')
    hori_idx = 700
    direction = 'horizontal_x'
    show1D(recon, [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))
    plt.show()

    P = recon.as_array()
    b = data_scatter.as_array()
    ny,nx = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    factor = [5,20,40,100,200][2]
    basis_params = [[0.15*1/factor,0,factor]]
    # basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    # factors = [40,100,200]
    # basis_params = [[1/factor,0,factor] for factor in factors]
    # basis_params[0][0]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if True:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[basis_idx,idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S = 0.05*I_S
        I_Q = trans-I_S
        s = -np.log(I_S+I_P) + np.log(I_P)
        ###

    print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    plt.title(f'basis_idx: {basis_idx}')
    plt.sca(ax[0])
    plt.plot(s[basis_idx,idx,:],label='s')
    plt.legend()

    plt.sca(ax[1])
    plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    plt.legend()
    plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    vg = VectorGeometry(length=nc+1)
    op1 = ESCMatrixOperator(Mext, domain_geometry=vg, range_geometry=ig)
    op2 = GradientOperator(domain_geometry=ig)
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = MixedL11Norm()

    # F = SmoothMixedL21Norm(epsilon=1e-2)

    cext = VectorData(array=np.hstack((1,np.full(nc, 0.1, dtype=np.float32))), geometry=vg)
    # id = op1.direct(cext)
    # vd = op1.adjoint(id)
    # print(np.linalg.norm(np.dot(Mext, cext.as_array()).reshape((ny,nx)) - id.as_array()))
    # print(np.linalg.norm(np.dot(Mext.T,id.as_array().flatten()) - vd.as_array()))

    l = -np.inf*np.ones(1+nc)
    l[0] = 1
    u = np.inf*np.ones(1+nc)
    u[0] = 1
    G = IndicatorBox(lower=l,upper=u)
    pdhg = PDHG(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=10)
    pdhg.run(iterations=150)

    # fista = FISTA(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=5)
    # fista.run(iterations=100)

    print(pdhg.solution.as_array())
    Q = op1.direct(pdhg.solution)
    show2D(Q, title=f"c_ext={pdhg.solution.as_array()}")
    show1D(Q, [(direction,hori_idx)], title=f'{direction}={hori_idx}', size=(8,3))

    F(op2.direct(op1.direct(pdhg.solution)))
    F(op2.direct(op1.direct(VectorData(np.array([1,0])))))
    F(op2.direct(op1.direct(VectorData(np.array([1,-4.47])))))

    c = np.array([1,-4.47])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    c = np.array([1,-30])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")


    # vs = np.arange(-15,20,0.5)
    vs = np.arange(-100,100,0.5)
    fs = np.zeros(vs.shape)
    for i,v in enumerate(vs):
        fs[i] = F(op2.direct(op1.direct(VectorData(np.array([1,v])))))
    plt.plot(vs,fs)
    plt.title("TV vs. c[1]")

def simulate_scatter2_2():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')
    
    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
    # I_S = 0.05*I_S*3
    # I_S = 0.05*I_S*3
    I = I_P + I_S_sum

    basis_idx = 0
    idx = 0

    ########################
    fig,ax = plt.subplots(figsize=(8,5))
    plt.sca(ax)
    plt.plot(I_P[idx,:],label=r'$I_P$')
    # plt.plot(I_S[basis_idx,idx,:],label=r'$I_S$')
    plt.plot(I_S_sum[idx,:],label=r'$I_S$')
    # plt.plot(I[idx,:],label=r'I=I_P+I_S')
    plt.grid(True)
    plt.legend()
    plt.title('Projection with simulated scatter')
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_proj.png'))
    plt.show()
    #####################

    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)

    ####################
    fig,ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,10))
    plt.sca(ax[0])
    im0 = plt.imshow(recon.as_array(), origin='lower', cmap='gray')#,vmin=-10, vmax=10)
    # plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title(r'Reconstruction of $-\ln(I_P+I_S)$')
    plt.colorbar()
    # plt.colorbar(im0,fraction=0.046, pad=0.04)

    plt.sca(ax[1])
    hori_x,hori_y = 700,650

    plt.plot(recon.as_array()[:,hori_x]) # horizontal_x fixed
    plt.title(f'Intensity profile for horizontal_x={hori_x} on the reconstruction')
    plt.xlabel('horizontal_y')

    # plt.plot(recon.as_array()[hori_y,:]) # horizontal_y fixed
    # plt.title(f'Intensity profile for horizontal_y={hori_y} on the reconstruction')
    # plt.xlabel('horizontal_x')
    
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_recon.pdf'))
    plt.show()
    ####################
    
    P = recon.as_array()
    b = data_scatter.as_array()
    ny,nx = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]
    # basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if False:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = -np.log(I_S+I_P) + np.log(I_P)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###

    # print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    # fig,ax = plt.subplots(1,2,figsize=(10,5))
    # plt.title(f'basis_idx: {basis_idx}')
    # plt.sca(ax[0])
    # plt.plot(s[basis_idx,idx,:],label='s')
    # plt.legend()

    # plt.sca(ax[1])
    # plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    # plt.legend()
    # plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    vg = VectorGeometry(length=nc+1)
    op1 = ESCMatrixOperator(Mext, domain_geometry=vg, range_geometry=ig)
    op2 = GradientOperator(domain_geometry=ig)
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = MixedL11Norm()
    # F = SmoothMixedL21Norm(epsilon=1e-2)

    cext = VectorData(array=np.hstack((1,np.full(nc, 0.1, dtype=np.float32))), geometry=vg)

    l = -np.inf*np.ones(1+nc)
    l[0] = 1
    u = np.inf*np.ones(1+nc)
    u[0] = 1
    G = IndicatorBox(lower=l,upper=u)
    upd_int = 3
    pdhg = PDHG(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=3)
    # pdhg.run(iterations=150)
    pdhg.run(iterations=34)

    print(pdhg.solution.as_array())
    Q = op1.direct(pdhg.solution)
    show2D(Q, title=f"c_ext={pdhg.solution.as_array()}")
    show1D(Q, [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))
    # show1D(Q, [('horizontal_y',hori_y)], title=f'horizontal_y={hori_y}', size=(8,3))

    F(op2.direct(op1.direct(pdhg.solution)))
    F(op2.direct(op1.direct(VectorData(np.array([1,0])))))
    F(op2.direct(op1.direct(VectorData(np.array([1,-4.47])))))

    c = np.array([1,-4.47])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    c = np.array([1,-1])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")
    show1D(op1.direct(VectorData(c)), [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))

    def total_variation1(image):
        diff1 = np.diff(image, axis=0)
        diff2 = np.diff(image, axis=1)
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
        return tv
    
    def total_variation2(image):
        dx = np.diff(image, axis=0)
        dy = np.diff(image, axis=1)
        dx = np.pad(dx, ((0, 1), (0, 0)), 'constant')
        dy = np.pad(dy, ((0, 0), (0, 1)), 'constant')
        tv = np.sum(np.sqrt(dx**2 + dy**2))
        return tv

    ##########
    plt.rcParams.update({'font.size': 16})
    vs = np.arange(-2.0,4.0,0.1)
    # vs = np.arange(-100,100,0.5)
    fs = np.zeros(vs.shape)
    for i,v in enumerate(vs):
        fs[i] = F(op2.direct(op1.direct(VectorData(np.array([1,v])))))
    
    fig,ax = plt.subplots(figsize=(8,5)); plt.sca(ax)
    plt.plot(vs,fs, color='black')
    plt.xlabel(r'$c_1$')
    plt.ylabel(r'Objective value')
    plt.plot(pdhg.solution.as_array()[1], pdhg.objective[-1], label='Minimizer',
        marker=".", markersize=15, color='red', linestyle='None')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_obj_curve.pdf'))
    plt.show()
    plt.rcParams.update({'font.size': 12})

    ##########
    plt.rcParams.update({'font.size': 16})
    import matplotlib as mpl
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    fig,ax = plt.subplots(1,2,figsize=(13,6))

    c = np.array([1,-1])
    Q_manual = op1.direct(VectorData(c))
    plt.sca(ax[0])
    plt.imshow(Q_manual.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    c_str = str(c[1:]).replace('[', '(').replace(']', ')')
    plt.title(rf'$\boldsymbol{{c}} = {c_str}$')

    Q = op1.direct(pdhg.solution)
    plt.sca(ax[1])
    plt.imshow(Q.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    rounded_c = [round(_, 3) for _ in pdhg.solution.as_array()[1:]]
    c_str = str(rounded_c).replace('[', '(').replace(']', ')')
    plt.title(rf'TV minimizer: $\boldsymbol{{c}} = {c_str}$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_sol_comparisons.pdf'))
    plt.show()
    #############

# class ESCMatrixOperator(LinearOperator):
#     def __init__(self, A, domain_geometry, range_geometry, order='C'):
#         """
#         Custom matrix operator for ESC.

#         Parameters:
#         A (ndarray): The matrix to apply in the direct and adjoint operations.
#         domain_geometry: VectorGeometry of the linear coefficients
#         range_geometry: ImageGeometry of the linear combination of basis images
#         order (str): The order of flattening and reshaping operations.
#                      'C' for row-major (C-style),
#                      'F' for column-major (Fortran-style).
#         """
#         super(ESCMatrixOperator, self).__init__(domain_geometry=domain_geometry, 
#                                                range_geometry=range_geometry)
#         self.A = A
#         self.order = order

#     def direct(self, x, out=None):
#         flattened_x = x.as_array().flatten(order=self.order)
#         result_1d = np.dot(self.A, flattened_x)
#         result_2d = result_1d.reshape((self.range_geometry().voxel_num_y, 
#                                         self.range_geometry().voxel_num_x), 
#                                        order=self.order)

#         if out is None:
#             result = self.range_geometry().allocate()
#             result.fill(result_2d)
#             return result
#         else:
#             out.fill(result_2d)

#     def adjoint(self, y, out=None):
#         flattened_y = y.as_array().flatten(order=self.order)
#         result_1d = np.dot(self.A.T, flattened_y)

#         if out is None:
#             result = self.domain_geometry().allocate()
#             result.fill(result_1d)
#             return result
#         else:
#             out.fill(result_1d)

class ESCMatrixOperator(LinearOperator):
    def __init__(self, A, domain_geometry, range_geometry, order='C'):
        """
        Custom matrix operator for ESC.

        Parameters:
        A (ndarray): The matrix to apply in the direct and adjoint operations.
        domain_geometry: VectorGeometry of the linear coefficients
        range_geometry: ImageGeometry of the linear combination of basis images
        order (str): The order of flattening and reshaping operations.
                     'C' for row-major (C-style),
                     'F' for column-major (Fortran-style).
        """
        super(ESCMatrixOperator, self).__init__(domain_geometry=domain_geometry, 
                                                range_geometry=range_geometry)
        self.A = A
        self.order = order

    def direct(self, x, out=None):
        result_1d = np.dot(self.A, x.as_array())
        result_2d = result_1d.reshape((self.range_geometry().voxel_num_y, 
                                       self.range_geometry().voxel_num_x), 
                                      order=self.order)

        if out is None:
            tmp = self.range_geometry().allocate()
            tmp.fill(result_2d)
            return tmp
        else:
            out.fill(result_2d)

    def adjoint(self, y, out=None):
        flattened_y = y.as_array().flatten(order=self.order)
        result = np.dot(self.A.T, flattened_y)

        if out is None:
            tmp = self.domain_geometry().allocate()
            tmp.fill(result)
            return tmp
        else:
            out.fill(result)

# class MyCustomOperator(LinearOperator):
#     def __init__(self, A, domain_geometry, range_geometry, order):
#         """
#         Custom linear operator that applies a matrix A to an input array.

#         Parameters:
#         A (ndarray): The matrix to apply in the direct and adjoint operations.
#         domain_geometry (ImageGeometry): The geometry of the input space.
#         range_geometry (ImageGeometry): The geometry of the output space.
#         order (str): The order of flattening and reshaping operations.
#                      'C' for row-major (C-style),
#                      'F' for column-major (Fortran-style).
#         """
#         super(MyCustomOperator, self).__init__(domain_geometry=domain_geometry, 
#                                                range_geometry=range_geometry)
#         self.A = A
#         self.order = order

#     def direct(self, x, out=None):
#         flattened_x = x.as_array().flatten(order=self.order)
#         result_1d = np.dot(self.A, flattened_x)
#         result_2d = result_1d.reshape((self.range_geometry().voxel_num_y, 
#                                         self.range_geometry().voxel_num_x), 
#                                        order=self.order)

#         if out is None:
#             result = self.range_geometry().allocate()
#             result.fill(result_2d)
#             return result
#         else:
#             out.fill(result_2d)

#     def adjoint(self, y, out=None):
#         flattened_y = y.as_array().flatten(order=self.order)
#         result_1d = np.dot(self.A.T, flattened_y)
#         result_2d = result_1d.reshape((self.domain_geometry().voxel_num_y, 
#                                         self.domain_geometry().voxel_num_x), 
#                                        order=self.order)

#         if out is None:
#             result = self.domain_geometry().allocate()
#             result.fill(result_2d)
#             return result
#         else:
#             out.fill(result_2d)

# def ESC(basis_params, num_iter=50, delta=1/11, c_l=1):
#     def compute_k(data,B,I,c,delta):
#         n_angle = data.shape[0] # ag.config.angles ?
#         I_over_sum_c_B = I / np.sum(c[:,None,None,None]*B, axis=0)
#         min_I_over_sum_c_B = np.min(I_over_sum_c_B, axis=(1, 2))
#         k = np.where(min_I_over_sum_c_B < 1, min_I_over_sum_c_B*(1-delta), 1)
#         return k

#     data_downsampled = downsample_and_subset_projections()
#     recon_init = FDK(data_downsampled)
#     tau_air, air_mask = segment_data(recon_init)
#     B_downsampled = compute_basis(data_downsampled, basis_params)
#     c = np.ones(B.shape[0]) / B.shape[0] * np.random.rand(B.shape[0])
#     for n in range(num_iter):
#         k = compute_k(data_downsampled,B_downsampled,I,c,delta)
#         p_downsampled = -np.log(data_downsampled.as_array() - k*np.sum(c[:,None,None,None]*B_downsampled, axis=0))
#         f = c_l * FDK(p_downsampled, geometry=ig)
#         c,c_l = update_cs(f,tau_air,air_mask)

#     B = upsample_basis_and_projections(data,B_downsampled)
#     data_cor = data.as_array() - k*np.sum(c[:,None,None,None]*B, axis=0)
#     return data_cor

if __name__ == '__main__':
    # testing()
    None