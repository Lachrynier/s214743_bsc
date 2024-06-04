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
                                       ZeroFunction

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
from matplotlib.colors import Normalize,LogNorm

hpc_cluster = 0

def load_centre(filename):
    base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
    file_path = os.path.join(base_dir,f'centres/{filename}')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data

def avg_out(path_lengths, data, num_bins):
    x = np.array(path_lengths.as_array().flatten())
    y = np.array(data.as_array().flatten())
    bin_edges = np.linspace(x.min(), x.max(), num_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(x, bin_edges)
    bin_means = np.array([y[bin_indices == i].mean() for i in range(1, len(bin_edges))])
    plt.scatter(x, y, alpha=0.1, color='black', label='Observed',s=2)
    # plt.plot(bin_centers, bin_means, color='red', label='Averaged out')
    plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    plt.xlabel('Path lengths (mm)')
    plt.ylabel('Absorption')
    plt.legend(loc='lower right')
    if hpc_cluster: plt.savefig(os.path.join(bfig_dir,'avg_out.png'))
    plt.show()

    return np.hstack(([0],bin_centers)), np.hstack(([0],bin_means))

def bhc(path_lengths, data, f_mono, f_poly, num_bins):
    bin_centers,bin_means = avg_out(path_lengths, data, num_bins=num_bins)
    popt_mono, pcov = curve_fit(f_mono, bin_centers, bin_means)
    popt_poly, pcov = curve_fit(f_poly, bin_means, bin_centers)
    xx = np.linspace(0, path_lengths.max())
    yy = np.linspace(0, data.max())
    y_mono = f_mono(xx, *popt_mono)
    x_poly = f_poly(yy, *popt_poly)
    plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    plt.plot(xx, y_mono, label='Mono fit')
    plt.plot(x_poly, yy, label='Poly fit')
    plt.legend(loc='lower right')
    plt.show()

    # gglin = g(yylin,popt_g[0],popt_g[1])
    yy_bhc = f_mono(x_poly, *popt_mono)
    plt.plot(yy, yy_bhc)
    plt.title('BHC absorptions vs Original absorptions')
    plt.show()

    data_x_poly = f_poly(data.as_array(), *popt_poly)
    data_y_mono = f_mono(data_x_poly, *popt_mono)
    data_bhc = AcquisitionData(array=np.array(data_y_mono,dtype='float32'),geometry=data.geometry)
    recon_bhc = FDK(data_bhc).run()
    return data_bhc, recon_bhc

def clip_otsu_segment(recon, ig, clip=True, title=''):
    if clip:
        tau = threshold_otsu(np.clip(recon.as_array(),a_min=0,a_max=None))
    else:
        tau = threshold_otsu(recon.as_array())
    
    segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    show2D(segmented, title=title)
    return segmented

def clip_otsu_segment2(recon, ig, clip=True, title=''):
    if clip:
        tau = threshold_otsu(np.clip(recon,a_min=0,a_max=None))
    else:
        tau = threshold_otsu(recon)
    
    segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    show2D(segmented, title=title)
    return segmented

def bhc_v2(path_lengths, data, f_mono, f_poly, num_bins, mask=None, weight_fun=np.sqrt, color_norm='log'):
    # mask: logical array of same shape as data
    if mask is None:
        x = np.array(path_lengths.as_array()).flatten()
        y = np.array(data.as_array()).flatten()
    else:
        x = np.array(path_lengths.as_array()[mask])
        y = np.array(data.as_array()[mask])
    
    norms = {'lin': Normalize(), 'log': LogNorm()}
    counts, x_edges, y_edges, _ = plt.hist2d(x,y,bins=100, range=[[0,x.max()],[0,y.max()]], norm=norms[color_norm])
    plt.xlabel('Path length')
    plt.ylabel('Absorption')
    plt.colorbar()

    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    y_centers, x_centers = np.meshgrid(y_centers, x_centers)
    x_centers_flat = x_centers.flatten()
    y_centers_flat = y_centers.flatten()
    counts_flat = counts.flatten()

    nonzero_counts = counts_flat > 0
    x_fit = x_centers_flat[nonzero_counts]
    y_fit = y_centers_flat[nonzero_counts]
    counts_fit = counts_flat[nonzero_counts]
    weights = weight_fun(counts_fit)
    popt_poly, p_cov = curve_fit(f_poly, y_fit, x_fit, sigma=1./weights, absolute_sigma=True)
    print(f'popt_poly: {popt_poly}')
    popt_mono, p_cov = curve_fit(f_mono, x_fit, y_fit, sigma=1./weights, absolute_sigma=True)
    # popt_mono, p_cov = curve_fit(f_mono, x_fit, y_fit, sigma=1./(1+np.log(counts_fit)), absolute_sigma=True)
    # popt_mono, p_cov = curve_fit(f_mono, x_fit, y_fit)

    # xx = np.linspace(0, path_lengths.max())
    xx = np.linspace(0, data.max()/popt_mono)
    yy = np.linspace(0, data.max())
    y_mono = f_mono(xx, *popt_mono)
    x_poly = f_poly(yy, *popt_poly)
    lw = 3
    # plt.plot(xx, y_mono, label='Mono fit', color='magenta', linewidth=lw, linestyle='dashed')
    plt.plot(x_poly, yy, label='Poly fit', color='red', linewidth=lw)
    plt.title('Fit used for linearization')
    # plt.legend(loc='lower right')
    plt.show()

    yy_bhc = f_mono(x_poly, *popt_mono)
    plt.plot(yy, yy_bhc)
    plt.title('BHC absorptions vs Original absorptions')
    plt.show()

    # return
    ### Data corrections
    data_x_poly = f_poly(data.as_array(), *popt_poly)
    data_y_mono = f_mono(data_x_poly, *popt_mono)
    data_bhc = AcquisitionData(array=np.array(data_y_mono,dtype='float32'),geometry=data.geometry)
    recon_bhc = FDK(data_bhc).run()
    return data_bhc, recon_bhc


class BHC:
    def __init__(self, path_lengths, data, f_mono, f_poly, num_bins, mask=None, weight_fun=np.sqrt, color_norm='log', n_poly=None):
        self.path_lengths = path_lengths
        self.data = data
        self.f_mono = f_mono
        self.f_poly = f_poly
        self.num_bins = num_bins
        self.mask = mask
        self.weight_fun = weight_fun
        self.color_norm = color_norm
        self.n_poly = n_poly
        self.popt_poly = None
        self.popt_mono = None
        self.data_bhc = None
        self.recon_bhc = None
        self.x_centers_flat = None
        self.y_centers_flat = None
        self.counts_flat = None

        if f_mono is None:
            def f_mono(x, a):
                return a*x
            self.f_mono = f_mono

    def prepare_data(self):
        if self.mask is None:
            x = np.array(self.path_lengths.as_array()).flatten()
            y = np.array(self.data.as_array()).flatten()
        else:
            x = np.array(self.path_lengths.as_array()[self.mask])
            y = np.array(self.data.as_array()[self.mask])
        return x, y

    def plot_histogram(self, x, y):
        norms = {'lin': Normalize(), 'log': LogNorm()}
        counts, x_edges, y_edges, _ = plt.hist2d(x, y, bins=self.num_bins, range=[[0, x.max()], [0, y.max()]], norm=norms[self.color_norm])
        plt.xlabel('Path length')
        plt.ylabel('Absorption')
        plt.colorbar()
        return counts, x_edges, y_edges

    def calculate_fit_data(self, counts, x_edges, y_edges):
        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
        y_centers, x_centers = np.meshgrid(y_centers, x_centers)
        self.x_centers_flat = x_centers.flatten()
        self.y_centers_flat = y_centers.flatten()
        self.counts_flat = counts.flatten()

    def perform_fit(self):
        nonzero_counts = self.counts_flat > 0
        x_fit = self.x_centers_flat[nonzero_counts]
        y_fit = self.y_centers_flat[nonzero_counts]
        counts_fit = self.counts_flat[nonzero_counts]
        weights = self.weight_fun(counts_fit)
        
        self.popt_mono, _ = curve_fit(self.f_mono, x_fit, y_fit, sigma=1./weights, absolute_sigma=True)
        if self.n_poly is not None:
            p0 = np.ones(self.n_poly)
        else:
            p0 = None
        self.popt_poly, _ = curve_fit(self.f_poly, y_fit, x_fit, p0=p0, sigma=1./weights, absolute_sigma=True)

    def plot_fits(self, show_hist=True, make_trans_plot=True, **kwargs):
        xx = np.linspace(0, self.data.max() / self.popt_mono)
        yy = np.linspace(0, self.data.max())
        y_mono = self.f_mono(xx, *self.popt_mono)
        x_poly = self.f_poly(yy, *self.popt_poly)
        lw = 3
        if not kwargs:
            plt.plot(x_poly, yy, label='Poly fit', color='red', linewidth=lw)
            plt.title('Fit used for linearization')
        else:
            plt.plot(x_poly, yy, **kwargs)
        
        if show_hist:
            plt.show()
        if make_trans_plot:
            plt.plot(yy, self.f_mono(x_poly, *self.popt_mono))
            plt.title('BHC absorptions vs Original absorptions')
            plt.grid(True)
            plt.show()

    def perform_correction(self):
        data_x_poly = self.f_poly(self.data.as_array(), *self.popt_poly)
        data_y_mono = self.f_mono(data_x_poly, *self.popt_mono)
        self.data_bhc = AcquisitionData(array=np.array(data_y_mono, dtype='float32'), geometry=self.data.geometry)
        self.recon_bhc = FDK(self.data_bhc).run()

    def run(self):
        x, y = self.prepare_data()
        counts, x_edges, y_edges = self.plot_histogram(x, y)
        self.calculate_fit_data(counts, x_edges, y_edges)
        self.perform_fit()
        self.plot_fits()
        self.perform_correction()
        return self.data_bhc, self.recon_bhc
    
    ### Auxiliary function for extended work on the hist plot
    def get_hist_fit_plot(self):
        x, y = self.prepare_data()
        counts, x_edges, y_edges = self.plot_histogram(x, y)
        self.calculate_fit_data(counts, x_edges, y_edges)
        self.perform_fit()
        # self.plot_fits(show_hist=False, make_trans_plot=False)

def testing():
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK', clip=0)
    path_lengths = A.direct(segmented)

    mask = (path_lengths.as_array() > 0) & (data.as_array() > 0.25)
    def f_mono(x, a):
        return a*x
    def f_poly1(x, a,b,c):
        return a*x**3 + b*x**2 + c*x
    def f_poly2(x, a,b):
        return a*x**5 + b*x
    def f_poly3(x, a,b):
        return a*x**3
    data_bhc, recon_bhc = bhc_v2(path_lengths, data, f_mono, f_poly3, num_bins=100, mask=mask)
    cmap = ['grey', 'viridis', 'nipy_spectral', 'turbo', 'gnuplot2'][0]
    # show2D(recon_bhc, cmap=cmap)
    show2D(recon_bhc, fix_range=(-0.5,1))

def testing2():
    # data = load_centre('X20_cor.pkl')
    data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK', clip=1)
    path_lengths = A.direct(segmented)

    mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
    def f_mono(x, a):
        return a*x
    def f_poly1(x, a,b,c):
        return a*x**3 + b*x**2 + c*x
    def f_poly2(x, a,b):
        return a*x**2 + b*x
    def f_poly3(x, a,b):
        return a*x**3
    data_bhc, recon_bhc = bhc_v2(path_lengths, data, f_mono, f_poly3, num_bins=100, mask=mask)
    cmap = ['grey', 'viridis', 'nipy_spectral', 'turbo', 'gnuplot2'][0]
    # show2D(recon_bhc, cmap=cmap)
    show2D(recon_bhc, fix_range=(-0.5,1))

    # segmented = np.abs(recon_bhc.as_array()) > 0.3
    # 0.3 for X16, 0.13 for X20
    seg_arr = recon_bhc.as_array() > 0.2
    # seg_arr = np.abs(recon_bhc.as_array()) > 0.2
    segmented = ImageData(array=np.array(
        seg_arr,
        dtype='float32'), geometry=ig)
    show2D(segmented)
    path_lengths = A.direct(segmented)
    
    data_bhc, recon_bhc = bhc_v2(path_lengths, data, f_mono, f_poly3, num_bins=100, mask=mask)
    cmap = ['grey', 'viridis', 'nipy_spectral', 'turbo', 'gnuplot2'][0]
    # show2D(recon_bhc, cmap=cmap)
    show2D(recon_bhc, fix_range=(-0.5,1))

def bhc_v2_iter():
    def bhc_iteration(recon, it):
        clip = 0
        segmented = clip_otsu_segment(recon, ig, title=f'Otsu segmentation on iteration={it}', clip=clip)
        path_lengths = A.direct(segmented)
        mask = (path_lengths.as_array() > 0) & (data.as_array() > 0.25)
        data_bhc, recon_bhc = bhc_v2(path_lengths, data, f_mono, f_poly2, num_bins=100, mask=mask)
        F = LeastSquares(A, data_bhc)
        G = IndicatorBox(lower=0.0)
        x0 = ig.allocate(0.1)
        myFISTANN = FISTA(f=F, g=G, initial=x0, 
                        max_iteration=1000,
                        update_objective_interval=10)
        return myFISTANN
    
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    def f_mono(x, a):
        return a*x
    def f_poly1(x, a):
        return a*x**2
    def f_poly2(x, a,b):
        return a*x**3 + b*x
    def f_poly3(x, a):
        return a*x**3

    ###
    # F = LeastSquares(A, data)
    # G = IndicatorBox(lower=0.0)
    # x0 = ig.allocate(0.1)
    # fistaNN = FISTA(f=F, g=G, initial=x0, 
    #                 max_iteration=1000,
    #                 update_objective_interval=10)
    # fistaNN.run(100, verbose=1)
    # recon = fistaNN.solution
    # show2D(recon,f'fistaNN BHC reconstruction iteration=0')
    ###

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    for i in range(1,3):
    # for i in range(5,9):
        fistaNN = bhc_iteration(recon,it=i)
        fistaNN.run(100, verbose=1)
        recon = fistaNN.solution
        show2D(recon,f'fistaNN BHC reconstruction iteration={i}')

def bhc_v2_iter_fdk():
    def bhc_iteration(recon, it):
        clip = 0
        segmented = clip_otsu_segment(recon, ig, title=f'Otsu segmentation on iteration={it}', clip=clip)
        path_lengths = A.direct(segmented)
        mask = (path_lengths.as_array() > 0) & (data.as_array() > 0.25)
        data_bhc, recon_bhc = bhc_v2(path_lengths, data, f_mono, f_poly2, num_bins=100, mask=mask)
        # return FDK(data_bhc).
    
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    def f_mono(x, a):
        return a*x
    def f_poly1(x, a):
        return a*x**2
    def f_poly2(x, a,b):
        return a*x**3 + b*x
    def f_poly3(x, a):
        return a*x**3

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    for i in range(1,3):
    # for i in range(5,9):
        fistaNN = bhc_iteration(recon,it=i)
        fistaNN.run(100, verbose=1)
        recon = fistaNN.solution
        show2D(recon,f'fistaNN BHC reconstruction iteration={i}')

def bhc_v2_dev():
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK', clip=1)
    path_lengths = A.direct(segmented)

    cond = (path_lengths.as_array() > 0) & (data.as_array() > 0.25)
    x = np.array(path_lengths.as_array()[cond])
    y = np.array(data.as_array()[cond])
    norms = {'lin': Normalize(), 'log': LogNorm()}
    counts, x_edges, y_edges, _ = plt.hist2d(x,y,bins=100, range=[[0,x.max()],[0,y.max()]], norm=norms['log'])
    plt.xlabel('Path length')
    plt.ylabel('Absorption')
    plt.colorbar()
    if hpc_cluster: plt.savefig(os.path.join(bfig_dir,'hist2d'))

    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])

    # Flatten the counts and create meshgrid of x and y centers
    y_centers, x_centers = np.meshgrid(y_centers, x_centers)
    x_centers_flat = x_centers.flatten()
    y_centers_flat = y_centers.flatten()
    counts_flat = counts.flatten()

    # Indices where counts are nonzero
    nonzero_indices = counts_flat > 0
    x_fit = x_centers_flat[nonzero_indices]
    y_fit = y_centers_flat[nonzero_indices]
    counts_fit = counts_flat[nonzero_indices]

    # The weights could be directly the counts, or some function of counts such as square root or logarithmic
    weights = np.sqrt(counts_fit)
    # weights = counts_fit
    # weights = np.log(1+np.sqrt(counts_fit))  # Using square root to stabilize large variations
    def f_mono(x, a):
        return a*x
    def f_poly(x, a,b,c,d):
        return a*x**4 + b*x**3 + c*x**2 + d*x
    popt_poly, p_cov = curve_fit(f_poly, y_fit, x_fit, sigma=1./weights, absolute_sigma=True)
    # popt_poly, p_cov = curve_fit(f_poly, y_fit, x_fit)


    xx = np.linspace(0, path_lengths.max())
    yy = np.linspace(0, data.max())
    # y_mono = f_mono(xx, *popt_mono)
    x_poly = f_poly(yy, *popt_poly)
    # plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    # plt.plot(xx, y_mono, label='Mono fit')
    plt.plot(x_poly, yy, label='Poly fit', color='red', linewidth=4)


if __name__ == '__main__':
    # bhc_v2_dev()
    None