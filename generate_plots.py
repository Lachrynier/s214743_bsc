import sys
sys.path.append('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc')
import sim_main
from sim_main import fun_attenuation, generate_spectrum, generate_triangle_image

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
import numpy as np
import math
import scipy as sp
from scipy import interpolate
from numpy.linalg import solve
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from time import time
import spekpy
import os
import skimage
import pickle

from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 

### CIL imports
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
from cil.processors import TransmissionAbsorptionConverter, Slicer, AbsorptionTransmissionConverter
from cil.optimisation.algorithms import CGLS, SIRT

from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry
from cil.utilities.noise import gaussian, poisson

base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')


def spectrum_penetration_plot():
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0,tube_potential=200,bin_width=0.01)
    num_bins = bin_centers.size

    max_length = 10
    # d = np.linspace(0,max_length, dtype='float64')
    d = np.array([0,0.4,2.5], dtype='float64')
    I_E = np.zeros((num_bins,d.size), dtype='float64')
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I_E[i,:] = I0_E * np.exp(-mu(E)*d)
    
    I_E = I_E / np.sum(I_E, axis=0)

    bin_edges = np.linspace(0, 200, 81)  # Creates 80 bins from 0 to 200

    plt.figure(figsize=(10, 6))  # Set the figure size
    # kwargs = dict(alpha=0.5, bins=80, density=True, stacked=True)  # Common kwargs for all histograms
    kwargs = dict(alpha=0.5, density=True, stacked=True)  # Removed 'bins=80'
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,0], color='r', label=fr'$d = 0$ mm')
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,1], color='g', label=fr'$d = {d[1]}$ mm')
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,2], color='b', label=fr'$d = {d[2]}$ mm')

    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,0], color='r', edgecolor='r', label=fr'$d = 0$ mm')
    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,1], color='g', edgecolor='g', label=fr'$d = {d[1]}$ mm')
    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,2], color='b', edgecolor='b', label=fr'$d = {d[2]}$ mm')

    plt.xlabel(r'$E$ [keV]')
    plt.title(r'Normalized X-ray spectrum histograms at different path lengths $d$')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'plots/spectrum_snapshots.pdf'))
    plt.show()
    print(np.sum(I_E[:,0]))
    print(np.sum(I_E[:,2]))

def random_plot():
    from matplotlib import rc
    import matplotlib
    d = np.linspace(0, 10, 100)  # Replace with your actual data
    mean_energies = np.sin(d)  # Replace with your actual data

    # Set up Matplotlib to use LaTeX for text rendering
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Palatino"],
    # })

    # font = {'family' : 'monospace',
    #     'weight' : 'bold',
    #     'size'   : 22}
    # rc('font', **font)

    plt.rcParams.update({"text.usetex": True})
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    # Create the plot
    plt.figure(figsize=(8, 6))  # Choose a good size for thesis or papers
    plt.plot(d, mean_energies, label=r'Mean Energies')  # Use raw string for LaTeX
    plt.title(r'\textbf{Energy Distribution}', fontsize=16)  # Bold title with LaTeX
    plt.xlabel(r'Distance ($m$)', fontsize=14)  # Label with units
    plt.ylabel(r'Energy $(J)$ $\sum \xi$', fontsize=14)  # Label with units
    plt.legend(frameon=False, fontsize=12)
    plt.grid(True)
    plt.tight_layout()  # Adjust the layout to not cut off anything
    
    # Set font size of the numbers on the x and y axes
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.tick_params(axis='both', which='minor', labelsize=10)  # If using minor ticks
    # plt.savefig('energy_distribution.pdf')
    plt.show()

def mean_energy_plot():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0,tube_potential=200,bin_width=0.5)
    num_bins = bin_centers.size


    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    max_lengths = [0.1,2.5,15]
    for k,max_length in enumerate(max_lengths):
        d = np.linspace(0,max_length,num=300, dtype='float64')
        I_E = np.zeros((num_bins,d.size), dtype='float64')
        for i in range(num_bins):
            E = bin_centers[i]
            I0_E = bin_heights[i]
            I_E[i,:] = I0_E * np.exp(-mu(E)*d)
        
        I_E = I_E / np.sum(I_E, axis=0)
        
        mean_energies = np.sum(bin_centers[:,None] * I_E, axis=0)

        ax[k].set_ylabel(r'[keV]')
        ax[k].set_xlabel(r'$d$ [mm]')
        # ax[k].set_title(r'$d\in [0,10]$')
        ax[k].set_title(fr'$d\in [0,{max_length}]$')
        ax[k].plot(d,mean_energies)
        ax[k].grid(True)
    
    ax[2].set_xticks(np.linspace(start=0,stop=15,num=7))
    # plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle(r'Mean energy of the X-ray spectrum as a function of the path length $d$')
    plt.savefig(os.path.join(base_dir, 'plots/mean_energy.pdf'))
    plt.show()

def attenuation_plots():
    # mu/rho in (cm2/g)
    # multiply by density of gold to get mu in (cm^{-1})
    # ORIGINAL DATA IN MEV!!!! SO CONVERT TO KEV
    data = []
    file_path = os.path.join(base_dir,'NIST_gold_only_dat.txt')
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split()

            # Check if the line has at least 3 elements and the first is a number
            if len(parts) >= 3 and parts[0].replace('.', '', 1).replace('E-', '', 1).replace('E+', '', 1).isdigit():
                energy = float(parts[-3])  # Energy value
                mu_rho = float(parts[-2])  # mu/rho value
                mu_en_rho = float(parts[-1])  # mu_en/rho value
                data.append((energy, mu_rho, mu_en_rho))

    data = np.array(data)

    # convert energies to keV
    energies = 1000*data[:,0]

    # calculate mu = mu/rho * rho
    rho_gold = 19.3 # g/cm^3
    # rho_gold = 1
    # rho_gold = 1
    print(f'rho: {rho_gold}')
    # divide by 10 to get it in mm
    mu = data[:,1] * rho_gold / 10 # mm^{-1}

    # Perform spline interpolation in log domain
    spline_log_domain = interpolate.InterpolatedUnivariateSpline(np.log10(energies), np.log10(mu), k=1)
    def estimate_attenuation(energy):
        return np.power(10, spline_log_domain(np.log10(energy)))
    
    energy_plot_range = np.logspace(np.log10(min(energies)), np.log10(max(energies)), 500)
    ##############################
    #####
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    ###
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].loglog(energy_plot_range, estimate_attenuation(energy_plot_range), label='Interpolation')
    ax[0].loglog(energies, mu, '.', label='Raw data',markersize=5,color='red')  # Data points

    ax[0].set_xlabel(r'$E$ [keV]')
    ax[0].set_ylabel(r'$\mu$ [mm$^{-1}$]')
    ax[0].set_title('Complete range of raw data')
    ax[0].grid(True)
    ax[0].legend()
    
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False)
    num_bins = bin_centers.size

    ax[1].semilogy(bin_centers, mu(bin_centers))
    ax[1].set_title('Range narrowed down to energies in gold scans')
    
    ax[1].set_xlabel(r'$E$ [keV]')
    ax[1].set_ylabel(r'$\mu$ [mm$^{-1}$]')
    ax[1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    fig.suptitle(r'Gold attenuation coefficient $\mu$ as a function of energy $E$')
    plt.savefig(os.path.join(base_dir, 'plots/gold_attenuations.pdf'))
    plt.show()

def transmission_plot():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    energies = [150,220]
    mu = fun_attenuation(plot=False)
    max_pen = 10
    xx = np.linspace(0,max_pen,num=200)
    plt.figure(figsize=(10,10))
    for E in energies:
        mu_E = mu(E)
        plt.plot(xx,np.exp(-mu_E*xx), label=f'{E} keV')
    plt.legend()
    plt.xlabel(r'Path length [mm]')
    plt.title(r'Transmission $I/I_0$ at fixed energies')
    plt.yscale('log')
    plt.grid(True)

    plt.yticks(10.0**np.arange(-15, 1, 1))  # Corrected line here
    plt.xticks(np.arange(0,max_pen+1,1))
    # plt.grid(True, which="both", ls="--")  # This ensures grid lines are drawn for both major and minor ticks
    plt.show()

def staircase_plot():
    physical_in_mm = 1
    voxel_num = 20
    angles = np.linspace(start=0, stop=180, num=1, endpoint=False)
    physical_size = physical_in_mm # mm
    voxel_size = physical_size/voxel_num

    ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)

    factor = 1
    panel_num_cells = math.ceil(factor*voxel_num)
    panel_cell_length = 1/factor * voxel_size
    ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
        .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
        .set_angles(angles=angles)

    tri_width = voxel_num
    corner = 0

    img_size = (voxel_num, voxel_num)  # Image size: 100x100 pixels
    triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
    corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
    triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
    im_arr = np.rot90(triangle_image, 1)
    im_arr = im_arr.astype('float32')
    # plt.imshow(im_arr,cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    image = ImageData(array=im_arr.T,geometry=ig)
    show2D(image)
    show2D(image, title=r'Staircase test image', size=(10,10)).save(os.path.join(base_dir, 'plots/staircase_test_image.pdf'))

def staircase_experiments():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    def staircase_bhc(physical_in_mm,voxel_num=1000,mono_E=None):
        ### Set up CIL geometries
        angles = np.linspace(start=0, stop=180, num=1, endpoint=False)
        physical_size = physical_in_mm # mm
        voxel_size = physical_size/voxel_num

        ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)

        factor = 1
        panel_num_cells = math.ceil(factor*voxel_num)
        panel_cell_length = 1/factor * voxel_size
        ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
            .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
            .set_angles(angles=angles)

        plot_size = 6
        # show_geometry(ag, ig, grid=True, figsize=(plot_size, plot_size),fontsize=plot_size)

        ### Generate staircase test image
        tri_width = voxel_num
        corner = 0

        img_size = (voxel_num, voxel_num)  # Image size: 100x100 pixels
        triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
        corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
        triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
        im_arr = np.rot90(triangle_image, 1)
        im_arr = im_arr.astype('float32')
        # plt.imshow(im_arr,cmap='gray')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

        image = ImageData(array=im_arr.T,geometry=ig)
        # show2D(image, size=(8,8))

        ###
        mu = fun_attenuation(plot=False)
        bin_centers, bin_heights = generate_spectrum(plot=False)
        num_bins = bin_centers.size

        A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
        d = A.direct(image)
        d = d.as_array()
        I = np.zeros(d.shape, dtype='float64')
        I0 = 0
        for i in range(num_bins):
            E = bin_centers[i]
            I0_E = bin_heights[i]
            I0 += I0_E
            I += I0_E * np.exp(-mu(E)*d)

        print(f'minmax D: {d.min()}, {d.max()}')
        print(f'minmax I: {I.min()}, {I.max()}')

        I = np.array(I,dtype='float32')
        data = AcquisitionData(array=-np.log(I/I0), geometry=ag)
        b = data.as_array()

        if mono_E is None:
            mu_eff = np.sum(bin_heights * mu(bin_centers))
            b_mono = d*mu_eff
        else:
            b_mono = d*mu(mono_E)
        

        # x = voxel_size * np.arange(1,b.size+1,1)
        # y = b_mono-b
        # corrections = np.column_stack((x,y))
        # plt.plot(b,b_mono-b)
        # plt.title('b_mono-b vs b')
        # plt.show()

        # plt.plot(voxel_size * np.arange(1,b.size+1,1),b_mono-b)
        # plt.title('b_mono-b vs path length')
        # plt.show()

        # corrections2 = np.column_stack((b,b_mono-b))
        return b,b_mono

    max_lengths = [0.01,0.1] # mm
    voxel_num = 1000
    fig, ax = plt.subplots(1, len(max_lengths), figsize=(10, 5))
    for k,max_length in enumerate(max_lengths):
        voxel_size = max_length/voxel_num
        b,b_mono = staircase_bhc(physical_in_mm=max_length,voxel_num=voxel_num,mono_E=None)
        ax[k].set_xlabel(r'$d$ [mm]')
        ax[k].set_title(fr'$d\in [0,{max_length}]$')
        ax[k].plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label=r'Monochromatic absorption at $\mu_{\text{eff}}(0)$')
        ax[k].plot(voxel_size * np.arange(1,b.size+1,1), b, label=r'Polychromatic absorption')
        ax[k].grid(True)
        ax[k].legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle(r'Beam hardening effect on absorption as a function of the path length $d$')
    plt.savefig(os.path.join(base_dir, 'plots/bh_effect.pdf'))
    plt.show()

# ax[k].set_ylabel(r'[keV]')
#         ax[k].set_xlabel(r'$d$ [mm]')
#         # ax[k].set_title(r'$d\in [0,10]$')
#         ax[k].set_title(fr'$d\in [0,{max_length}]$')
#         ax[k].plot(d,mean_energies)
#         ax[k].grid(True)
    
#     ax[2].set_xticks(np.linspace(start=0,stop=15,num=7))
#     # plt.xscale('log')
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.85)
#     fig.suptitle(r'Mean energy of the X-ray spectrum as a function of the path length $d$')
#     plt.savefig(os.path.join(base_dir, 'plots/mean_energy.pdf'))
#     plt.show()