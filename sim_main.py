# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
### Imports
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
bfig_dir = os.path.join(base_dir,'bjobs/figs')

### Test image functions
def create_circle_image(image_size, radius, center):
    image = np.zeros((image_size, image_size))
    for x in range(image_size):
        for y in range(image_size):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < (radius+0.5) ** 2:
                image[x, y] = 1
    
    return image

def generate_triangle_image(img_size, triangle_size, corner_coords):
    """
    Generate a right triangle test image.

    Parameters:
    img_size (tuple): Size of the image (width, height)
    triangle_size (tuple): Size of the right triangle (base, height)
    corner_coords (tuple): Coordinates of the right-angled corner (x, y)
    """
    # Create a blank image
    image = np.zeros(img_size)

    # Define the triangle
    for x in range(corner_coords[0], min(corner_coords[0] + triangle_size[0], img_size[0])):
        for y in range(corner_coords[1], min(corner_coords[1] + triangle_size[1], img_size[1])):
            if (x - corner_coords[0]) + (y - corner_coords[1]) < triangle_size[0]:
                image[y, x] = 1  # Set pixel value

    return image

def create_rings_image(size, ring_width, spacing, radius, center):
    """
    Create a square image with concentric rings.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - ring_width: The width of each ring in pixels.
    - spacing: The spacing between rings in pixels.
    - radius: The overall radius of the circles (from the center to the outer edge) in pixels.
    - center: A tuple (x, y) representing the center of the circles.
    """

    # Create an empty image
    image = np.zeros((size, size))

    # Generate a grid of distances from the center
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create rings by selecting pixels within specific distance ranges
    for r in np.arange(radius, 0, -(ring_width + spacing)):
        inner_radius = r - ring_width
        mask = (dist_from_center <= r) & (dist_from_center > inner_radius)
        image[mask] = 1  # Set pixels within the ring to white

    return image

### Attenuation and X-ray spectrum
def fun_attenuation(plot=False):
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

    if plot == True:
        plt.loglog(energy_plot_range, estimate_attenuation(energy_plot_range), label='Spline Interpolation (Linear)')
        plt.loglog(energies, mu, 'o', label='Data Points',markersize=4)#,color='black')  # Data points

        plt.xlabel('Energy (keV)')
        plt.ylabel('Attenuation coefficient (mm^{-1})')
        plt.title('Attenuation coefficient vs energy')
        plt.legend()
        plt.show()

    return estimate_attenuation

def generate_spectrum(plot=False,filter=0,tube_potential=220,bin_width=2):
    # tube_potential = 120 # keV
    # tube_potential = 220
    # anode_angle = 90 # deg
    anode_angle = 12
    # bin_width = 2 # keV
    # bin_width = 0.5
    # data_source = 'pene'
    data_source = 'nist'
    physics_arr = ['diff','spekpy-v1','casim']
    physics = physics_arr[2]
    # spectrum = spekpy.Spek(kvp=tube_potential, th=anode_angle, dk=bin_width, mu_data_source='nist', physics='casim')
    spectrum = spekpy.Spek(kvp=tube_potential, th=anode_angle, dk=bin_width, mu_data_source=data_source, physics=physics)
    spectrum.filter('Sn',filter)
    spectrum = np.array(spectrum.get_spectrum())
    spectrum[1,:] /= np.sum(spectrum[1,:])

    num_bins = spectrum.shape[1]

    bin_centers = spectrum[0,:]
    bin_heights = spectrum[1,:]
    bin_edges = np.array(bin_centers) - bin_width/2
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

    if plot == True:
        plt.bar(bin_centers, bin_heights, align='center', width=np.diff(bin_edges))
        plt.title(f'Normalized histogram of X-ray spectrum at {tube_potential} keV')
        # plt.yscale('log')
        # plt.ylim(0,bin_width/10*0.1)
        plt.show()

        # plt.bar(bin_centers, bin_heights, align='center', width=np.diff(bin_edges))
        # plt.plot(bin_centers, mu(bin_centers))
    
    return bin_centers, bin_heights

###
physical_in_mm = 100
def staircase_bhc(physical_in_mm,mono_E=None):
    ### Set up CIL geometries
    angles = np.linspace(start=0, stop=180, num=1, endpoint=False)
    physical_size = physical_in_mm # mm
    voxel_num = 1000
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
    show2D(image, size=(8,8))

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
        I += I0_E * np.exp(-mu(E)*d) #+ 1e-40
        

    # plt.hist(np.log10(mu(bin_centers)),bins=100)
    # plt.title('log10 mu(bin_centers)')
    # # plt.xscale('log')
    # plt.show()

    print(f'minmax D: {d.min()}, {d.max()}')
    print(f'minmax I: {I.min()}, {I.max()}')
    # plt.hist(I.flatten(),bins=100)
    # plt.show()

    I = np.array(I,dtype='float32')
    data = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    b = data.as_array()

    # plt.hist(b.flatten(),bins=100)
    # plt.show()

    plt.plot(voxel_size * np.arange(1,b.size+1,1),b,label='Polychromatic absorption')
    if mono_E is None:
        mu_eff = np.sum(bin_heights * mu(bin_centers))
        b_mono = d*mu_eff
        plt.plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label='Monochromatic absorption at effective initial attenuation')
    else:
        b_mono = d*mu(mono_E)
        plt.plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label=f'Monochromatic absorption at {mono_E} keV')

    # plt.plot(voxel_size * np.arange(1,b.size+1,1),b,label='Polychromatic absorption')
    # plt.title('polychromatic absorption vs path length')
    # plt.loglog(voxel_size * np.arange(1,b.size+1,1),b,label='Polychromatic absorption')
    # plt.loglog(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label='Monochromatic absorption at effective initial attenuation')
    plt.xlabel('Pathlength (mm)')
    plt.title('Beam hardening effect for different pathlengths of gold')
    plt.legend()
    plt.show()

    # plt.loglog(voxel_size * np.arange(1,b.size+1,1), b_mono-b)
    # plt.title("polychromatic vs monochromatic absorption")
    # plt.show()

    x = voxel_size * np.arange(1,b.size+1,1)
    y = b_mono-b
    # y_spline = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    # y_fit = y_spline(x)
    # file_path = os.path.join(base_dir,'bh_absorption_corrections.txt')
    corrections = np.column_stack((x,y))
    # np.savetxt(file_path,corrections)



    plt.plot(b,b_mono-b)
    # plt.xlim(0,3)
    # plt.ylim(0,20)
    plt.title('b_mono-b vs b')
    plt.show()

    plt.plot(voxel_size * np.arange(1,b.size+1,1),b_mono-b)
    plt.title('b_mono-b vs path length')
    plt.show()

    corrections2 = np.column_stack((b,b_mono-b))
    return corrections,corrections2

# staircase_bhc(0.01)
    

def bhc_test1():
    physical_in_mm = 1
### Set up CIL geometries
    angles = np.linspace(start=0, stop=180, num=3*180//1, endpoint=False) #6
    physical_size = physical_in_mm # mm
    voxel_num = 1000
    voxel_size = physical_size/voxel_num

    ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)

    factor = 1 #4
    panel_num_cells = math.ceil(np.sqrt(2)*factor*voxel_num)
    # panel_num_cells = math.ceil(2*factor*voxel_num)

    # panel_num_cells = math.ceil(factor*voxel_num)
    panel_cell_length = 1/factor * voxel_size
    ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
        .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
        .set_angles(angles=angles)
    
    
    print(ig)
    show_geometry(ag,ig, grid=True)
    
### Generate test image
    ## circle with hole
    if True:
        # im1 = create_circle_image(image_size=voxel_num, radius=voxel_num//3, center=[voxel_num//2, voxel_num//2])
        im1 = create_circle_image(image_size=voxel_num, radius=voxel_num//2.5, center=[voxel_num//2, voxel_num//2])
        im2 = create_circle_image(image_size=voxel_num, radius=voxel_num//5.5, center=[voxel_num//2, voxel_num//2])
        # im_arr = im1
        im_arr = im1-0*im2

    ## concentric rings
    if False:
        # im_arr = create_rings_image(size=voxel_num, spacing=10, ring_width=15, radius=voxel_num//3, center=[voxel_num//2, voxel_num//2])
        im_arr = create_rings_image(size=voxel_num, spacing=20, ring_width=30, radius=voxel_num//3, center=[voxel_num//2, voxel_num//2])

    
    ##
    im_arr = im_arr.astype('float32')
    plt.imshow(im_arr,cmap='grey')
    plt.title('test image')
    plt.show()

    image = ImageData(array=im_arr,geometry=ig)

### Call other functions
    mu = fun_attenuation(plot=True)
    bin_centers, bin_heights = generate_spectrum(plot=True)
    num_bins = bin_centers.size
    plt.semilogy(bin_centers, mu(bin_centers))
    plt.xlabel('Energy (keV)')
    plt.ylabel('mu (mm^{-1})')
    plt.title('Attenuation coefficients narrowed down to spectrum')
    plt.show()
    
    corrections,corrections2 = staircase_bhc(physical_in_mm)
    spline_corrections = interpolate.InterpolatedUnivariateSpline(corrections[:,0], corrections[:,1], k=1)
    # plt.plot(corrections[:,0], corrections[:,1],'r', label='Original Data')

    spline_corrections2 = interpolate.InterpolatedUnivariateSpline(corrections2[:,0], corrections2[:,1], k=1)
    # return

### Simulate BH effect
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    d = A.direct(image)
    d = d.as_array()
    I = np.zeros(d.shape, dtype='float32')
    I0 = 0
    # print(d)
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        I += I0_E * np.exp(-mu(E)*d)

    b = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    # I = AcquisitionData(array=I, geometry=ag)

    ##
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    print(f'mu_eff: {mu_eff}')
    # b_mono = -np.log(np.exp(-mu_eff*d))
    b_mono = d*mu_eff
    # b_mono = AcquisitionData(array=np.exp(-b_mono), geometry=ag)
    b_mono = AcquisitionData(array=b_mono, geometry=ag)

### Reconstructions
    ## raw
    b.reorder(order='tigre')
    recon = FBP(b, image_geometry=ig, backend='tigre').run(verbose=0)
    # show2D(recon, title='FDK reconstruction')
    # show1D(recon, slice_list=[('horizontal_y', recon.shape[0]//2)])

####################    ###
    # fig,ax = plt.subplots(2,1, figsize=(8,12),gridspec_kw={'height_ratios': [2, 1]})
    # ax[0].imshow(recon.as_array(), origin='lower', cmap='gray')
    # ax[1].plot(recon.as_array()[recon.shape[0]//2,:])

    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 3
    ax[0] = fig.add_subplot(top_row_gs[0, 0])
    ax[1] = fig.add_subplot(top_row_gs[0, 1])
    ax[2] = fig.add_subplot(gs[1])

    plt.sca(ax[0])
    plt.imshow(im_arr, origin='lower',cmap='grey')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Disk test image')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    plt.colorbar()

    plt.sca(ax[2])
    plt.plot(recon.as_array()[recon.shape[0]//2,:])
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={recon.shape[0]//2} on the reconstruction')
    plt.grid(True)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.savefig(os.path.join(base_dir, 'plots/bh_disk.pdf'))
    plt.show()
####################################

    ## mono
    recon_mono = FBP(b_mono, image_geometry=ig, backend='tigre').run(verbose=0)
    # show2D(recon_mono, title='recon_mono')
    # show1D(recon_mono, slice_list=[('horizontal_y', recon_mono.shape[0]//2)])

### bhc
    plt.hist(recon.as_array().flatten(), bins=100)
    plt.title('recon raw hist')
    plt.show()

    # tau = 1
    tau = 7
    recon_segmented = ImageData(array=np.array(recon > tau,dtype='float32'), geometry=ig)
    show2D(recon_segmented,title='recon_segmented')

    path_lengths = A.direct(recon_segmented)
    # print(path_lengths.as_array()[500,500])
    # show2D(path_lengths)
    # np.max(path_lengths)
    plt.hist(path_lengths.as_array().flatten(), bins=200)
    plt.title('path_lengths hist')
    plt.show()

    # b_corrected = b.as_array() + spline_corrections(path_lengths.as_array())
    b_corrected = b.as_array() + spline_corrections2(b.as_array())
    b_corrected = AcquisitionData(array=np.array(b_corrected,dtype='float32'), geometry=ag)
    # print(np.sum(np.abs(b.as_array()-b_corrected.as_array())))

    plt.imshow(spline_corrections(path_lengths.as_array()),cmap='grey')
    plt.title('spline_corrections')
    plt.colorbar()
    plt.show()
    # show2D([b.as_array(),b_corrected.as_array()],num_cols=1)

    plt.plot(path_lengths.as_array().flatten(),b.as_array().flatten())
    plt.title('absorption vs path_lengths')
    plt.show()

    b_corrected.reorder('tigre')
    recon_bhc = FBP(b_corrected, image_geometry=ig, backend='tigre').run(verbose=0)
    show2D(recon_bhc, title='recon_bhc')
    show1D(recon_bhc, slice_list=[('horizontal_y', recon.shape[0]//2)])

### bhc on noisy data
    # 400
    b_noisy = np.clip(a=skimage.util.random_noise(b.as_array(), mode='gaussian',clip=False, mean=0, var=1/(0.5*400*I)), a_min=0, a_max=None)
    b_noisy = AcquisitionData(array=np.array(b_noisy,dtype='float32'), geometry=ag)
    # show2D([b,b_noisy],title='b and b_noisy',num_cols=1)

    recon_noisy = FBP(b_noisy, image_geometry=ig, backend='tigre').run(verbose=0)
    show2D(recon_noisy, title='recon noisy')
    # show1D(recon_noisy, slice_list=[('horizontal_y', recon.shape[0]//2)])

    plt.hist(np.array(recon_noisy.as_array().flatten()), bins=100)
    plt.title('recon_noisy raw hist')
    plt.show()

    # tau = 1
    tau = 7
    segmented = ImageData(array=np.array(recon_noisy > tau,dtype='float32'), geometry=ig)
    show2D(segmented,title='recon_noisy segmented')
    path_lengths = A.direct(segmented)

    plt.hist(path_lengths.as_array().flatten(), bins=200)
    plt.title('noisy path_lengths hist')
    plt.show()

    # b_corrected = b_noisy.as_array() + spline_corrections(path_lengths.as_array())
    b_corrected = b_noisy.as_array() + spline_corrections2(b_noisy.as_array())
    b_corrected = AcquisitionData(array=np.array(b_corrected,dtype='float32'), geometry=ag)

    plt.plot(path_lengths.as_array().flatten(),b_noisy.as_array().flatten())
    plt.title('noisy: absorption vs path_lengths')
    plt.show()

    b_corrected.reorder('tigre')
    recon_bhc = FBP(b_corrected, image_geometry=ig, backend='tigre').run(verbose=0)
    show2D(recon_bhc, title='noisy recon_bhc')
    show1D(recon_bhc, slice_list=[('horizontal_y', recon.shape[0]//2)])

def bhc_test2():
    base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
    file_path = os.path.join(base_dir,'centres/X20.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    voxel_num = ig.voxel_num_x
    voxel_size = ig.voxel_size_x
    physical_in_mm = voxel_num*voxel_size
    
    ##
    # im1 = create_circle_image(image_size=voxel_num, radius=voxel_num//3, center=[voxel_num//2, voxel_num//2])
    # im2 = create_circle_image(image_size=voxel_num, radius=voxel_num//3-10, center=[voxel_num//2, voxel_num//2])
    # im_arr = im1-im2

    # im_arr = create_circle_image(image_size=voxel_num, radius=voxel_num//2, center=[voxel_num//2, voxel_num//2])

    # im1 = io.imread(os.path.join(base_dir,'test_images/test_image_shapes2.png'))
    # im1 = io.imread(os.path.join(base_dir,'test_images/test_image_X20_imitation.png'))
    # im2 = color.rgb2gray(im1) == 0
    # im3 = erosion(im2,footprint=disk(1))
    # im3 = erosion(im3,footprint=disk(1))
    # #im3 = erosion(im3,footprint=disk(1))

    # im2 = io.imread(os.path.join(base_dir,'test_images/state_of_colorado.png')) < 1
    im2 = io.imread(os.path.join(base_dir,'test_images/X20_2_imitation.png'))
    im_arr = im2[:,:,0] > 100

    im_arr = im_arr.astype('float32')
    # plt.imshow(im_arr,cmap='grey')
    # plt.title('test image')
    # plt.show()
    # show2D(im_arr,'test image')

    image = ImageData(array=im_arr,geometry=ig)
    # image.apply_circular_mask(radius=0.69, in_place=True)
    # image.apply_circular_mask(radius=0.6, in_place=True)
    # image.apply_circular_mask(radius=0.1, in_place=True)
    show2D(image)
    # return

### Call other functions
    mu = fun_attenuation(plot=True)
    bin_centers, bin_heights = generate_spectrum(plot=True)
    num_bins = bin_centers.size
    plt.semilogy(bin_centers, mu(bin_centers))
    plt.xlabel('Energy (keV)')
    plt.ylabel('mu (mm^{-1})')
    plt.title('Attenuation coefficients narrowed down to spectrum')
    plt.show()
    
    corrections,corrections2 = staircase_bhc(6)
    # return
    spline_corrections = interpolate.InterpolatedUnivariateSpline(corrections[:,0], corrections[:,1], k=1)
    # plt.plot(corrections[:,0], corrections[:,1],'r', label='Original Data')

    spline_corrections2 = interpolate.InterpolatedUnivariateSpline(corrections2[:,0], corrections2[:,1], k=1)
    # return

### Simulate BH effect
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    d = A.direct(image)
    d = d.as_array()

    plt.hist(d.flatten(),bins=100)
    plt.show()

    I = np.zeros(d.shape, dtype='float32')
    I0 = 0
    # print(d)
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        # print(I0)
        I += I0_E * np.exp(-mu(E)*d)

    # return
    b = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    # I = AcquisitionData(array=I, geometry=ag)

    ##
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    print(f'mu_eff: {mu_eff}')
    # b_mono = -np.log(np.exp(-mu_eff*d))
    b_mono = d*mu_eff
    # b_mono = AcquisitionData(array=np.exp(-b_mono), geometry=ag)
    b_mono = AcquisitionData(array=b_mono, geometry=ag)

### Reconstructions
    ## raw
    b_noisy = np.clip(a=skimage.util.random_noise(b.as_array(), mode='gaussian',clip=False, mean=0, var=(1/(0.5*400*I))**2), a_min=0, a_max=None)
    b = AcquisitionData(array=np.array(b_noisy,dtype='float32'), geometry=ag)
    
    b.reorder(order='tigre')
    recon = FDK(b, image_geometry=ig).run(verbose=0)
    show2D(recon, title='recon')
    # show1D(recon, slice_list=[('horizontal_y', recon.shape[0]//2)])
    plt.plot(recon.as_array()[recon.shape[0]//2,:])
    plt.show()

    ## mono
    recon_mono = FDK(b_mono, image_geometry=ig).run(verbose=0)
    # show2D(recon_mono, title='recon_mono')
    # show1D(recon_mono, slice_list=[('horizontal_y', recon_mono.shape[0]//2)])

### bhc
    tau = 0.2
    recon_segmented = ImageData(array=np.array(recon > tau,dtype='float32'), geometry=ig)
    # recon_segmented = ImageData(array=im_arr, geometry=ig)
    show2D(recon_segmented,title='recon_segmented')

    # path_lengths = A.direct(recon_segmented)
    path_lengths = A.direct(recon)


    # print(path_lengths.as_array()[500,500])
    # show2D(path_lengths)
    # np.max(path_lengths)
    plt.hist(path_lengths.as_array().flatten(), bins=200)
    plt.title('path_lengths hist')
    plt.show()
    
    b_corrected = b.as_array() + spline_corrections(path_lengths.as_array())
    # b_corrected = b.as_array() + spline_corrections2(b.as_array())
    b_corrected = AcquisitionData(array=np.array(b_corrected,dtype='float32'), geometry=ag)
    # print(np.sum(np.abs(b.as_array()-b_corrected.as_array())))

    # show2D([b.as_array(),b_corrected.as_array()],num_cols=1)

    plt.plot(path_lengths.as_array().flatten(),b.as_array().flatten(),'.')
    plt.title('absorption vs path_lengths')
    plt.show()

    # plt.plot(path_lengths.as_array().flatten(),b_corrected.as_array().flatten())
    # plt.title('spline_corrections vs path_lengths')
    # plt.show()

    b_corrected.reorder('tigre')
    recon_bhc = FDK(b_corrected, image_geometry=ig).run(verbose=0)
    show2D(recon_bhc, title='recon_bhc')
    show1D(recon_bhc, slice_list=[('horizontal_y', recon.shape[0]//2)])
    

def lin_interp_sino2D(data,tau):
    def lin_interp_proj(proj):
        above_tau_indices = np.where(proj > tau)[0]
        diffs = np.diff(above_tau_indices)
        breaks = np.where(diffs > 1)[0]
        groups = np.split(above_tau_indices, breaks+1)
        if not list(groups[0]):
            return proj
        for group in groups:
            if group[0] == 0:
                lval = tau
            else:
                lval = proj[group[0]-1]
            if group[-1] == proj.size-1:
                rval = tau
            else:
                rval = proj[group[-1]+1]
            
            proj[group] = (rval-lval)/((group[-1]+1) - (group[0]-1)) * (group - (group[0]-1)) + lval
        return proj
    
    # data_interp = np.apply_along_axis(func1d=lin_interp_proj, axis=1, arr=data.copy().as_array())
    data_interp = data.copy().as_array()
    for i in range(data.shape[0]):
        data_interp[i,:] = lin_interp_proj(data_interp[i,:])
    return AcquisitionData(array=data_interp, geometry=data.geometry)

def test_linear_interpolation():
    def compare_lin_interp(ang_idx,indices=None):
        n_rays = data.shape[1]
        if indices is None:
            indices = np.arange(0,n_rays)
        plt.figure(figsize=(10,8))
        plt.title(f'Single projection at angle_index={ang_idx}')
        plt.xlabel('Panel index')
        plt.ylabel('Absorption')
        plt.plot([indices.min(),indices.max()],[tau,tau],label='tau (threshold)')
        plt.plot(indices,data.as_array()[ang_idx,indices], label='data', marker='.')
        plt.plot(indices,data_interp.as_array()[ang_idx,indices], label='data_interp')
        plt.legend()
        plt.show()

    # file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    data_trans = AbsorptionTransmissionConverter()(data)
    show2D(data_trans, cmap='nipy_spectral', fix_range=(data_trans.min(),0.5))

    tau = -np.log(0.115)
    # tau = -np.log(0.13)
    # tau = -np.log(0.15)
    # tau = -np.log(0.07)
    # show1D(data_trans, slice_list=[('angle', 480)])
    # show1D(data, slice_list=[('angle', 800)])

    data_interp = lin_interp_sino2D(data, tau)
    ang_idx = 480
    # show1D([data,data_interp], slice_list=[[('angle', ang_idx)],[('angle', ang_idx)]], line_styles=['-','-'])
    indices = np.arange(275,750)
    compare_lin_interp(ang_idx,indices=None)

    recon = FDK(data).run(verbose=0)
    recon_interp = FDK(data_interp).run(verbose=0)
    show2D([recon,recon_interp],title=f'recon vs recon_interp for tau={tau}',fix_range=(-0.15,0.6))
    show2D([recon,recon_interp],title=f'recon vs recon_interp for tau={tau}')

    show2D(AbsorptionTransmissionConverter()(data_interp), cmap='nipy_spectral', fix_range=(data_trans.min(),0.5))


def back_projection():
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ###
    # tau = -np.log(0.13)
    # data = lin_interp_sino2D(data, tau)

    ###
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    tau = 0.3
    air_proj_mask = 1000*np.array(data.as_array() < 0.3, dtype=np.float32)
    air_proj_mask = AcquisitionData(array=air_proj_mask, geometry=ag)

    # fdk = FDK(data)
    # recon = FDK(data, filter=np.ones(2**11,dtype=np.float32)).run(verbose=0)
    # show2D(recon).save(os.path.join(bfig_dir,'backprojection.png'))

    fft_order = 12
    recon_air = FDK(air_proj_mask, filter=np.ones(2**12,dtype=np.float32)).run(verbose=0)
    recon_mask = recon_air < 1
    show2D(recon_air, fix_range=(0,10), cmap='nipy_spectral').save(os.path.join(bfig_dir,'recon_air_mask.png'))
    show2D(recon_air < 1).save(os.path.join(bfig_dir,'recon_air_mask_bin.png'))

    ub = np.array(recon_air < 1, dtype=np.float32)
    ub[recon_air < 0.5] = np.inf
    G = IndicatorBox(lower=0, upper=ub)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    F = LeastSquares(A, data)
    x0 = ig.allocate(0.0)
    # x0 = FDK(data).run(verbose=0)
    show2D(x0)
    myFISTAmask = FISTA(f=F, g=G, initial=x0, 
                      max_iteration=1000,
                      update_objective_interval = 10)
    myFISTAmask.run(50, verbose=1)
    show2D(myFISTAmask.solution,'Convex hull FISTA').save(os.path.join(bfig_dir,'fista_hull.png'))
    show2D(myFISTAmask.solution,fix_range=(0,1.25))


physical_in_mm = 1

if __name__ == "__main__":
    # staircase_bhc(0.001)
    # bhc_test1()
    # bhc_test2()
    # back_projection()
    # test_linear_interpolation()
    None