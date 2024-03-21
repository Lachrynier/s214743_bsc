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

from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry
from cil.utilities.noise import gaussian, poisson

print(os.getcwd())
if os.getcwd() == "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev":
    os.chdir('analysis/s214743_bsc')
    print(os.getcwd())

from sim_main import fun_attenuation, generate_spectrum, generate_triangle_image

from scipy.optimize import curve_fit


def staircase_bhc(physical_in_mm):
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
        I += I0_E * np.exp(-mu(E)*d) + 1e-40
        

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

    # mu_eff = np.sum(bin_heights * mu(bin_centers))
    # mono_E = 170
    # mono_E = 105
    mono_E = 55
    mu_eff = mu(mono_E)
    # b_mono = -np.log(np.exp(-mu_eff*d))
    b_mono = d*mu_eff

    plt.plot(voxel_size * np.arange(1,b.size+1,1),b,label='Polychromatic absorption')
    plt.title('polychromatic absorption vs path length')
    plt.show()

    plt.plot(voxel_size * np.arange(1,b.size+1,1),b,label='Polychromatic absorption')
    # plt.plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label='Monochromatic absorption at effective initial attenuation')
    plt.plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label=f'Monochromatic absorption at {mono_E} keV')
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


def main():
    base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    # print(ag)
    # print(ig)
    # show_geometry(ag, ig)
    # return##
    print((data.min(),data.max()))
    # data /= data.max()
    # print((data.min(),data.max()))
    # data = TransmissionAbsorptionConverter()(data)
    print((data.min(),data.max()))
    plt.hist(data.as_array().flatten(),bins=100)
    plt.yscale('log')
    plt.title('absorption hist')
    plt.show()

    data.reorder('tigre')
    # fdk = FDK(data, ig,backend='tigre')
    fdk = FDK(data)
    recon = fdk.run(verbose=0)
    show2D(recon)

    plt.hist(recon.as_array().flatten(), bins=100)
    plt.title('recon raw hist')
    plt.yscale('log')
    plt.show()
    return

    tau = 0.3
    # tau = 0.25
    # tau = 0.15
    # recon_segmented = ImageData(array=np.array(recon > tau,dtype='float32'), geometry=ig)
    # clipped_arr = np.array(np.clip(recon.as_array(), a_min=0, a_max=100),dtype='float32')
    # recon_segmented = ImageData(array=clipped_arr, geometry=ig)
    # show2D(recon_segmented,title='recon_segmented')

    # A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    # # A = ProjectionOperator(ig, ag, device='gpu')
    # path_lengths = A.direct(recon_segmented)


    # plt.plot(path_lengths.as_array().flatten(),data.as_array().flatten(),'.',markersize=2)
    # plt.title('absorption vs path_lengths')
    # plt.show()

    # ##Average out plot
    # if True:
    #     x = np.array(path_lengths.as_array().flatten())
    #     y = np.array(data.as_array().flatten())
    #     num_bins = 25
    #     bin_edges = np.linspace(x.min(), x.max(), num_bins+1)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     bin_indices = np.digitize(x, bin_edges)
    #     bin_means = np.array([y[bin_indices == i].mean() for i in range(1, len(bin_edges))])
    #     plt.scatter(x, y, alpha=0.2, label='Observed',s=3)
    #     plt.plot(bin_centers, bin_means, color='red', label='Averaged out')
    #     plt.xlabel('Path lengths (mm)')
    #     plt.ylabel('Absorption')
    #     plt.title('')
    #     plt.legend()
    #     # plt.show()

    # def f(x, c):
    #     return c*x
    
    # popt, pcov = curve_fit(f, bin_centers, bin_means)
    # b_corrected = f(path_lengths.as_array(),popt[0])
    # plt.plot(bin_centers, f(bin_centers,popt[0]), color='green', label='line_fit')
    # plt.show()

    # print((np.max(x),np.max(y)))
    # plt.scatter(x, y, alpha=0.2, label='Observed')
    # plt.plot(bin_centers, np.sqrt(bin_centers * np.max(y)**2/np.max(x)), color='red', label='Averaged out')
    # plt.plot(bin_centers, 1.2*np.max(y)*np.power(,4), color='red', label='Averaged out')
    # plt.plot(bin_centers, np.power(bin_centers * bin_means[-1]**4/bin_means[-1],4), color='red', label='Averaged out')
    # plt.xlabel('Path lengths (mm)')
    # plt.ylabel('Absorption')
    # plt.title('')
    # plt.legend()
    # plt.show()
    
    # x2 = np.hstack(([0],bin_centers))
    # y2 = np.hstack(([0],np.power(bin_centers * bin_means[-1]**4/bin_means[-1],4)))
    # spline_corrections = interpolate.InterpolatedUnivariateSpline(x2,y2, k=1)
    # return

    # b_corrected = f(spline_corrections(data.as_array()), popt[0])
    # b_corrected = (data.as_array() / 1.8)** 6
    b_corrected = (data.as_array() / 2.3)** 6

    # plt.plot(data.as_array().flatten(),b_corrected.flatten(),'.')
    # plt.show()

    
    _,corrections = staircase_bhc(8)
    # staircase_bhc(0.1)
    # return

    # spline_corrections = interpolate.InterpolatedUnivariateSpline(corrections[:,0], corrections[:,1], k=1)
    # b_corrected = data.as_array() + 2*spline_corrections(data.as_array())

    # plt.plot(data.as_array().flatten(),spline_corrections(data.as_array()).flatten(),'.')
    # plt.show()

    b_corrected = AcquisitionData(array=np.array(b_corrected,dtype='float32'), geometry=ag)

    plt.hist(b_corrected.as_array().flatten(),bins=100)
    plt.title('absorption hist bhc')
    plt.show()

    # return
    b_corrected.reorder('tigre')
    fdk = FDK(b_corrected, ig)
    recon_bhc = fdk.run()
    show2D(recon_bhc, title='recon_bhc')
    show1D(recon_bhc, slice_list=[('horizontal_y', recon.shape[0]//2)])
    show1D(recon, slice_list=[('horizontal_y', recon.shape[0]//2)])
    # recon_bhc
    # show2D(recon_bhc, title='recon_bhc')

def main2():
    base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
    # file_path = os.path.join(base_dir,'centres/X20.pkl')
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    data = AbsorptionTransmissionConverter()(data)
    # data = TransmissionAbsorptionConverter()(data)

    ag = data.geometry
    ig = ag.get_ImageGeometry()
    # show_geometry(ag, ig)
    show2D(data,fix_range=(0.9,1.02))
    print(data.max())
    # show2D(Slicer(roi={'horizontal':(0,50)})(data))
    hori = 0
    show1D(data,slice_list=[('horizontal',hori)],title=f'horizontal={hori}')
    # show1D(data)
    # print(data.__dir__())

    data.reorder('tigre')
    fdk = FDK(data)
    recon = fdk.run(verbose=0)
    show2D(recon)

    ###
    # padsize = 200
    # # data2 = Padder.edge(pad_width={'horizontal': padsize})(data)
    # data2 = Padder.constant(pad_width={'horizontal': padsize}, constant_values=0)(data)

    # show2D(data2)
    # show1D(data2,slice_list=[('horizontal',0)])
    # fdk2 = FDK(data2)
    # recon2 = fdk2.run(verbose=0)
    # show2D(recon2)


if __name__ == "__main__":
    main()
    # main2()
    None