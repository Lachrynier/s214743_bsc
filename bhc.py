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
import itertools

print(os.getcwd())
if os.getcwd() == "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev":
    os.chdir('analysis/s214743_bsc')
    print(os.getcwd())

base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
bfig_dir = os.path.join(base_dir,'bjobs/figs')


from sim_main import fun_attenuation, generate_spectrum, generate_triangle_image, staircase_bhc

from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu


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

def calc_line_params_from_two_points(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if np.abs(dy) < 1e-10:
        theta = np.pi / 2
    else:
        theta = np.arctan(-dx/dy)
    
    rho = p1[0]*np.cos(theta) + p1[1]*np.sin(theta)
    return theta,rho

def show_lines(data,image,idx_pairs,ray_opacity=0.5,ray_thickness=1.5):
    def R_to_plot(vec):
        return axis_scalings*(vec + shift_R_to_I)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    ### initializations
    voxel_num_x, voxel_num_y, voxel_num_z = ig.voxel_num_x, ig.voxel_num_y, ig.voxel_num_z
    voxel_size_x, voxel_size_y, voxel_size_z = ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_z
    center_x, center_y, center_z = ig.center_x, ig.center_y, ig.center_z
    image_max_side = np.max([voxel_num_x*voxel_size_x, voxel_num_y*voxel_size_y])

    axis_scalings = 1/np.array([voxel_size_x,voxel_size_y])
    shift_image_origin = 1/axis_scalings * np.array([voxel_num_x,voxel_num_y]) / 2

    source_position = ag.config.system.source.position
    rotation_axis_position = ag.config.system.rotation_axis.position
    detector_position = ag.config.system.detector.position
    detector_direction_x = ag.config.system.detector.direction_x

    shift_rot_axis_origin = -rotation_axis_position

    shift_R_to_I = -shift_rot_axis_origin + shift_image_origin

    panel_num_pixels = ag.config.panel.num_pixels
    panel_pixel_size = ag.config.panel.pixel_size
    panel_origin = ag.config.panel.origin

    plt.figure(figsize=[10,10])
    # plt.figure(figsize=[20,20])
    # plt.imshow(ig.allocate(0.0).as_array(),cmap='gray')
    plt.imshow(image,cmap='gray',vmin=image.min(),vmax=image.max())
    xlims = plt.xlim()
    ylims = plt.ylim()

    for angle_idx,horizontal_idx in idx_pairs:
        angle = ag.angles[angle_idx]
        # print(angle)
        angle *= np.pi/180
        rot_mat_angle = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        rot_source_R = rot_mat_angle.T @ (source_position + shift_rot_axis_origin)

        panel_pixel_position = detector_position + detector_direction_x * panel_pixel_size * ( -(panel_num_pixels[0]-1)/2 + horizontal_idx )
        rot_panel_pixel_R = rot_mat_angle.T @ (panel_pixel_position + shift_rot_axis_origin)

        plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(rot_panel_pixel_R)),'r-',linewidth=ray_thickness, alpha=ray_opacity)
    
    plt.xlim(xlims)
    plt.ylim(ylims)
    # plt.legend(loc='upper left')
    plt.show()

def show_lines2(data,image,idx_pairs,ray_opacity=0.5,ray_thickness=1.5,random_colors=True):
    max_abs = 1
    scale_proj_axis=2
    def R_to_plot(vec):
        return axis_scalings*(vec + shift_R_to_I)
    
    def horizontal_to_proj(horizontal,scale):
        panel_pixel_position = detector_position + detector_direction_x * panel_pixel_size * ( -(panel_num_pixels[0]-1)/2 + horizontal )
        rot_panel_pixel_R = rot_mat_angle.T @ (panel_pixel_position + shift_rot_axis_origin)
        rot_panel_pixel = rot_panel_pixel_R - shift_rot_axis_origin
        rot_panel_pixel_I = rot_panel_pixel_R + shift_image_origin

        H = np.linalg.norm(panel_pixel_position - detector_position, 2)
        scale_n_3hat = H/L * scale_n
        if horizontal < (panel_num_pixels[0] // 2):
            scale_n_3hat *= (-1)
    
        # proj_x is "x-axis" of fan-beam projection plot
        proj_x_R = rot_source_R + scale_n*n + scale_n_3hat*n_3hat
        if scale == None:
            proj_y_R = None
        else:
            proj_abs = data.as_array()[angle_idx,horizontal]
            # proj_y_R = proj_x_R + 0.1*image_max_side * n
            proj_y_R = proj_x_R + scale*image_max_side * proj_abs/max_abs * n

        # print(f'Horizontal {horizontal}: {proj_x_R,proj_y_R}')
        return proj_x_R,proj_y_R
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    ### initializations
    voxel_num_x, voxel_num_y, voxel_num_z = ig.voxel_num_x, ig.voxel_num_y, ig.voxel_num_z
    voxel_size_x, voxel_size_y, voxel_size_z = ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_z
    center_x, center_y, center_z = ig.center_x, ig.center_y, ig.center_z
    image_max_side = np.max([voxel_num_x*voxel_size_x, voxel_num_y*voxel_size_y])

    axis_scalings = 1/np.array([voxel_size_x,voxel_size_y])
    shift_image_origin = 1/axis_scalings * np.array([voxel_num_x,voxel_num_y]) / 2

    source_position = ag.config.system.source.position
    rotation_axis_position = ag.config.system.rotation_axis.position
    detector_position = ag.config.system.detector.position
    detector_direction_x = ag.config.system.detector.direction_x

    shift_rot_axis_origin = -rotation_axis_position

    shift_R_to_I = -shift_rot_axis_origin + shift_image_origin

    panel_num_pixels = ag.config.panel.num_pixels
    panel_pixel_size = ag.config.panel.pixel_size
    panel_origin = ag.config.panel.origin


    plt.figure(figsize=[10,10])
    # plt.figure(figsize=[20,20])
    # plt.imshow(ig.allocate(0.0).as_array(),cmap='gray')
    plt.imshow(image,cmap='gray',vmin=image.min(),vmax=image.max())
    xlims = plt.xlim()
    ylims = plt.ylim()

    color_list = list(matplotlib.colors.cnames)
    for angle_idx,horizontal_idx in idx_pairs:
        angle = ag.angles[angle_idx]
        angle *= np.pi/180
        rot_mat_angle = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        rot_source_R = rot_mat_angle.T @ (source_position + shift_rot_axis_origin)

        rot_panel_center_R = rot_mat_angle.T @ (detector_position + shift_rot_axis_origin)
        rot_panel_center = rot_panel_center_R - shift_rot_axis_origin
        rot_panel_center_I = rot_panel_center_R + shift_image_origin

        n = (rot_panel_center_R - rot_source_R) / np.linalg.norm(rot_panel_center_R - rot_source_R,2)
        n_3hat = -np.array([n[1],-n[0]])
        scale_n = np.linalg.norm(source_position, 2) + scale_proj_axis * image_max_side/2
        L = np.linalg.norm(detector_position-source_position,2)
        proj_x_R,proj_y_R = horizontal_to_proj(horizontal_idx,0.1)

        n_proj = (proj_x_R - rot_source_R) / np.linalg.norm(proj_x_R - rot_source_R,2)
        if random_colors:
            color_ray = random.choice(color_list)
            # plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'-',linewidth=ray_thickness, alpha=ray_opacity,color=color_ray)
            plt.plot(*zip(R_to_plot(proj_x_R - 3*image_max_side*n_proj),R_to_plot(proj_x_R)),'-',linewidth=ray_thickness, alpha=ray_opacity,color=color_ray)
        else:
            # plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'r-',linewidth=ray_thickness, alpha=ray_opacity)
            plt.plot(*zip(R_to_plot(proj_x_R - 3*image_max_side*n_proj),R_to_plot(proj_x_R)),'r-',linewidth=ray_thickness, alpha=ray_opacity)
    
    plt.xlim(xlims)
    plt.ylim(ylims)
    # plt.legend(loc='upper left')
    plt.show()

def make_projection_plot(
        data,image,angle_idx,horizontal_idxs,
        show_source=False,scale_proj_axis=1.2,scale_factor=1.8,
        ray_opacity=0.5,ray_thickness=1.5,scale_abs=0.3,
        random_colors=True,max_type='absorption',axis_color='black',
        bound_color='red', scale_factor_x=None,scale_factor_y=None,lims=None,
        show=True
    ):
    # currently assumes orthogonality of center ray with detector panel. that is detector_direction_x = [1,0]
    def R_to_plot(vec):
        return axis_scalings*(vec + shift_R_to_I)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    angle = ag.angles[angle_idx]

    ### initializations
    voxel_num_x, voxel_num_y, voxel_num_z = ig.voxel_num_x, ig.voxel_num_y, ig.voxel_num_z
    voxel_size_x, voxel_size_y, voxel_size_z = ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_z
    center_x, center_y, center_z = ig.center_x, ig.center_y, ig.center_z
    image_max_side = np.max([voxel_num_x*voxel_size_x, voxel_num_y*voxel_size_y])

    axis_scalings = 1/np.array([voxel_size_x,voxel_size_y])
    shift_image_origin = 1/axis_scalings * np.array([voxel_num_x,voxel_num_y]) / 2

    source_position = ag.config.system.source.position
    rotation_axis_position = ag.config.system.rotation_axis.position #+ np.array([20000,0])
    print(source_position, rotation_axis_position)
    detector_position = ag.config.system.detector.position
    detector_direction_x = ag.config.system.detector.direction_x

    shift_rot_axis_origin = -rotation_axis_position
    angle *= np.pi/180
    rot_mat_angle = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    shift_R_to_I = -shift_rot_axis_origin + shift_image_origin

    panel_num_pixels = ag.config.panel.num_pixels
    panel_pixel_size = ag.config.panel.pixel_size
    panel_origin = ag.config.panel.origin

    rot_source_R = rot_mat_angle.T @ (source_position + shift_rot_axis_origin)
    rot_source = rot_source_R - shift_rot_axis_origin
    rot_source_I = rot_source_R + shift_image_origin

    ### line towards detector position
    rot_panel_center_R = rot_mat_angle.T @ (detector_position + shift_rot_axis_origin)
    rot_panel_center = rot_panel_center_R - shift_rot_axis_origin
    rot_panel_center_I = rot_panel_center_R + shift_image_origin

    # theta,rho = calc_line_params_from_two_points(rot_source_R,rot_panel_pixel_R)
    # rot_mat_theta = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    n = (rot_panel_center_R - rot_source_R) / np.linalg.norm(rot_panel_center_R - rot_source_R,2)
    n_3hat = -np.array([n[1],-n[0]])
    scale_n = np.linalg.norm(source_position, 2) + scale_proj_axis * image_max_side/2
    L = np.linalg.norm(detector_position-source_position,2)
    
    ###
    def horizontal_to_proj(horizontal,scale,proj_abs=None):
        panel_pixel_position = detector_position + detector_direction_x * panel_pixel_size * ( -(panel_num_pixels[0]-1)/2 + horizontal )
        rot_panel_pixel_R = rot_mat_angle.T @ (panel_pixel_position + shift_rot_axis_origin)
        rot_panel_pixel = rot_panel_pixel_R - shift_rot_axis_origin
        rot_panel_pixel_I = rot_panel_pixel_R + shift_image_origin

        H = np.linalg.norm(panel_pixel_position - detector_position, 2)
        scale_n_3hat = H/L * scale_n
        if horizontal < (panel_num_pixels[0] // 2):
            scale_n_3hat *= (-1)
    
        # proj_x is "x-axis" of fan-beam projection plot
        proj_x_R = rot_source_R + scale_n*n + scale_n_3hat*n_3hat
        if scale == None:
            proj_y_R = None
        else:
            if not proj_abs:
                proj_abs = data.as_array()[angle_idx,horizontal]
            # proj_y_R = proj_x_R + 0.1*image_max_side * n
            proj_y_R = proj_x_R + scale*image_max_side * proj_abs/max_abs * n

        # print(f'Horizontal {horizontal}: {proj_x_R,proj_y_R}')
        return proj_x_R,proj_y_R

    if show:
        plt.figure(figsize=[10,10])
        # plt.figure(figsize=[20,20])
    # plt.imshow(ig.allocate(0.0).as_array(),cmap='gray')
    plt.imshow(image.as_array(),cmap='gray',vmin=image.min(),vmax=image.max())
    xlims = plt.xlim()
    ylims = plt.ylim()

    # plt.plot(*zip(R_to_plot(rot_panel_center_R - panel_num_pixels[0]//2*n_3hat),R_to_plot(rot_panel_center_R + panel_num_pixels//2*n_3hat)),'k-')#,linewidth=5)

    # max_abs = data.as_array().max()
    max_abs = np.max(data.as_array()[angle_idx,horizontal_idxs])
    print(f'Max {max_type} for projection: {np.max(data.as_array()[angle_idx,:])}')
    plot_xs = []
    plot_ys = []
    color_abs = 'magenta'
    color_list = list(matplotlib.colors.cnames)
    color_list = [_ for _ in color_list if _ not in ['cyan',axis_color,bound_color]]

    # color_cycle = itertools.cycle(color_list)
    for horizontal in horizontal_idxs:
        proj_x_R,proj_y_R = horizontal_to_proj(horizontal,scale=scale_abs)

        if random_colors:
            color_ray = random.choice(color_list)
            # color_ray = next(color_cycle)
            color_abs = color_ray
            plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'-',linewidth=ray_thickness, alpha=ray_opacity,color=color_ray)
            plt.plot(*zip(R_to_plot(proj_x_R),R_to_plot(proj_y_R)),'-',linewidth=ray_thickness,color=color_abs)
        else:
            plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'r-',linewidth=ray_thickness, alpha=ray_opacity)
            plt.plot(*zip(R_to_plot(proj_x_R),R_to_plot(proj_y_R)),'b-',linewidth=ray_thickness,color=color_abs)

        # print(f'Magenta: {R_to_plot(proj_x_R),R_to_plot(proj_y_R)}')
        plot_xs.append(R_to_plot(proj_y_R)[0])
        plot_ys.append(R_to_plot(proj_y_R)[1])
        # plt.legend(loc='upper left')

    # plt.plot(plot_xs,plot_ys,'-o',color='cyan',markersize=2,label=f'Maximum visible absorption: {max_abs:.3f}')
    plt.plot(plot_xs,plot_ys,'-o',color='cyan',markersize=2,label=f'Absorption values')

    left_end,_ = horizontal_to_proj(0-0.5,scale=None)
    right_end,_ = horizontal_to_proj(panel_num_pixels[0]+0.5,scale=None)
    plt.plot(*zip(R_to_plot(left_end),R_to_plot(right_end)),'--',color=axis_color,label=f'Absorption = {0*max_abs:.2f}')

    ### upper bar
    max_pad = 1.1
    _,top_left_end = horizontal_to_proj(0-0.5,scale=scale_abs,proj_abs=1.1*max_abs)
    _,top_right_end = horizontal_to_proj(panel_num_pixels[0]+0.5,scale=scale_abs,proj_abs=1.1*max_abs)
    plt.plot(*zip(R_to_plot(top_left_end),R_to_plot(top_right_end)),'--',color=bound_color,label=f'Absorption = {max_pad*max_abs:.2f}')
    ###

    # Calculate new limits
    if not scale_factor_x: scale_factor_x = scale_factor
    if not scale_factor_y: scale_factor_y = scale_factor
    new_xlims = ((xlims[0] + xlims[1]) / 2 - (xlims[1] - xlims[0]) / 2 * scale_factor_x, 
                (xlims[0] + xlims[1]) / 2 + (xlims[1] - xlims[0]) / 2 * scale_factor_x)

    new_ylims = ((ylims[0] + ylims[1]) / 2 - (ylims[1] - ylims[0]) / 2 * scale_factor_y, 
                (ylims[0] + ylims[1]) / 2 + (ylims[1] - ylims[0]) / 2 * scale_factor_y)
    if lims is not None:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    elif not show_source:
        plt.xlim(new_xlims)
        plt.ylim(new_ylims)
    
    # plt.xlim(xlims)
    # plt.ylim(ylims)
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    if show:
        plt.show()


    print(n_3hat)
    return
    # return axis_scalings * rot_source_I, axis_scalings * rot_panel_pixel_I

def calculate_line_parameters(ag,ig,angle,horizontal_idx):
    # image geometry
    voxel_num_x = ig.voxel_num_x
    voxel_num_y = ig.voxel_num_y
    voxel_num_z = ig.voxel_num_z
    voxel_size_x = ig.voxel_size_x
    voxel_size_y = ig.voxel_size_y
    voxel_size_z = ig.voxel_size_z
    center_x = ig.center_x
    center_y = ig.center_y
    center_z = ig.center_z

    # in order to do the rotation of the chosen angle we shift
    # so rotation_axis_position is at the center of coordinate system
    # because we can then in this reference multiply by a rotation matrix

    # image plot has voxel_size_* all 1 and origin in panel_origin which is usually 'bottom-left'
    # therefore scale and shift all other quantities. first scale and shift to rotation origin but wait till end with the image origin shifting
    # subscript I for image origin and R for rotation axis origin
    axis_scalings = 1/np.array([voxel_size_x,voxel_size_y])
    shift_image_origin = 1/axis_scalings * np.array([voxel_num_x,voxel_num_y]) / 2

    # system configuration (scaled and in original reference)
    source_position = ag.config.system.source.position
    rotation_axis_position = ag.config.system.rotation_axis.position
    # rotation_axis_position = [0,0]
    detector_position = ag.config.system.detector.position
    detector_direction_x = ag.config.system.detector.direction_x

    shift_rotation_origin = -rotation_axis_position
    angle *= np.pi/180
    rot_mat_angle = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    # panel configuration
    panel_num_pixels = ag.config.panel.num_pixels
    panel_pixel_size = ag.config.panel.pixel_size
    panel_origin = ag.config.panel.origin

    panel_pixel_position = detector_position + detector_direction_x * panel_pixel_size * ( -(panel_num_pixels[0]-1)/2 + horizontal_idx )

    rotated_source_position = rot_mat_angle.T @ (source_position + shift_rotation_origin) - shift_rotation_origin
    rotated_panel_pixel_position = rot_mat_angle.T @ (panel_pixel_position + shift_rotation_origin) - shift_rotation_origin

    rotated_source_position_I = rotated_source_position + shift_image_origin
    rotated_panel_pixel_position_I = rotated_panel_pixel_position + shift_image_origin
    # 2 points uniquely define a line. we now know: source_position and panel
    return axis_scalings * rotated_source_position_I, axis_scalings * rotated_panel_pixel_position_I


def choose_random_rows(arr,count):
    nrows = arr.shape[0]
    n = min(count,nrows)
    choose_idxs = random.sample(range(nrows),n)
    return arr[choose_idxs,:]


def sim_ebhc():
    # factor = 500
    factor = 1
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    # ig.voxel_size_x *= 0.1/2; ig.voxel_size_y *= 0.1/2
    ## factor = 0.05
    ## ig = ImageGeometry(voxel_num_x=ig.voxel_num_x,voxel_num_y=ig.voxel_num_y,voxel_size_x=factor*ig.voxel_size_x,voxel_size_y=factor*ig.voxel_size_y)
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')

    im = io.imread(os.path.join(base_dir,'test_images/X20_imi.png'))[::-1,:,0] > 0
    image = ImageData(array=im.astype('float32'), geometry=ig)
    show2D(image)

### Call other functions
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0.00)
    num_bins = bin_centers.size
    # plt.semilogy(bin_centers, mu(bin_centers))
    # plt.xlabel('Energy (keV)')
    # plt.ylabel('mu (mm^{-1})')
    # plt.title('Attenuation coefficients narrowed down to spectrum')
    # plt.show()

    # path_lengths = A.direct(image)
    # plt.hist(path_lengths.as_array().flatten(),bins=100)
    # plt.yscale('log')
    # corrections,corrections2 = staircase_bhc(20)

### Simulate BH effect
    d = A.direct(image)
    d = d.as_array()/factor

    # plt.hist(d.flatten(),bins=100)
    # plt.show()

    I = np.zeros(d.shape, dtype='float64')
    I0 = 0
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        I += I0_E * np.exp(-mu(E)*d) + 5*1e-4
        # I += I0_E * np.exp(-mu(E)*d) + 1e-40

    I = np.array(I,dtype='float32')
    b = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    # b = data.as_array()

    ##
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    print(f'mu_eff: {mu_eff}')
    # b_mono = -np.log(np.exp(-mu_eff*d))
    b_mono = d*mu_eff
    # b_mono = AcquisitionData(array=np.exp(-b_mono), geometry=ag)
    b_mono = AcquisitionData(array=b_mono, geometry=ag)

### Reconstructions
    ## raw
    cap = None
    # cap = b.max()
    b_noisy = skimage.util.random_noise(b.as_array(), mode='gaussian',clip=False, mean=0, var=1/(1e4*I))
    b_noisy = np.clip(a=b_noisy, a_min=0, a_max=cap)
    b_noisy = AcquisitionData(array=np.array(b_noisy,dtype='float32'), geometry=ag)

    # plt.scatter(A.direct(image).as_array().flatten()/factor, b_noisy.as_array().flatten(), alpha=0.2, color='black', label='Observed',s=3)
    plt.scatter(A.direct(image).as_array().flatten()/factor, b_noisy.as_array().flatten(), alpha=0.2, color='black', label='Observed',s=3)
    plt.title('b_noisy vs true path lengths')
    plt.xlabel('mm')
    
    recon = FDK(b_noisy, image_geometry=ig).run(verbose=0)
    show2D(recon, title='recon')
    # show1D(recon, slice_list=[('horizontal_y', recon.shape[0]//2)])
    # plt.plot(recon.as_array()[recon.shape[0]//2,:])
    # plt.show()

    # ## mono
    # recon_mono = FDK(b_mono, image_geometry=ig).run(verbose=0)
    # show2D(recon_mono, title='recon')

    # idx_chosen_angle = 40; horizontal_idxs = np.arange(300,700)
    # make_projection_plot(
    #     path_lengths,segmented,idx_chosen_angle,horizontal_idxs,scale_abs=0.05,
    #     scale_proj_axis=0.9,ray_opacity=0.3,ray_thickness=1*3,scale_factor=1.1,axis_color='magenta')

    # show2D(prewitt())
    tau = 2.5
    b_cap = (b_noisy < tau) * b_noisy.as_array() + (b_noisy >= tau) * 1.5
    b_cap = AcquisitionData(array=np.array(b_cap,dtype='float32'),geometry=ag)
    recon_cap = FDK(b_cap, image_geometry=ig).run(verbose=0)
    show2D(recon_cap, title='recon')
    
    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
    
    for i in range(1):
        path_lengths = A.direct(segmented)/factor
        data_bhc, recon_bhc = bhc(path_lengths, b_noisy, f_mono, f_poly, num_bins=25)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
    
    _ = avg_out(path_lengths,data_bhc,10)
    _ = avg_out(path_lengths,b_noisy,10)
    
    path_lengths = A.direct(segmented)/factor

def detect_tails():
    # data = load_centre('X12-X22_cor.pkl')
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
    
    for i in range(2):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
    
    _ = avg_out(path_lengths,data_bhc,10)
    _ = avg_out(path_lengths,data,10)
    
    path_lengths = A.direct(segmented)

###
    #(377, 488)
    detector_num_pixels = data.get_dimension_size('horizontal')
    chosen_angle = 0
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,300)).astype(int)
    # horizontal_idxs = [488]
    # horizontal_idxs = np.arange(300,700)
    # horizontal_idxs = np.arange(498,500)
    # horizontal_idxs = [498,499,500,501,502]
    # segmented2 = ImageData(np.array(segmented.as_array()[::-1,:],dtype='float32'),geometry=ig)

    idx_chosen_angle = 1171; horizontal_idxs = [504]
    make_projection_plot(
        path_lengths,segmented,idx_chosen_angle,horizontal_idxs,scale_abs=0.05,
        scale_proj_axis=0.9,ray_opacity=0.3,ray_thickness=1*3,scale_factor=1.1,axis_color='magenta')

    show_lines2(data,segmented.as_array(),tail_idxs)
###
    # path_tau,data_tau = 7,0.3
    path_tau,data_tau = 2,0.3
    tail_idxs = np.array(np.where((path_lengths > path_tau) & (data < data_tau))).T
    tail_idxs = np.array(np.where((path_lengths > 2) & (data < 0.3))).T
    tail_idxs = np.array(np.where((path_lengths < 1) & (data > 2.2))).T
    tail_idxs = np.array(np.where((path_lengths < 0.1) & (data > 2.5))).T
    tail_idxs = np.array(np.where((path_lengths < 0.1) & (data > 2))).T
    tail_idxs = np.array(np.where((path_lengths < 0.01) & (data > 1.5))).T
    show_lines2(data,segmented.as_array(),tail_idxs)
    # show_lines(data,segmented.as_array()[:,::-1],tail_idxs[:5,:])

    rows = 400
    test_idxs = np.zeros((rows,2),dtype=int)
    for i in range(rows):
        test_idxs[i,:] = [idx_chosen_angle,i+300]
    
    show_lines(data,segmented.as_array(),test_idxs)
    # show_lines(data,segmented.as_array()[:,::-1],choose_random_rows(tail_idxs,20))

    tail_idxs = np.array(np.where((path_lengths > 2.5) & (path_lengths < 6) & (data > 2.5))).T
    show_lines2(data,segmented.as_array(),choose_random_rows(tail_idxs,40),ray_opacity=0.3)

    tail_idxs = np.array(np.where(path_lengths.as_array() > 15)).T
    show_lines2(data,segmented.as_array(),choose_random_rows(tail_idxs,100))


    path_tau,data_tau = 15,10
    tail_idxs = np.array(np.where((path_lengths > path_tau) & (data < data_tau))).T
    show_lines(data,segmented.as_array()[:,::-1],choose_random_rows(tail_idxs,50))

    show_lines(data,segmented.as_array(),[[377,510]])



def main():
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X12-X22_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    # data_mask = data > 0.25
    # data = data * data_mask

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

### ray path
    show_geometry(ag, ig)
    print(ig); print(ag); print(ag.dimension_labels)
    # pixel_intersections = A.direct(ig.allocate(1.0))
    # ag_chosen_angle = ag.copy()
    # ag_chosen_angle.set_angles([ag.angles[idx_chosen_angle]])
    # plt.plot(pixel_intersections)

    ###
    detector_num_pixels = data.get_dimension_size('horizontal')
    # chosen_angle = 90
    # chosen_angle = 45
    chosen_angle = 270
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    # make_projection_plot(data,None,0,np.round(np.linspace(0,detector_num_pixels,10)))
    # make_projection_plot(data,recon,idx_chosen_angle,np.round(np.linspace(0,detector_num_pixels-1,10)).astype(int))
    # make_projection_plot(data,recon,idx_chosen_angle,np.round(np.linspace(0,detector_num_pixels-1,20)).astype(int))

    ###
    if False:
        horizontal_idx = detector_num_pixels//2+600
        # source,panel_pixel = calculate_line_parameters(ag,ig,chosen_angle,detector_num_pixels//2-10)
        source,panel_pixel = calculate_line_parameters(ag,ig,ag.angles[chosen_angle],horizontal_idx)

        plt.figure(figsize=[10,10])
        # plt.imshow(recon.as_array().T[::-1,:],cmap='gray')
        plt.imshow(np.flip(recon.as_array().T,0),cmap='gray')
        xlims = plt.xlim()
        ylims = plt.ylim()
        plt.plot(*zip(source,panel_pixel),'r-', label=f'absorption: {data.as_array()[idx_chosen_angle,horizontal_idx]}')
        plt.legend(loc='upper left')

        # scale_factor = 2
        # # Calculate new limits
        # new_xlims = ((xlims[0] + xlims[1]) / 2 - (xlims[1] - xlims[0]) / 2 * scale_factor, 
        #             (xlims[0] + xlims[1]) / 2 + (xlims[1] - xlims[0]) / 2 * scale_factor)

        # new_ylims = ((ylims[0] + ylims[1]) / 2 - (ylims[1] - ylims[0]) / 2 * scale_factor, 
        #             (ylims[0] + ylims[1]) / 2 + (ylims[1] - ylims[0]) / 2 * scale_factor)
        # plt.xlim(new_xlims)
        # plt.ylim(new_ylims)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.show()


    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    # tau = 0.2
    # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(2):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        # segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        tau = 0.2
        segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')

    F = LeastSquares(A, data_bhc)
    G = IndicatorBox(lower=0.0)
    # x0 = recon_bhc
    x0 = ig.allocate(0.1)
    myFISTANN = FISTA(f=F, g=G, initial=x0, 
                      max_iteration=1000,
                      update_objective_interval = 10)
    myFISTANN.run(50, verbose=1)
    show2D(myFISTANN.solution,'fistaNN recon_bhc')

    # myFISTANN.run(500, verbose=1)
    # show2D(myFISTANN.solution,'fistaNN recon_bhc')

def test_proj_plot_X16():
    data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    recon = FDK(data).run()
    detector_num_pixels = data.get_dimension_size('horizontal')
    chosen_angle = -3.2
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))

    # horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,20)).astype(int)
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,100)).astype(int)
    # horizontal_idxs = np.round(np.linspace(800,1200,20)).astype(int)
    # make_projection_plot(data,recon,idx_chosen_angle,horizontal_idxs)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.1,
        scale_proj_axis=0.13,ray_opacity=0.7,ray_thickness=1,scale_factor=0.9
    )

    chosen_angle = -3.2+90
    # chosen_angle = -3.2
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,100)).astype(int)
    horizontal_idxs = np.arange(1010-10,1125+10)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.1,
        scale_factor_y=0.4
    )
    ###
    # chosen_angle = -3.2+90+180 -0
    # chosen_angle = -3.2+70
    chosen_angle = -3.2+60+180
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,500)).astype(int)
    # horizontal_idxs = np.arange(1010,1125)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.3
    )

    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.14,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.3
    )

    show1D(data,slice_list=[('angle', 300)])

def test_proj_plot_X20():
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    recon = FDK(data).run()
    detector_num_pixels = data.get_dimension_size('horizontal')
    chosen_angle = 0
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))

    # horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,20)).astype(int)
    # horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,100)).astype(int)
    # horizontal_idxs = np.round(np.linspace(800,1200,20)).astype(int)
    # make_projection_plot(data,recon,idx_chosen_angle,horizontal_idxs)
    # make_projection_plot(
    #     data,recon,idx_chosen_angle,horizontal_idxs,
    #     scale_proj_axis=0.3,ray_opacity=0.7,ray_thickness=1,scale_factor=1.1)

    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,300)).astype(int)
    # horizontal_idxs = np.arange(300,700)
    # horizontal_idxs = np.arange(498,500)
    # horizontal_idxs = [498,499,500,501,502]
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.2,
        scale_proj_axis=0.3,ray_opacity=0.3,ray_thickness=1,scale_factor=0.9)
    
    ###
    chosen_angle = 180
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.arange(300,700)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.05,
        scale_proj_axis=0.08,ray_opacity=0.2,ray_thickness=1,scale_factor=0.3)
    
    ###
    idx_chosen_angle = 500
    horizontal_idxs = np.arange(600,700)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.05,
        scale_proj_axis=-0.4,ray_opacity=0.2,ray_thickness=1,scale_factor=0.8)

def ebhc_TV():
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X12-X22_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    # tau = 0.2
    # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**2
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(2):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        # segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        tau = 0.2
        segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')

    # F = LeastSquares(A, data_bhc)
    # G = IndicatorBox(lower=0.0)
    # # x0 = recon_bhc
    # x0 = ig.allocate(0.1)
    # myFISTANN = FISTA(f=F, g=G, initial=x0, 
    #                   max_iteration=1000,
    #                   update_objective_interval = 10)
    # myFISTANN.run(50, verbose=1)
    # show2D(myFISTANN.solution,'fistaNN recon_bhc')

    alpha = 30
    F = LeastSquares(A, data_bhc)
    TV = FGP_TV(alpha=alpha, nonnegativity=True, device='gpu')
    #fista_TV = FISTA(initial=recont, f=F, g=TV, max_iteration=1000, update_objective_interval=10)
    fista_TV = FISTA(initial=ig.allocate(0.0), f=F, g=TV, max_iteration=1000, update_objective_interval=10)
    fista_TV.run(100, verbose=1)
    show2D(fista_TV.solution,title=f'EBHC + NN + TV with alpha={alpha}')

def conservation_principle():
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    # tau = 0.5
    # proj_mask = data.as_array() > tau
    # proj_sums = np.sum(proj_mask * data.as_array(), axis=1)[:,None]
    # proj_counts = np.sum(proj_mask, axis=1)[:,None]
    # proj_weights = proj_mask * data.as_array() / proj_sums
    # data_cons = proj_mask*data.as_array() + proj_weights*(np.max(proj_sums) - proj_sums)
    # data_cons = AcquisitionData(array=np.array(data_cons, dtype='float32'), geometry=ag)

    proj_sums = np.sum(data.as_array(), axis=1)[:,None]
    proj_weights = data.as_array() / proj_sums
    data_cons = data.as_array() + proj_weights*(np.max(proj_sums) - proj_sums)
    data_cons = AcquisitionData(array=np.array(data_cons, dtype='float32'), geometry=ag)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(data.as_array().flatten(),bins=100)
    ax[0].set_yscale('log')
    ax[0].set_title('hist of data')
    ax[1].hist(data_cons.as_array().flatten(),bins=100)
    ax[1].set_yscale('log')
    ax[1].set_title('hist of data_cons')

    # data_cons = data.as_array() + proj_mask*(np.max(proj_sums) - proj_sums) / proj_counts
    # data_cons = data.as_array() + proj_mask*(np.max(proj_sums) - proj_sums) / data.shape[1]

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    recon_cons = FDK(data_cons).run()
    show2D([recon,recon_cons],size=(16,8),title=['recon','recon_cons'])

    segmented = clip_otsu_segment(recon_cons, ig, title='Otsu segmentation on initial FDK')
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
    
    for i in range(1):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data_cons, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
    
    _ = avg_out(path_lengths,data_bhc,10)
    _ = avg_out(path_lengths,data_cons,10)
    
    path_lengths = A.direct(segmented)

def ebhc_backproj_X20():
    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    ###
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    # tau = 0.2
    # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(1):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        data_bhc = (data + data_bhc)/2
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        # tau = 0.2
        # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')

    data = data_bhc
    ###
    tau = 0.3
    air_proj_mask = 1000*np.array(data.as_array() < 0.3, dtype=np.float32)
    air_proj_mask = AcquisitionData(array=air_proj_mask, geometry=ag)

    # fdk = FDK(data)
    # recon = FDK(data, filter=np.ones(2**11,dtype=np.float32)).run(verbose=0)
    # show2D(recon).save(os.path.join(bfig_dir,'backprojection.png'))

    fft_order = 11
    recon_air = FDK(air_proj_mask, filter=np.ones(2**fft_order,dtype=np.float32)).run(verbose=0)
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
    myFISTAmask.run(70, verbose=1)
    show2D(myFISTAmask.solution,'Convex hull FISTA').save(os.path.join(bfig_dir,'fista_hull.png'))
    show2D(myFISTAmask.solution,fix_range=(0,0.4))

def mask_ebhc_X16():
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ###
    tau = -np.log(0.07)
    data = lin_interp_sino2D(data, tau)

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

    ###
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = ImageData(np.array(recon_mask,dtype=np.float32),geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(1):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        data_bhc = (data + data_bhc)/2
        show2D(recon_bhc,f'recon_bhc {i+1}')
    
    show2D(recon_bhc,f'recon_bhc {i+1}',fix_range=(-0.1,0.6))


def lin_interp_backproj_X20():
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')

    tau = -np.log(0.13)
    data_interp = lin_interp_sino2D(data, tau)

    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
    
    for i in range(1):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        # tau = 0.2
        # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')

def ebhc_backproj_X16():
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    ###
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    # tau = 0.2
    # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(2):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        # tau = 0.2
        # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')

    data = data_bhc
    ###
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
    myFISTAmask.run(70, verbose=1)
    show2D(myFISTAmask.solution,'Convex hull FISTA').save(os.path.join(bfig_dir,'fista_hull.png'))
    show2D(myFISTAmask.solution,fix_range=(0,1.25))

def backproj_ebhc_X16():
    file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    ###
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
    myFISTAmask.run(70, verbose=1)
    show2D(myFISTAmask.solution,'Convex hull FISTA').save(os.path.join(bfig_dir,'fista_hull.png'))
    show2D(myFISTAmask.solution,fix_range=(0,1.25))

    ###
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    show2D(recon)

    segmented = clip_otsu_segment(recon, ig, title='Otsu segmentation on initial FDK')
    # tau = 0.2
    # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        return a*x**3
        # return a*np.sqrt(x+0.5)
        # return a*np.exp(x)-1
    
    for i in range(2):
        path_lengths = A.direct(segmented)
        data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=10)
        show2D(recon_bhc,f'recon_bhc {i+1}')
        segmented = clip_otsu_segment(recon_bhc, ig, title=f'Otsu segmentation on BHC {i+1}')
        # tau = 0.2
        # segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
        show2D(segmented, title=f'{i+1}')


if __name__ == "__main__":
    # main()
    # test_proj_plot_X16()
    # test_proj_plot_X20()
    None