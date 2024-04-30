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

###

def avg_out(path_lengths, data, num_bins):
    x = np.array(path_lengths.as_array().flatten())
    y = np.array(data.as_array().flatten())
    bin_edges = np.linspace(x.min(), x.max(), num_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(x, bin_edges)
    bin_means = np.array([y[bin_indices == i].mean() for i in range(1, len(bin_edges))])

    scatter_subset = np.random.choice(len(x), 500000, replace=False)

    plt.figure()
    plt.scatter(x[scatter_subset], y[scatter_subset], alpha=0.1, color='black', label='Observed',s=2)
    # plt.plot(bin_centers, bin_means, color='red', label='Averaged out')
    plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    plt.xlabel('Path lengths (mm)')
    plt.ylabel('Absorption')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(bfig_dir,'avg_out_plot.png'))
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

    plt.figure()
    plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    plt.plot(xx, y_mono, label='Mono fit')
    plt.plot(x_poly, yy, label='Poly fit')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(bfig_dir,'bin_fits.png'))
    plt.show()

    # gglin = g(yylin,popt_g[0],popt_g[1])
    yy_bhc = f_mono(x_poly, *popt_mono)

    plt.figure()
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
    # show2D(segmented, title=title)
    return segmented

def main():
    path = "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/raw_data_3DIM/VKH8206-X20 [2022-02-08 10.27.26]/VKH8206-X20_recon.xtekct"
    reader = NikonDataReader(file_name=path)
    data = reader.read()
    data = TransmissionAbsorptionConverter()(data)
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run(verbose=0)
    segmented = clip_otsu_segment(recon, ig)
    
    def f_mono(x, a):
        return a*x
    def f_poly(x, a):
        # return a*x**3
        return a*x**5
    
    path_lengths = A.direct(segmented)
    data_bhc, recon_bhc = bhc(path_lengths, data, f_mono, f_poly, num_bins=25)

    file_path = os.path.join(base_dir, 'bjobs/X20_bhc_recon_full')
    with open(file_path, 'wb') as file:
        pickle.dump(recon_bhc, file)
    return

    vert_slice = recon.get_dimension_size('vertical')//2
    hori_y_slice = recon.get_dimension_size('horizontal_y')//2
    hori_x_slice = [recon.get_dimension_size('horizontal_x')//2, 400][1]
    bfig_dir = os.path.join(base_dir,'bjobs/figs/bhc_deg5')
    for idx in np.arange(start=200, stop=850, step=100, dtype=np.int32):
        show2D(recon, title=f'vertical slice {idx}', slice_list=[('vertical', idx)]).save(os.path.join(bfig_dir,f'recon_v_idx{idx}.png'))
        show2D(recon, title=f'horizontal_y slice {idx}', slice_list=[('horizontal_y', idx)]).save(os.path.join(bfig_dir,f'recon_y_idx{idx}.png'))
        show2D(recon, title=f'horizontal_x slice {idx}', slice_list=[('horizontal_x', idx)]).save(os.path.join(bfig_dir,f'recon_x_idx{idx}.png'))
        show2D(recon_bhc, title=f'vertical slice {idx}', slice_list=[('vertical', idx)]).save(os.path.join(bfig_dir,f'recon_bhc_v_idx{idx}.png'))
        show2D(recon_bhc, title=f'horizontal_y slice {idx}', slice_list=[('horizontal_y', idx)]).save(os.path.join(bfig_dir,f'recon_bhc_y_idx{idx}.png'))
        show2D(recon_bhc, title=f'horizontal_x slice {idx}', slice_list=[('horizontal_x', idx)]).save(os.path.join(bfig_dir,f'recon_bhc_x_idx{idx}.png'))
        
    # show2D(recon, slice_list=[('vertical', vert_slice)]).save(os.path.join(bfig_dir,'recon_v.png'))
    # show2D(recon, slice_list=[('horizontal_y', hori_y_slice)]).save(os.path.join(bfig_dir,'recon_y.png'))
    # show2D(recon, slice_list=[('horizontal_x', hori_x_slice)]).save(os.path.join(bfig_dir,'recon_x.png'))
    # show2D(recon_bhc, slice_list=[('vertical', vert_slice)]).save(os.path.join(bfig_dir,'recon_bhc_v.png'))
    # show2D(recon_bhc, slice_list=[('horizontal_y', hori_y_slice)]).save(os.path.join(bfig_dir,'recon_bhc_y.png'))
    # show2D(recon_bhc, slice_list=[('horizontal_x', hori_x_slice)]).save(os.path.join(bfig_dir,'recon_bhc_x.png'))
    
def test():
    path = "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/raw_data_3DIM/VKH8206-X20 [2022-02-08 10.27.26]/VKH8206-X20_recon.xtekct"
    reader = NikonDataReader(file_name=path)
    data = reader.read()
    data = TransmissionAbsorptionConverter()(data)
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    recon = FDK(data).run(verbose=0)
    print(recon)

if __name__ == '__main__':
    main()
    # test()
    None