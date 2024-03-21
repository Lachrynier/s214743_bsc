import sys
sys.path.append('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc')
import sim_main
from sim_main import fun_attenuation, generate_spectrum
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
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