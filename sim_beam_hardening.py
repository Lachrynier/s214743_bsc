# import matplotlib
# matplotlib.use('TkAgg')

import os
import numpy as np
import math
import scipy as sp
from scipy import interpolate
from numpy.linalg import solve
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from time import time

# A = np.zeros((M,M))
# A = sp.sparse.lil_matrix((M,M))
# u = solve(A,b)
# A = A.tocsc()
# B = sp.sparse.linalg.splu(A)
# u = B.solve(b)

print(os.getcwd())
if os.getcwd() == "/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev":
    os.chdir('analysis/s214743_bsc')
    print(os.getcwd())

def calc_intersection_length(ray_direction, rho, im_width, im_pixel, coordinate_shift):
    # assume theta in [0,Pi]
    # checks if \ell_{\theta,\rho} intersects im_pixel (row-major) in im
    # convert im_pixel to 2D index
    i,j = divmod(im_pixel, im_width)
    # coordinates of the edges of pixel
    x_left = i + coordinate_shift - 0.5
    x_right = x_left + 1
    y_bottom = j + coordinate_shift - 0.5
    y_top = y_bottom + 1
    
    # ray_intersects_pixel = 0
    # ray_x > 1 corresponds to theta < pi/2
    if ray_direction[0] > 0:
        rho_top_right = x_right * ray_direction[0] + y_top * ray_direction[1]
        rho_bottom_left = x_left * ray_direction[0] + y_bottom * ray_direction[1]
        if (rho_top_right < rho) or (rho < rho_bottom_left):
            return None
    else:
        rho_top_left = x_left * ray_direction[0] + y_top * ray_direction[1]
        rho_bottom_right = x_right * ray_direction[0] + y_bottom * ray_direction[1]
        if (rho_top_left < rho) or (rho < rho_bottom_right):
            return None
    

    if np.sum(np.isclose(ray_direction,0)) > 0:
        return 1
    
    x_crossings = np.zeros(2)
    y_crossings = np.zeros(2)
    y_left = (rho - x_left*ray_direction[0]) / ray_direction[1]
    y_right = (rho - x_right*ray_direction[0]) / ray_direction[1]
    if y_left > y_top:
        y_crossings[0] = y_top
        x_crossings[0] = (rho - y_top*ray_direction[1]) / ray_direction[0]
    elif y_left < y_bottom:
        y_crossings[0] = y_bottom
        x_crossings[0] = (rho - y_bottom*ray_direction[1]) / ray_direction[0]
    else:
        y_crossings[0] = y_left
        x_crossings[0] = x_left

    if y_right > y_top:
        y_crossings[1] = y_top
        x_crossings[1] = (rho - y_top*ray_direction[1]) / ray_direction[0]
    elif y_right < y_bottom:
        y_crossings[1] = y_bottom
        x_crossings[1] = (rho - y_bottom*ray_direction[1]) / ray_direction[0]
    else:
        y_crossings[1] = y_right
        x_crossings[1] = x_right
    
    return np.linalg.norm([x_crossings[1]-x_crossings[0], y_crossings[1]-y_crossings[0]])

def construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width):
    # assume square image where pixels are of units 1.0 x 1.0
    # assume rotation axis at the enter of the image
    # x axis vertically down and y axis horizontally right. origin at geometric center of image.
    # lines parametrized by (theta,rho)
    # where theta is rotation angle (angle 0 points in the y-axis direction) and rho is distance from origin

    # from geometrical center of image to center of im[0,0]
    coordinate_shift = -(im_width-1)/2

    # symmetric panel and detector rotation axis is centered at
    panel_pixel_centers = panel_cell_length * ( -(panel_num_cells-1)/2 + np.arange(0, panel_num_cells) )

    num_angles = len(angles)
    im_size = im_width**2
    # A = sp.sparse.csr_matrix((panel_num_cells * num_angles, im_size))
    # A = sp.sparse.lil_matrix((panel_num_cells*num_angles, im_size))
    A = np.zeros((panel_num_cells*num_angles, im_size))
    for proj_idx in range(num_angles):
        theta = np.deg2rad(angles[proj_idx])
        # ray_direction = [-np.sin(angle_rad), np.cos(angle_rad)]
        ray_direction = [np.cos(theta), np.sin(theta)]
        # print(ray_direction)
        idx_shift = proj_idx * panel_num_cells
        for ray_idx in range(panel_num_cells):
            rho = panel_pixel_centers[ray_idx]
            # print(rho)
            for im_pixel in range(im_size):
                intersection_length = calc_intersection_length(ray_direction, rho, im_width, im_pixel, coordinate_shift)
                if intersection_length != None:
                    A[idx_shift + ray_idx, im_pixel] = intersection_length
    
    return A

import numpy as np
import concurrent.futures
import itertools

def worker(proj_idx, angles, panel_num_cells, im_width, coordinate_shift, panel_pixel_centers, im_size):
    theta = np.deg2rad(angles[proj_idx])
    ray_direction = [np.cos(theta), np.sin(theta)]
    idx_shift = proj_idx * panel_num_cells
    result = np.zeros((panel_num_cells, im_size))

    for ray_idx in range(panel_num_cells):
        rho = panel_pixel_centers[ray_idx]
        for im_pixel in range(im_size):
            intersection_length = calc_intersection_length(ray_direction, rho, im_width, im_pixel, coordinate_shift)
            if intersection_length is not None:
                result[ray_idx, im_pixel] = intersection_length
    return idx_shift, result

def parallelized_construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width):
    coordinate_shift = -(im_width-1)/2
    panel_pixel_centers = panel_cell_length * ( -(panel_num_cells-1)/2 + np.arange(0, panel_num_cells) )
    num_angles = len(angles)
    im_size = im_width**2
    A = np.zeros((panel_num_cells*num_angles, im_size))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = ((proj_idx, angles, panel_num_cells, im_width, coordinate_shift, panel_pixel_centers, im_size) for proj_idx in range(num_angles))
        for idx_shift, result in executor.map(worker, *zip(*args)):
            A[idx_shift:idx_shift + panel_num_cells] = result

    return A

def create_circle_image(image_size, radius, center):
    # Initialize a square image with zeros (black)
    image = np.zeros((image_size, image_size))

    # Create the circle
    for x in range(image_size):
        for y in range(image_size):
            # Check if the pixel is inside the circle
            # x is now the vertical coordinate, and y is the horizontal coordinate
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < (radius+0.5) ** 2:
                image[x, y] = 1  # Set pixel value to 1 (white) inside the circle
    
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

def simulate_BH2():
    ###
    num_samples = 10**6
    mean_energy = 175  # in keV
    std_deviation = 50  # in keV
    sampled_energies = fun_xray_initial_dist(mean_energy,std_deviation).rvs(num_samples)
    num_bins = 30  # Number of bins
    E_max = 350
    bins = np.linspace(0, E_max, num_bins+1)
    hist, bin_edges = np.histogram(sampled_energies, bins=bins, density=True)
    # normalize
    hist = hist / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist, align='center', width=np.diff(bin_edges))
    plt.title("Histogram of X-ray Energy Distribution")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Probability Density")

    plt.plot(bin_centers,hist,'r.')
    plt.show()

    mu = fun_attenuation()

    ### Generate triangle test image
    im_width = 20
    ##
    # tri_width = im_width//2
    # corner = im_width//4
    ##
    tri_width = im_width
    corner = 0

    img_size = (im_width, im_width)  # Image size: 100x100 pixels
    triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
    corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
    triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
    image = np.rot90(triangle_image, 1)

    # just to make the image not white
    # image[im_width//2,im_width//2] = 2

    ###
    # angles = np.arange(0,180,0.5)
    angles = np.array([0])
    # panel_total_length = panel_num_cells*panel_cell_length

    factor = 1
    # panel_num_cells = math.ceil(np.sqrt(2)*factor*im_width)
    panel_num_cells = math.ceil(factor*im_width)
    panel_cell_length = 1/factor
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)

    d = np.dot(A, image.flatten())
    I = np.zeros(A.shape[0])
    test = 0
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = hist[i]
        mu_E = mu(E)
        I += I0_E * np.exp(-mu_E*d)
        test += I0_E
    
    print(f'should sum to 1: {test}')
    b = -np.log(I)

    E_eff = np.sum(bin_centers*hist)
    print(np.sum(hist))
    print(f'E_eff: {E_eff}')
    b_mono = -np.log(np.exp(-mu(E_eff)*d))

    # Display the image
    # plt.imshow(triangle_image, cmap='gray')
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.plot(np.arange(1,b.size+1,1),b,label='Polychromatic absorption for test image')
    plt.plot(np.arange(1,b_mono.size+1,1),b_mono,'--',label='Monochromatic absorption at effective energy')
    plt.xlabel('Pathlength')
    plt.title('Beam hardening effect for different pathlengths')
    plt.legend()
    plt.show()

def simulate_BH():
    ###
    num_samples = 10**6
    mean_energy = 175  # in keV
    std_deviation = 50  # in keV
    E_max = 350
    # mean_energy =   # in keV
    # std_deviation =   # in keV
    # std_deviation = 1000
    # E_max = 200
    sampled_energies = fun_xray_initial_dist(mean_energy,std_deviation).rvs(num_samples)
    num_bins = 50  # Number of bins
    bins = np.linspace(0, E_max, num_bins+1)
    hist, bin_edges = np.histogram(sampled_energies, bins=bins, density=True)
    # normalize
    hist = hist / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist, align='center', width=np.diff(bin_edges))
    plt.title("Histogram of X-ray Energy Distribution")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Probability Density")

    plt.plot(bin_centers,hist,'r.')
    plt.show()

    mu = fun_attenuation()

    ### Generate triangle test image
    # im_width = 40
    # tri_width = im_width//2
    # corner = im_width//4
    # img_size = (im_width, im_width)  # Image size: 100x100 pixels
    # triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
    # corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
    # triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
    # image = np.rot90(triangle_image, 1)

    ### Generate circle test image
    im_width = 40
    image = create_circle_image(im_width, radius=im_width//2.5, center=[im_width/2,im_width/2])
    image -= create_circle_image(im_width, radius=im_width//5.5, center=[im_width/2,im_width/2])
    # image += create_circle_image(im_width, radius=im_width//9, center=[im_width/2,im_width/2])
    ###
    angles = np.arange(0,180,1)
    # angles = np.array([0])
    # panel_total_length = panel_num_cells*panel_cell_length

    factor = 2
    # factor = 1.3
    # panel_num_cells = math.ceil(np.sqrt(2)*factor*im_width)
    # panel_cell_length = 1/factor
    # panel_num_cells = math.ceil(factor*im_width)
    panel_num_cells = math.ceil(0.8*factor*im_width)
    panel_cell_length = 1/factor

    t = time()
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    # A = parallelized_construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    print(f'construct A: {time()-t}')
    print(f'A.shape: {A.shape}')
    # print(A)
    # plt.spy(A)
    
    print(type(A))
    # d = np.dot(A, image.flatten())
    d = A @ image.flatten() #*0.1#* 1000/im_width
    I = np.zeros(A.shape[0])
    test = 0
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = hist[i]
        mu_E = mu(E)
        I += I0_E * np.exp(-mu_E*d)
        test += I0_E
    
    print(f'test: {test}')
    b = -np.log(I)

    ###
    t = time()
    # x = (A^T A)^(-1) A^T b
    At = A.T
    AtA = np.dot(At, A)
    Atb = np.dot(At, b)

    # Tikhonov
    # alpha = 0.001
    alpha = 0.001
    x = np.linalg.solve(AtA+alpha*np.identity(n=AtA.shape[0]), Atb)
    print(f'solve: {time()-t}')

    # Display the image
    # plt.imshow(triangle_image, cmap='gray')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the figsize as needed

    # Plotting the original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original image')
    axs[0].axis('off')
    fig.colorbar(axs[0].imshow(image, cmap='gray'), ax=axs[0])

    # Plotting the reconstructed image
    axs[1].imshow(x.reshape(image.shape), cmap='gray')
    axs[1].set_title('Reconstructed image')
    axs[1].axis('off')
    fig.colorbar(axs[1].imshow(x.reshape(image.shape), cmap='gray'), ax=axs[1])

    plt.show()
    plt.savefig('plots/gt_vs_recon.png')

    plt.figure()
    print(((image-x.reshape(image.shape))**2).mean())
    p = profile_line(x.reshape(image.shape), src=(im_width//2,0), dst=(im_width//2,im_width-1))
    plt.plot(p)
    plt.title('Profile line through center')
    plt.ylabel('Reconstructed values')
    plt.xlabel('Distance along line')
    plt.show()
    plt.savefig('plots/pline.png')

def fun_xray_initial_dist(mean_energy,std_deviation):
    from scipy.stats import truncnorm

    # Parameters for the truncated Gaussian distribution
    # mean_energy = 150  # in keV
    # std_deviation = 50  # in keV
    lower, upper = 0, np.inf  # Lower and upper bounds for truncation

    # Calculate the bounds for the truncation in standard deviation units
    a, b = (lower - mean_energy) / std_deviation, (upper - mean_energy) / std_deviation

    # Generate the truncated normal distribution
    truncated_dist = truncnorm(a, b, loc=mean_energy, scale=std_deviation)
    energies = truncated_dist.rvs(100000)

    # Plotting the distribution
    # plt.hist(energies, bins=30, density=True, alpha=0.6, color='g')
    # plt.xlabel('Energy (keV)')
    # plt.ylabel('Probability Density')
    # plt.title('Truncated Gaussian X-ray Energy Distribution')
    # plt.show()
    return truncated_dist

def fun_attenuation():
    data = []
    # ORIGINAL DATA IN MEV!!!! SO CONVERT TO KEV
    with open('NIST_gold_only_dat.txt', 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split()

            # Check if the line has at least 3 elements and the first is a number
            if len(parts) >= 3 and parts[0].replace('.', '', 1).replace('E-', '', 1).replace('E+', '', 1).isdigit():
                energy = float(parts[-3])  # Energy value
                mu_rho = float(parts[-2])  # mu/rho value
                mu_en_rho = float(parts[-1])  # mu_en/rho value
                data.append((energy, mu_rho, mu_en_rho))

    # Example: Print the first few entries
    data = np.array(data)
    # print(data)
    # plt.loglog(data[:,0],data[:,1])
    # plt.show()
    # plt.loglog(data[:,0],data[:,2])
    # plt.show()


    # Assuming 'data' is a list of tuples (energy, mu_rho, mu_en_rho)
    # convert to KEV
    energies = 1000*data[:,0]
    mu_rhos = data[:,1]

    # Create a spline interpolator
    spline = interpolate.InterpolatedUnivariateSpline(energies, mu_rhos, k=1)

    # Define a function that uses this interpolator
    def estimate_attenuation(energy):
        return spline(energy)

    # Generate a log-spaced range of energy values for plotting
    energy_plot_range = np.logspace(np.log10(min(energies)), np.log10(max(energies)), 500)

    # Example usage
    # energy = 0.005  # 5 keV
    # attenuation = estimate_attenuation(energy)
    # print(f"Estimated attenuation at {energy} keV is {attenuation}")

    # Plot using log-log scale
    plt.loglog(energies, mu_rhos, 'o', label='Data Points')  # Data points
    plt.loglog(energy_plot_range, spline(energy_plot_range), label='Spline Interpolation')  # Spline curve

    plt.xlabel('Energy (keV)')
    plt.ylabel('Attenuation Coefficient')
    plt.title('Attenuation Coefficient vs Energy')
    plt.legend()
    plt.show()

    return spline

def test_triangle():
    # Example usage
    im_width = 20
    tri_width = im_width//2
    corner = im_width//4
    # im_width = 50
    # tri_width = 25
    # corner = 12
    img_size = (im_width, im_width)  # Image size: 100x100 pixels
    triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
    corner_coords = (corner, corner)  # Right-angled corner at (25, 25)

    # Generate the image
    triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
    image = np.rot90(triangle_image, 1)
    angles = np.arange(80,100,0.5)
    # angles = np.array([0,90])
    # panel_total_length = panel_num_cells*panel_cell_length
    panel_num_cells = 2*im_width
    panel_cell_length = 0.5
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    b = np.dot(A, image.flatten())

    # x = (A^T A)^(-1) A^T b
    At = A.T
    AtA = np.dot(At, A)
    Atb = np.dot(At, b)

    # Tikhonov
    alpha = 0.01
    x = np.linalg.solve(AtA+alpha*np.identity(n=AtA.shape[0]), Atb)


    # Display the image
    # plt.imshow(triangle_image, cmap='gray')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(x.reshape(image.shape), cmap='gray')
    plt.title('Reconstructed image')
    plt.axis('off')  # Turn off axis numbers
    plt.show()

    print(((image-x.reshape(image.shape))**2).mean())

def test_circle2():
    # Create the image
    im_width = 40
    angles = np.arange(0,180,3)
    panel_num_cells = 2*im_width
    panel_cell_length = 0.5
    A = parallelized_construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    # image = create_circle_image(im_width, radius=im_width//3, center=[im_width/2,im_width/2])

    image = create_circle_image(im_width, radius=im_width//2.5, center=[im_width/2,im_width/2])
    image -= create_circle_image(im_width, radius=im_width//6, center=[im_width/2,im_width/2])
    b = np.dot(A, image.flatten())

    # x = (A^T A)^(-1) A^T b
    At = A.T
    AtA = np.dot(At, A)
    Atb = np.dot(At, b)
    x = np.linalg.solve(AtA, Atb)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the figsize as needed
    # Plotting the original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original image')
    axs[0].axis('off')
    fig.colorbar(axs[0].imshow(image, cmap='gray'), ax=axs[0])

    # Plotting the reconstructed image
    axs[1].imshow(x.reshape(image.shape), cmap='gray')
    axs[1].set_title('Reconstructed image')
    axs[1].axis('off')
    fig.colorbar(axs[1].imshow(x.reshape(image.shape), cmap='gray'), ax=axs[1])

    plt.show()

    print(((image-x.reshape(image.shape))**2).mean())

def test_circle():
    # Create the image
    im_width = 50
    angles = np.arange(0,180,3)
    panel_num_cells = 2*im_width
    panel_cell_length = 0.5
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    image = create_circle_image(im_width, radius=im_width//4, center=[im_width//3,im_width//1.5])
    image2 = create_circle_image(im_width, radius=im_width//4, center=[im_width//2.5,im_width//3])
    combined_image = image+image2
    b = np.dot(A, combined_image.flatten())

    # x = (A^T A)^(-1) A^T b
    At = A.T
    AtA = np.dot(At, A)
    Atb = np.dot(At, b)
    x = np.linalg.solve(AtA, Atb)

    # Display the image
    plt.imshow(combined_image, cmap='gray')
    plt.title('Square Image with Custom Circle')
    plt.axis('off')  # Turn off axis numbers
    plt.show()

    plt.imshow(x.reshape(image.shape), cmap='gray')
    plt.title('Reconstructed image')
    plt.axis('off')  # Turn off axis numbers
    plt.show()

    print(((combined_image-x.reshape(image.shape))**2).mean())

def test_A_2():
    angles = np.array([90])
    panel_num_cells = 2
    panel_cell_length = 1
    im_width = 4
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)
    # print(A)

def test_A_1():
    angles = np.array([0,45,90,135])
    panel_num_cells = 3
    panel_cell_length = 1
    im_width = 3
    A = construct_A_parallel2D(angles, panel_num_cells, panel_cell_length, im_width)

def test_1():
    # 3: 1 = ; 5: 2 = ; 7: 3 = (7-1)/2
    # 6: 2.5 = (6-1)/2; 4: 1.5 = (4-1)/2
    panel_num_pixels = 10
    panel_pixel_size = 1
    s = panel_pixel_size * ( -(panel_num_pixels-1)/2 + np.arange(0, panel_num_pixels) )
    print(s)

def test_2():
    im_size = 2
    num_angles = 3
    panel_num_pixels = 10
    for proj_idx in range(num_angles):
        idx_shift = proj_idx * panel_num_pixels
        for ray_idx in range(panel_num_pixels):
            for im_pixel in range(im_size):
                print(idx_shift + ray_idx)

def test_3():
    im_pixel = 24
    im_width = 5
    i,j = divmod(im_pixel, im_width)
    coordinate_shift = -(im_width-1)/2
    pixel_coordinates_upper_left = np.array([i,j]) + coordinate_shift - 0.5
    pixel_coordinates_bottom_right = pixel_coordinates_upper_left + 1
    print(pixel_coordinates_upper_left)
    print(pixel_coordinates_bottom_right)

    x_left = i + coordinate_shift - 0.5
    y_top = j + coordinate_shift - 0.5
    x_right,y_bottom = x_left + 1, y_top + 1
    print([x_left,x_right])
    print([y_top,y_bottom])

if __name__ == "__main__":
    # test_1()
    # test_2()
    # test_3()
    # test_A_1()
    # test_A_2()
    # test_circle()
    # fun_attenuation()
    # fun_xray_initial_dist()
    # test_circle2()
    # test_triangle()
    simulate_BH()
    # simulate_BH2()
    None