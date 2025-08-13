import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import center_of_mass
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.optimize import curve_fit


# File containing helper function related to analyzing original mcp image and extracting radial profile
# Functions contained: mcp_image_analyzer

def mcp_image_analyzer(original_mcp_path, run_number):
    """
    Given raw mcp image file, analyzes the image, generates auto mask,
    detects plasma edge, fits ellipse, and returns radial density profile.
    """
    mcp_tif = os.path.join(os.getcwd(), "mcp_results",
                           "originals", original_mcp_path)
    mcp_analysis_dir = os.path.join(
        os.getcwd(), "mcp_results", f"{run_number}")
    os.makedirs(mcp_analysis_dir, exist_ok=True)

    print(f"Reading MCP image: {mcp_tif}")

    # Read mcp file, convert it to a 2d numpy array with values as float64 (originally uint16)
    mcp_image = plt.imread(mcp_tif).astype(np.float64)

    # Subtract minimum (offset) from whole image
    # MCP offset is usually high, signal covers a small portion on top of the offset
    mcp_image -= np.amin(mcp_image)

    # Optional
    # Crop
    # mcp_image = mcp_image[380:600, 500:700]

    # Optional
    # Remove bright positron image before continuing Be+ plasma extraction
    # Needs trial and error
    # Upground estimate and set to 0
    # ug_estimate = np.percentile(mcp_image, 83)
    # mcp_image[mcp_image > ug_estimate] = 0

    # Stats
    n, m = mcp_image.shape
    print("Analyzing image...")
    print(f"Image dimensions: {n} x {m}")

    # === AUTOMATIC MASK CREATION ===

    # 1. Estimate background using low percentile (avoid using mean, as plasma can bias it)
    # Lower 10th percentile is good background estimation, also calculate standard deviation of the background
    bg_estimate = np.percentile(mcp_image, 10)
    std_estimate = np.std(mcp_image[mcp_image < bg_estimate + 1e-7])

    # 2. Threshold: anything much brighter than background is considered plasma
    # Background plus 4sigma is common threshold estimate at plasma imaging
    threshold = bg_estimate + 4 * std_estimate
    initial_mask = (mcp_image > threshold).astype(np.uint8)

    # 3. Connected components: label blobs
    labeled, num_features = label(initial_mask)
    if num_features == 0:
        print("No plasma region detected. Exiting.")
        exit()

    # 4. Select the largest blob as the plasma, label numbers start at 1
    sizes = [np.sum(labeled == i) for i in range(1, num_features+1)]
    biggest = np.argmax(sizes) + 1
    mask_image = (labeled == biggest).astype(np.uint8)

    # 5. Fill holes inside the mask (for hollow plasma regions)
    mask_image = binary_fill_holes(mask_image).astype(np.uint8)

    # MCP image with background substracted
    mcp_image_clean = mcp_image - threshold

    # Invert mask
    antimask = 1 - mask_image

    # Masked and antimasked mcp image
    masked_mcp = mask_image * mcp_image
    antimasked_mcp = antimask * mcp_image

    # Non-mask (background) region, and total, and net (plasma) intensity
    total_intensity = np.sum(mcp_image)
    background = np.mean(mcp_image[antimask != 0])
    background_int = background * n * m
    intensity = total_intensity - background_int

    print(f"Net Intensity (NNI): {intensity:.2e}")

    # Plot
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title('Auto-mask')
    plt.imshow(mask_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('MCP with Mask')
    plt.imshow(masked_mcp, cmap='plasma')
    plt.contour(mask_image, colors='r')
    plt.subplot(1, 3, 3)
    plt.title('Original Image')
    plt.imshow(mcp_image, cmap='plasma')
    plt.tight_layout()
    plt.savefig(os.path.join(mcp_analysis_dir, "auto_mask.png"))
    plt.close()
    print("Auto-mask plots saved to " + mcp_analysis_dir)

    # Estimate plasma center using mask center of mass
    y_c_rough, x_c_rough = center_of_mass(masked_mcp)
    print(f"Plasma center estimated at (x={x_c_rough:.2f}, y={y_c_rough:.2f})")

    # === AUTOMATE CHORDLENGTH ===

    # 1. Mask coordinates and center coordinate in tuples of (x, y)
    mask_coords = np.column_stack(np.nonzero(mask_image))
    center_coord = np.array([[y_c_rough, x_c_rough]])

    # 2. Euclidean distance between center_coord and every plasma pixel coordinate in mask_coords.
    dists = cdist(center_coord, mask_coords)

    # 3. Maximum euclidian distance from the center and little bit buffer into background
    chordlength = int(np.max(dists)) + 30
    print(f"Automated chordlength = {chordlength}")

    # Chord integration from center
    N_chords = 300
    phi = np.linspace(0, 2*np.pi, N_chords)

    chord = []
    for angle in phi:
        values = []
        for step in range(chordlength):
            x = int(round(x_c_rough + np.cos(angle) * step))
            y = int(round(y_c_rough + np.sin(angle) * step))
            if 1 < x < (m-2) and 1 < y < (n-2):
                region = mcp_image_clean[y-1:y+2, x-1:x+2]
                values.append(region.mean())
            else:
                values.append(0)
        chord.append(values)

    chord = np.array(chord)

    # Smooth each chord using running mean
    def running_mean(x, N):
        return np.convolve(x, np.ones(N)/N, mode='valid')

    Nrun1 = 16
    chord_rm = np.array([running_mean(c, Nrun1) for c in chord])

    # Compute gradient and smooth again
    chord_grad = np.diff(chord_rm, axis=1)
    Nrun2 = 16
    chord_grad_rm = np.array([running_mean(g, Nrun2) for g in chord_grad])

    # Find the max gradient halft the way along each chord (plasma edge)
    cmax_radius = chord_grad_rm[:, :chordlength//2].argmax(axis=1)

    # Calculate (x, y) coordinates of edge points (the detected ellipse)
    ex_coord = x_c_rough + cmax_radius * np.cos(phi)
    ey_coord = y_c_rough + cmax_radius * np.sin(phi)

    # Plot the edge points over the MCP image
    plt.figure(figsize=(8, 8))
    plt.imshow(mcp_image_clean, cmap='plasma', origin='upper')
    plt.plot(ex_coord, ey_coord, 'r.', markersize=1,
             label='Detected Plasma Edge')
    plt.scatter([x_c_rough], [y_c_rough], color='white',
                s=5, label='Estimated Center')
    plt.title("Detected Plasma Boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mcp_analysis_dir, "edge_detection.png"))
    plt.close()
    print("Edge detection plot saved to " + mcp_analysis_dir)

    # This includes the mm/pixels calibration
    # The fringe factor in B=1T
    # Therefore converts pixels -> size in trap at B=1T
    # Feel free to adjust with this if you are unsatisfied
    # b_calibration_factor = 0.086/8.74 (From Jack's analyses)
    # b_calibration_factor = 0.086/9.74 (Best for 2019 data)
    # b_calibration_factor = 2.5 * 0.086/9.74 (Best for 2025 data)
    b_calibration_factor = 0.086/9.74

    # Copy edge points into NDArrays
    x = np.array(ex_coord)
    y = np.array(ey_coord)

    # Stack x & y coordinates into two columns: [[x0,y0], ...]
    a_points = np.column_stack((x, y))

    # Fit ellipse to plasma edge coordinates
    ell = EllipseModel()
    ell.estimate(a_points)

    # Extract ellipse center coordinates, semi-major/minor axes, & counterclockwise rotation of major axis in radians
    xc, yc, a, b, theta = ell.params

    print(f"Ellipse center = (x={xc:.2f}, y={yc:.2f})")
    print(f"Angle of rotation = {theta:.4f} rad")
    print(f"Axes = a: {a:.2f}, b: {b:.2f}")

    # Plot detected edge points and fitted ellipse
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(x, y, 'x', color='black')
    axs[0].set_title('Plasma edge points')

    axs[1].plot(x, y, 'x', color='black', label='Edge')
    axs[1].scatter(xc, yc, color='red', s=100, label='Center')
    # Ellipse overlay
    ell_patch = Ellipse(
        (xc, yc), width=2*a, height=2*b, angle=np.degrees(theta),
        edgecolor='red', facecolor='none', linewidth=2, label='Fitted ellipse'
    )
    axs[1].add_patch(ell_patch)
    axs[1].legend()
    axs[1].set_title('Fitted Ellipse Overlay')
    plt.tight_layout()
    plt.savefig(os.path.join(mcp_analysis_dir, "elliptical_fit.png"))
    plt.close()
    print("Elliptical fit plots saved to " + mcp_analysis_dir)

    # Overlay fitted ellipse on MCP image
    fig2 = plt.figure(figsize=(8, 8))
    plt.imshow(mcp_image_clean, cmap='plasma')
    ellipse_patch = Ellipse(
        (xc, yc), width=2*a, height=2*b, angle=np.degrees(theta),
        edgecolor='cyan', facecolor='none', linewidth=1
    )
    plt.gca().add_patch(ellipse_patch)
    plt.scatter(xc, yc, color='red', s=10)
    plt.title("Ellipse Overlay on MCP Image")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(mcp_analysis_dir, "elliptical_fit_on_image.png"))
    plt.close()
    print("Elliptical fit on mcp image saved to " + mcp_analysis_dir)

    print("Computing circularized radial density profile...")

    # Chord integration from center
    N_chords = 300
    phis = np.linspace(0, 2*np.pi, N_chords)

    def relative_stretch_factor(a, b, phi, theta):
        """
        Stretch factor is 1 when the chord is at the “average” direction,
        >1 on the major axis (longest chord), <1 on the minor axis (shortest chord).
        """
        # phi: chord angle in image coordinates (from x-axis)
        # theta: rotation of major axis (from x-axis)
        phi_rel = phi - theta
        r_ellipse = (a * b) / np.sqrt((b * np.cos(phi_rel))
                                      ** 2 + (a * np.sin(phi_rel))**2)
        return r_ellipse / np.sqrt(a * b)

    chords = []
    for angle in phis:
        stretch = relative_stretch_factor(a, b, angle, theta)
        values = []
        for step in range(chordlength):
            radius = (step + 1e-7) * stretch
            x = int(round(xc + np.cos(angle) * radius))
            y = int(round(yc + np.sin(angle) * radius))
            if 1 < x < (m-2) and 1 < y < (n-2):
                region = mcp_image_clean[y-1:y+2, x-1:x+2]
                values.append(region.mean())
            else:
                values.append(0)
        chords.append(values)

    chords = np.array(chords)

    # Sum over all chords at each radius (i.e., for each annulus)
    summed = chords.sum(axis=0)

    # Baseline subtraction
    bg = np.percentile(summed, 25)
    summed -= bg
    summed[summed < 0] = 0

    # Smooth profile
    Nrun3 = 8
    summed_rm = running_mean(summed, Nrun3)

    # Compute physical radius in mm for each bin (step)
    max_radius = chordlength * b_calibration_factor
    radius_array = np.linspace(0, max_radius, len(summed))
    radius_bin_edges = np.linspace(0, max_radius, len(summed_rm) + 1)
    radius_centers = 0.5 * (radius_bin_edges[:-1] + radius_bin_edges[1:])

    # Annulus areas
    # annulus_areas = np.pi * (radius_bin_edges[1:]**2 - radius_bin_edges[:-1]**2)

    # Density: total signal per unit area (in each annulus)
    # density = summed_rm / annulus_areas
    density = summed_rm

    # Smooth profile again
    Nrun4 = 4
    density = running_mean(density, Nrun4)

    # Adjust radius array
    radius_centers = radius_centers[:len(density)]

    # Normalize
    # density = density / np.max(density)

    # Clip Noisy Edges
    density = density[5:-3]
    radius_centers = radius_centers[5:-3]

    # Adding bins to radius_centers (and pad density with the last value) until you reach at least 1.5 mm.
    while radius_centers[-1] < 1.5:
        radius_centers = np.append(
            radius_centers, radius_centers[-1] + b_calibration_factor)
        density = np.append(density, density[-1])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(radius_centers, density, '+', color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Axially Integrated Density")
    plt.grid(True)
    plt.title("MCP Radial Density Profile")
    plt.tight_layout()
    plt.savefig(os.path.join(mcp_analysis_dir, "radial_profile.png"))
    plt.show()
    plt.close()
    print("Radial profile plot saved to " + mcp_analysis_dir)

    print("Started experiment")

    xc, yc, a, b, theta = ell.params
    center = np.array([xc, yc])

    # Here follows a later edition by Rhys Brown to get the radial density profile by improving on the code written by the prophet Deniz Yoldaz
    ############################

    NNIs = []
    Rs = []

    max_r = int(chordlength)
    spacing = 4

    bg = np.percentile(masked_mcp, 25)
    masked_mcp -= bg

    H, W = mcp_image_clean.shape
    for r in range(int(a*5/6), int(max_r * 5/6)):

        circum = 2 * np.pi * r

        numPoints = max(4, int(circum / spacing))

        phi = np.linspace(0, 2 * np.pi, numPoints, endpoint=False)

        x = xc + r * np.cos(phi) * np.cos(theta) - r * b/a * \
            np.sin(phi) * np.sin(theta)
        y = yc + r * np.cos(phi) * np.sin(theta) + r * b/a * \
            np.sin(phi) * np.cos(theta)

        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if valid.mean() < 0.7:  # If > 30% is out of bounds, *Brooklyn accent* forgettt about itttt
            continue

        vals = map_coordinates(
            # interpolates decimal image positions
            mcp_image_clean, [y[valid], x[valid]], order=1, mode="constant", cval=np.nan)

        ringMean = max(0, np.nanmean(vals))

        NNIs.append(max(0, ringMean))
        Rs.append(np.sqrt(r**2 * b / a)*b_calibration_factor)

    NNIs_sm = gaussian_filter1d(np.array(NNIs), sigma=3.0)

    def getPositronRadius(path):
        img = plt.imread(path).astype(float)

        bg = np.percentile(img, 10)
        thr = bg + 4*np.std(img[img <= bg])
        mask = img > thr

        lab, n = label(mask)
        if n == 0:
            return 0.0

        # counts how many pixels belong to each label and then picks the one with the most assigned to it
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = binary_fill_holes(lab == sizes.argmax())

        cy, cx = center_of_mass(mask)
        H, W = mask.shape

        # simple marching algorithm,
        ang = np.linspace(0, np.pi, 5, endpoint=False)
        radii = []
        for a in ang:
            c, s = np.cos(a), np.sin(a)
            r1 = r2 = 0.0
            while True:
                x = int(cx + r1*c)
                y = int(cy + r1*s)
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    r1 += 1.0
                else:
                    break
            while True:
                x = int(cx - r2*c)
                y = int(cy - r2*s)
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    r2 += 1.0
                else:
                    break
            radii.append(0.5*(r1 + r2))

        # median not mean because some chords go crazy wrong
        return float(np.median(radii))

    positronRadius = getPositronRadius(
        "C:/Users/Acer Predator/Documents/ALPHA/PROJECXT 3/gitRepository/ALPHA_N2DEC/mcp_results/originals/1129_40.883.tif", )

    print("Positron Radius: " + str(positronRadius))

    magStrength_i = 0.01  # ATDS MCP
    magStrength_f = 3.1  # In ATM trap

    expansionFactor = np.sqrt(magStrength_i / magStrength_f)

    positronRadius = positronRadius * expansionFactor

    pixelCalibration = 20e-3  # https://alphacpc05.cern.ch/elog/ALPHA/34611

    pixelCalibration = pixelCalibration * 2 / (H + W)

    beExpansion = np.sqrt(a * b) / positronRadius

    Rs = np.array(Rs) / beExpansion * pixelCalibration

    def gaussian(x, A, mu, sigma, C):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

    params, opt = curve_fit(gaussian, Rs, NNIs)

    xfit = np.linspace(min(Rs), max(Rs), 500)
    yfit = [gaussian(x, params[0], params[1], params[2], params[3])
            for x in xfit]

    plt.plot(Rs, NNIs_sm, '+', color='black')
    plt.plot(xfit, yfit)
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Net Intensity")
    plt.title("Radial NNI Profile")
    plt.grid(True)
    plt.show()

    ############

    # Split .tif extension (eg. 1027_36.322.tif to 1027_36.322)
    base_name = os.path.splitext(original_mcp_path)[0]

    # Save radial profile as txt
    out_path = os.path.join(mcp_analysis_dir, f"radial_profile{base_name}.txt")
    np.savetxt(out_path, np.column_stack(
        [radius_centers, density]), fmt="%.10f\t%.10f")
    print("Radial profile saved to", out_path)

    summary_path = os.path.join(mcp_analysis_dir, "image_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== MCP Image Fitting Summary ===\n")
        f.write(f"Original MCP image used {original_mcp_path}\n")
        f.write(f"Ellipse center (xc, yc): ({xc:.4f}, {yc:.4f})\n")
        f.write(f"Semi-axes (a, b): ({a:.4f}, {b:.4f})\n")
        f.write(f"Ellipse angle (theta, degrees): {np.degrees(theta):.4f}\n")
        f.write(f"Net intensity (NNI): {intensity:.4e}\n")

    print("MCP image analysis summary saved to", summary_path)
