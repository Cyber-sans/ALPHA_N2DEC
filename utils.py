import os
import shutil
import yaml
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import concurrent.futures
from dataclasses import dataclass
from dacite import from_dict
from typing import Literal


# File containing helper functions related to plotting & fitting
# Functions contained: plot_radial_profile_sim, plot_radial_profile_mcp, plot_radial_profile, bin_fit, interpolate_fit, plot_radial_profile_multiple, plot_radial_profile_multiple_species

# === Plot 1D Radial Density Profile for the N2DEC ===
def plot_radial_profile_sim(out_dir):
    """
    Plot the normalized 1D radial Be+ density profile from simulation output.
    Expects '2spc_z_int00.dat' in the given output directory.
    """
    # Load the Be+ 1D density profile from simulation output
    file = os.path.join(out_dir, '2spc_z_int00.dat')
    data = np.loadtxt(file)

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r = data[:, 0] * 1e3   # m → mm 
    beplus_dens = data[:, -1]

    # Normalize
    beplus_dens_norm = (beplus_dens)/ np.max(beplus_dens)

    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(r, beplus_dens_norm, label="N2DEC Be+ Density", linewidth=2, color='blue')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title('Be$^+$ Normalized 1D Radial Density Profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beplus_density_sim.png"))
    plt.close()
    print(f'N2DEC Be+ Normalized 1D Radial Density Profile saved to: {out_dir}')


# === Plot 1D Radial Density Profile for the MCP Data ===
def plot_radial_profile_mcp(out_dir, mcp_path):
    """
    Plot the normalized 1D radial Be+ density profile from MCP output.
    Expects 'mcp_path' in the 'mcp_results' directory.
    """
    # Load the Be+ 1D density profile from MCP output
    mcp_file = mcp_path
    data = np.loadtxt(mcp_file)

    # Column 0: radius [mm], Column 1: Be+ density
    r = data[:, 0]
    beplus_dens = data[:, -1]

    # Normalize
    beplus_dens_norm = (beplus_dens)/ np.max(beplus_dens)

    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(r, beplus_dens_norm, '+', label="MCP Be+ Density", color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title('Be$^+$ Normalized 1D Radial Density Profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beplus_density_mcp.png"))
    plt.close()
    print(f'MCP Be+ Normalized 1D Radial Density Profile saved to: {out_dir}')


# === Plot 1D Radial Density Profiles for the N2DEC & MCP Data ===
def plot_radial_profile(out_dir, mcp_path):
    """
    Plot the normalized 1D radial Be+ density profiles from the MCP & N2DEC outputs.
    Note: chops the MCP data wrt r-grid of N2DEC output.
    """
    # Load the Be+ 1D density profile from MCP output
    mcp_file = mcp_path
    data_mcp = np.loadtxt(mcp_file)

     # Load the Be+ 1D density profile from simulation output
    sim_file = os.path.join(out_dir, '2spc_z_int00.dat')
    data_sim = np.loadtxt(sim_file)

    # Column 0: radius [mm], Column 1: Be+ density
    r_mcp = data_mcp[:, 0]
    beplus_dens_mcp = data_mcp[:, -1]

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r_sim = data_sim[:, 0] * 1e3   # m → mm 
    beplus_dens_sim = data_sim[:, -1]

    # Normalize
    beplus_dens_sim_norm = beplus_dens_sim / np.max(beplus_dens_sim)
    beplus_dens_mcp_norm = beplus_dens_mcp / np.max(beplus_dens_mcp)

    # Chop mcp 'r' values higher than simualtion's max 'r' value.
    mask = r_mcp <= np.max(r_sim)
    chopped_r_mcp = r_mcp[mask]

    # Apply the same mask to beplus_dens_mcp_norm
    chopped_beplus_dens_mcp_norm = beplus_dens_mcp_norm[mask]

    # Calculate chi2
    chi2 = interpolate_fit(out_dir, mcp_path)
    
    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(r_sim, beplus_dens_sim_norm, label="N2DEC Be+ Density", linewidth=2 ,color='blue')
    plt.plot(chopped_r_mcp, chopped_beplus_dens_mcp_norm, '+',label="MCP Be+ Density", color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title(f'Be$^+$ Normalized 1D Radial Density Profile ($\\chi^2$: {chi2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beplus_density.png"))
    plt.close()
    print(f'Be+ Normalized 1D Radial Density Profiles saved to: {out_dir}')


# === Calculate Chi-Square Between Simulation and MCP Data via Binning ===
def bin_fit(out_dir, mcp_path):
    """
    Uses binning technique to compare N2DEC & MCP radial profiles and calculate chi2.
    """
    # Load the Be+ 1D density profile from MCP output
    mcp_file = mcp_path
    data_mcp = np.loadtxt(mcp_file)

    # Load the Be+ 1D density profile from simulation output
    sim_file = os.path.join(out_dir, '2spc_z_int00.dat')
    data_sim = np.loadtxt(sim_file)

    # Column 0: radius [mm], Column 1: Be+ density
    r_mcp = data_mcp[:, 0]
    beplus_dens_mcp = data_mcp[:, -1]

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r_sim = data_sim[:, 0] * 1e3   # m → mm 
    beplus_dens_sim = data_sim[:, -1]

    # Normalize
    beplus_dens_sim_norm = beplus_dens_sim / np.max(beplus_dens_sim)
    beplus_dens_mcp_norm = beplus_dens_mcp / np.max(beplus_dens_mcp)

    # Binning, [r_min, ..., r_max]
    r_min = max(np.min(r_sim), np.min(r_mcp))
    r_max = min(np.max(r_sim), np.max(r_mcp))
    bins = np.linspace(r_min, r_max, 90)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Distribute average densities over given bins
    sim_binned, _ = np.histogram(r_sim, bins=bins, weights=beplus_dens_sim_norm)
    sim_counts, _ = np.histogram(r_sim, bins=bins)
    sim_profile = sim_binned / np.maximum(sim_counts, 1)

    mcp_binned, _ = np.histogram(r_mcp, bins=bins, weights=beplus_dens_mcp_norm)
    mcp_counts, _ = np.histogram(r_mcp, bins=bins)
    mcp_profile = mcp_binned / np.maximum(mcp_counts, 1)

    # Only compare bins with both data for chi2
    valid = (sim_counts > 0) & (mcp_counts > 0)
    chi2 = np.sum((mcp_profile[valid] - sim_profile[valid])**2)

    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(bin_centers, sim_profile, label="N2DEC (binned)\n Be+ Density", linewidth=2 ,color='blue')
    plt.plot(bin_centers, mcp_profile, '+', label="MCP (binned)\n Be+ Density", color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title(f'Binned Be$^+$ Normalized 1D Radial Density Profiles ($\\chi^2$: {chi2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fit.png"))
    plt.close()
    print(f'Binning fit results saved to: {out_dir}')

    return chi2


# === Calculate Chi-Square Between Simulation and MCP Data via Interpolatation ===
def interpolate_fit(out_dir, mcp_path):
    """
    Uses interpolatation technique to compare N2DEC & MCP radial profiles and calculate chi2.
    """
    # Load the Be+ 1D density profile from MCP output
    mcp_file = mcp_path
    data_mcp = np.loadtxt(mcp_file)

    # Load the Be+ 1D density profile from simulation output
    sim_file = os.path.join(out_dir, '2spc_z_int00.dat')
    data_sim = np.loadtxt(sim_file)

    # Column 0: radius [mm], Column 1: Be+ density
    r_mcp = data_mcp[:, 0]
    beplus_dens_mcp = data_mcp[:, -1]

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r_sim = data_sim[:, 0] * 1e3   # m → mm
    beplus_dens_sim = data_sim[:, -1]

    # Normalize
    beplus_dens_sim_norm = (beplus_dens_sim) / np.max(beplus_dens_sim)
    beplus_dens_mcp_norm = (beplus_dens_mcp) / np.max(beplus_dens_mcp)

    # Interpolate simulation data to a function
    f = interp1d(r_sim, beplus_dens_sim_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Chop mcp 'r' values higher than simualtion's max 'r' value.
    mask = r_mcp <= np.max(r_sim)
    chopped_r_mcp = r_mcp[mask]

    # Apply the same mask to beplus_dens_mcp_norm and compare
    chopped_beplus_dens_mcp_norm = beplus_dens_mcp_norm[mask]
    chi2 = np.sum((chopped_beplus_dens_mcp_norm - f(chopped_r_mcp))**2)
    
    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(chopped_r_mcp, f(chopped_r_mcp), label="Interpolated N2DEC\n on MCP r-grid", linewidth=4, color='red')
    plt.plot(r_sim, beplus_dens_sim_norm, label="N2DEC Be+ Density", linewidth=2 ,color='blue')
    plt.plot(chopped_r_mcp, chopped_beplus_dens_mcp_norm, '+', label='MCP Be+ Density', color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title(f'Interpolated Be$^+$ Normalized 1D Radial Density Profile ($\\chi^2$: {chi2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fit.png"))
    plt.close()
    print(f'Interpolate fit results saved to: {out_dir}')

    return chi2


# === Plot 1D Radial Density Profiles for the N2DEC & MCP Data, for Multiple N2DEC ===
def plot_radial_profile_multiple(out_dir1, out_dir2, out_dir3, mcp_path, best_dir, label: Literal["Temp", "Pos", "Be", "Den"], values):
    """
    Plot the normalized 1D radial Be+ density profiles from the MCP & N2DEC outputs.
    Mutliple N2DEC outputs are plotted for given values of a parameter for better analysis of that parameter's error bar.
    Note: chops the MCP data wrt r-grid of N2DEC output.
    """
    # Load the Be+ 1D density profile from MCP output
    mcp_file = mcp_path
    data_mcp = np.loadtxt(mcp_file)

     # Load the Be+ 1D density profiles from simulation outputs
    sim_file1 = os.path.join(out_dir1, '2spc_z_int00.dat')
    data_sim1 = np.loadtxt(sim_file1)

    sim_file2 = os.path.join(out_dir2, '2spc_z_int00.dat')
    data_sim2 = np.loadtxt(sim_file2)

    sim_file3 = os.path.join(out_dir3, '2spc_z_int00.dat')
    data_sim3 = np.loadtxt(sim_file3)

    # Column 0: radius [mm], Column 1: Be+ density
    r_mcp = data_mcp[:, 0]
    beplus_dens_mcp = data_mcp[:, -1]

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r_sim1 = data_sim1[:, 0] * 1e3   # m → mm 
    beplus_dens_sim1 = data_sim1[:, -1]

    r_sim2 = data_sim2[:, 0] * 1e3   # m → mm 
    beplus_dens_sim2 = data_sim2[:, -1]

    r_sim3 = data_sim3[:, 0] * 1e3   # m → mm 
    beplus_dens_sim3 = data_sim3[:, -1]

    # Normalize
    beplus_dens_sim_norm1 = beplus_dens_sim1 / np.max(beplus_dens_sim1)
    beplus_dens_sim_norm2 = beplus_dens_sim2 / np.max(beplus_dens_sim2)
    beplus_dens_sim_norm3 = beplus_dens_sim3 / np.max(beplus_dens_sim3)
    beplus_dens_mcp_norm = beplus_dens_mcp / np.max(beplus_dens_mcp)

    # Chop mcp 'r' values higher than simualtion's max 'r' value.
    mask = r_mcp <= np.max(r_sim1)
    chopped_r_mcp = r_mcp[mask]

    # Apply the same mask to beplus_dens_mcp_norm
    chopped_beplus_dens_mcp_norm = beplus_dens_mcp_norm[mask]

    # Calculate chi2
    chi2_1 = interpolate_fit(out_dir1, mcp_path)
    chi2_2 = interpolate_fit(out_dir2, mcp_path)
    chi2_3 = interpolate_fit(out_dir3, mcp_path)

    # Naming and labels for given labels
    if label == "Temp":
        name = "Temp (K)"
    elif label == "Pos":
        name = "# of Positrons"
    elif label == "Be":
        name = "# of Be+ ions"
    else:
        name = "Density"
    
    if label == "Temp":
        label_1 = f"N2DEC {name} = {values[0]:.2f}\n $\\chi^2$ = {chi2_1:.2f}"
        label_2 = f"N2DEC {name} = {values[1]:.2f}\n $\\chi^2$ = {chi2_2:.2f}"
        label_3 = f"N2DEC {name} = {values[2]:.2f}\n $\\chi^2$ = {chi2_3:.2f}"
    else:
        label_1 = f"N2DEC {name} = {values[0]:.2e}\n $\\chi^2$ = {chi2_1:.2f}"
        label_2 = f"N2DEC {name} = {values[1]:.2e}\n $\\chi^2$ = {chi2_2:.2f}"
        label_3 = f"N2DEC {name} = {values[2]:.2e}\n $\\chi^2$ = {chi2_3:.2f}"

    # filename
    filename = f"beplus_density_err_{label}.png"
    
    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(r_sim1, beplus_dens_sim_norm1, label=label_1, linewidth=2 ,color='blue')
    plt.plot(r_sim2, beplus_dens_sim_norm2, label=label_2, linewidth=2 ,color='red')
    plt.plot(r_sim3, beplus_dens_sim_norm3, label=label_3, linewidth=2 ,color='green')
    plt.plot(chopped_r_mcp, chopped_beplus_dens_mcp_norm, '+',label="MCP Be+ Density", color='black')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title(f'Be$^+$ Normalized 1D Radial Density Profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(best_dir, filename))
    plt.close()
    print(f'Be+ Normalized 1D Radial Density Profiles for error bars saved to: {best_dir}')


# === Plot 1D Radial Density Profile for Multiple Species for the N2DEC ===
def plot_radial_profile_multiple_species(out_dir):
    """
    Plot the normalized 1D radial Be+ & e+ density profiles from simulation output.
    Expects '2spc_z_int00.dat' in the given output directory.
    """
    # Load the Be+ 1D density profile from simulation output
    file = os.path.join(out_dir, '2spc_z_int00.dat')
    data = np.loadtxt(file)

    # Column 0: radius [m], Column 1: e+ density, Column 2: Be+ density
    r = data[:, 0] * 1e3   # m → mm 
    pos_dens = data[:, -2]
    beplus_dens = data[:, -1]

    # Normalize
    pos_dens_norm = (pos_dens) / np.max(pos_dens) 
    beplus_dens_norm = (beplus_dens) / np.max(beplus_dens)

    # Plot and save
    plt.figure(figsize=(7, 4))
    plt.plot(r, pos_dens_norm, label="N2DEC e+ Density", linewidth=2, color='red')
    plt.plot(r, beplus_dens_norm, label="N2DEC Be+ Density", linewidth=2, color='blue')
    plt.xlabel("Distance from Plasma Center [mm]")
    plt.ylabel("Normalized Axially Integrated Density [arb. units]")
    plt.title('Be$^+$ & e$^+$ Normalized 1D Radial Density Profiles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multiple_density_sim.png"))
    plt.close()
    print(f'N2DEC Be+ & e+ Normalized 1D Radial Density Profiles saved to: {out_dir}')
