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
from utils import plot_radial_profile_sim, plot_radial_profile_mcp, plot_radial_profile, bin_fit, interpolate_fit, plot_radial_profile_multiple, plot_radial_profile_multiple_species
from utils2 import generate_field_inp_2sp_file, compile_n2dec, run_n2dec, run_and_fit_param, parallel_optimizer, calculate_error
from mcp_image_analysis import mcp_image_analyzer

# === Input Configurations Dataclass ===


@dataclass
class InputConfig:
    """
    Input configurations for the N2DEC simulation.
    """
    temperature: float
    num_electrodes: int
    center_index: int
    density: float
    num_positrons: float
    num_be_ions: float
    temp_factor: float
    accuracy: float
    diameter: float
    default_length: float
    electrode_profile: list
    mcp_path: str
    optimize_temp: bool
    optimize_num_pos: bool
    optimize_num_be_ions: bool
    optimize_density: bool
    fit_method: str
    calculate_error: bool
    use_processed_mcp: bool
    processed_mcp_path: str


# === Helper for Loading Input Configurations ===
def load_config(path):
    """
    Create and return InputConfig object from a YAML file.
    """
    with open(path, "r") as f:
        yaml_data = yaml.safe_load(f)

    if "inputs" not in yaml_data:
        raise KeyError(f"Expected top_level key 'inputs'. Got keys: {
                       list(yaml_data.keys())}")

    # Force float values to float, for extra sanity!
    # YAML sometimes parses these values (i.e. 6.0e2) as strings for some reason
    # So section is necessary for returning flawless InputConfig object
    inputs = yaml_data["inputs"]
    try:
        inputs["temperature"] = float(inputs["temperature"])
        inputs["density"] = float(inputs["density"])
        inputs["num_positrons"] = float(inputs["num_positrons"])
        inputs["num_be_ions"] = float(inputs["num_be_ions"])
        inputs["temp_factor"] = float(inputs["temp_factor"])
        inputs["diameter"] = float(inputs["diameter"])
        inputs["default_length"] = float(inputs["default_length"])
        inputs["accuracy"] = float(inputs["accuracy"])
    except Exception:
        pass

    return from_dict(data_class=InputConfig, data=inputs)


# === Run Everything ===
def main():
    # Folder structure (relative paths)
    base_dir = os.getcwd()

    # input_config.yaml path
    input_path = os.path.join(base_dir, 'input_config.yaml')

    # N2DEC executable directory
    exe_dir = os.path.join(base_dir, 'executable')

    # Base results directory (create if missing)
    base_res_dir = os.path.join(base_dir, 'results')
    os.makedirs(base_res_dir, exist_ok=True)

    # MCP results directory
    mcp_dir = os.path.join(base_dir, 'mcp_results')

    # Get user input, ask run number (assign 0 if invalid entry).
    try:
        run_number = int(input("Enter the run number: "))
    except Exception:
        print("Invalid input. Using default run number 0.")
        run_number = 0

    # Compile the c++ n2dec source code
    # compile_n2dec(exe_dir, 1)

    # Unique results/<run_number>/ directory for given run
    res_dir = os.path.join(base_res_dir, f'{run_number}')
    os.makedirs(res_dir, exist_ok=True)

    # Copy input_config.yaml file to results/<run_number>/
    dest_input_path = os.path.join(res_dir, 'input_config.yaml')
    if os.path.exists(input_path):
        shutil.copyfile(input_path, dest_input_path)
        print(f"input_config.yaml file for given run copied to {
              dest_input_path}")

    # Create InputConfig object from yaml file.
    input_config = load_config(input_path)

    # Load chosen MCP path
    # Check if use already processed mcp or analyze raw mcp image
    if input_config.use_processed_mcp:
        # Make the run directory within mcp_results/
        mcp_analysis_dir = os.path.join(mcp_dir, f"{run_number}")
        os.makedirs(mcp_analysis_dir, exist_ok=True)

        # Copy the already processed mcp radial profile from mcp_results/processed/ to mcp_results/<run_number>/
        src_processed_mcp = os.path.join(
            mcp_dir, "processed", input_config.processed_mcp_path)
        dest_processed_mcp = os.path.join(
            mcp_analysis_dir, input_config.processed_mcp_path)
        if os.path.exists(src_processed_mcp):
            shutil.copyfile(src_processed_mcp, dest_processed_mcp)
        else:
            raise FileNotFoundError(
                f"Source file '{src_processed_mcp}' does not exist.")

        mcp_path = dest_processed_mcp
        print("Already processed MCP image being used for optimization: " + mcp_path)
    else:
        original_mcp_path = input_config.mcp_path
        print("Raw MCP image being used for optimization: " + original_mcp_path)

        # Analyze the MCP image
        mcp_image_analyzer(original_mcp_path, run_number)

        # Split .tif extension (eg. 1027_36.322.tif to 1027_36.322)
        base_name = os.path.splitext(original_mcp_path)[0]

        # Extracted radial profile to be used in fitting!!!
        mcp_path = os.path.join(
            mcp_dir, f"{run_number}", f"radial_profile{base_name}.txt")
        print("Extracted radial profile from image: " + mcp_path)

    print("Quitting early for debugging purposes...if this isn't expected go to line 155")
    quit()

    # Choose fitting function
    if input_config.fit_method == "interpolate":
        fit = interpolate_fit
        print("Interpolatation technique will be used in fitting N2DEC to MCP data")
    elif input_config.fit_method == "bin":
        fit = bin_fit
        print("Binning technique will be used in fitting N2DEC to MCP data")
    else:
        fit = interpolate_fit
        print("Invalid fit_method configuration, interpolatation technique will be used in fitting N2DEC to MCP data as default")

    # Load initial guesses
    temperature_guess = input_config.temperature
    num_positrons_guess = input_config.num_positrons
    num_be_ions_guess = input_config.num_be_ions
    density_guess = input_config.density

    # Manual num_positron, num_be_ion, & density sweeps:
    # For sweep = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], and sweep_ratio = 0.02:
    # Builds an sweeping array for given parameter: [par - %10, par - %8, ..., par, ..., par + %10]
    sweep = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    sweep_ratio = 0.02

    # Optimization arrays
    num_positrons_array = np.array(
        [num_positrons_guess + (num_positrons_guess*sweep_ratio*i) for i in sweep], dtype=float)
    num_be_ions_array = np.array(
        [num_be_ions_guess + (num_be_ions_guess*sweep_ratio*i) for i in sweep], dtype=float)
    density_array = np.array(
        [density_guess + (density_guess*sweep_ratio*i) for i in sweep], dtype=float)

    # Deltas array, will be used for calculating error bars
    deltas = [1, num_be_ions_guess*sweep_ratio,
              num_be_ions_guess*sweep_ratio, density_guess*sweep_ratio]

    # Optimization flags
    optimize_temp = input_config.optimize_temp
    optimize_num_pos = input_config.optimize_num_pos
    optimize_num_be_ions = input_config.optimize_num_be_ions
    optimize_density = input_config.optimize_density

    # Initialize best guesses
    best_temp = temperature_guess
    best_num_pos = num_positrons_guess
    best_num_be_ions = num_be_ions_guess
    best_density = density_guess
    best_chi2 = np.inf

    # Full temperature optimization, find the best temperature in the order of a Kelvin, no initial guess needed.
    if optimize_temp:
        # First find the most optimal temperature in the order of ten Kelvins,
        # Then find the most optimal temperature in the order of a Kelvin,
        # coarse_temp_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        coarse_temp_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                     160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300], dtype=float)

        # Coarse run
        best_coarse_temp, _ = parallel_optimizer(run_number, temperature_guess, num_positrons_guess, num_be_ions_guess,
                                                 density_guess, input_config, exe_dir, res_dir, mcp_path, coarse_temp_array, "Temp", fit)

        # Tuning for fine precision temperature array
        fine_tune = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        fine_temp_array = np.array(
            [best_coarse_temp + i for i in fine_tune], dtype=float)

        # Check if the Temperatue is lower than 6 K
        # If the best-fit temperature is found to be 5 K, you need to run N2DEC on 4 K to calculate error bars
        # 4 K do not converge on N2DEC so, lowest temperature threshold should be 6 K to be able to calculate error bars
        if (int(best_coarse_temp) == 10) and (input_config.calculate_error == True):
            fine_temp_array = np.array(
                [6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)

        # Medium run
        best_fine_temp, best_fine_chi2 = parallel_optimizer(run_number, temperature_guess, num_positrons_guess,
                                                            num_be_ions_guess, density_guess, input_config, exe_dir, res_dir, mcp_path, fine_temp_array, "Temp", fit)

        # Update best_temp & best_chi2
        best_temp = best_fine_temp
        best_chi2 = best_fine_chi2

        # results
        print(
            f'Fine best-fit temperature: {best_fine_temp:.2f}, with chi2: {best_fine_chi2:.2f}')

    # Optimize num_positrons choice, using optimized temperature. Directly update best_num_pos & best_chi2.
    if optimize_num_pos:
        best_num_pos, best_chi2 = parallel_optimizer(run_number, best_temp, num_positrons_guess, num_be_ions_guess,
                                                     density_guess, input_config, exe_dir, res_dir, mcp_path, num_positrons_array, "Pos", fit)

    # Optimize density choice, using optimized temp & num_pos. Directly update best_density & best_chi2.
    if optimize_density:
        best_density, best_chi2 = parallel_optimizer(
            run_number, best_temp, best_num_pos, num_be_ions_guess, density_guess, input_config, exe_dir, res_dir, mcp_path, density_array, "Den", fit)

    # Optimize num_be_ions choice, using optimized temp & num_pos & density. Directly update best_num_be_ions & best_chi2.
    if optimize_num_be_ions:
        best_num_be_ions, best_chi2 = parallel_optimizer(
            run_number, best_temp, best_num_pos, num_be_ions_guess, best_density, input_config, exe_dir, res_dir, mcp_path, num_be_ions_array, "Be", fit)

    # Print out optimizations applied
    print(f'Full Temperature Optimization: {
          optimize_temp}, Number of Positrons Optimization: {optimize_num_pos}')
    print(f'Number of Be+ ions Optimization: {
          optimize_num_be_ions}, Density Optimization: {optimize_density}')

    # Extract the best run_id
    best_run_id = f"{run_number}_T{best_temp:.2f}_P{
        best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"

    # Output directory of best run: results/<run_number>/<run_number>_T<best_temp>_P<best_num_pos>_B<best_num_be>_D<best_density>/
    out_dir = os.path.join(res_dir, best_run_id)

    # Create new directory for the best-fit summary
    best_dir = os.path.join(res_dir, "best_fit")
    os.makedirs(best_dir, exist_ok=True)

    # If no optimization is done, run and fit the simulation using initial best guess inputs
    if (optimize_temp == False) and (optimize_num_pos == False) and (optimize_num_be_ions == False) and (optimize_density == False):
        _ = run_and_fit_param(run_number, best_temp, best_num_pos, best_num_be_ions,
                              best_density, input_config, exe_dir, res_dir, mcp_path, fit)

    # Plottings for the best-fit so far
    plot_radial_profile_sim(out_dir)
    plot_radial_profile_mcp(out_dir=out_dir, mcp_path=mcp_path)
    plot_radial_profile(out_dir=out_dir, mcp_path=mcp_path)
    plot_radial_profile_multiple_species(out_dir=out_dir)

    # Name of summary plots
    summaryfiles = ["beplus_density_mcp.png", "beplus_density_sim.png",
                    "beplus_density.png", "fit.png", "multiple_density_sim.png"]

    # Move the best-fit plots to the results/<run_number>/best_fit
    for filename in summaryfiles:
        src_path = os.path.join(out_dir, filename)
        dest_path = os.path.join(best_dir, filename)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
            if not filename == "fit.png":
                os.remove(src_path)

    # Calculate the error bars to the best fit
    if input_config.calculate_error:
        uncertainties = calculate_error(run_number, deltas, best_temp, best_num_pos, best_num_be_ions,
                                        best_density, best_chi2, input_config, exe_dir, res_dir, mcp_path, fit)

    # Save best fit summary to results/<run_number>/best_fit/best_fit_summary.txt
    best_fit_summary = os.path.join(best_dir, "best_fit_summary.txt")
    with open(best_fit_summary, "w") as f:
        print(f"Saving best-fit summary to {best_fit_summary}")
        f.write("=== Best Run ===\n")
        f.write(f"Results for the best run are in: {out_dir}\n")
        f.write("=== Summary ===\n")
        if input_config.calculate_error:
            f.write(
                f"Best-fit temperature in Kelvins: {best_temp:.2f} ± {uncertainties[0]:.2f}\n")
            f.write(
                f"Best-fit number of positrons: {best_num_pos:.2e} ± {uncertainties[1]:.2e}\n")
            f.write(
                f"Best-fit number of Be+ ions: {best_num_be_ions:.2e} ± {uncertainties[2]:.2e}\n")
            f.write(
                f"Best-fit density: {best_density:.2e} ± {uncertainties[3]:.2e}\n")
        else:
            f.write(f"Best-fit temperature in Kelvins: {best_temp:.2f}\n")
            f.write(f"Best-fit number of positrons: {best_num_pos:.2e}\n")
            f.write(f"Best-fit number of Be+ ions: {best_num_be_ions:.2e}\n")
            f.write(f"Best-fit density: {best_density:.2e}\n")

        f.write(f"Best-fit chi2: {best_chi2:.2f}\n")
        f.write("MCP radial profile used: " + mcp_path)


# Run main if executed as script
if __name__ == "__main__":
    main()
