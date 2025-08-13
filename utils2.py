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
from utils import plot_radial_profile_multiple

# File containing helper functions related to compiling & parallel running N2DEC
# Functions contained: generate_field_inp_2sp_file, compile_n2dec, run_n2dec, run_and_fit_param, parallel_optimizer, calculate_error

# === Create input parameters file (.dat) for N2DEC ===


def generate_field_inp_2sp_file(
    out_dir,
    run_number,
    num_electrodes=14,
    center_index=5,
    density=62e12,
    num_positrons=0.11e6,
    num_be_ions=0.33e6,
    temp_factor=0.50,
    accuracy=2e-8,
    diameter=44.55e-3,
    default_length=20.05e-3,
    electrode_profile=[0, 72.0, 72.0, 72.0, -72.0, 72.0,
                       72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 0]
):
    """
    Create a field_inp_2sp_<run_number>.dat file in the results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/ directory.
    One line for parameters, followed by num_electrodes lines of <length> <voltage>.
    """
    # Configuration file name for the given run
    filename = f'field_inp_2sp_{run_number}.dat'
    file_path = os.path.join(out_dir, filename)

    with open(file_path, 'w') as f:
        # Header line
        f.write(f"{num_electrodes} {center_index} {density:.3e} {num_positrons:.3e} {num_be_ions:.3e} "
                f"{temp_factor:.3f} {accuracy:.3e} {diameter:.4e}\n")

        # Electrode profile
        for i in range(1, num_electrodes + 1):
            voltage = electrode_profile[i - 1]
            f.write(f"{default_length:.5e} {voltage:.3f}\n")

    print(f"Created: {file_path}")


# === Compile N2DEC ===
def compile_n2dec(exe_dir, bfield):
    """
    Helper function to compile n2dec c++ appropriately.
    Uses -O2 optimizer in compiling the binary.
    """
    # Path to your C++ source code
    if bfield == 3:
        cpp_file = os.path.join(
            exe_dir, "pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp")
    else:
        cpp_file = os.path.join(
            exe_dir, "pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp")

    exe_name = "n2dec.exe"
    exe_path = os.path.join(exe_dir, exe_name)

    cpp_file = os.path.join(
        exe_dir, "pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp")

    compile_command = ["g++", "-O2", cpp_file, "-o", exe_path]

    print("Compiling C++ source before execution...")
    compile_result = subprocess.run(
        compile_command, capture_output=True, text=True)

    if compile_result.stdout:
        print(compile_result.stdout)
    if compile_result.stderr:
        print(compile_result.stderr)

    if compile_result.returncode != 0 or not os.path.exists(exe_path):
        raise RuntimeError("Compilation failed! Check above for errors.")
    else:
        print("Compilation succeeded.\n")


# === Run N2DEC ===
def run_n2dec(run_number, temperature, num_pos, num_be, density, exe_dir, out_dir):
    """
    Run the n2dec executable from the 'results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/' folder.
    Input: run_number selects input file, temperature in Kelvin.
    Output files are in the 'results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/' folder.
    """

    # Handling the executable
    exe_name = 'n2dec.exe'

    # Source path: executable/n2dec
    src_exe = os.path.join(exe_dir, exe_name)

    # Destination path: results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/n2dec
    dest_exe = os.path.join(out_dir, exe_name)

    # Copy from source to destination
    shutil.copyfile(src_exe, dest_exe)

    # Make sure the binary is executable
    os.chmod(dest_exe, 0o755)

    print(f"Running N2DEC at T: {temperature:.2f} K, Num_pos: {
          num_pos:.2e}, Num_be: {num_be:.2e}, Density: {density:.2e}")

    # Save copy (Kelvin)
    temp_copy = temperature

    # Convert to electronvolts
    # temperature = temperature * 8.617333262145e-5

    # Launch an external process in the terminal under results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/ directory to run the simulation executable
    # Example: ./n2dec <run_number> <temperature in K>
    # Catches all standard output and error as text
    result = subprocess.run(
        [dest_exe, str(run_number), str(f"{temperature:.1f}")],
        cwd=out_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Save stdout and stderr to a .txt file results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/
    stdout_file = os.path.join(out_dir, f"n2dec_stdout_{run_number}_T{
                               temp_copy:.2f}_P{num_pos:.2e}_B{num_be:.2e}_D{density:.2e}.txt")
    with open(stdout_file, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    print(f'stdout & stderr logged to n2dec_stdout_{run_number}_T{
          temp_copy:.2f}_P{num_pos:.2e}_B{num_be:.2e}_D{density:.2e}.txt')

    # If the binary crashed (non-zero exit), raise an exception and print the error trace
    if result.returncode != 0:
        raise RuntimeError(f"N2DEC failed:\n{result.stderr}")

    # Check if the output files exist in results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/
    output_files = [
        "outpotzfft_00.dat",
        "outpotrfft_00.dat",
        "2spc_z_int00.dat",
        "contourfft_00.dat",
    ]

    for file in output_files:
        dst = os.path.join(out_dir, file)
        if os.path.exists(dst):
            # Optional line, for debugging
            # print(f"{file} is in results/{run_number}/{run_number}_T{temp_copy:.2f}_P{num_pos:.2e}_B{num_be:.2e}_D{density:.2e}")
            pass
        else:
            print(f"Warning: {file} not found after simulation.")

    # Clean up temporary executable copy in results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/
    if os.path.exists(dest_exe):
        os.remove(dest_exe)


# === Run N2DEC & Fit ===
def run_and_fit_param(run_number, temp, num_positrons, num_be_ions, density, input_config, exe_dir, res_dir, mcp_path, fit):
    """
    Generates a unique run directory: results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/.
    Generates appropriate input file with chosen num_positrons & num_be_ions & density.
    Runs the simulation and fits it to the mcp data, returns tuple of important parameters & chi2.
    """

    # Create unique run ID using run number, temp, num_positrons, num_be_ions, and density
    run_id = f"{run_number}_T{temp:.2f}_P{
        num_positrons:.2e}_B{num_be_ions:.2e}_D{density:.2e}"
    out_dir = os.path.join(res_dir, run_id)

    # Check if the simulation was already been executed for given parameters, happens sometimes
    if os.path.exists(out_dir):
        chi2 = fit(out_dir=out_dir, mcp_path=mcp_path)
        return (temp, num_positrons, num_be_ions, density, chi2)

    # If not, create the unique run directory
    os.makedirs(out_dir, exist_ok=True)

    # Generate unique .dat file per param set
    generate_field_inp_2sp_file(
        out_dir=out_dir,
        run_number=run_number,
        num_electrodes=input_config.num_electrodes,
        center_index=input_config.center_index,
        density=density,
        num_positrons=num_positrons,
        num_be_ions=num_be_ions,
        temp_factor=input_config.temp_factor,
        accuracy=input_config.accuracy,
        diameter=input_config.diameter,
        default_length=input_config.default_length,
        electrode_profile=input_config.electrode_profile,
    )

    # Run simulation and fit
    run_n2dec(run_number=run_number, temperature=temp, num_pos=num_positrons,
              num_be=num_be_ions, density=density, exe_dir=exe_dir, out_dir=out_dir)
    chi2 = fit(out_dir=out_dir, mcp_path=mcp_path)

    return (temp, num_positrons, num_be_ions, density, chi2)


# === Parallel N2DEC Execution Helper ===
def parallel_optimizer(run_number, temperature_choice, num_positrons_choice, num_be_ions_choice, density_choice, input_config, exe_dir, res_dir, mcp_path, array, label: Literal["Temp", "Pos", "Be", "Den"], fit):
    """
    Runs parallel cpu processes to execute N2DEC for each parameter in a given parameter array (temperature, num_pos, num_be_ions, or density).
    Results of parallel runs are sorted with respect to chi2 results.
    Returns the best-fit value with its chi2.
    """

    # Log which parameter is being optimized, save its index in the return tuple of run_and_fit_param (temp, num_pos, num_be, chi2)
    i = 0
    if label == "Temp":
        name = "temperature"
        i = 0
    elif label == "Pos":
        name = "number of positrons"
        i = 1
    elif label == "Be":
        name = "number of Be+ ions"
        i = 2
    elif label == "Den":
        name = "density"
        i = 3

    print(f"Optimizing {name}...")

    # Run in parallel using all available cores
    # To set limit on maximum number of cpu processes that can run in parallel add max_workers argument
    # Example: with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Parallel runs wrt temperature
        if label == "Temp":
            futures = [
                executor.submit(run_and_fit_param, run_number, temp, num_positrons_choice, num_be_ions_choice, density_choice,
                                input_config, exe_dir, res_dir, mcp_path, fit)
                for temp in array
            ]

        # Parallel runs wrt num_pos
        elif label == "Pos":
            futures = [
                executor.submit(run_and_fit_param, run_number, temperature_choice, num_pos, num_be_ions_choice, density_choice,
                                input_config, exe_dir, res_dir, mcp_path, fit)
                for num_pos in array
            ]

        # Parallel runs wrt num_be_ions
        elif label == "Be":
            futures = [
                executor.submit(run_and_fit_param, run_number, temperature_choice, num_positrons_choice, num_be, density_choice,
                                input_config, exe_dir, res_dir, mcp_path, fit)
                for num_be in array
            ]

        elif label == "Den":
            futures = [
                executor.submit(run_and_fit_param, run_number, temperature_choice, num_positrons_choice, num_be_ions_choice, dens,
                                input_config, exe_dir, res_dir, mcp_path, fit)
                for dens in array
            ]

        # result is list of tuples in the form [(temp, num_positrons, num_be_ions, density, chi2), ...]
        results = [f.result()
                   for f in concurrent.futures.as_completed(futures)]

        # Sort results by chi2, last element of the tuple
        results.sort(key=lambda x: x[-1])

        # Extract best tuple with lowest chi2
        best = results[0]

        # Best parameter (temperature, num_pos, or num_be_ions) and its chi2
        best_par = best[i]
        best_chi2 = best[-1]
        print(f"Best fit {name} = {best[i]:.2e} with chi2 = {best[-1]:.2f}")

        return best_par, best_chi2


# === Error Calculation for Best Fit Parameters ===
def calculate_error(run_number, deltas, best_temp, best_num_pos, best_num_be_ions, best_density, best_chi2, input_config, exe_dir, res_dir, mcp_path, fit):
    """
    Calculates parameter uncertainties using finite differences and full correlation matrix.
    """
    # Array of best parameters
    best_pars = [best_temp, best_num_pos, best_num_be_ions, best_density]

    # Initialize array for perturbated chi2 results
    perturbed_chis = [[None, None] for _ in range(4)]

    # Step 1: Run the simulation at +- delta for each parameter
    jobs = []

    # jobs: [[i, 0 or 1, (temp, num_pos, num_be, dens)], ...]
    for i in range(4):
        # Perturb +delta
        plus = best_pars.copy()
        plus[i] += deltas[i]

        # param_index, sign = 0 (plus)
        jobs.append((i, 0, tuple(plus)))

        # Perturb -delta
        minus = best_pars.copy()
        minus[i] -= deltas[i]

        # param_index, sign = 1 (minus)
        jobs.append((i, 1, tuple(minus)))

        # run_ids and parameter values for error bar analysis plots
        if i == 0:
            run_id_temp1 = f"{run_number}_T{plus[i]:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_temp2 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_temp3 = f"{run_number}_T{minus[i]:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            values_temp = [plus[i], best_temp, minus[i]]
        elif i == 1:
            run_id_pos1 = f"{run_number}_T{best_temp:.2f}_P{
                plus[i]:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_pos2 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_pos3 = f"{run_number}_T{best_temp:.2f}_P{
                minus[i]:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            values_pos = [plus[i], best_num_pos, minus[i]]
        elif i == 2:
            run_id_be1 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{plus[i]:.2e}_D{best_density:.2e}"
            run_id_be2 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_be3 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{minus[i]:.2e}_D{best_density:.2e}"
            values_be = [plus[i], best_num_be_ions, minus[i]]
        else:
            run_id_den1 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{plus[i]:.2e}"
            run_id_den2 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{best_density:.2e}"
            run_id_den3 = f"{run_number}_T{best_temp:.2f}_P{
                best_num_pos:.2e}_B{best_num_be_ions:.2e}_D{minus[i]:.2e}"
            values_den = [plus[i], best_density, minus[i]]

    # Step 2: Run all 8 perturbed simulations in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_and_fit_param, run_number, temp, num_pos, num_be, dens,
                            input_config, exe_dir, res_dir, mcp_path, fit)
            for (_, _, (temp, num_pos, num_be, dens)) in jobs
        ]

        results = [f.result() for f in futures]

    # Step 3: Reassociate results with job metadata
    for job, res in zip(jobs, results):
        i, sign, _ = job
        chi2 = res[-1]
        perturbed_chis[i][sign] = chi2

    # Step 4: Build C matrix
    # Approximate gradient of chi2 with respect to each parameter using central difference:
    # d_chi2_i = (chi2(par_i + deltas[i], rest_best_pars) - chi2(par_i - deltas[i], rest_best_pars)) / (2*deltas[i])
    # C_ij = d_chi2_i * d_chi2_j
    # This captures how chi2 changes with parameters, including their correlations
    C = np.zeros((4, 4))

    for i in range(4):
        dchi_i = (perturbed_chis[i][0] -
                  perturbed_chis[i][1]) / (2 * deltas[i])
        for j in range(4):
            dchi_j = (perturbed_chis[j][0] -
                      perturbed_chis[j][1]) / (2 * deltas[j])
            C[i, j] = dchi_i * dchi_j

    # Step 5: Invert C matrix and compute uncertainties
    # Pseudo-inverse for robust error bars
    C_inv = np.linalg.pinv(C)

    # Estimate degrees of freedom from MCP file length
    mcp_file = os.path.join(os.getcwd(), "mcp_results", mcp_path)
    data = np.loadtxt(mcp_file)
    N_points = len(data[:, 0])
    dof = N_points - 4

    # Uncertainy calculation, in the order of [temp, num_pos, num_be_ions, density]
    # sig_i = sqrt((best_chi2/dof) * C_inv_ii)
    uncertainties = np.sqrt((best_chi2 / dof) * np.diag(C_inv))

    # Error bar analysis plots
    out_dir_temp1 = os.path.join(res_dir, run_id_temp1)
    out_dir_temp2 = os.path.join(res_dir, run_id_temp2)
    out_dir_temp3 = os.path.join(res_dir, run_id_temp3)

    out_dir_pos1 = os.path.join(res_dir, run_id_pos1)
    out_dir_pos2 = os.path.join(res_dir, run_id_pos2)
    out_dir_pos3 = os.path.join(res_dir, run_id_pos3)

    out_dir_be1 = os.path.join(res_dir, run_id_be1)
    out_dir_be2 = os.path.join(res_dir, run_id_be2)
    out_dir_be3 = os.path.join(res_dir, run_id_be3)

    out_dir_den1 = os.path.join(res_dir, run_id_den1)
    out_dir_den2 = os.path.join(res_dir, run_id_den2)
    out_dir_den3 = os.path.join(res_dir, run_id_den3)

    best_dir = os.path.join(res_dir, "best_fit")

    plot_radial_profile_multiple(
        out_dir_temp1, out_dir_temp2, out_dir_temp3, mcp_path, best_dir, "Temp", values_temp)
    plot_radial_profile_multiple(
        out_dir_pos1, out_dir_pos2, out_dir_pos3, mcp_path, best_dir, "Pos", values_pos)
    plot_radial_profile_multiple(
        out_dir_be1, out_dir_be2, out_dir_be3, mcp_path, best_dir, "Be", values_be)
    plot_radial_profile_multiple(
        out_dir_den1, out_dir_den2, out_dir_den3, mcp_path, best_dir, "Den", values_den)

    return uncertainties
