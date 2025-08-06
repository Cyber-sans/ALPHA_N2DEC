# N2DEC_ANALYSIS_V1 #

**Project: `N2DEC_ANALYSIS_V1`**

**Author: Deniz Yoldas**

**Colloboration: ALPHA**

This project is designed to estimate the temperature of Be+ ions & positrons in a sympathetically cooled Be+ / positron plasma by fitting simulated density profiles to experimental MCP image from Be+ cold dump.

It executes **N2DEC** simulation multiple times to find best-fit with the MCP data.

***Note: I would recommend you to use `VScode`, layout will be clean and it supports .md & .ipynb file types.***

***Note: The N2DEC is a heavy simulation, so program may take hours to run!***

***Note: For a cold dump of estimated temperature of ~5 Kelvins, with all optimization flags set to `True` and error bar calculation implemented, it takes ~6 hours to run on a 15 core CPU!***

***Recommended number of CPU cores to exploit parallel optimization: >=10***

***Note: If you have a slow computer with few CPU cores, check out `How to Use LXPLUS Clusters of CERN` section.***

---

## Structure ##
```
N2DEC_ANALYSIS_V1/
├── README.md                       # Project overview and user guide
├── my_n2dec_script.py              # Main script
├── utils.py                        # Plotting and fitting utility functions for the main script
├── utils2.py                       # Optimization & error calculation utility functions for the main script
├── mcp_image_analysis.py           # Helper function for MCP image analysis
├── input_config.yaml               # Initial input configurations for the program 
├── mcp_analysis_test.ipynb         # Jupyter notebook for interactive MCP image analysis tests
│
├── executable/                     # Contains the N2DEC source code and compiled binary
│   ├── pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp  # C++ N2DEC code
│   └── n2dec                       # Compiled executable to run simulation (will be compiled automatically when you run the program)
│
├── results/<run_number>/
│   ├── input_config.yaml           # Copy of the input_config.yaml for given run
│   │
│   ├── <run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/  # Output from simulation runs
│   │   ├── outpotzfft_00.dat              # Axial (z) cuts: potential and densities vs. z (fixed r)
│   │   ├── outpotrfft_00.dat              # Radial (r) cuts: potential and densities vs. r (fixed z)
│   │   ├── 2spc_z_int00.dat               # Line-integrated densities vs. radius (projection to 1D)
│   │   ├── contourfft_00.dat              # 2D grid: densities of both species for full plasma map (z, r)
│   │   ├── field_inp_2sp_<run_number>.dat # Input file fed into N2DEC for given run
│   │   ├── n2dec_stdout_<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>.txt
│   │   └── fit.png                        # Plot of fitting results for given run
│   │
│   └── best_fit                    # Plots and summary for the best-fit simulation run
│       ├── beplus_density_mcp.png         # Integrated (Phi & z) 1D radial density profile for the mcp image choosen for the fitting
│       ├── beplus_density_sim.png         # Integrated (Phi & z) 1D radial density profile for the best-fit simulation output
│       ├── beplus_density.png             # MCP and N2DEC density profiles fitted to single plot
│       ├── multiple_density_sim.png       # Best-fit 1D radial density profiles for multiple species
│       ├── fit.png
│       ├── best_fit_summary.txt           # Summary of the best-fit run, error bars, best fit parameters
│       ├── beplus_density_err_Temp.png    # Error bar results: Fits for multiple temperatures plotted
│       ├── beplus_density_err_Pos.png     # Error bar results: Fits for multiple number of positrons
│       ├── beplus_density_err_Be.png      # Error bar results: Fits for multiple number of Be+ ion
│       └── beplus_density_err_Den.png     # Error bar results: Fits for multiple densities
│
├── mcp_results/                    
│   ├── originals/                  # Original MCP images (.tif files) to be analyzed
│   ├── processed/                  # Already processed & exctracted radial density profiles (.txt files)
│   │
│   └── <run_number>
│       ├── radial_profile<file_name>.txt  # Radial density profile
│       ├── image_analysis_summary.txt     # MCP analysis results: Summary
│       ├── auto_mask.png                  # MCP analysis results: Automatic masks generated
│       ├── edge_detection.png             # MCP analysis results: Detected edges on image
│       ├── elliptical_fit.png             # MCP analysis results: Ellipse fits to edges
│       ├── elliptical_fit_on_image.png    # MCP analysis results: Ellipse fits on image
│       └── radial_profile.png             # MCP analysis results: Radial density profile plotted
│
└── ___pycache__/                   # Ignore
```

---

## Getting Started ##

Before you run the program make sure that the MCP image (.tif file) you want to analyze is in the `mcp_results/originals/` directory.
If you are going to use already extracted radial profile (.txt file), then make sure that it is in the `mcp_results/processed/` directory.

---

## Input Configurations, `input_config.yaml` ##

For desired input parameters, modify the values inside `input_config.yaml` file accordingly. Inside the file, you will see the following structure:
```
inputs:
  temperature: 6.0              # Temperature guess (in Kelvin, current N2DEC only accepts integer part)
  num_electrodes: 14            # Number of electrodes
  center_index: 5               # Index of the center electrode
  density: 62e12                # Initial density
  num_positrons: 0.11e6         # Positron number guess
  num_be_ions: 0.33e6           # Be+ ion number guess
  temp_factor: 0.50             # Temperature iteration factor
  accuracy: 2e-8                # Resolution for the simulaton iterations
  diameter: 44.55e-3            # Diameter of electrodes in meters
  default_length: 20.05e-3      # Length of electrodes in meters
  electrode_profile: [0, 72.0, 72.0, 72.0, -72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 0]      # Voltage profile of the electrodes
  mcp_path: '0840_38.764.tif'   # MCP image used for fitting
  optimize_temp: True           # Do temperature optimization to find the best-fit temperature in the order of a Kelvin
  optimize_num_pos: False       # Do positron number optimization around the initial guess
  optimize_num_be_ions: False   # Do Be+ ion number optimization around the initial guess
  optimize_density: False       # Do density optimization around the initial guess
  fit_method: "interpolate"     # Choose between "interpolate" & "bin" method for fitting N2DEC to MCP data
  calculate_error: False        # Calculate error bars for best fit parameters
  use_processed_mcp: False      # If you want to use already extracted radial density profile.
  processed_mcp_path: 'radial_profile0840_38.764.txt'
```

**After you execute the program, for each simulation call, the `generate_field_inp_2sp_file` function will create `field_inp_2sp_<run_number>.dat` file within the `results/<run_number>/<run_number>_T<temp>_P<num_pos>_B<num_be>_D<density>/` directory for that specific run.**

**This file is used as the inputs for the simulation executable, it includes first a single line containing:**

- (number of electrodes), (index of central electrode relative to plasma position), (est. density of central species), (number of positrons), (number of Be ions), (temperature factor), (acc), (diameter of electrodes)

**This is followed by a list of electrodes where each line give:**

- (length), (voltage)

**Explanation:**

- **number of electrodes:** gives the length of the list of electrodes

- **index (1-based) of central electrode (plasma position)** indicates in which electrode the plasma is centered (approximately)

- **est. density of central species:** is the initial estimated central density m^-3.

- **number of positrons:** the number of positrons

- **number of Be ions:** the number of ions

- **temperature factor:** iteration factor, should be below 1, 0.5 is a good start

- **acc:** is the resolution to which the iteration should go 2e-8 is reasonable

- **diameter of electrodes:** in unit of m. 

---

## Running the Program ##
- **Temperature optimization:** To find the best-fit temperature in the order of a Kelvin, given the range 6-305 Kelvins, set `optimize_temp` to `True` in your input configuration file. (Default is `False`.)

- **Positron number optimization:** To find the best-fit positron number near your initial guess (±%10 default), set `optimize_num_pos` to `True` in your input configuration file. (Default is `False`.)

- **Be+ ion number optimizatiom:** To find the best-fit Be+ ion number near your initial guess (±%10 default), set `optimize_num_be_ions` to `True` in your input configuration file. (Default is `False`.)

- **Density optimization:** To find the best-fit density near your initial guess (±%10 default), set `optimize_density` to `True` in your input configuration file. (Default is `False`.)

*In default `input_config.yaml` configurations, where all optimization flags are set to `False`, the program will just execute one N2DEC simulation at the given initial guesses and will compare it to the MCP image.*

To run the program, execute the following command in your terminal:
```python
python my_n2dec_script.py
```
Follow the terminal prompts to enter the **run number**.

***Note: Do not forget to run the `caffeinate -id` command (MacOS) on a seperate terminal window, the program may take hours to run, display or idle sleep can distrupt the process.***

***Note: The program will compile `pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp` automatically, ignore the error message of the -O2 compiler if you see the message `Compilation succeeded` afterwards.***

---

## How to Use LXPLUS Clusters of CERN
This program requires multiple runs of the heavy `pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp` simulation. 

If your computer is not feasible for heavy simulations, or have few available cpu cores, you can exploit LXPLUS (Linux Public Login User Service) service of CERN. They provide clusters where you can run this program remotely at.

For more info on how to set up LXPLUS, please see [this website](https://abpcomputing.web.cern.ch/computing_resources/lxplus/). Normally, you need to request the activation of "AFS Workspaces" and of "LXPLUS and linux" for your account on [CERN Resource Portal](https://resources.web.cern.ch/resources/Manage/ListServices.aspx).

***Note: The default AFS storage quota is 2 GB, increase it to 10 GB from settings tab [here](sources.web.cern.ch/resources/Manage/AFS). The python environment and the program's result folder will take up some space so enough storage is necessary.***

After obtaining your account and increasing your storage quota, make sure that you have appropriate python environment to run the program. You can find the steps for installation [here](https://abpcomputing.web.cern.ch/guides/python_inst/). Also make sure to install numpy, scipy, scikit-image, matplotlib, pyyaml, and dacite packages to your environment. Usually, you should run these commands from your LXPLUS environment:
```bash
conda install numpy matplotlib scipy pyyaml scikit-image
pip install dacite
```

### Copying the Pipeline ###
Copy the full pipeline to the LXPLUS server safely. To do so, run the following command from the directory where your `n2dec_analysis_v1` folder is on your local machine:
```bash
scp -r n2dec_analysis_v1 <your_username>@lxplus.cern.ch:~
```
***Enter your password after prompted.***

Then, login to remote machine using following command:
```bash
ssh -X <your_username>@lxplus.cern.ch
```
***Enter your password after prompted.***

You should see the `n2dec_analysis_v1` folder in your home AFS directory on CERN LXPLUS.

### Modifying Inputs ###
I would prefer modifying inputs before copying it to remote machine, but here is the way to modify `input_config.yaml` file from the LXPLUS terminal.

Install simple yaml parsing:
```bash
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ~/yq
chmod +x ~/yq
```
You should see `yq` in the remote home directory.

To update a parameter, such as the mcp_path, run the following command:
```bash
~/yq -i '.inputs.mcp_path = "1046_49.355.tif"' input_config.yaml
```

### Running the Program ###
Execute the following command to run:
```python
python my_n2dec_script.py
```
Follow the terminal prompts to enter the **run number**.

***Note: Do not forget to run the `caffeinate -id` command (for MacOS) on a seperate terminal window, display or idle sleep can cause disconnection to the LXPLUS machine.***

***Note: Another option is to simply connect to `alphacpc28` using general `alpha` username, then connect to LXPLUS from there to run your program. `alphacpc28` is always awake so you won't get any connection distruptions to LXPLUS. `tmux` is another option, but has some problems...***

### Pulling Results ###
To pull the results, execute the following command:
```bash
scp -r <your_username>@lxplus.cern.ch:~/n2dec_analysis_v1/results/<run_number>/ ~/Desktop/
```

If you also want the image analysis results:
```bash
scp -r <your_username>@lxplus.cern.ch:~/n2dec_analysis_v1/mcp_results/<run_number>/ ~/Desktop/
```
***Enter your password after prompted.***

---

## Current N2DEC Version ##
For this project, the version of the N2DEC used, `pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp`, is originally from [this repository](https://gitlab.cern.ch/alpha/Beryllium/N2DEC_Repo/-/blob/main/pbarcoolpot_fft_wn_2spec_nonpert_Bex3_lxp.cpp?ref_type=heads). However, I made some edits to it such as changing the Bfield to 1 Tesla & extra filewriting at the end. It should be under `executable` folder. If missing, please download it from the project repository and add it to the `executable` folder.

---

## TL;DR ##
You have a good and stationed computer (>=10 CPU cores), don't have time to read the manual, and just want to extract best-fit result:

0) Ensure that you are within `n2dec_analysis_v1` directory and the MCP image (.tif file) you want to analyze is in `n2dec_analysis_v1/mcp_results/originals/` directory.

1) Go to `input_config.yaml` and set the input paramerers, also set the flags `optimize_temp`, `optimize_num_pos`, `optimize_num_be_ions`, `optimize_density` to `True`.

2) Execute the following command to run (enter the run number when asked):
```python
python my_n2dec_script.py
```

3) Navigate to the `results/<run_number>/best_fit` directory to see the fitting results.

---

## Optional Tricks ##

To capture all terminal output, run your Python script with redirection in the terminal:
```python
python my_n2dec_script.py > my_program_log.txt 2>&1
```
- `>` writes all standard output to `my_program_log.txt`
- `2>&1` redirects all error messages (stderr) to the same file