# DecodingAnalysis

Tools for preprocessing neural & behavioral data, training & testing LSTM-based decoders and postprocessing results.

Code written by **Sidney Rafilson** in the laboratory of **Matt Smear**.

## Overview
Decoding analysis for the clickbait-ephys experiments designed by **Nate Hess**: freely moving simultanous tetrode recordings from the olfactory bulb and hippocampus with thermistor respiratory measurement and bottom up video tracking.
- **core.py** – functions for loading Kilosort outputs, computing smoothed spike rates, and preparing datasets.
- **LSTM_decoding.py** – training script for LSTM models that decode behavioral variables from neural activity.
- **envirnment.yml** – conda environment with required dependencies.
- **example_notebook.ipynb** and **postprocessing.ipynb** – notebooks for exploratory analysis and post-processing.
- **submit_LSTM.sh** – SLURM job array script for the University of Oregon's Talapas cluster, allowing each session and hyperparameter combination to run on parallel GPU cores.

## Setup
Create the conda environment and activate it:
```bash
conda env create -f envirnment.yml
conda activate decoding-env
```

Next install the approprate version of PyTorch for the machine. For example on Linux:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Usage
Run the LSTM decoding script specifying data and output directories and optional parameters:
```bash
python LSTM_decoding.py --dir /path/to/data --save_dir /path/to/output \
    --mouse 6002 --session 9 --use_behaviors "['position_x','position_y']" \
    --window_size 0.1 --step_size 0.1
```
See `python LSTM_decoding.py -h` for the full list of options.

Or submit a SLURM job-array to parallelize across GPU nodes:
```bash
sbatch submit_LSTM.sh
```

## Notebooks
Open the notebooks in JupyterLab to explore preprocessing steps and analyze model outputs:
```bash
jupyter lab example_notebook.ipynb
```

## Contributing
Issues and pull requests are welcome. Please ensure that any code changes are accompanied by appropriate tests or validation.

