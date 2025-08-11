# DecodingAnalysis

Tools for preprocessing neural data and training LSTM-based decoders of behavior.

## Overview
- **core.py** – functions for loading Kilosort outputs, computing smoothed spike rates, and preparing datasets.
- **LSTM_decoding.py** – training script for LSTM models that decode behavioral variables from neural activity.
- **decoding_env_backup.yml** – conda environment with required dependencies.
- **example_notebook.ipynb** and **posprocessing.ipynb** – notebooks for exploratory analysis and post-processing.
- **submit_LSTM.sh** – SLURM job array script for the University of Oregon's Talapas cluster, allowing each session and hyperparameter combination to run on parallel GPU cores.

## Setup
Create the conda environment and activate it:
```bash
conda env create -f decoding_env_backup.yml
conda activate decoding
```

## Usage
Run the LSTM decoding script specifying data and output directories and optional parameters:
```bash
python LSTM_decoding.py --dir /path/to/data --save_dir /path/to/output \
    --mouse 6002 --session 9 --use_behaviors "['position_x','position_y']" \
    --window_size 0.1 --step_size 0.1
```
See `python LSTM_decoding.py -h` for the full list of options.

## Notebooks
Open the notebooks in JupyterLab to explore preprocessing steps and analyze model outputs:
```bash
jupyter lab example_notebook.ipynb
```

## Contributing
Issues and pull requests are welcome. Please ensure that any code changes are accompanied by appropriate tests or validation.

