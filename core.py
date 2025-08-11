import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import subprocess
import time



"""
Data loading and preprocessing
"""
def compute_spike_rates(kilosort_dir: str, sampling_rate: int, window_size: float = 1.0, step_size: float = 0.5, use_units: str = 'all', sigma: float = 2.5, zscore: bool = True):
    
    """
    Compute smoothed spike rates for neural units in OB (olfactory bulb) and HC (hippocampus) regions 
    using a sliding window approach from Kilosort output data.
    
    This function processes spike times and cluster assignments from Kilosort/Phy2, separates units by 
    brain region based on channel mapping, calculates firing rates within sliding time windows, and 
    applies Gaussian smoothing. Optionally, z-scoring can be applied to normalize firing rates.
    
    Parameters
    ----------
    kilosort_dir : str
        Path to the directory containing Kilosort output files.
    sampling_rate : int
        Sampling rate of the recording in Hz.
    window_size : float, optional
        Size of the sliding window in seconds, default is 1.0.
    step_size : float, optional
        Step size for sliding window advancement in seconds, default is 0.5.
    use_units : str, optional
        Filter for unit types to include:
        - 'all': Include all units
        - 'good': Include only good units
        - 'mua': Include only multi-unit activity
        - 'good/mua': Include both good units and multi-unit activity
        - 'noise': Include only noise units
        Default is 'all'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing kernel, default is 2.5.
    zscore : bool, optional
        Whether to z-score the firing rates, default is True.
    
    Returns
    -------
    spike_rate_matrix_OB : ndarray
        Matrix of spike rates for OB units (shape: num_OB_units × num_windows).
    spike_rate_matrix_HC : ndarray
        Matrix of spike rates for HC units (shape: num_HC_units × num_windows).
    time_bins : ndarray
        Array of starting times for each window.
    ob_units : ndarray
        Array of unit IDs for OB region.
    hc_units : ndarray
        Array of unit IDs for HC region.
    
    Notes
    -----
    - OB units are assumed to be on channels 16-31
    - HC units are assumed to be on channels 0-15
    - Firing rates are computed in Hz (spikes per second)
    
    Raises
    ------
    FileNotFoundError
        If any required Kilosort output files are missing.
    """    

    # Load spike times and cluster assignments
    spike_times_path = os.path.join(kilosort_dir, "spike_times.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")  # Cluster assignments from Phy2 manual curation
    templates_path = os.path.join(kilosort_dir, "templates.npy")
    templates_ind_path = os.path.join(kilosort_dir, "templates_ind.npy")
    cluster_groups_path = os.path.join(kilosort_dir, "cluster_group.tsv")

    # Ensure all required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path, templates_path, templates_ind_path, cluster_groups_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Loading the data
    templates = np.load(templates_path)  # Shape: (nTemplates, nTimePoints, nChannels)
    templates_ind = np.load(templates_ind_path)  # Shape: (nTemplates, nChannels)
    spike_times = np.load(spike_times_path) / sampling_rate  # Convert to seconds
    spike_clusters = np.load(spike_clusters_path)
    cluster_groups = np.loadtxt(cluster_groups_path, dtype=str, skiprows=1, usecols=[1])

    # Find peak amplitude channel for each template and assign to unit
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    unit_best_channels = {unit: templates_ind[unit, peak_channels[unit]] for unit in range(len(peak_channels))}
    
    # Filter units based on use_units parameter
    if use_units == 'all':
        unit_best_channels = unit_best_channels
    elif use_units == 'good':
        unit_indices = np.where(cluster_groups == 'good')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'mua':
        unit_indices = np.where(cluster_groups == 'mua')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'good/mua':
        unit_indices = np.where(np.isin(cluster_groups, ['good', 'mua']))[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'noise':
        unit_indices = np.where(cluster_groups == 'noise')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}


    # Get total duration of the recording
    recording_duration = np.max(spike_times)

    # Define time windows
    time_bins = np.arange(0, recording_duration - window_size, step_size)
    num_windows = len(time_bins)

    # Separate OB and HC units
    hc_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(0, 16)])
    ob_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(16, 32)])
    num_ob_units = len(ob_units)
    num_hc_units = len(hc_units)

    # Initialize spike rate matrices
    spike_rate_matrix_OB = np.zeros((num_ob_units, num_windows))
    spike_rate_matrix_HC = np.zeros((num_hc_units, num_windows))

    # Compute spike counts in each window
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size

        # Find spikes in this window
        in_window = (spike_times >= t_start) & (spike_times < t_end)
        spike_clusters_in_window = spike_clusters[in_window]

        # Compute spike rates for OB
        for j, unit in enumerate(ob_units):
            spike_rate_matrix_OB[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

        # Compute spike rates for HC
        for j, unit in enumerate(hc_units):
            spike_rate_matrix_HC[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

    # Apply Gaussian smoothing
    for j in range(num_ob_units):
        if sigma > 0:
            spike_rate_matrix_OB[j, :] = gaussian_filter1d(spike_rate_matrix_OB[j, :], sigma=sigma)

    for j in range(num_hc_units):
        if sigma > 0:
            spike_rate_matrix_HC[j, :] = gaussian_filter1d(spike_rate_matrix_HC[j, :], sigma=sigma)

    # Apply Z-scoring (optional)
    if zscore:
        def z_score(matrix):
            mean_firing = np.mean(matrix, axis=1, keepdims=True)
            std_firing = np.std(matrix, axis=1, keepdims=True)
            std_firing[std_firing == 0] = 1  # Prevent division by zero
            return (matrix - mean_firing) / std_firing

        spike_rate_matrix_OB = z_score(spike_rate_matrix_OB)
        spike_rate_matrix_HC = z_score(spike_rate_matrix_HC)

    return spike_rate_matrix_OB, spike_rate_matrix_HC, time_bins, ob_units, hc_units



def compute_sniff_freqs_bins(sniff_params_file: str, time_bins: np.ndarray, window_size: float, sfs: int):
    inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(sniff_params_file)
    inhalation_times = inhalation_times / sfs  # Convert to seconds

    # Compute sniff frequencies
    freqs = 1 / np.diff(inhalation_times)  # Instantaneous frequency between inhalations

    # Remove unrealistic frequencies
    bad_indices = np.where((freqs > 14) | (freqs < 0.8))[0]
    freqs = np.delete(freqs, bad_indices)
    inhalation_times = np.delete(inhalation_times[:-1], bad_indices)  # Align with freqs

    # Initialize outputs
    mean_freqs = np.full(len(time_bins), np.nan)
    inhalation_latencies = np.full(len(time_bins), np.nan)
    phases = np.full(len(time_bins), np.nan)

    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        in_window = (inhalation_times >= t_start) & (inhalation_times < t_end)

        # Find last inhalation before the center of the bin
        last_idx = np.where(inhalation_times < middle)[0]
        if len(last_idx) > 0:
            last_idx = last_idx[-1]
            last_inh_time = inhalation_times[last_idx]
            inhalation_latencies[i] = middle - last_inh_time

            # Phase: time since last inhalation / duration of current sniff
            if last_idx < len(freqs):  # Ensure freq is defined
                sniff_duration = 1 / freqs[last_idx]
                phase_fraction = (middle - last_inh_time) / sniff_duration
                phases[i] = (phase_fraction % 1) * 2 * np.pi  # Convert to radians
        else:
            inhalation_latencies[i] = np.nan
            phases[i] = np.nan

        # Mean frequency in the bin
        if np.any(in_window):
            mean_freqs[i] = np.nanmean(freqs[in_window])

    return mean_freqs, inhalation_latencies, phases



def align_brain_and_behavior(events: pd.DataFrame, spike_rates: np.ndarray, units: np.ndarray, time_bins: np.ndarray, window_size: float = 0.1, speed_threshold: float = 100, interp_method = 'linear', order = None):
    
    """
    Align neural spike rate data with behavioral tracking data using time windows.
    
    This function matches neural activity from spike rates with behavioral metrics (position, velocity, speed)
    by finding the closest behavioral event to the middle of each time bin. It creates a unified dataframe
    containing both neural and behavioral data, removes outliers based on speed threshold, and interpolates
    missing values.
    
    Parameters
    ----------
    events : pd.DataFrame
        Behavioral tracking data containing columns:
        - 'timestamps_ms': Timestamps in milliseconds
        - 'position_x', 'position_y': Position coordinates
        - 'velocity_x', 'velocity_y': Velocity components
        - 'speed': Overall movement speed
        - 'reward_state': Reward state indicator
    
    spike_rates : np.ndarray
        Matrix of spike rates with shape (n_units, n_time_bins).
    
    units : np.ndarray
        Array of unit IDs corresponding to rows in spike_rates.
    
    time_bins : np.ndarray
        Array of starting times for each time bin in seconds.
    
    window_size : float, optional
        Size of each time window in seconds, default is 0.1.
    
    speed_threshold : float, optional
        Threshold for removing speed outliers, expressed as multiplier of standard deviation, 
        default is 4.0 (values > 4 × std are treated as outliers).
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with aligned neural and behavioral data containing:
        - Unit columns: Spike rates for each neural unit
        - 'x', 'y': Position coordinates
        - 'v_x', 'v_y': Velocity components
        - 'speed': Movement speed
        - 'time': Time bin start times
        
    Notes
    -----
    - For each time bin, the behavioral event closest to the middle of the bin is selected
    - Speed outliers are removed using a threshold based on standard deviation
    - Missing values are interpolated using linear interpolation
    - Rows with missing behavioral data (typically at beginning/end of recording) are removed
    """

    # Initialize arrays for holding aligned data
    mean_positions_x = np.full(len(time_bins), np.nan)
    mean_positions_y = np.full(len(time_bins), np.nan)
    mean_velocities_x = np.full(len(time_bins), np.nan)
    mean_velocities_y = np.full(len(time_bins), np.nan)
    mean_speeds = np.full(len(time_bins), np.nan)
    mean_rewards = np.full(len(time_bins), np.nan)

    # getting event times in seconds
    event_times = events['timestamp_ms'].values / 1000

    # Calculate mean behavior in each time bin
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        if np.any(event_times < middle):
            nearest_event_index = np.argmin(np.abs(event_times - middle))
            mean_positions_x[i] = events['position_x'].iloc[nearest_event_index]
            mean_positions_y[i] = events['position_y'].iloc[nearest_event_index]
            mean_velocities_x[i] = events['velocity_x'].iloc[nearest_event_index]
            mean_velocities_y[i] = events['velocity_y'].iloc[nearest_event_index]
            mean_speeds[i] = events['speed'].iloc[nearest_event_index]
            mean_rewards[i] = events['reward_state'].iloc[nearest_event_index]
        else:
            mean_positions_x[i] = np.nan
            mean_positions_y[i] = np.nan
            mean_velocities_x[i] = np.nan
            mean_velocities_y[i] = np.nan
            mean_speeds[i] = np.nan
            mean_rewards[i] = np.nan


    # converting the spike rate matrix to a DataFrame
    data = pd.DataFrame(spike_rates.T, columns=[f"Unit {i}" for i in units])

    # adding the tracking data to the DataFrame
    conversion = 5.1
    data['x'] = mean_positions_x / conversion # convert to cm
    data['y'] = mean_positions_y / conversion # convert to cm
    data['v_x'] = mean_velocities_x / conversion # convert to cm/s
    data['v_y'] = mean_velocities_y / conversion # convert to cm/s
    data['speed'] = mean_speeds / conversion # convert to cm/s
    data['time'] = time_bins # in seconds
    data['reward_state'] = mean_rewards

    
    data.loc[data['speed'] > speed_threshold, ['x', 'y', 'v_x', 'v_y', 'speed']] = np.nan

    # interpolating the tracking data to fill in NaN values
    data.interpolate(method=interp_method, inplace=True, order = order)

    # Finding the trial number and getting the click time
    trial_ids = np.zeros(data.shape[0])
    click_event = np.zeros(data.shape[0])
    for i in range(1, len(data)):
        trial_ids[i] = trial_ids[i-1]
        if data['reward_state'].iloc[i-1] and not data['reward_state'].iloc[i]:
            trial_ids[i] += 1
            click_event[i] = 1
    data = data.assign(trial_id = trial_ids, click = click_event)

    return data



def load_behavior(behavior_file: str, tracking_file: str = None) -> pd.DataFrame:

    """
    Load and preprocess behavioral tracking data from a CSV file.
    
    This function loads movement tracking data, normalizes spatial coordinates by
    centering them around zero, calculates velocity components and overall speed,
    and returns a filtered dataframe with relevant movement metrics.
    
    Parameters
    ----------
    behavior_file : str
        Path to the CSV file containing behavioral tracking data. The file should
        include columns for 'centroid_x', 'centroid_y', and 'timestamps_ms'.
        
    Returns
    -------
    events : pandas.DataFrame
        Processed dataframe containing the following columns:
        - 'time': Original time values
        - 'centroid_x': Zero-centered x coordinates 
        - 'centroid_y': Zero-centered y coordinates
        - 'velocity_x': Rate of change in x position
        - 'velocity_y': Rate of change in y position
        - 'speed': Overall movement speed (Euclidean norm of velocity components)
        - 'timestamps_ms': Timestamps in milliseconds
        
    Notes
    -----
    - Position coordinates are normalized by subtracting the mean to center around zero
    - Velocity is calculated using first-order differences (current - previous position)
    - The first velocity value uses the first position value as the "previous" position
    - Speed is calculated as the Euclidean distance between consecutive positions
    """

    # Load the behavior data
    events = pd.read_csv(os.path.join(behavior_file, 'events.csv'))

    if tracking_file:
        # Load the SLEAP tracking data from the HDF5 file
        f = h5py.File(tracking_file, 'r')
        nose = f['tracks'][:].T[:, 0, :]
        nose = nose[:np.shape(events)[0], :]
        mean_x, mean_y = np.nanmean(nose[:, 0]), np.nanmean(nose[:, 1])
        events['position_x'] = nose[:, 0] - mean_x
        events['position_y'] = nose[:, 1] - mean_y
        
    else:
        # zero-mean normalize the x and y coordinates
        mean_x, mean_y = np.nanmean(events['centroid_x']), np.nanmean(events['centroid_y'])
        events['position_x'] = events['centroid_x'] - mean_x
        events['position_y'] = events['centroid_y'] - mean_y

    # Estimating velocity and speed
    events['velocity_x'] = np.diff(events['position_x'], prepend=events['position_x'].iloc[0])
    events['velocity_y'] = np.diff(events['position_y'], prepend=events['position_y'].iloc[0])
    events['speed'] = np.sqrt(events['velocity_x']**2 + events['velocity_y']**2)



    # keeping only the columns we need
    events = events[['position_x', 'position_y', 'velocity_x', 'velocity_y', 'reward_state', 'speed', 'timestamp_ms']]
    return events



def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array
    '''

    mat = loadmat(file)
    sniff_params = mat['sniff_params']

    # loading sniff parameters
    inhalation_times = sniff_params[:, 0]
    inhalation_voltage = sniff_params[:, 1]
    exhalation_times = sniff_params[:, 2]
    exhalation_voltage = sniff_params[:, 3]

    # bad sniffs are indicated by 0 value in exhalation_times
    bad_indices = np.where(exhalation_times == 0)


    # removing bad sniffs
    inhalation_times = np.delete(inhalation_times, bad_indices)
    inhalation_voltage = np.delete(inhalation_voltage, bad_indices)
    exhalation_times = np.delete(exhalation_times, bad_indices)
    exhalation_voltage = np.delete(exhalation_voltage, bad_indices)

    return inhalation_times.astype(np.int32), inhalation_voltage, exhalation_times.astype(np.int32), exhalation_voltage

# Main preprocessing function to load and align data

def preprocess(data_dir, save_dir, mouse, session, window_size, step_size, sigma_smooth, use_units, nfs = 30_000, sfs = 1_000):

    # Loading the neural data and computing the spike rates
    kilosort_dir = os.path.join(data_dir, 'kilosorted', mouse, session)
    rates_OB, rates_HC, time_bins, ob_units, hc_units = compute_spike_rates(kilosort_dir, nfs, window_size, step_size, use_units=use_units, sigma = sigma_smooth, zscore=False)
    rates = np.concatenate((rates_HC, rates_OB), axis=0)
    units = np.concatenate((hc_units, ob_units), axis=0)


    # Loading the sniffing data
    sniff_params_file = os.path.join(data_dir, 'sniff', mouse, session, 'sniff_params')
    mean_freqs, latencies, phases = compute_sniff_freqs_bins(sniff_params_file, time_bins, window_size, sfs)


    # Loading the behavior (tracking & task variable) data
    behavior_dir = os.path.join(data_dir, 'behavior_data', mouse, session)
    tracking_dir = os.path.join(data_dir, 'sleap_predictions', mouse, session)
    tracking_file = os.path.join(tracking_dir, next(f for f in os.listdir(tracking_dir) if f.endswith('.analysis.h5')))
    events = load_behavior(behavior_dir, tracking_file)


    # Aligning the neural and behavior data
    rates_data = align_brain_and_behavior(events, rates, units, time_bins, window_size)
    rates_data = rates_data.assign(sns=mean_freqs, latency=latencies, phase=phases)
    rates_data['sns'] = rates_data['sns'].interpolate(method='linear')
    rates_data.dropna(subset=['x', 'y', 'v_x', 'v_y'], inplace=True)



    # Converting the data to numpy arrays for PGAM standardized input
    counts = np.array(rates_data.drop(columns=['x', 'y', 'v_x', 'v_y', 'sns', 'speed', 'latency', 'phase', 'reward_state', 'time', 'trial_id', 'click']).values) * window_size
    variables = [
        rates_data['x'].to_numpy(),
        rates_data['y'].to_numpy(),
        rates_data['v_x'].to_numpy(),
        rates_data['v_y'].to_numpy(),                             
        rates_data['sns'].to_numpy(),           
        rates_data['latency'].to_numpy(),
        rates_data['phase'].to_numpy(),
        rates_data['speed'].to_numpy(),
        rates_data['click'].to_numpy()
    ]
    rates_data.drop(columns=['x', 'y', 'v_x', 'v_y'], inplace=True)

    variable_names = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'sns', 'latency', 'phase', 'speed', 'click']

    trial_ids = np.array(rates_data['trial_id'].values)

    neu_names = np.array(rates_data.columns[:len(counts[0])])

    neu_info = {}
    for i, name in enumerate(neu_names):
        neu_info[name] = {'area': 'HC' if i < len(hc_units) else 'OB', 'id': units[i]}



    # Plot the variables

    plot_dir = os.path.join(save_dir, 'behaior_figs')
    os.makedirs(plot_dir, exist_ok=True)
    for i, name in enumerate(variable_names):
        if name in ['position', 'velocity']:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i][:, 0], label=f'{name} x')
            plt.plot(variables[i][:, 1], label=f'{name} y')
            plt.title(name)
            plt.legend()
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f'{name}.png'))
            plt.close()
        else:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i])
            plt.title(name)
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f'{name}.png'))
            plt.close()




    #np.savez(os.path.join(save_dir, f'data.npz'), counts=counts, variables=variables, variable_names=variable_names, trial_ids = trial_ids, neu_names = neu_names, neu_info=neu_info)
    return counts, variables, variable_names, trial_ids, neu_names, neu_info




"""
Models and training utilities
"""

def cv_split(data, k, k_CV=10, n_blocks=10):
    '''
    Perform cross-validation split of the data, following the Hardcastle et 
    al paper.
    
    Parameters
    --
    data : An array of data.
    
    k : Which CV subset to hold out as testing data (integer from 0 to k_CV-1).
    
    k_CV : Number of CV splits (integer).
        
    n_blocks : Number of blocks for initially partitioning the data. The testing
        data will consist of a fraction 1/k_CV of the data from each of these
        blocks.
        
    Returns
    --
    data_train, data_test, switch_indices : 
        - Data arrays after performing the train/test split
        - Indices in the train and test data where new blocks begin
    '''

    block_size = len(data)//n_blocks
    mask_test = [False for _ in data]
    
    # Keep track of which indices in the original data are the start of test blocks
    test_block_starts = []
    
    for block in range(n_blocks):
        i_start = int((block + k/k_CV)*block_size)
        i_stop = int(i_start + block_size//k_CV)
        mask_test[i_start:i_stop] = [True for _ in range(block_size//k_CV)]
        test_block_starts.append(i_start)
        
    mask_train = [not a for a in mask_test]
    data_test = data[mask_test]
    data_train = data[mask_train]

    train_switch_indices = [0]
    test_switch_indices = [0]
    train_count = 0
    test_count = 0
    for i in range(len(data)-1):
        if mask_train[i]:
            train_count += 1
        if mask_test[i]:
            test_count += 1
        if not mask_train[i] and mask_train[i + 1]:
            train_switch_indices.append(train_count)
        if not mask_test[i] and mask_test[i + 1]:
            test_switch_indices.append(test_count)

    train_switch_indices = np.unique(train_switch_indices)
    test_switch_indices = np.unique(test_switch_indices)

    
    return data_train, data_test, train_switch_indices, test_switch_indices


class SequenceDataset(Dataset):
    def __init__(self, X, y, blocks, sequence_length, target_index=-1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length



        # Validate range and allow negative indexing
        if target_index < 0:
            target_index = target_index + sequence_length
        if not (-sequence_length <= target_index < sequence_length):
            raise ValueError(f"target_index must be in range [-{sequence_length}, {sequence_length - 1}], got {target_index}")

        self.target_index = target_index

        self.indices = []
        for start, end in blocks:
            for i in range(start, end - sequence_length + 1):
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, idx):
        i = self.indices[idx]
        X_seq = self.X[i: i + self.sequence_length, :]
        y_target = self.y[i + self.target_index, :]
        return X_seq, y_target

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1, dropout = 0.1):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_LSTM(model, train_loader, device, lr=0.01, epochs=1000, patience=50, min_delta=0.1, factor=0.1, verbose=False):
    """
    Train the LSTM model with early stopping and learning rate scheduling.
    
    Parameters
    ----------
    model : MLPModel
        The model to train
    X : torch.Tensor
        Input features
    y : torch.Tensor
        Target values
    lr : float
        Initial learning rate
    epochs : int
        Maximum number of epochs
    patience : int
        Number of epochs with no improvement after which training will be stopped
    min_delta : float
        Minimum change in loss to qualify as an improvement
    factor : float
        Factor by which the learning rate will be reduced
        
    Returns
    -------
    model : MLPModel
        The trained model
    history : list
        Training loss history
    """
    # Initialize the training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience)
    
    best_loss = float('inf')
    best_model_state = model.state_dict().copy()  # Save a copy of the model state
    counter = 0

    optimizer.zero_grad()

    # Training the model
    history = []
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type = device.type):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * X_batch.size(0)

        # Average loss
        epoch_loss /= len(train_loader.dataset)

        # Evaluation and early stopping
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()  # Save a copy of the model state
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        # Learning rate scheduler and history
        scheduler.step(epoch_loss)
        history.append(epoch_loss)

        if verbose:
            if epoch % 250 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}")
                # GPU usage (nvidia-smi)
                print("---- GPU Usage ----")
                try:
                    gpu_info = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
                    print(gpu_info)
                except Exception as e:
                    print(f"Could not get GPU info: {e}")


 

    # Load the best model
    model.load_state_dict(best_model_state)

    
    return model, history
    


def process_fold(X_train, X_test, y_train, y_test, train_switch_ind, test_switch_ind, current_save_path,
        device = None, shift = 0, k = 0, hidden_dim = 8, num_layers = 2, dropout = 0.1, sequence_length = 3, target_index = -1, 
        batch_size = 64, lr = 0.01, num_epochs = 300, patience = 10, min_delta = 0.01, factor = 0.5, plot_predictions = True):
    
    # Set the device
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    # Create the model
    lstm_model = LSTMDecoder(input_dim= X_train.shape[1], hidden_dim=hidden_dim, output_dim=y_train.shape[1], num_layers=num_layers, dropout = dropout).to(device)


    # Prepare the training data for LSTM
    blocks = [(train_switch_ind[i], train_switch_ind[i + 1]) for i in range(len(train_switch_ind) - 1)]
    train_dataset = SequenceDataset(X_train, y_train, blocks, sequence_length, target_index)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False, num_workers=0, pin_memory=True, prefetch_factor=None)


    # Training the LSTM model
    start_train = time.time()
    trained_lstm_model, lstm_history = train_LSTM(lstm_model, train_loader, device, lr=lr, epochs=num_epochs, patience=patience, min_delta=min_delta, factor=factor, verbose=False)
    print(f"\nTraining time: {time.time() - start_train:.2f}s fold={k} shift={shift}", flush=True)
    # Free up memory
    del lstm_model, train_dataset, train_loader


    # Plot loss
    if plot_predictions:
        plot_training(lstm_history, current_save_path, shift, k)


    # Prepare the test data for LSTM
    test_blocks = [(test_switch_ind[i], test_switch_ind[i + 1]) for i in range(len(test_switch_ind) - 1)]
    test_dataset = SequenceDataset(X_test, y_test, test_blocks, sequence_length, target_index)
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), num_workers=0, pin_memory=True)


    # Predict on the test set
    trained_lstm_model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = trained_lstm_model(X_batch)
            predictions.append(preds.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Clean up
    del trained_lstm_model, test_dataset, test_loader

    plot_preds(targets, predictions, test_switch_ind, sequence_length, current_save_path, k, shift)

    diffs = predictions - targets
    rmse = np.sqrt(np.mean(diffs ** 2, axis=0))


    torch.cuda.empty_cache()
    print(f"Total time {time.time() - start_train:.2f}s for fold={k} shift={shift}\n\n\n", flush=True)


    return rmse, targets, predictions




def plot_training(lstm_history, save_path, shift, k):

    optimal_loss = min(lstm_history)
    model_used_index = lstm_history.index(optimal_loss)

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(lstm_history)), y=lstm_history, linewidth=4, color='blue')
    plt.title(f'LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.scatter(model_used_index, optimal_loss, color='red', s=100)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'lstm_loss_{k}_shift_{shift}.png'), dpi=300)
    plt.close()

def plot_preds(targets, predictions, test_switch_ind, sequence_length, save_path, k, shift):
    adjusted_test_switch_ind = [ind - sequence_length * k for k, ind in enumerate(test_switch_ind)]
    behavior_dim = targets.shape[1]
    _, ax = plt.subplots(behavior_dim, 1, figsize=(20, 10), sharex= True)
    if behavior_dim == 1:
        ax = [ax]
    for i in range(behavior_dim):
        ax[i].plot(targets[:, i], label='True', color = 'crimson')
        ax[i].plot(predictions[:, i], label='Predicted')
        for ind in adjusted_test_switch_ind:
            ax[i].axvline(ind, color='grey', linestyle = '--', alpha=0.5)

    if behavior_dim > 4:
        # remove the y-axis ticks 
        for a in ax:
            a.set_yticks([])
    plt.xlabel('Time')
    ax[0].legend(loc = 'upper right')

    sns.despine()
    plt.savefig(os.path.join(save_path, f'lstm_predictions_k_{k}_shift_{shift}.png'), dpi=300)
    plt.close()