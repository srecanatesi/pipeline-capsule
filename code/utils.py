"""
Common utility functions used across the neural states dimensionality analysis pipeline.

This module consolidates shared functions to avoid code duplication.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import xarray as xr
import ssm
from sklearn.model_selection import KFold
from math import isnan
import os
import time
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# Import configuration for reproducibility settings
import config

# Set numpy random seed for reproducibility
np.random.seed(config.RANDOM_SEED)


def midbins(x):
    """
    Calculate midpoints of bins from bin edges.
    
    Args:
        x: Array of bin edges
        
    Returns:
        Array of bin midpoints
    """
    return x[:-1] + (x[1] - x[0]) / 2


def find_permutation_seq(x, y):
    """
    Find permutation to match sequences x and y.
    
    Args:
        x: First sequence
        y: Second sequence
        
    Returns:
        Permutation array
    """
    x_el = np.unique(x)
    y_el = np.unique(y)
    x_elpos = [np.nanmean([np.mean(np.where(row == el)[0]) for row in x]) for el in x_el]
    y_elpos = [np.nanmean([np.mean(np.where(row == el)[0]) for row in y]) for el in y_el]
    x_elids = np.argsort(x_elpos)
    y_elids = np.argsort(y_elpos)
    permutation = [np.where(x_elids == el)[0][0] for el in y_elids]
    return permutation


def sequence2midpoints(sequence):
    """
    Find midpoint indices of state transitions in a sequence.
    
    Args:
        sequence: Array of state labels
        
    Returns:
        Tuple of (midpoint indices, midpoint values)
    """
    final_points = np.array(np.where(np.diff(sequence) != 0)[0])
    initial_points = final_points+1
    initial_points = np.insert(initial_points, 0, 0)
    final_points = np.append(final_points, len(sequence)-1)
    mid_points = np.round((initial_points + final_points) /2).astype('int')
    # This fixes initial and last state to have middle point respectively at beginning and end of array
    mid_points = np.insert(mid_points, 0, 0)
    mid_points = np.append(mid_points, len(sequence)-1)
    mid_values = sequence[mid_points]
    return mid_points, mid_values


def isininterval(x, a, b, y=None, axis=0):
    """
    Filter array to values within interval [a, b].
    
    Args:
        x: Input array
        a: Lower bound
        b: Upper bound
        y: Optional array for filtering
        axis: Axis along which to filter
        
    Returns:
        Filtered array
    """
    if len(x.shape) > 0:
        x = np.swapaxes(x, 0, axis)
    if y is None:
        x = x[(x >= a) & (x <= b)]
    else:
        x = x[(y >= a) & (y <= b)]
    if len(x.shape) > 0:
        x = np.swapaxes(x, 0, axis)
    return x


def area2region(units, field):
    """
    Map brain areas to brain regions.
    
    Args:
        units: DataFrame with unit information
        field: Column name containing area acronyms
        
    Returns:
        DataFrame with region assignments
    """
    dict = {'Thalamus': ['LGd', 'LGn', 'LP', 'LD', 'POL', 'MD', 'VPL', 'PO', 'VPM', 'RT', 'MG', 'MGv', 'MGd', 'Eth', 'SGN', 'TH'],
           'others': ['RSP', 'OLF', 'BLA', 'ZI', 'grey'],
            'Hippocampus': ['DG', 'CA3', 'CA1', 'SUB', 'POST', 'ProS'],
            'FrontalCortex': ['ACA', 'MOs', 'PL', 'ILA', 'ORB', 'MOp', 'SSp'],
            'VisualCortex' : ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'VISa', 'VISal', 'VIS', 'VISli', 'VISlm'],
            'Midbrain' : ['SCs', 'SCm', 'MRN', 'APN', 'PAG', 'MB'],
            'BasalGanglia' : ['CP', 'GPe', 'SNr', 'ACB', 'LS']}

    df = pd.DataFrame.from_dict(dict.items())
    df = df.explode(1)
    df = df.rename(columns= {0:'region', 1:'area'})
    df = df.merge(units, left_on = 'area', right_on = field)
    return df


def grpBySameConsecutiveItem(l, max_length=15, min_length=3, value=True):
    """
    Group consecutive identical items in a list.
    
    Finds sequences of consecutive identical items and groups them if they
    meet the length criteria. Useful for finding state transitions or other
    consecutive patterns.
    
    Args:
        l: Input list or array
        max_length: Maximum group length (groups longer than this are split)
        min_length: Minimum group length (only groups >= this length are returned)
        value: Value to filter for (only groups with this value are returned)
        
    Returns:
        tuple: (grouped_items, grouped_indices) where:
            - grouped_items: List of lists containing consecutive items
            - grouped_indices: List of lists containing indices of grouped items
    """
    rv= []
    rv_idx = []
    last = None
    last_idx = None
    for i_elem, elem in enumerate(l):
        if last == None:
            last = [elem]
            last_idx = [i_elem]
            continue
        if (elem == last[0]) & (len(last) < max_length):
            last.append(elem)
            last_idx.append(i_elem)
            continue
        if (len(last) >= min_length) & (last[0]==value):
            rv.append(last)
            rv_idx.append(last_idx)
        last = [elem]
        last_idx = [i_elem]
    return rv, rv_idx


def classify_waveform(units_details):
    """
    Classify units as excitatory (Exc), inhibitory (Ini), or other (Oth) based on waveform duration.
    
    Uses Gaussian Mixture Model with 3 components to classify units.
    
    Args:
        units_details: DataFrame with unit details including 'waveform_duration'
        
    Returns:
        DataFrame with added 'EI_type' column
    """
    X = units_details['waveform_duration'].values
    X = X.reshape(-1,1)
    # Use config random state for reproducibility
    gm = GaussianMixture(n_components=config.WAVEFORM_N_COMPONENTS, 
                         random_state=config.WAVEFORM_RANDOM_STATE, 
                         covariance_type=config.WAVEFORM_COVARIANCE_TYPE).fit(X)
    clu = gm.predict(X)
    clu_p = np.max(gm.predict_proba(X), axis=1)

    ini_idx = (clu == np.argmin(gm.means_)) & (clu_p > 0.95)
    exc_idx = (clu == np.argsort(gm.means_)[1][0]) & (clu_p > 0.95)
    oth_idx = (clu == np.argmax(gm.means_)) & (clu_p > 0.95)
    units_details['EI_type'] = np.nan
    units_details['EI_type'][exc_idx] = 'Exc'
    units_details['EI_type'][ini_idx] = 'Ini'
    units_details['EI_type'][oth_idx] = 'Oth'
    return units_details


def optotagging_spike_counts(bin_edges, trials, session_or_nwb, units, use_nwb=False):
    """
    Calculate spike counts around optogenetic stimulation trials.
    
    Args:
        bin_edges: Time bin edges
        trials: Trial information DataFrame
        session_or_nwb: Session object (for Allen SDK) or NWB file object
        units: Units DataFrame
        use_nwb: If True, expects NWB file; if False, expects Allen SDK session object
        
    Returns:
        numpy array or xarray DataArray with spike counts
    """
    time_resolution = np.mean(np.diff(bin_edges))
    spike_matrix = np.zeros((len(trials), len(bin_edges)-1, len(units)))
    
    for unit_idx, unit_id in enumerate(units.index.values):
        if use_nwb:
            # Extract spike times from NWB file
            unit_position = None
            for i in range(len(session_or_nwb.units)):
                unit_data = session_or_nwb.units[i]
                unit_id_nwb = unit_data.index[0] if hasattr(unit_data, 'index') and len(unit_data.index) > 0 else i
                if unit_id_nwb == unit_id:
                    unit_position = i
                    break
            
            if unit_position is not None:
                unit_data = session_or_nwb.units[unit_position]
                spike_times = unit_data['spike_times'].values[0] if hasattr(unit_data['spike_times'], 'values') else unit_data['spike_times']
            else:
                spike_times = np.array([])
        else:
            # Use Allen SDK session object
            spike_times = session_or_nwb.spike_times[unit_id]
        
        for trial_idx, trial_start in enumerate(trials.start_time.values):
            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))
            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            binned_times, counts = np.unique(binned_times, return_counts=True)
            spike_matrix[trial_idx, binned_times, unit_idx] = counts
    
    # Return xarray DataArray if not using NWB (for compatibility with Allen SDK code)
    if not use_nwb:
        return xr.DataArray(
            name='spike_counts',
            data=spike_matrix,
            coords={'trial_id': trials.index.values, 
                   'time_relative_to_stimulus_onset': (bin_edges[:-1]+bin_edges[1:])/2, 
                   'unit_id': units.index.values},
            dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id'])
    else:
        return spike_matrix


def classify_10mspulses(units_details, session_or_nwb, use_nwb=False):
    """
    Classify units based on responses to 10ms optogenetic pulses.
    
    Args:
        units_details: DataFrame with unit details
        session_or_nwb: Session object (for Allen SDK) or NWB file object
        use_nwb: If True, expects NWB file; if False, expects Allen SDK session object
        
    Returns:
        DataFrame with added 'opto_10ms' column
    """
    units_details['opto_10ms'] = np.nan
    
    if use_nwb:
        # NWB file version
        try:
            if hasattr(session_or_nwb, 'processing') and 'optotagging' in session_or_nwb.processing:
                opto_module = session_or_nwb.processing['optotagging']
                if 'optogenetic_stimulation' in opto_module.data_interfaces:
                    stim_epochs = opto_module.data_interfaces['optogenetic_stimulation'].to_dataframe()
                    ten_ms_pulses = stim_epochs[(stim_epochs['duration'] > 0.009) & (stim_epochs['duration'] < 0.02)]
                    
                    if len(ten_ms_pulses) > 0:
                        genotype = 'wt/wt'
                        if hasattr(session_or_nwb, 'subject'):
                            subject = session_or_nwb.subject
                            if hasattr(subject, 'genotype'):
                                genotype = subject.genotype
                        genotype_short = genotype[:3] if len(genotype) >= 3 else genotype
                        # Note: Full implementation would require spike count analysis
                        print(f'Genotype: {genotype} -> {genotype_short}')
                        print('Note: Full optogenetic analysis requires implementing spike count analysis')
        except Exception as e:
            print(f'Error in optogenetic analysis: {e}')
    else:
        # Allen SDK session version
        trials = session_or_nwb.optogenetic_stimulation_epochs[
            (session_or_nwb.optogenetic_stimulation_epochs.duration > 0.009) & 
            (session_or_nwb.optogenetic_stimulation_epochs.duration < 0.02)]
        time_resolution = 0.0005  # 0.5 ms bins
        bin_edges = np.arange(-0.01, 0.025, time_resolution)
        da = optotagging_spike_counts(bin_edges, trials, session_or_nwb, 
                                     units_details.set_index('unit_id'), use_nwb=False)
        baseline = da.sel(time_relative_to_stimulus_onset=slice(-0.01, -0.002))
        baseline_rate = baseline.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
        evoked = da.sel(time_relative_to_stimulus_onset=slice(0.001, 0.009))
        evoked_rate = evoked.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
        idxs_opto_10ms = ((evoked_rate / (baseline_rate + (baseline_rate == 0))) > 2.).values
        genotype = session_or_nwb.full_genotype[:3]
        units_details['opto_10ms'][idxs_opto_10ms] = genotype
    
    return units_details


def get_spikecounts_during_spontaneous_epochs_session(session, uID_list, bSize=0.5, binarize=False, dtype=None):
    """
    Extract spike counts during spontaneous epochs from Allen SDK session object.
    
    Only includes epochs longer than 1500 seconds (25 minutes).
    
    Args:
        session: Allen SDK session object
        uID_list: List of unit IDs to extract
        bSize: Bin size in seconds (default: 0.5)
        binarize: Whether to binarize spike counts (default: False)
        dtype: Optional data type for output arrays
        
    Returns:
        Tuple of (spikecount_list, timecourse_list)
    """
    # Get start & end times
    spontaneous_df = session.get_stimulus_table("spontaneous")
    start_times = spontaneous_df['start_time'].values
    stop_times = spontaneous_df['stop_time'].values
    durations = spontaneous_df['duration'].values
    # Ensure it was for long enough
    iBlocks = np.where(durations > 1500)[0]
    nBlocks = len(iBlocks)
    # Get spike times
    spike_times = session.spike_times
    spikecount_list = []
    timecourse_list = []
    # Loop through spontaneous blocks
    for iEpoch in iBlocks:
        tStart = start_times[iEpoch]
        tStop = stop_times[iEpoch]
        duration = tStop-tStart
        # Bin spikes into windows to calculate simple FR vector for each neuron
        bin_edges = np.arange(tStart, tStop, bSize)
        starts = bin_edges[:-1]
        ends = bin_edges[1:]
        tiled_data = np.zeros((bin_edges.shape[0] - 1, len(uID_list)),
                              dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype)
        # Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            dTmp = np.array(spike_times[unit_id])
            # ignore invalid spike times
            pos = np.where(dTmp > 0)[0]
            data = dTmp[pos]
            # Ensure spike times are sorted
            sort_indices = np.argsort(data)
            start_positions = np.searchsorted(data, starts.flat, sorter=sort_indices)
            end_positions = np.searchsorted(data, ends.flat, side="right", sorter=sort_indices)
            counts = (end_positions - start_positions)
            tiled_data[:, ii].flat = counts > 0 if binarize else counts
        # Save matrix to list
        spikecount_list.append(tiled_data.T)
        timecourse_list.append(bin_edges[:-1] + np.diff(bin_edges) / 2)
    return spikecount_list, timecourse_list


def get_spikecounts_during_spontaneous_epochs_nwb(nwb_file, uID_list, bSize=0.5, binarize=False, dtype=None):
    """
    Extract spike counts during spontaneous epochs from NWB file.
    
    Only includes epochs longer than 1500 seconds (25 minutes).
    
    Args:
        nwb_file: NWB file object
        uID_list: List of unit IDs to extract
        bSize: Bin size in seconds (default: 0.5)
        binarize: Whether to binarize spike counts (default: False)
        dtype: Optional data type for output arrays
        
    Returns:
        Tuple of (spikecount_list, timecourse_list)
    """
    # Extract stimulus table from NWB intervals
    if hasattr(nwb_file, 'intervals') and 'spontaneous_presentations' in nwb_file.intervals:
        spontaneous_table = nwb_file.intervals['spontaneous_presentations']
        start_times = spontaneous_table.start_time[:] if hasattr(spontaneous_table.start_time, '__getitem__') else spontaneous_table.start_time
        stop_times = spontaneous_table.stop_time[:] if hasattr(spontaneous_table.stop_time, '__getitem__') else spontaneous_table.stop_time
        durations = stop_times - start_times
    else:
        print("No spontaneous presentations found in NWB file")
        return [], []
    
    # Ensure it was for long enough (1500 seconds = 25 minutes)
    iBlocks = np.where(durations > 1500)[0]
    nBlocks = len(iBlocks)
    print(f"Found {nBlocks} epochs longer than 1500 seconds")
    
    if nBlocks == 0:
        print("No spontaneous epochs longer than 1500 seconds found, skipping session")
        return [], []
    
    # Create mapping from unit ID to position in NWB units table
    unit_id_to_position = {}
    for i in range(len(nwb_file.units)):
        unit_data = nwb_file.units[i]
        unit_id = unit_data.index[0] if hasattr(unit_data, 'index') and len(unit_data.index) > 0 else i
        unit_id_to_position[unit_id] = i
    
    spikecount_list = []
    timecourse_list = []
    
    # Loop through spontaneous blocks
    for iEpoch in iBlocks:
        tStart = start_times[iEpoch]
        tStop = stop_times[iEpoch]
        duration = tStop - tStart
        
        # Bin spikes into windows to calculate simple FR vector for each neuron
        bin_edges = np.arange(tStart, tStop, bSize)
        starts = bin_edges[:-1]
        ends = bin_edges[1:]
        tiled_data = np.zeros((bin_edges.shape[0] - 1, len(uID_list)),
                              dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype)
        
        # Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            if unit_id in unit_id_to_position:
                # Get spike times for this unit from NWB using position
                unit_position = unit_id_to_position[unit_id]
                unit_data = nwb_file.units[unit_position]
                spike_times = unit_data['spike_times'].values[0] if hasattr(unit_data['spike_times'], 'values') else unit_data['spike_times']
                
                # Ignore invalid spike times
                pos = np.where(spike_times > 0)[0]
                data = spike_times[pos]
                
                # Ensure spike times are sorted
                sort_indices = np.argsort(data)
                start_positions = np.searchsorted(data, starts.flat, sorter=sort_indices)
                end_positions = np.searchsorted(data, ends.flat, side="right", sorter=sort_indices)
                counts = (end_positions - start_positions)
                
                tiled_data[:, ii].flat = counts > 0 if binarize else counts
            else:
                print(f"Warning: Unit ID {unit_id} not found in NWB file")
                tiled_data[:, ii].flat = 0
        
        # Save matrix to list
        spikecount_list.append(tiled_data.T)
        timecourse_list.append(bin_edges[:-1] + np.diff(bin_edges) / 2)
    
    return spikecount_list, timecourse_list


def check_convergence(train_lls, num_iters, tolerance):
    """
    Check if HMM fitting converged early due to tolerance.
    
    Args:
        train_lls: Array of log-likelihoods at each iteration
        num_iters: Maximum number of iterations requested
        tolerance: Convergence tolerance threshold
        
    Returns:
        dict with convergence information:
        - converged: bool, whether convergence occurred
        - reason: str, 'tolerance' if converged early, 'max_iterations' if reached max
        - n_iters_used: int, number of iterations actually used
        - final_ll_change: float, change in log-likelihood at last iteration
    """
    train_lls = np.array(train_lls)
    n_iters_used = len(train_lls)
    
    # Calculate change in log-likelihood at the last iteration
    if n_iters_used >= 2:
        final_ll_change = abs(train_lls[-1] - train_lls[-2])
    else:
        final_ll_change = np.inf
    
    # Check if converged early (stopped before max iterations)
    if n_iters_used < num_iters:
        # Check if the change was below tolerance (converged due to tolerance)
        if final_ll_change < tolerance:
            return {
                'converged': True,
                'reason': 'tolerance',
                'n_iters_used': n_iters_used,
                'final_ll_change': final_ll_change
            }
        else:
            # Stopped early but not due to tolerance (shouldn't happen normally)
            return {
                'converged': False,
                'reason': 'early_stop_other',
                'n_iters_used': n_iters_used,
                'final_ll_change': final_ll_change
            }
    else:
        # Reached max iterations
        if final_ll_change < tolerance:
            # Could have converged but didn't check in time
            return {
                'converged': True,
                'reason': 'tolerance_at_max',
                'n_iters_used': n_iters_used,
                'final_ll_change': final_ll_change
            }
        else:
            return {
                'converged': False,
                'reason': 'max_iterations',
                'n_iters_used': n_iters_used,
                'final_ll_change': final_ll_change
            }


def format_convergence_info(conv_info, max_iters=None, prefix=""):
    """
    Format convergence information as a readable string.
    
    Args:
        conv_info: Convergence dict from check_convergence()
        max_iters: Maximum iterations (for display purposes)
        prefix: Optional prefix string for the output
        
    Returns:
        Formatted string describing convergence status
    """
    reason_map = {
        'tolerance': 'Converged early due to tolerance',
        'tolerance_at_max': 'Converged at max iterations (tolerance met)',
        'max_iterations': 'Reached max iterations (tolerance not met)',
        'early_stop_other': 'Stopped early for unknown reason'
    }
    
    reason_str = reason_map.get(conv_info['reason'], conv_info['reason'])
    max_iters_str = f"/{max_iters}" if max_iters is not None else ""
    return (f"{prefix}Iterations: {conv_info['n_iters_used']}{max_iters_str}, "
            f"Final LL change: {conv_info['final_ll_change']:.6f}, "
            f"Status: {reason_str}")


def hmm_fit(data, num_states, num_iters, true_states=None, tolerance=0.01, return_convergence=False):
    """
    Fit Hidden Markov Model to spike count data.
    
    Args:
        data: List of trial data arrays (n_neurons x n_timepoints)
        num_states: Number of hidden states
        num_iters: Number of EM iterations
        true_states: Optional true state sequence (for evaluation)
        tolerance: Convergence tolerance for EM algorithm
        return_convergence: If True, also return convergence information
        
    Returns:
        Tuple of (inferred states, log likelihoods, posterior probabilities)
        If return_convergence=True, also returns convergence dict as 4th element
    """
    # Ensure random seed is set before HMM initialization for reproducibility
    # Note: ssm library uses autograd.numpy.random internally, which should already be seeded
    num_trials = len(data)
    num_neurons = data[0].shape[0]
    hmm = ssm.HMM(num_states, num_neurons, observations="poisson")
    train_data = [data[i].transpose().astype(np.int8) for i in range(num_trials)]
    train_lls = hmm.fit(train_data, method="em", num_iters=num_iters, tolerance=tolerance)
    hmm_z = np.array([hmm.most_likely_states(train_data[i_trial]) for i_trial in range(num_trials)])
    hmm_ll = np.array([hmm.observations.log_likelihoods(train_data[i_trial], None, None, None) for i_trial in range(num_trials)])
    hmm_ps = np.array([hmm.filter(train_data[i_trial]) for i_trial in range(num_trials)])
    
    if return_convergence:
        conv_info = check_convergence(train_lls, num_iters, tolerance)
        return hmm_z, hmm_ll, hmm_ps, conv_info
    else:
        return hmm_z, hmm_ll, hmm_ps


# ============================================================================
# S3 Download Functions
# ============================================================================

# S3 bucket configuration for Allen Brain Observatory
S3_BUCKET = "allen-brain-observatory"
S3_REGION = "us-west-2"
SESSIONS_TABLE_KEY = "visual-coding-neuropixels/ecephys-cache/sessions.csv"
SESSION_NWB_KEY = "visual-coding-neuropixels/ecephys-cache/session_{sid}/session_{sid}.nwb"


def s3_client():
    """
    Create and return an S3 client configured for public access.
    
    Returns:
        boto3.client: Configured S3 client with unsigned requests
    """
    return boto3.client("s3", region_name=S3_REGION, config=Config(signature_version=UNSIGNED))


def download_file(key: str, dest_path: str, show_progress: bool = True, session_id: int = None):
    """
    Download a file from S3 to local filesystem.
    
    Args:
        key: S3 object key (path within bucket)
        dest_path: Local destination file path
        show_progress: Whether to print download progress
        session_id: Optional session ID for progress messages
        
    Raises:
        FileNotFoundError: If S3 object doesn't exist
        IOError: If downloaded file size doesn't match expected size
    """
    # Create destination directory if it doesn't exist
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:  # Only create if path has a directory component
        os.makedirs(dest_dir, exist_ok=True)
    
    s3 = s3_client()
    
    # Check if file exists and get size
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        raise FileNotFoundError(f"S3 object not found: s3://{S3_BUCKET}/{key}") from e
    
    size = head.get("ContentLength", None)
    bytes_read = 0
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    
    try:
        with open(dest_path, "wb") as f:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = resp["Body"]
            t0 = time.time()
            
            # Download file in chunks
            while True:
                data = body.read(chunk_size)
                if not data:
                    break
                f.write(data)
                bytes_read += len(data)
                
                # Show progress every 2 seconds
                if size and show_progress and (time.time() - t0 > 2.0):
                    pct = 100.0 * bytes_read / size
                    print(f"[Session {session_id}] … {os.path.basename(dest_path)}: {pct:5.1f}%")
            
            # Print completion message
            if show_progress:
                elapsed = time.time() - t0
                print(f"[Session {session_id}] ✓ {os.path.basename(dest_path)} done in {elapsed:.1f}s")
                
    except Exception:
        # Clean up partial download on error
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    
    # Verify file size matches expected size
    if size and os.path.getsize(dest_path) != size:
        os.remove(dest_path)
        raise IOError("Size mismatch after download")

