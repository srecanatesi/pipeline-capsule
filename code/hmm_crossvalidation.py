#!/usr/bin/env python3
"""
HMM cross-validation and analysis script.

This script performs cross-validation to determine optimal number of HMM states
and then fits the final HMM model.

The script processes all preprocessed session files found in the preprocessed
directory and performs:
1. Cross-validation to determine optimal number of states
2. Final HMM fit with optimal number of states
3. Saves results to HMM cross-validation directory

Usage:
    python hmm_crossvalidation.py

The script will automatically discover and process all preprocessed session files.
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from pathlib import Path
import warnings
import ssm
import multiprocessing as mp
import pandas as pd
import numpy as np_std  # Standard numpy for non-gradient operations like random sampling

# Import configuration for reproducibility settings
import config

# Set seeds for reproducibility at module level
npr.seed(config.AUTOGRAD_SEED)  # Seed for autograd numpy random (used in HMM)
np_std.random.seed(config.RANDOM_SEED)  # Seed for standard numpy random operations
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)
pd.set_option("display.max_columns", None)
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from kneed import KneeLocator
from sklearn.model_selection import KFold
from math import isnan
import pickle
import subprocess
import sys
import argparse

# Import shared utilities and configuration
from utils import (
    hmm_fit,
    get_spikecounts_during_spontaneous_epochs_session,
)
import config

# Set up plotting colors and colormaps
colors = sns.xkcd_palette(config.COLOR_NAMES)
from ssm.plots import gradient_cmap
cmap = gradient_cmap(colors)

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
statescmp = ListedColormap(config.COOL_COLORS, name='states', N=config.COOL_COLORS.shape[0])

# Register colormaps with alpha transparency
for mapname in config.COLORMAPS_NAMES:
    color_array = plt.get_cmap(mapname)(range(config.N_COLORS))
    color_array[:, -1] = np.linspace(.0, 1., config.N_COLORS)
    map_object = LinearSegmentedColormap.from_list(name=mapname + 'alpha', colors=color_array)
    plt.register_cmap(cmap=map_object)

warnings.filterwarnings("ignore")


def hmm_xval_core(K_states, data, nKfold=None, N_iters=None, tolerance=None, return_convergence=False):
    """
    Core cross-validation function for a single number of states.
    
    Performs k-fold cross-validation for a given number of states by:
    1. Splitting data into train/test folds
    2. Fitting HMM on training data
    3. Evaluating log-likelihood on test data
    4. Returning mean test log-likelihood across folds
    
    Args:
        K_states: Number of states to test
        data: List of trial data arrays (each array is n_neurons x n_timepoints)
        nKfold: Number of cross-validation folds (uses config default if None)
        N_iters: Number of EM iterations (uses config default if None)
        tolerance: Convergence tolerance (uses config default if None)
        return_convergence: If True, also return convergence information
        
    Returns:
        float: Mean test log-likelihood across folds
        If return_convergence=True, also returns list of convergence dicts (one per fold)
    """
    if nKfold is None:
        nKfold = config.get_hmm_n_kfold()
    if N_iters is None:
        N_iters = config.get_hmm_n_iters_xval()
    if tolerance is None:
        tolerance = config.get_hmm_tolerance_xval()
    
    test_lls = np.zeros((nKfold))
    convergence_info = []
    # Use random_state for reproducibility in cross-validation
    kfold = KFold(n_splits=nKfold, shuffle=True, random_state=config.KFOLD_RANDOM_STATE)
    obs_dim = data[0].shape[0]
    for iX, (train_index, test_index) in enumerate(kfold.split(data)):
        train_data = [data[i].transpose().astype(np.int8) for i in train_index]
        test_data = [data[i].transpose().astype(np.int8) for i in test_index]
        # HMM initialization uses autograd.numpy.random which is already seeded at module level
        hmm_data = ssm.HMM(K_states, obs_dim, observations='poisson')
        train_lls = hmm_data.fit(train_data, method='em', num_iters=N_iters, tolerance=tolerance)
        test_lls[iX] = hmm_data.log_likelihood(test_data)
        
        if return_convergence:
            from utils import check_convergence, format_convergence_info
            conv_info = check_convergence(train_lls, N_iters, tolerance)
            convergence_info.append(conv_info)
            # Print convergence info for each fold
            print(f"    Fold {iX+1}/{nKfold} (K={K_states}): {format_convergence_info(conv_info, max_iters=N_iters)}")
    
    if return_convergence:
        return np.mean(test_lls), convergence_info
    else:
        return np.mean(test_lls)


def hmm_xval(spkcnts_split, nKfold=None, N_iters=None, tolerance=None, return_convergence=False):
    """
    Perform cross-validation across a range of state numbers.
    
    Tests multiple numbers of states in parallel and returns cross-validation
    scores for each. Uses multiprocessing to speed up computation.
    
    Args:
        spkcnts_split: List of spike count arrays (one per trial/part)
        nKfold: Number of cross-validation folds (uses config default if None)
        N_iters: Number of EM iterations (uses config default if None)
        tolerance: Convergence tolerance (uses config default if None)
        return_convergence: If True, also return convergence information for each fold
        
    Returns:
        tuple: (states_space, llhood_mean) where:
            - states_space: Array of state numbers tested
            - llhood_mean: Array of mean test log-likelihoods for each state number
        If return_convergence=True, also returns:
            - convergence_info_xval: List of lists, where each inner list contains
              convergence info dicts for each fold of that state number
    """
    if nKfold is None:
        nKfold = config.HMM_N_KFOLD
    if N_iters is None:
        N_iters = config.HMM_N_ITERS_XVAL
    if tolerance is None:
        tolerance = config.HMM_TOLERANCE_XVAL
    
    # Use dry run parameters if enabled
    states_space = np.arange(config.get_hmm_k_min(), config.get_hmm_k_max(), 1)
    xval_core_partial = partial(hmm_xval_core, data=spkcnts_split, nKfold=nKfold, 
                                 N_iters=N_iters, tolerance=tolerance, return_convergence=return_convergence)
    
    # Use multiprocessing to test multiple state numbers in parallel
    # Note: Each worker process will inherit the seeded random state from the main process
    # The autograd.numpy.random seed is set at module level, so it should be inherited
    with mp.Pool(processes=config.HMM_N_PROCESSES) as pool:
        results = pool.map(xval_core_partial, states_space)
    
    if return_convergence:
        # Unpack results: each element is (mean_ll, convergence_info_list)
        llhood_mean = [r[0] for r in results]
        convergence_info_xval = [r[1] for r in results]
        return states_space, llhood_mean, convergence_info_xval
    else:
        llhood_mean = results
        return states_space, llhood_mean


def hmm_analysis(spkcnts, nKfold=None, N_itersxv=None, N_iters=None, N_states=np.nan, 
                 N_final_fit=None, tolerance=None, hmm_true4colors=None, save=np.nan, session_id=None):
    """
    Complete HMM analysis: cross-validation followed by final fit.
    
    Args:
        spkcnts: List of spike count arrays
        nKfold: Number of cross-validation folds (uses config default if None)
        N_itersxv: Iterations for cross-validation (uses config default if None)
        N_iters: Iterations for final fit (uses config default if None)
        N_states: Number of states (if np.nan, determined via cross-validation)
        N_final_fit: Number of final fits to perform (uses config default if None)
        tolerance: Convergence tolerance (uses config default if None)
        hmm_true4colors: Unused parameter (for compatibility)
        save: Path to save results (or np.nan to skip saving)
        session_id: Optional session ID string to append to figure filenames
        
    Returns:
        Dictionary with HMM analysis results
    """
    if nKfold is None:
        nKfold = config.get_hmm_n_kfold()
    if N_itersxv is None:
        N_itersxv = config.get_hmm_n_iters_xval()
    if N_iters is None:
        N_iters = config.get_hmm_n_iters_final()
    if N_final_fit is None:
        N_final_fit = config.get_hmm_n_final_fit()
    if tolerance is None:
        tolerance = config.get_hmm_tolerance()
    
    llhood_mean = np.nan
    states_space = np.nan
    convergence_info_xval = None
    
    if isnan(N_states):
        xval_result = hmm_xval(spkcnts, nKfold, N_itersxv, tolerance=tolerance, return_convergence=True)
        if len(xval_result) == 3:
            states_space, llhood_mean, convergence_info_xval = xval_result
        else:
            states_space, llhood_mean = xval_result
        try:
            kneedle = KneeLocator(states_space, llhood_mean, S=config.get_knee_locator_s(), 
                                 curve=config.KNEE_CURVE, direction=config.KNEE_DIRECTION)
            # Create results directory if it doesn't exist
            results_folder = config.get_results_folder()
            results_folder.mkdir(exist_ok=True)
            
            # Create filename with session_id if provided
            if session_id:
                fig_name_normalized = f'crossvalidation_meanLLhood_normalized_{session_id}.pdf'
                fig_name = f'crossvalidation_meanLLhood_{session_id}.pdf'
            else:
                fig_name_normalized = 'crossvalidation_meanLLhood_normalized.pdf'
                fig_name = 'crossvalidation_meanLLhood.pdf'
            
            kneedle.plot_knee_normalized(figsize=config.FIG_SIZE_LARGE)
            sns.despine()
            plt.xticks(np.linspace(0, 1, len(states_space)), states_space)
            plt.savefig(results_folder / fig_name_normalized)
            plt.close()
            
            kneedle.plot_knee(figsize=config.FIG_SIZE_LARGE)
            sns.despine()
            plt.savefig(results_folder / fig_name)
            plt.close()
            N_states = round(kneedle.elbow, 3)
        except Exception as e:
            # If knee detection fails, use maximum number of states tested
            print(f"Warning: Knee detection failed ({e}), using maximum states: {states_space[-1]}")
            N_states = states_space[-1]
    else:
        llhood_mean = np.nan
    
    print(f"Optimal number of states: {N_states}")

    # Perform multiple fits and select the best one (lowest log-likelihood)
    hmm_states_all, hmm_ll_all, hmm_posterior_all = [], [], []
    convergence_info_all = []
    for i_fold in range(N_final_fit):
        result = hmm_fit(spkcnts, N_states, N_iters, tolerance=tolerance, return_convergence=True)
        if len(result) == 4:
            hmm_states_i, hmm_ll_i, hmm_posterior_i, conv_info = result
            convergence_info_all.append(conv_info)
            # Print convergence info
            from utils import format_convergence_info
            print(f"  Final fit {i_fold+1}/{N_final_fit}: {format_convergence_info(conv_info, max_iters=N_iters)}")
        else:
            hmm_states_i, hmm_ll_i, hmm_posterior_i = result
        hmm_states_all.append(hmm_states_i)
        hmm_ll_all.append(hmm_ll_i)
        hmm_posterior_all.append(hmm_posterior_i)
    
    # Select fit with lowest (most negative) log-likelihood
    i_min = np.argmin([np.nanmean(hmm_ll_all[i]) for i in range(N_final_fit)])
    hmm_states, hmm_ll, hmm_posterior = hmm_states_all[i_min], hmm_ll_all[i_min], hmm_posterior_all[i_min]

    hmm_analysis_data = {
        'states': hmm_states, 
        'll': hmm_ll, 
        'posterior': hmm_posterior, 
        'N_states': N_states, 
        'll_xval': llhood_mean,
        'convergence_info_final': convergence_info_all,  # Convergence info for all final fits
        'convergence_info_xval': convergence_info_xval  # Convergence info for cross-validation (if available)
    }
    if isinstance(save, str):
        with open(save, "wb") as f:
            pickle.dump(hmm_analysis_data, f)
    return hmm_analysis_data


def process_session(df_iteration, N_min_neurons=None):
    """
    Process a single session by running hmm_crossvalidation.py script.
    
    This function runs the hmm_crossvalidation.py script as a subprocess,
    which will process all preprocessed session files including the one
    corresponding to df_iteration.
    
    Note: This function is kept for backward compatibility but is not typically
    used, as the main script processes all sessions directly.
    
    Args:
        df_iteration: DataFrame with session data (used to identify session)
        N_min_neurons: Minimum number of neurons required (unused, kept for compatibility)
        
    Returns:
        None (processing is done by the subprocess script)
        
    Raises:
        subprocess.CalledProcessError: If the subprocess fails
    """
    import subprocess
    import sys
    
    # Run hmm_crossvalidation.py script
    script_path = Path(__file__).resolve()
    print(f"Running HMM cross-validation script: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print("HMM cross-validation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running HMM cross-validation: {e}")
        raise


# ============================================================================
# Main Execution Block
# ============================================================================

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run HMM cross-validation analysis')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Process only this specific session ID (e.g., "767871931"). If not provided, processes all preprocessed sessions.')
    args = parser.parse_args()
    
    print('=' * 60)
    print('HMM Cross-validation Analysis')
    print('=' * 60)
    
    # Find preprocessed session files
    preprocessed_folder = config.get_preprocessed_folder()
    all_files = list(preprocessed_folder.glob('df_*.pkl'))
    
    # Filter by session_id if provided
    if args.session_id:
        session_id_str = str(args.session_id)
        files = [f for f in all_files if f.stem == f'df_{session_id_str}']
        if not files:
            print(f"[ERROR] No preprocessed file found for session {session_id_str}")
            print(f"Available preprocessed files:")
            for f in all_files:
                print(f"  {f.stem}")
            sys.exit(1)
        print(f"Processing only session {session_id_str}")
    else:
        files = all_files
        print(f"Processing all preprocessed sessions")
    
    print(f"Found {len(files)} preprocessed session file(s) to process")
    if len(files) <= 5:
        print(files)
    else:
        print(f"  (showing first 5): {files[:5]} ...")
    
    if len(files):
        hmm_crossval_folder = config.get_hmm_crossval_folder()
        hmm_crossval_folder.mkdir(exist_ok=True)
        
        for filet in files:
            iteration = filet.stem.replace('df_', '')  # Extract session ID
            
            # Check if HMM results already exist for this session
            output_file = hmm_crossval_folder / f'hmm_{iteration}.pkl'
            if output_file.exists():
                print(f"Skipping {filet.name} - HMM results already exist: {output_file.name}")
                continue
            
            print(f"Processing {filet.name}")
            df_iteration = pd.read_pickle(filet)
            warnings.warn("starting HMM", FutureWarning)
            print("starting HMM")
            
            # Process session using the inline processing logic
            spkcnts = df_iteration['spkcnts'].values[0]
            all_times = df_iteration['times'].values[0]
            running = df_iteration['running'].values[0]
            
            # Extract numpy arrays from lists if needed
            if isinstance(spkcnts, list) and len(spkcnts) > 0:
                spkcnts = spkcnts[0]
            if isinstance(all_times, list) and len(all_times) > 0:
                all_times = all_times[0]
            if isinstance(running, list) and len(running) > 0:
                running = running[0]
            
            print(f"  Session data shape: {spkcnts.shape}")
            print(f"  Time points: {len(all_times)}")
            print(f"  Running data shape: {running.shape}")

            # In dry run mode, randomly sample a subset of neurons
            n_neurons_dry_run = config.get_dry_run_n_neurons()
            if n_neurons_dry_run is not None and spkcnts.shape[0] > n_neurons_dry_run:
                n_neurons_available = spkcnts.shape[0]
                # Use specific seed for neuron sampling to ensure reproducibility
                np_std.random.seed(config.NEURON_SAMPLING_SEED)
                selected_neuron_indices = np_std.random.choice(
                    n_neurons_available, 
                    size=n_neurons_dry_run, 
                    replace=False
                )
                selected_neuron_indices = np.sort(selected_neuron_indices)  # Sort for consistency
                spkcnts = spkcnts[selected_neuron_indices, :]
                print(f"  [DRY RUN] Randomly selected {n_neurons_dry_run} neurons from {n_neurons_available} available")
                print(f"  Session data shape after sampling: {spkcnts.shape}")
                # Reset random state after sampling to avoid affecting other operations
                np_std.random.seed(config.RANDOM_SEED)

            # Split data into parts for cross-validation
            spkcnts_split = spkcnts[:, :config.N_PARTS * (spkcnts.shape[1] // config.N_PARTS)].reshape(
                spkcnts.shape[0], config.N_PARTS, (spkcnts.shape[1] // config.N_PARTS))
            spkcnts_split = [spkcnts_split[:, i] for i in range(config.N_PARTS)]
            spkcnts = [spkcnts]
            
            print('HMM crossvalidation')
            # Perform cross-validation (using dry run parameters if enabled)
            data_xval = hmm_analysis(spkcnts_split, nKfold=config.get_hmm_n_kfold(), 
                                     N_itersxv=config.get_hmm_n_iters_xval_process(), 
                                     N_iters=1, tolerance=config.get_hmm_tolerance_xval_process(),
                                     session_id=iteration)
            
            print('HMM final fit')
            # Perform final fit with optimal number of states (using dry run parameters if enabled)
            data = hmm_analysis(spkcnts, nKfold=1, N_itersxv=None, 
                                N_iters=config.get_hmm_n_iters_final_process(), 
                                N_states=data_xval['N_states'], 
                                tolerance=config.get_hmm_tolerance_final_process(),
                                session_id=iteration)
            
            posterior = np.max(data['posterior'].squeeze(), axis=1)
            df_hmm = pd.DataFrame({
                'N_states': data['N_states'], 
                'states_sequence': [data['states']], 
                'posterior': [posterior], 
                'll_xval': [data_xval['ll_xval']],
                'convergence_info_final': [data.get('convergence_info_final', None)],  # Convergence info for final fits
                'convergence_info_xval': [data_xval.get('convergence_info_xval', None)]  # Convergence info for cross-validation
            })
            
            # Save results
            if df_hmm is not None and not df_hmm.empty:
                output_file = hmm_crossval_folder / f'hmm_{iteration}.pkl'
                df_hmm.to_pickle(output_file)
                print(f"[OK] Saved HMM results for session {iteration} to {output_file}")
            else:
                # Save empty marker if no valid results
                df_hmm = pd.DataFrame({'empty_dataframe': True}, index=[0])
                output_file = hmm_crossval_folder / f'hmm_empty_{iteration}.pkl'
                df_hmm.to_pickle(output_file)
                print(f"[WARNING] No valid HMM results for session {iteration}, saved empty marker")
    else:
        # No preprocessed files found
        print("[WARNING] No preprocessed session files found")
        hmm_crossval_folder = config.get_hmm_crossval_folder()
        hmm_crossval_folder.mkdir(exist_ok=True)
        df_hmm = pd.DataFrame({'empty_dataframe': True}, index=[0])
        df_hmm.to_pickle(hmm_crossval_folder / 'hmm_empty.pk')
    
    print('=' * 60)
    print('HMM Cross-validation Analysis Complete')
    print('=' * 60)
