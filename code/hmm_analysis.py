#!/usr/bin/env python3
"""
Post-HMM analysis script for neural states dimensionality analysis.

This module provides functions for analyzing fitted HMM models, including:
- State duration analysis
- Dimensionality reduction analysis (PCA, Factor Analysis)
- Covariance statistics computation
- Firing rate analysis by state
- Behavioral data analysis (running, pupil)

Note: This file was renamed from hmm_final_fit.py. The main HMM fitting
and cross-validation functionality is in hmm_crossvalidation.py.

Usage:
    python hmm_analysis.py --session-id <session_id>

This script processes a single session and saves results to:
    results/sessions_hmm_analysis/hmm_analysis_<session_id>.pkl
"""

import sys
import argparse
import os
import warnings
from pathlib import Path
from functools import partial

# Standard scientific computing libraries
import numpy as np_std
import pandas as pd
import scipy.linalg as la
import scipy.stats
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, skew

# Autograd numpy for HMM operations (must be imported separately)
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

# Machine learning and dimensionality reduction
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Statistical analysis
from statsmodels.stats.proportion import proportion_confint

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# HMM library
import ssm
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap

# Other utilities
from kneed import KneeLocator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure pandas display options
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)

# Configure matplotlib/seaborn style
sns.set_style("white")
sns.set_context("talk")

# ============================================================================
# Color and Visualization Setup
# ============================================================================

# Color palette for plots
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

# State colormap (cool colors)
CoolColors = np.array([
    [1, 1, 1, 1],
    [141, 211, 199, 255],
    [255, 255, 179, 255],
    [190, 186, 218, 255],
    [251, 128, 114, 255],
    [128, 177, 211, 255],
    [253, 180, 98, 255],
    [179, 222, 105, 255],
    [252, 205, 229, 255],
    [217, 217, 217, 255],
    [188, 128, 189, 255],
    [204, 235, 197, 255],
    [255, 237, 111, 255]
]) / 255

# Event colors
EventsColors = np.array([
    [0, 128, 0],
    [255, 211, 0],
    [0, 158, 255],
    [255, 0, 0],
    [255, 128, 0],
    [42, 82, 255]
]) / 256

statescmp = ListedColormap(CoolColors, name='states', N=CoolColors.shape[0])

# Register alpha-transparent colormaps
ncolors = 256
colormaps_names = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 'gist_rainbow']
for mapname in colormaps_names:
    color_array = plt.get_cmap(mapname)(range(ncolors))
    color_array[:, -1] = np.linspace(.0, 1., ncolors)
    map_object = LinearSegmentedColormap.from_list(name=mapname + 'alpha', colors=color_array)
    plt.register_cmap(cmap=map_object)


# ============================================================================
# Core Analysis Functions
# ============================================================================

def cov_stats(C, spkcounts=None, n=None, type='LFA', saveflag=False, titlestring=''):
    """
    Compute covariance statistics from a covariance matrix.
    
    This function computes various statistics from a covariance matrix including:
    - Mean and variance of diagonal (mii, sii) and off-diagonal (mij, sij) elements
    - Dimensionality metrics (dim_70, dim_80, dim_90, dim_95) based on explained variance
    - Participation ratio (PR_empirical)
    
    Note: This is a simplified implementation. Full bias correction functionality
    (when spkcounts is provided) is not yet implemented.
    
    Args:
        C: Covariance matrix (N x N array)
        spkcounts: Optional spike counts matrix for bias correction (not yet implemented)
        n: Optional number of components for LFA (not yet used)
        type: Type of analysis ('LFA' or 'PCA') - not yet used
        saveflag: Whether to save diagnostic plots (not yet implemented)
        titlestring: Title string for saved plots (not yet used)
        
    Returns:
        Dictionary containing covariance statistics:
        - mii, sii: Mean and std of diagonal elements
        - mij, sij: Mean and std of off-diagonal elements
        - ds: Normalized off-diagonal std (sij/mii)
        - PR_empirical: Participation ratio
        - dim_70, dim_80, dim_90, dim_95: Dimensionality at different variance thresholds
        - N: Number of neurons/dimensions
        - Other fields set to np.nan (for future implementation)
    """
    # Handle empty or invalid covariance matrices
    if C is None or np.sum(C) == 0:
        return {}
    
    # Extract diagonal and off-diagonal elements
    ofdiag = np.triu(np.ones_like(C, dtype=bool), k=1)  # Upper triangular (excluding diagonal)
    ondiag = np.eye(C.shape[0], dtype=bool)  # Diagonal elements
    N = C.shape[0]
    
    # Compute mean values
    mii = C[ondiag].mean() if np.any(ondiag) else np.nan
    mij = C[ofdiag].mean() if np.any(ofdiag) else np.nan
    
    # Compute standard deviations
    if np.any(ondiag) and N > 1:
        sii = np.sqrt(1 / (N - 1) * np.sum((C[ondiag] - mii) ** 2))
    else:
        sii = np.nan
    
    if np.any(ofdiag) and N > 1:
        sij = np.sqrt(1 / (N * (N - 1) - 2) * np.sum((C[ofdiag] - mij) ** 2))
    else:
        sij = np.nan
    
    # Normalized off-diagonal standard deviation
    ds = sij / mii if mii != 0 else np.nan
    
    # Participation ratio: measures effective dimensionality
    # PR = (trace(C))^2 / trace(C^2)
    PR_empirical = (np.trace(C)) ** 2 / (np.trace(C @ C)) if np.trace(C @ C) != 0 else np.nan
    
    # Compute eigenvalues and explained variance for dimensionality analysis
    try:
        C_eigs, C_eigv = la.eigh(C)
        # Sort eigenvalues in descending order
        C_eigs = C_eigs[::-1]
        # Normalize to get explained variance
        C_explained_variance = C_eigs / np.sum(C_eigs) if np.sum(C_eigs) != 0 else np.zeros_like(C_eigs)
        C_cum_explained_variance = np.cumsum(C_explained_variance)
        
        # Find dimensionality at different variance thresholds
        dim_70 = np.where(C_cum_explained_variance > 0.7)[0][0] if np.any(C_cum_explained_variance > 0.7) else np.nan
        dim_80 = np.where(C_cum_explained_variance > 0.8)[0][0] if np.any(C_cum_explained_variance > 0.8) else np.nan
        dim_90 = np.where(C_cum_explained_variance > 0.9)[0][0] if np.any(C_cum_explained_variance > 0.9) else np.nan
        dim_95 = np.where(C_cum_explained_variance > 0.95)[0][0] if np.any(C_cum_explained_variance > 0.95) else np.nan
    except Exception:
        dim_70 = dim_80 = dim_90 = dim_95 = np.nan
    
    # Return statistics dictionary
    # Note: Fields with '_fit' suffix are for bias-corrected statistics (future implementation)
    stats = {
        'mii': mii, 'sii': sii, 'mij': mij, 'sij': sij,
        'miit': np.nan, 'siit': np.nan, 'mijt': np.nan, 'sijt': np.nan,
        'ds': ds, 'dst': np.nan, 'N': N, 'T': np.nan,
        'PR_empirical': PR_empirical, 'PR_fit': np.nan,
        'dim_70': dim_70, 'dim_80': dim_80, 'dim_90': dim_90, 'dim_95': dim_95,
        'sij_r2': np.nan
    }
    return stats


def rebin2states(spkcnts, seq_state, bin_size, new_bin_size, return_averages=False, min_cum_bin=1):
    """
    Rebin spike counts according to state boundaries.
    
    This function takes spike counts and a state sequence, and rebins the data
    within each state according to the new bin size. It can either sum spikes
    within bins (for counts) or average them (for rates).
    
    Args:
        spkcnts: Spike counts array (neurons x timepoints)
        seq_state: Boolean array indicating state membership for each timepoint
        bin_size: Original bin size in seconds
        new_bin_size: Target bin size in seconds (must be >= bin_size)
        return_averages: If True, return averages instead of sums
        min_cum_bin: Minimum number of bins required for a state to be included
        
    Returns:
        Rebinned spike counts array (neurons x new_timepoints) or None if no
        valid states found
    """
    # Convert spkcnts to numpy array if it's a list
    if isinstance(spkcnts, list):
        spkcnts = spkcnts[0] if len(spkcnts) > 0 else np.array([])
    
    # Calculate conversion factor (how many old bins per new bin)
    conv_bin_step = int(np.round(new_bin_size // bin_size))
    
    # Find state boundaries: detect transitions in state sequence
    # Add False at start and end to capture states at boundaries
    starts_ends = np.diff(np.concatenate([[False], seq_state, [False]]))
    # Reshape to get (start, end) pairs for each state segment
    seq_states_ends = np.where(starts_ends)[0].reshape(-1, 2)
    
    # Calculate number of new bins for each state segment
    if not return_averages:
        num_new_bins = (seq_states_ends[:, 1] - seq_states_ends[:, 0]) // conv_bin_step
    else:
        num_new_bins = (seq_states_ends[:, 1] - seq_states_ends[:, 0])
        conv_bin_step = 1  # Use single bin when averaging
    
    # Filter out states that are too short
    idxs2keep = num_new_bins >= min_cum_bin
    seq_states_ends = seq_states_ends[idxs2keep]
    num_new_bins = num_new_bins[idxs2keep]
    
    # Rebin each state segment
    new_spks = []
    for i_seq in range(len(num_new_bins)):
        # Extract spike counts for this state segment
        start_idx = seq_states_ends[i_seq, 0]
        end_idx = start_idx + conv_bin_step * num_new_bins[i_seq]
        spkcnts_seq = spkcnts[:, np.arange(start_idx, end_idx)]
        
        # Reshape to (neurons, num_new_bins, conv_bin_step)
        spkcnts_seq = spkcnts_seq.reshape(-1, num_new_bins[i_seq], conv_bin_step)
        
        # Sum or average across the binning dimension
        if not return_averages:
            new_spkcnts = spkcnts_seq.sum(axis=2)  # Sum spikes in each new bin
        else:
            new_spkcnts = spkcnts_seq.mean(axis=1)  # Average rates
        
        new_spks.append(new_spkcnts)
    
    # Concatenate all rebinned segments
    if len(new_spks):
        spkcnts_state = np.concatenate(new_spks, axis=1)
    else:
        spkcnts_state = None
    
    return spkcnts_state


def hmm_spontaneous_sessionwise(df_ses, saveflag=False):
    """
    Perform comprehensive HMM analysis on a single session.
    
    This is the main analysis function that processes HMM-fitted data to compute:
    - Covariance statistics for each state
    - Dimensionality metrics using PCA and Factor Analysis
    - Firing rates by neuron type (excitatory/inhibitory)
    - Behavioral correlates (running speed, pupil size)
    - Analysis across different brain areas and layers
    
    The function iterates over:
    - All HMM states (and overall state -1)
    - Different brain areas and layers
    - Time packets within states (for temporal analysis)
    
    Args:
        df_ses: DataFrame containing session data with columns:
            - spkcnts: Spike counts array (neurons x timepoints)
            - states_sequence: State assignment for each timepoint
            - posterior: Posterior probability of state assignment
            - areas: Brain area labels for each neuron
            - layers: Cortical layer labels for each neuron
            - EI_type: Excitatory/Inhibitory classification
            - opto_10ms: Optogenetic stimulation data
            - running: Running speed time series
            - pupil: Pupil size time series (optional)
            - session_id: Session identifier
        saveflag: If True, save diagnostic plots (not yet fully implemented)
        
    Returns:
        DataFrame with one row per (state, area/layer, packet) combination,
        containing all computed statistics and properties.
    """
    # Extract session ID
    if 'session_id' in df_ses.columns:
        session_id = df_ses['session_id'].values[0]
    else:
        session_id = 'unknown'
    
    i_session = df_ses.index.values[0] if len(df_ses) > 0 else 0
    
    # Analysis parameters
    n_components_max = 30  # Maximum number of components for dimensionality analysis
    n_components_ses = np.arange(1, n_components_max, 1)
    bin_size = 0.005  # Original bin size (5 ms)
    new_bin_size = 0.1  # Target bin size for state analysis (100 ms)
    min_state_cumulative_duration = 30.0  # Minimum state duration in seconds
    
    print(new_bin_size)  # Debug output
    
    # Extract brain areas
    if 'areas' in df_ses.columns and len(df_ses.areas[0]) > 0:
        areas = np.unique(df_ses.areas[0])
        area_neurons = df_ses.areas[0]
    else:
        areas = ['unknown']
        area_neurons = np.array(['unknown'] * df_ses.spkcnts.values[0].shape[0])
    
    # Extract cortical layers
    if 'layers' in df_ses.columns and len(df_ses.layers[0]) > 0:
        layers = np.unique(df_ses.layers[0])
        layer_neurons = df_ses.layers[0]
    else:
        layers = ['unknown']
        layer_neurons = np.array(['unknown'] * df_ses.spkcnts.values[0].shape[0])
    
    # Extract HMM states
    if 'states_sequence' in df_ses.columns and len(df_ses.states_sequence[0]) > 0:
        states = np.unique(df_ses.states_sequence[0])
        state_times = df_ses.states_sequence[0]
    else:
        print("Warning: No states_sequence found, using default states")
        states = np.array([0, 1, 2])  # Default states
        state_times = np.zeros(df_ses.spkcnts.values[0].shape[1])
    
    spkcnts = df_ses.spkcnts.values[0]
    
    # Extract neuron type classifications (excitatory/inhibitory)
    if 'EI_type' in df_ses.columns and len(df_ses.EI_type[0]) > 0:
        idxs_Ini = df_ses.EI_type.values[0] == 'Ini'
        idxs_Exc = df_ses.EI_type.values[0] == 'Exc'
    else:
        # Default: assume all neurons are excitatory
        idxs_Ini = np.zeros(spkcnts.shape[0], dtype=bool)
        idxs_Exc = np.ones(spkcnts.shape[0], dtype=bool)
    
    # Extract optogenetic data (cre-line information)
    if 'opto_10ms' in df_ses.columns and len(df_ses.opto_10ms[0]) > 0:
        opto_data = df_ses.opto_10ms.values[0]
        try:
            if isinstance(opto_data, np.ndarray):
                creline = np.unique(opto_data.flatten())
            else:
                creline = np.unique(opto_data)
            # Get the first non-nan value
            valid_creline = creline[~np.isnan(creline)] if creline.dtype in [np.float64, np.float32] else creline
            creline = valid_creline[0] if len(valid_creline) > 0 else 'unknown'
            idxs_creline = opto_data == creline
        except Exception:
            creline = 'unknown'
            idxs_creline = np.zeros(spkcnts.shape[0], dtype=bool)
    else:
        creline = 'unknown'
        idxs_creline = np.zeros(spkcnts.shape[0], dtype=bool)
    
    # Initialize output dataframe
    dfc = pd.DataFrame()
    
    # Add overall state (-1) to analyze all data together
    states = np.insert(states, 0, -1)
    
    # Create output directory for diagnostic plots if requested
    if saveflag:
        figurespath = f'./results/bias_removal_statistics_session_{session_id}'
        isExist = os.path.exists(figurespath)
        if not isExist:
            os.makedirs(figurespath)
    
    # Iterate over all states (including overall state -1)
    for state in states[:]:
        # Determine timepoints belonging to this state
        if state >= 0:
            # For specific states: use posterior probability > 0.5
            idxs_state = (df_ses.posterior.values[0] * (state_times == state) > 0.5).flatten()
        else:
            # For overall state (-1): use all timepoints
            idxs_state = np.ones_like(state_times).flatten() == 1
        
        # Rebin spike counts for this state
        spkcnts_state_all = rebin2states(spkcnts, idxs_state, bin_size, new_bin_size)
        
        # Calculate state duration
        if spkcnts_state_all is not None:
            state_duration = spkcnts_state_all.shape[1] * new_bin_size
        else:
            state_duration = 0
        
        # Skip states that are too short
        if state_duration < min_state_cumulative_duration:
            continue
        
        # Determine number of time packets for temporal analysis
        # Split long states into ~30 second packets
        if state >= 0:
            N_packets = np.max([state_duration // 30., 1])
            size_packet = int(spkcnts_state_all.shape[1] // N_packets)
        else:
            N_packets = 1
            size_packet = spkcnts_state_all.shape[1]
        
        # Create list of packets to analyze
        list_packets = ['all']  # Always analyze all data together
        if N_packets > 1:
            list_packets.extend(list(np.arange(N_packets)))  # Also analyze individual packets
        
        # Iterate over packets
        for i_packet, packet in enumerate(list_packets):
            if packet != 'all':
                # Analyze individual packet
                idxs_packet = np.arange(size_packet * (i_packet - 1), size_packet * i_packet).astype(int)
                spkcnts_state = spkcnts_state_all[:, idxs_packet]
                
                # Compute firing rates by neuron type
                Ini_FRs = spkcnts_state[idxs_Ini] / new_bin_size
                Exc_FRs = spkcnts_state[idxs_Exc] / new_bin_size
                creline_FRs = spkcnts_state[idxs_creline] / new_bin_size
                
                Ini_FR = spkcnts_state[idxs_Ini].mean() / new_bin_size
                Exc_FR = spkcnts_state[idxs_Exc].mean() / new_bin_size
                creline_FR = spkcnts_state[idxs_creline].mean() / new_bin_size
                
                # No behavioral data for individual packets
                pupil_state, pupil_state_averages, running_state, running_state_averages = np.nan, np.nan, np.nan, np.nan
                iterall = ['all'] + list(areas) + list(layers)
            else:
                # Analyze all data together
                spkcnts_state = spkcnts_state_all[:, :]
                
                # Compute averaged spike counts (for rate analysis)
                spkcnts_state_averages = rebin2states(
                    spkcnts, idxs_state, bin_size, new_bin_size,
                    return_averages=True, min_cum_bin=.1 / bin_size
                )
                
                # Compute firing rates (note: using bin_size for averages, new_bin_size for totals)
                Ini_FRs = spkcnts_state_averages[idxs_Ini] / bin_size
                Exc_FRs = spkcnts_state_averages[idxs_Exc] / bin_size
                creline_FRs = spkcnts_state_averages[idxs_creline] / bin_size
                
                # Mean firing rates
                Ini_FR = spkcnts_state[idxs_Ini].mean() / new_bin_size
                Exc_FR = spkcnts_state[idxs_Exc].mean() / new_bin_size
                creline_FR = spkcnts_state[idxs_creline].mean() / new_bin_size
                
                # Extract behavioral data
                pupil = df_ses['pupil'].values[0]
                if np.any(~np.isnan(pupil)):
                    pupil_state = df_ses['pupil'].values[0][idxs_state].mean()
                    pupil_state_averages = rebin2states(
                        df_ses['pupil'].values[0][np.newaxis, :], idxs_state,
                        bin_size, new_bin_size, return_averages=True,
                        min_cum_bin=.1 / bin_size
                    )
                else:
                    pupil_state = np.nan
                    pupil_state_averages = np.nan
                
                running_state = df_ses['running'].values[0][idxs_state].mean()
                running_state_averages = rebin2states(
                    df_ses['running'].values[0][np.newaxis, :], idxs_state,
                    bin_size, new_bin_size, return_averages=True,
                    min_cum_bin=.1 / bin_size
                )
                iterall = ['all'] + list(areas) + list(layers)
            
            # Iterate over brain areas and layers
            for item in iterall:
                print([i_session, state, item, packet])  # Debug output
                
                # Determine which neurons to include based on area/layer
                if item in areas:
                    idxs_item = area_neurons == item
                    area, layer = item, None
                elif item in layers:
                    idxs_item = layer_neurons == item
                    area, layer = None, item
                elif item == 'all':
                    idxs_item = np.ones_like(layer_neurons) == 1
                    area, layer = item, item
                else:
                    continue  # Skip unknown items
                
                # Skip if too few neurons
                if np.sum(idxs_item) < 35:
                    continue
                
                # Extract spike counts for this subset of neurons
                spkcnts_stateneu = spkcnts_state[idxs_item]
                
                # Compute covariance matrix
                X = spkcnts_stateneu.T  # Transpose: timepoints x neurons
                C = np.cov(X.T)  # Covariance: neurons x neurons
                
                # Determine if we should compute noise-corrected LFA
                # Only for: all neurons, all packets, specific states (not overall)
                compute_noise_LFA = (packet == 'all') and (item == 'all') and (state != -1)
                
                if compute_noise_LFA:
                    # Factor Analysis to separate shared and independent variance
                    solver = FactorAnalysis(random_state=0)
                    
                    if X.sum() == 0:
                        continue
                    
                    # Try to load previously computed number of factors
                    if (item == 'all') and (packet == 'all'):
                        try:
                            import config
                            analysis_dir = config.get_results_dir() / 'sessions_hmm_analysis'
                            files = list(analysis_dir.glob(f'hmm_analysis_*sessionwise*VisualCor*{i_session}.pk*'))
                            if files:
                                dfnew = pd.read_pickle(files[0])
                                n = dfnew[dfnew['state'] == state]['N_components'].unique()[0]
                                print([i_session, 'loading LFA number of factors'])
                            else:
                                raise FileNotFoundError("No previous analysis file found")
                        except Exception:
                            # Compute optimal number of factors using cross-validation
                            print([i_session, 'recomputing LFA number of factors'])
                            cv_scores = []
                            for i_n in n_components_ses:
                                solver.n_components = i_n
                                cv_scores.append(np.mean(cross_val_score(solver, X, cv=5)))
                            
                            # Find elbow point: where improvement rate drops below 5%
                            n = np.where(np.diff(cv_scores) / np.cumsum(np.diff(cv_scores)) < 0.05)[0][0]
                    
                    # Fit Factor Analysis model
                    solver.n_components = n
                    X_transformed = solver.fit_transform(X)
                    loadings = solver.components_
                    cov_shared = np.matmul(loadings[:n].T, loadings[:n])
                    
                    # Compute noise covariance (total - shared)
                    Cnoise = C - cov_shared
                    
                    # Compute statistics on noise covariance
                    stats_LFA = cov_stats(Cnoise)
                    stats_LFA_fit = cov_stats(Cnoise, X.T, n=n, type='LFA', saveflag=saveflag,
                                             titlestring=f'{figurespath}/layarea_{item}_state_{state}_packet_{packet}_LFA_fit')
                else:
                    # No LFA analysis for subsets
                    stats_LFA = {}
                    stats_LFA_fit = {}
                
                # Compute statistics on full covariance
                stats = cov_stats(C)
                statsfit = cov_stats(C, X.T, saveflag=saveflag,
                                    titlestring=f'{figurespath}/layarea_{item}_state_{state}_packet_{packet}_fit')
                
                # Organize statistics with appropriate suffixes
                stats = {key: stats[key] for key in stats.keys()}
                stats_fit = {key + '_fit': statsfit[key] for key in statsfit.keys()}
                stats_LFA = {key + '_LFA': stats_LFA[key] for key in stats_LFA.keys()}
                stats_LFA_fit = {key + '_LFA_fit': stats_LFA_fit[key] for key in stats_LFA_fit.keys()}
                
                # Compile all data properties
                data_properties = {
                    'area': area, 'layer': layer, 'state': state, 'creline': creline,
                    'packet': packet, 'pupil': pupil_state, 'running': running_state,
                    'pupil_xstate': [pupil_state_averages],
                    'running_xstate': [running_state_averages],
                    'session': session_id, 'i_session': i_session,
                    'state_duration': state_duration
                }
                
                # Compile firing rate data
                firing_rates = {
                    'Exc_FR': Exc_FR, 'Ini_FR': Ini_FR, 'creline_FR': creline_FR,
                    'Exc_FRs': [Exc_FRs], 'Ini_FRs': [Ini_FRs], 'creline_FRs': [creline_FRs]
                }
                
                # Combine all data into single row
                dfnew = pd.DataFrame({
                    **data_properties, **firing_rates, **stats, **stats_fit,
                    **stats_LFA, **stats_LFA_fit
                }, index=[0])
                
                # Append to results dataframe
                dfc = pd.concat([dfc, dfnew], ignore_index=True)
    
    return dfc


def fun_significance(x):
    """
    Convert p-values to significance symbols.
    
    Args:
        x: Array of p-values
        
    Returns:
        Array of strings with significance symbols:
        - '' for p >= 0.05
        - '*' for p < 0.05
        - '**' for p < 0.01
        - '***' for p < 0.005
    """
    significance = np.char.replace(np.char.mod('%d', np.zeros_like(x)), '0', '').astype(object)
    significance[np.where(x < 0.05)] = '*'
    significance[np.where(x < 0.01)] = '**'
    significance[np.where(x < 0.005)] = '***'
    return significance


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    import config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run HMM analysis on a single session')
    parser.add_argument('--session-id', type=str, required=True,
                        help='Session ID to process (e.g., "767871931")')
    args = parser.parse_args()
    
    session_id = args.session_id
    print(f'============================================================')
    print(f'HMM Analysis')
    print(f'============================================================')
    print(f'Processing session: {session_id}')
    
    # Get directories using config helper functions
    preprocessed_dir = config.get_preprocessed_folder()
    hmm_crossval_dir = config.get_hmm_crossval_folder()
    results_dir = config.get_results_dir()
    
    # Create output directory
    output_dir = results_dir / 'sessions_hmm_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find preprocessed session file
    file_data = preprocessed_dir / config.PREPROCESSED_FILE_PATTERN.format(session_id=session_id)
    if not file_data.exists():
        print(f"[ERROR] Preprocessed file not found: {file_data}")
        print(f"Available preprocessed files:")
        for f in sorted(preprocessed_dir.glob('df_*.pkl')):
            print(f"  {f.name}")
        sys.exit(1)
    
    # Find corresponding HMM file
    file_hmm = hmm_crossval_dir / config.HMM_CROSSVAL_FILE_PATTERN.format(session_id=session_id)
    if not file_hmm.exists():
        print(f"[ERROR] HMM crossvalidation file not found: {file_hmm}")
        print(f"Available HMM files:")
        for f in sorted(hmm_crossval_dir.glob('hmm_*.pkl')):
            print(f"  {f.name}")
        sys.exit(1)
    
    # Load data
    print(f"Loading preprocessed data: {file_data.name}")
    df_data = pd.read_pickle(file_data)
    
    print(f"Loading HMM results: {file_hmm.name}")
    df_hmm = pd.read_pickle(file_hmm)
    
    # Combine dataframes
    df_ses = pd.concat([df_data, df_hmm], axis=1)
    
    # Run analysis
    print(f"Running HMM analysis for session {session_id}...")
    saveflag = True
    dfc_ses = hmm_spontaneous_sessionwise(df_ses, saveflag=saveflag)
    
    # Save results
    output_file = output_dir / f'hmm_analysis_{session_id}.pkl'
    dfc_ses.to_pickle(output_file)
    print(f"[OK] Saved HMM analysis results for session {session_id} to {output_file}")
    print(f'============================================================')
    print(f'HMM Analysis Complete')
    print(f'============================================================')
