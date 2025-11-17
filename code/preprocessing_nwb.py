#!/usr/bin/env python3
"""
Preprocessing script for NWB files from Allen Brain Observatory.

This script processes NWB files to extract neural spike data during spontaneous
epochs, filters units by brain region/area/layer, and prepares data for HMM analysis.

Main processing steps:
1. Extract units from NWB file
2. Merge with unit metadata from CSV
3. Classify waveforms and optogenetic responses
4. Filter units by region/area/layer
5. Extract spike counts during spontaneous epochs
6. Extract behavioral data (running speed, pupil area)
7. Save preprocessed data as pickle file
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# Import shared utilities and configuration
from utils import (
    midbins,
    find_permutation_seq,
    sequence2midpoints,
    isininterval,
    area2region,
    grpBySameConsecutiveItem,
    classify_waveform,
    optotagging_spike_counts,
    classify_10mspulses as classify_10mspulses_util,
    get_spikecounts_during_spontaneous_epochs_nwb,
)
import config

# Set numpy random seed for reproducibility
np.random.seed(config.RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore')

# Set up paths using config
datafolder = config.get_data_dir()
results_dir = config.get_results_dir()
savefolder = config.get_preprocessed_folder()
nwb_folder = config.get_nwb_folder()
sessions_file = config.get_sessions_csv()
unit_table_file = config.get_unit_table_csv()

# Create output directory
savefolder.mkdir(parents=True, exist_ok=True)

# Load sessions table to filter for functional_connectivity sessions
def load_functional_connectivity_sessions():
    """
    Load session IDs that are labeled as functional_connectivity.
    
    Reads the sessions.csv file and filters for sessions with
    session_type == 'functional_connectivity'.
    
    Returns:
        set: Set of session ID strings, or None if sessions.csv doesn't exist
    """
    if not sessions_file.exists():
        print(f"Warning: {sessions_file} not found, processing all NWB files")
        return None
    
    sessions_df = pd.read_csv(sessions_file)
    fc_sessions = sessions_df[sessions_df['session_type'] == 'functional_connectivity']
    # Use 'id' column (not 'ecephys_session_id')
    fc_session_ids = set(fc_sessions['id'].astype(str))
    print(f"Found {len(fc_session_ids)} functional_connectivity sessions")
    return fc_session_ids


def get_spikecounts_during_spontaneous_epochs(nwb_file, uID_list, bSize=None, binarize=False, dtype=None):
    """
    Extract spike counts during spontaneous epochs from NWB file.
    
    Wrapper function that calls the utility function from utils.
    
    Args:
        nwb_file: NWB file object
        uID_list: List of unit IDs to extract
        bSize: Bin size in seconds (uses config default if None)
        binarize: Whether to binarize spike counts
        dtype: Optional data type for output arrays
        
    Returns:
        Tuple of (spikecount_list, timecourse_list)
    """
    if bSize is None:
        bSize = config.BIN_SIZE_SPONTANEOUS
    return get_spikecounts_during_spontaneous_epochs_nwb(nwb_file, uID_list, bSize, binarize, dtype)


def classify_10mspulses_nwb(units_details, nwb_file):
    """
    Classify units based on responses to 10ms optogenetic pulses.
    
    Wrapper function that calls the utility function from utils.
    
    Args:
        units_details: DataFrame with unit details
        nwb_file: NWB file object
        
    Returns:
        DataFrame with added 'opto_10ms' column
    """
    return classify_10mspulses_util(units_details, nwb_file, use_nwb=True)

def create_units_dataframe(nwb_file, session_id):
    """
    Create units DataFrame from NWB file, extracting unit information.
    
    Extracts spike times, waveform properties, quality metrics, and other
    unit metadata from the NWB file.
    
    Args:
        nwb_file: NWB file object
        session_id: Session ID string
        
    Returns:
        DataFrame with unit information
    """
    units = nwb_file.units
    unit_data = []
    
    for i, unit in enumerate(units):
        unit_info = {}
        
        # Basic unit information - use the actual unit ID from NWB DataFrame index
        unit_info['unit_id'] = unit.index[0] if hasattr(unit, 'index') and len(unit.index) > 0 else i
        unit_info['session_id'] = session_id
        
        # Spike times - access correctly from the units table
        if hasattr(unit, 'spike_times') and unit.spike_times is not None:
            spike_times = unit.spike_times
            if isinstance(spike_times, pd.Series):
                spike_times = spike_times.values[0] if len(spike_times.values) > 0 else np.array([])
            unit_info['spike_times'] = spike_times
            unit_info['num_spikes'] = len(spike_times)
        else:
            unit_info['spike_times'] = np.array([])
            unit_info['num_spikes'] = 0
        
        # Waveform information
        unit_info['waveform_mean'] = unit['waveform_mean'].values[0] if hasattr(unit, 'waveform_mean') else None
        unit_info['waveform_std'] = None  # Not available in NWB
        unit_info['waveform_duration'] = unit['waveform_duration'].values[0] if hasattr(unit, 'waveform_duration') else None
        unit_info['waveform_halfwidth'] = unit['waveform_halfwidth'].values[0] if hasattr(unit, 'waveform_halfwidth') else None
        
        # Electrode information
        unit_info['electrodes'] = None  # Not directly available
        unit_info['peak_channel_id'] = unit['peak_channel_id'].values[0] if hasattr(unit, 'peak_channel_id') else None
        
        # Quality metrics
        unit_info['quality'] = unit['quality'].values[0] if hasattr(unit, 'quality') else None
        unit_info['isi_violations'] = unit['isi_violations'].values[0] if hasattr(unit, 'isi_violations') else None
        unit_info['firing_rate'] = unit['firing_rate'].values[0] if hasattr(unit, 'firing_rate') else None
        unit_info['amplitude_cutoff'] = unit['amplitude_cutoff'].values[0] if hasattr(unit, 'amplitude_cutoff') else None
        unit_info['presence_ratio'] = unit['presence_ratio'].values[0] if hasattr(unit, 'presence_ratio') else None
        unit_info['isolation_distance'] = unit['isolation_distance'].values[0] if hasattr(unit, 'isolation_distance') else None
        unit_info['l_ratio'] = unit['l_ratio'].values[0] if hasattr(unit, 'l_ratio') else None
        unit_info['d_prime'] = unit['d_prime'].values[0] if hasattr(unit, 'd_prime') else None
        unit_info['snr'] = unit['snr'].values[0] if hasattr(unit, 'snr') else None
        
        # Additional metrics
        unit_info['halfwidth'] = unit['waveform_halfwidth'].values[0] if hasattr(unit, 'waveform_halfwidth') else None
        unit_info['PT_ratio'] = unit['PT_ratio'].values[0] if hasattr(unit, 'PT_ratio') else None
        unit_info['repolarization_slope'] = unit['repolarization_slope'].values[0] if hasattr(unit, 'repolarization_slope') else None
        unit_info['recovery_slope'] = unit['recovery_slope'].values[0] if hasattr(unit, 'recovery_slope') else None
        unit_info['amplitude'] = unit['amplitude'].values[0] if hasattr(unit, 'amplitude') else None
        
        # Depth information
        unit_info['depth'] = None  # Not directly available in NWB
        
        # Probe information
        unit_info['probe_id'] = None  # Not directly available
        unit_info['probe_description'] = None  # Not directly available
        
        # Location information
        unit_info['location'] = None
        unit_info['ecephys_structure_acronym'] = None
        unit_info['ecephys_structure_id'] = None
        
        # CCF coordinates
        unit_info['anterior_posterior_ccf_coordinate'] = None
        unit_info['dorsal_ventral_ccf_coordinate'] = None
        unit_info['left_right_ccf_coordinate'] = None
        
        # Layer and area
        unit_info['layer'] = None
        unit_info['area'] = None
        
        unit_data.append(unit_info)
    
    # Create DataFrame
    df_units = pd.DataFrame(unit_data)
    return df_units

def preprocess_data(nwb_file, session_id, region=None, area=None, layer=None, N_min_neurons=None, bin_size=None):
    """
    Main preprocessing function: extract and prepare neural data from NWB file.
    
    Processing pipeline:
    1. Extract units from NWB file
    2. Load and merge unit metadata from CSV
    3. Classify waveforms and optogenetic responses
    4. Assign region/area information
    5. Filter units by region/area/layer
    6. Extract spike counts during spontaneous epochs
    7. Extract behavioral data (running speed, pupil area)
    8. Prepare output DataFrame
    
    Args:
        nwb_file: NWB file object
        session_id: Session ID string
        region: Brain region to filter (e.g., 'VisualCortex')
        area: Brain area to filter (e.g., 'VISp')
        layer: Cortical layer to filter (0 = all layers)
        N_min_neurons: Minimum number of neurons required
        bin_size: Bin size in seconds for spike counts
        
    Returns:
        DataFrame with preprocessed data, or None if processing fails
    """
    # Use config defaults if not specified
    if region is None:
        region = config.DEFAULT_REGION
    if area is None:
        area = config.DEFAULT_AREA
    if layer is None:
        layer = config.DEFAULT_LAYER
    if N_min_neurons is None:
        N_min_neurons = config.N_MIN_NEURONS_PREPROCESS
    if bin_size is None:
        bin_size = config.BIN_SIZE
    
    preprocess_start = time.time()
    
    print(f"\n[PREPROCESS {session_id}] Starting preprocessing pipeline")
    print(f"[PREPROCESS {session_id}] Parameters: region={region}, area={area}, layer={layer}, bin_size={bin_size}")
    print(f"[PREPROCESS {session_id}] {'-'*50}")
    
    # Create units dataframe from NWB file
    print(f"[PREPROCESS {session_id}] Step 1: Extracting units from NWB file...")
    step_start = time.time()
    units = create_units_dataframe(nwb_file, session_id)
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Extracted {len(units)} units in {step_time:.1f}s")
    
    if units.empty:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No units data available")
        return None
    
    # Load unit details from CSV and merge with NWB data
    print(f"[PREPROCESS {session_id}] Step 2: Loading and merging unit details...")
    step_start = time.time()
    if unit_table_file.exists():
        print(f"[PREPROCESS {session_id}]   Loading unit details from {unit_table_file}")
        units_details = pd.read_csv(unit_table_file)
        units_details = units_details.rename(columns={'Unnamed: 0': 'unit_id'})
        
        print(f"[PREPROCESS {session_id}]   Units details shape: {units_details.shape}")
        print(f"[PREPROCESS {session_id}]   Units dataframe shape before merge: {units.shape}")
        common_ids = len(set(units['unit_id']) & set(units_details['unit_id']))
        print(f"[PREPROCESS {session_id}]   Common unit_ids: {common_ids}")
        
        # Classify waveforms and optogenetic responses on units_details before merge
        print(f"[PREPROCESS {session_id}]   Classifying waveforms...")
        if 'waveform_duration' in units_details.columns:
            units_details = classify_waveform(units_details)
            print(f"[PREPROCESS {session_id}]   [OK] Waveform classification complete")
        else:
            units_details['EI_type'] = 'Unknown'
            print(f"[PREPROCESS {session_id}]   ⚠ No waveform_duration column, skipping classification")
        
        # Classify optogenetic responses (for NWB files, this will set all to NaN)
        print(f"[PREPROCESS {session_id}]   Classifying optogenetic responses...")
        units_details = classify_10mspulses_nwb(units_details, nwb_file)
        
        # Merge on unit_id
        print(f"[PREPROCESS {session_id}]   Merging unit data...")
        units = pd.merge(units, units_details, on=['unit_id'], how='inner')
        print(f"[PREPROCESS {session_id}]   Units dataframe shape after merge: {units.shape}")
        
        if units.shape[0] == 0:
            print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No matching units found after merge, using basic unit data")
            units['EI_type'] = 'Unknown'
            units['opto_10ms'] = np.nan
    else:
        print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No unit_table_all.csv found, using basic unit data")
        units['EI_type'] = 'Unknown'
        units['opto_10ms'] = np.nan
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 2 completed in {step_time:.1f}s")
    
    # Add region information if ecephys_structure_acronym exists
    print(f"[PREPROCESS {session_id}] Step 3: Assigning region and area information...")
    step_start = time.time()
    ecephys_col = None
    if 'ecephys_structure_acronym_y' in units.columns:
        ecephys_col = 'ecephys_structure_acronym_y'
    elif 'ecephys_structure_acronym_x' in units.columns:
        ecephys_col = 'ecephys_structure_acronym_x'
    elif 'ecephys_structure_acronym' in units.columns:
        ecephys_col = 'ecephys_structure_acronym'
    
    if ecephys_col is not None:
        valid_acronyms = units[ecephys_col].dropna()
        if len(valid_acronyms) > 0:
            # Use the original ecephys_structure_acronym as area
            units['area'] = units[ecephys_col]
            
            # Create region mapping manually
            region_mapping = {
                'LGd': 'Thalamus', 'LGn': 'Thalamus', 'LP': 'Thalamus', 'LD': 'Thalamus', 'POL': 'Thalamus', 
                'MD': 'Thalamus', 'VPL': 'Thalamus', 'PO': 'Thalamus', 'VPM': 'Thalamus', 'RT': 'Thalamus', 
                'MG': 'Thalamus', 'MGv': 'Thalamus', 'MGd': 'Thalamus', 'Eth': 'Thalamus', 'SGN': 'Thalamus', 'TH': 'Thalamus',
                'RSP': 'others', 'OLF': 'others', 'BLA': 'others', 'ZI': 'others', 'grey': 'others',
                'DG': 'Hippocampus', 'CA3': 'Hippocampus', 'CA1': 'Hippocampus', 'SUB': 'Hippocampus', 'POST': 'Hippocampus', 'ProS': 'Hippocampus',
                'ACA': 'FrontalCortex', 'MOs': 'FrontalCortex', 'PL': 'FrontalCortex', 'ILA': 'FrontalCortex', 'ORB': 'FrontalCortex', 'MOp': 'FrontalCortex', 'SSp': 'FrontalCortex',
                'VISp': 'VisualCortex', 'VISl': 'VisualCortex', 'VISpm': 'VisualCortex', 'VISam': 'VisualCortex', 'VISrl': 'VisualCortex', 
                'VISa': 'VisualCortex', 'VISal': 'VisualCortex', 'VIS': 'VisualCortex', 'VISli': 'VisualCortex', 'VISlm': 'VisualCortex',
                'SCs': 'Midbrain', 'SCm': 'Midbrain', 'MRN': 'Midbrain', 'APN': 'Midbrain', 'PAG': 'Midbrain', 'MB': 'Midbrain',
                'CP': 'BasalGanglia', 'GPe': 'BasalGanglia', 'SNr': 'BasalGanglia', 'ACB': 'BasalGanglia', 'LS': 'BasalGanglia'
            }
            
            # Map regions
            units['region'] = units[ecephys_col].map(region_mapping).fillna('Unknown')
            unique_regions = units['region'].unique()
            print(f"[PREPROCESS {session_id}]   Found {len(unique_regions)} unique regions: {', '.join(unique_regions)}")
        else:
            print(f"[PREPROCESS {session_id}]   ⚠ WARNING: All ecephys_structure_acronym values are None, skipping region assignment")
            units['region'] = 'Unknown'
            units['area'] = 'Unknown'
    else:
        print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No ecephys_structure_acronym column found, skipping region assignment")
        units['region'] = 'Unknown'
        units['area'] = 'Unknown'
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 3 completed in {step_time:.1f}s")
    
    # Set index
    units = units.set_index('unit_id')
    
    # Filter by region
    print(f"[PREPROCESS {session_id}] Step 4: Filtering units by region/area/layer...")
    step_start = time.time()
    if region is not None:
        unitsregion = units[units['region'] == region]
        print(f"[PREPROCESS {session_id}]   Filtered by region '{region}': {len(unitsregion)} units")
    else:
        unitsregion = units
        print(f"[PREPROCESS {session_id}]   No region filter: {len(unitsregion)} units")
    
    if area is not None:
        unitsregion = unitsregion[unitsregion['area'] == area]
        print(f"[PREPROCESS {session_id}]   Filtered by area '{area}': {len(unitsregion)} units")
    
    if layer == 0:
        unitssel = unitsregion
    else:
        unitssel = unitsregion[unitsregion['cortical_layer'] == layer]
        print(f"[PREPROCESS {session_id}]   Filtered by layer {layer}: {len(unitssel)} units")

    print(f"[PREPROCESS {session_id}]   Final selected units: {len(unitssel)}")
    if unitssel.shape[0] < N_min_neurons:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: Number of neurons ({unitssel.shape[0]}) below minimum ({N_min_neurons})")
        return None
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 4 completed in {step_time:.1f}s")

    # Extract spike counts during spontaneous epochs
    print(f"[PREPROCESS {session_id}] Step 5: Extracting spike counts from spontaneous epochs...")
    step_start = time.time()
    spkspont = get_spikecounts_during_spontaneous_epochs(nwb_file, unitssel.index.values, bSize=bin_size)
    
    if not spkspont[0]:  # No spontaneous epochs found
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No spontaneous epochs found, skipping session")
        return None
    
    num_epochs = len(spkspont[0])
    total_timepoints = sum([epoch.shape[1] for epoch in spkspont[0]])
    print(f"[PREPROCESS {session_id}]   Found {num_epochs} spontaneous epochs")
    print(f"[PREPROCESS {session_id}]   Total timepoints: {total_timepoints}")
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 5 completed in {step_time:.1f}s")
    
    print(f"[PREPROCESS {session_id}] Step 6: Concatenating spike counts and extracting behavioral data...")
    step_start = time.time()
    spkcnts = np.concatenate(spkspont[0], axis=1)
    all_times = np.concatenate([spkspont[1][i] for i in range(len(spkspont[1]))])
    print(f"[PREPROCESS {session_id}]   Concatenated spike counts shape: {spkcnts.shape} (neurons x timepoints)")
    
    # Extract running speed (if available)
    running = np.nan
    if hasattr(nwb_file, 'processing') and 'running' in nwb_file.processing:
        running_module = nwb_file.processing['running']
        if 'running_speed' in running_module.data_interfaces:
            running_data = running_module.data_interfaces['running_speed']
            if hasattr(running_data, 'data') and hasattr(running_data, 'timestamps'):
                running_times = running_data.timestamps[:]
                running_values = running_data.data[:]
                # Interpolate to match spike times
                running = np.interp(all_times, running_times, running_values)
                print(f"[PREPROCESS {session_id}]   [OK] Extracted running speed data")
    else:
        print(f"[PREPROCESS {session_id}]   No running speed data available")
    
    # Extract pupil data (if available)
    pupil = np.nan
    if hasattr(nwb_file, 'processing') and 'filtered_gaze_mapping' in nwb_file.processing:
        gaze_module = nwb_file.processing['filtered_gaze_mapping']
        if 'pupil_area' in gaze_module.data_interfaces:
            pupil_data = gaze_module.data_interfaces['pupil_area']
            if hasattr(pupil_data, 'data') and hasattr(pupil_data, 'timestamps'):
                pupil_times = pupil_data.timestamps[:]
                pupil_values = pupil_data.data[:]
                # Interpolate to match spike times
                pupil = np.interp(all_times, pupil_times, pupil_values)
                print(f"[PREPROCESS {session_id}]   [OK] Extracted pupil area data")
    else:
        print(f"[PREPROCESS {session_id}]   No pupil area data available")
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 6 completed in {step_time:.1f}s")
    
    # Prepare output data
    print(f"[PREPROCESS {session_id}] Step 7: Preparing output dataframe...")
    step_start = time.time()
    EI_type = unitssel['EI_type'].values if 'EI_type' in unitssel.columns else np.array(['Unknown'] * len(unitssel))
    opto_10ms = unitssel['opto_10ms'].values if 'opto_10ms' in unitssel.columns else np.array([np.nan] * len(unitssel))
    
    # Use ecephys_structure_acronym for areas (this comes from the merge with unit_table_all.csv)
    if 'ecephys_structure_acronym' in unitssel.columns:
        areas_values = unitssel['ecephys_structure_acronym'].values
    elif 'area' in unitssel.columns:
        areas_values = unitssel['area'].values
    else:
        areas_values = np.array(['Unknown'] * len(unitssel))
    
    layers_values = unitssel['cortical_layer'].values if 'cortical_layer' in unitssel.columns else np.array([0] * len(unitssel))
    
    df = pd.DataFrame({
        'session_id': session_id, 
        'stimulus': 'spontaneous', 
        'region': region, 
        'area': area, 
        'layer': layer,
        'epoch': 'all', 
        'state': 'all', 
        'spkcnts': [spkcnts], 
        'times': [all_times], 
        'EI_type': [EI_type], 
        'opto_10ms': [opto_10ms], 
        'areas': [areas_values], 
        'layers': [layers_values],
        'running': [running], 
        'pupil': [pupil], 
        'N_trials': np.nan
    })
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 7 completed in {step_time:.1f}s")
    
    total_preprocess_time = time.time() - preprocess_start
    print(f"\n[PREPROCESS {session_id}] {'='*50}")
    print(f"[PREPROCESS {session_id}] [OK] PREPROCESSING COMPLETE")
    print(f"[PREPROCESS {session_id}] Total preprocessing time: {total_preprocess_time:.1f}s ({total_preprocess_time/60:.1f} minutes)")
    print(f"[PREPROCESS {session_id}] Output shape: {df.shape}")
    print(f"[PREPROCESS {session_id}] {'='*50}\n")
    
    return df

# Main execution block
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=str, help='Session ID (e.g., "766640955") or index (integer)')
    parser.add_argument('--all-sessions', action='store_true', 
                       help='Process all sessions, not just functional_connectivity')
    args = parser.parse_args()
    session_arg = args.session
    process_all_sessions = args.all_sessions

    # Preprocessing parameters (use config defaults)
    bin_size = config.BIN_SIZE
    region = config.DEFAULT_REGION
    area = config.DEFAULT_AREA
    layers = [config.DEFAULT_LAYER]
    regions_of_interest = config.REGIONS_OF_INTEREST
    areas_of_interest = config.AREAS_OF_INTEREST
    
    # Load functional connectivity sessions
    fc_session_ids = load_functional_connectivity_sessions()
    
    # Find NWB files and filter for functional_connectivity sessions
    all_nwb_files = list(nwb_folder.glob('*.nwb'))
    if not all_nwb_files:
        print("No NWB files found in", nwb_folder)
        exit(1)
    
    # Filter NWB files based on session type
    nwb_files = []
    if process_all_sessions:
        print("Processing all available NWB files (--all-sessions flag used)")
        nwb_files = all_nwb_files
    else:
        # Filter NWB files to only include functional_connectivity sessions
        for nwb_file in all_nwb_files:
            session_id = nwb_file.stem.replace('session_', '')
            if fc_session_ids is None or session_id in fc_session_ids:
                nwb_files.append(nwb_file)
    
    if not nwb_files:
        print(f"No functional_connectivity NWB files found. Available files: {len(all_nwb_files)}")
        print("Available NWB files:")
        for nwb_file in all_nwb_files:
            session_id = nwb_file.stem.replace('session_', '')
            print(f"  {session_id}")
        print("\nFunctional connectivity session IDs from CSV:")
        if fc_session_ids:
            for session_id in sorted(list(fc_session_ids)):
                print(f"  {session_id}")
        print("\nTo process functional_connectivity sessions, you need to download the corresponding NWB files first.")
        exit(1)
    
    print(f"Found {len(nwb_files)} functional_connectivity NWB files out of {len(all_nwb_files)} total files")
    print(f"NWB folder: {nwb_folder}")
    print(f"Output folder: {savefolder}")
    
    # Determine if session_arg is a session_id or an index
    # First, check if it's a session_id by looking for matching NWB file
    session_id_str = str(session_arg)
    nwb_file_path = nwb_folder / f'session_{session_id_str}.nwb'
    
    print(f"\n[PREPROCESS] Determining session from argument: '{session_arg}'")
    if nwb_file_path.exists():
        # It's a session_id and the file exists
        session_id = session_id_str
        file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS] [OK] Found NWB file: {nwb_file_path.name} ({file_size_mb:.1f} MB)")
        print(f"[PREPROCESS] Processing session_id: {session_id}")
    else:
        # Try to parse as integer index
        try:
            i_iterator = int(session_arg)
            if i_iterator < 0 or i_iterator >= len(nwb_files):
                print(f"[PREPROCESS] [ERROR] ERROR: Session index {i_iterator} out of range. Available functional_connectivity sessions: {len(nwb_files)}")
                print(f"[PREPROCESS] Available NWB files:")
                for nwb_file in nwb_files:
                    print(f"[PREPROCESS]   {nwb_file.stem.replace('session_', '')}")
                exit(1)
            nwb_file_path = nwb_files[i_iterator]
            session_id = nwb_file_path.stem.replace('session_', '')
            file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
            print(f"[PREPROCESS] [OK] Using index {i_iterator} -> {nwb_file_path.name} ({file_size_mb:.1f} MB)")
            print(f"[PREPROCESS] Processing functional_connectivity session {i_iterator + 1}/{len(nwb_files)}")
            print(f"[PREPROCESS] Session ID: {session_id}")
        except ValueError:
            # Not a valid integer and file doesn't exist
            print(f"[PREPROCESS] [ERROR] ERROR: NWB file not found: {nwb_file_path}")
            print(f"[PREPROCESS] Available NWB files:")
            for nwb_file in nwb_files:
                print(f"[PREPROCESS]   {nwb_file.stem.replace('session_', '')}")
            exit(1)
    
    # Load NWB file
    print(f"\n[PREPROCESS {session_id}] Loading NWB file: {nwb_file_path}")
    load_start = time.time()
    try:
        io = NWBHDF5IO(str(nwb_file_path), mode='r')
        nwb_file = io.read()
        load_time = time.time() - load_start
        file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS {session_id}] [OK] NWB file loaded ({file_size_mb:.1f} MB) in {load_time:.1f}s")
    except Exception as e:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR loading NWB file {nwb_file_path}: {e}")
        exit(1)
    
    # Process data
    df = preprocess_data(nwb_file, session_id, region=region, area=area, layer=layers[0], bin_size=bin_size)
    
    # Save results
    if df is not None:
        print(f"\n[PREPROCESS {session_id}] Saving preprocessed data...")
        save_start = time.time()
        output_file = savefolder / config.PREPROCESSED_FILE_PATTERN.format(session_id=session_id)
        df.to_pickle(output_file)
        save_time = time.time() - save_start
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS {session_id}] [OK] Saved preprocessed data: {output_file}")
        print(f"[PREPROCESS {session_id}]   - File size: {output_size_mb:.2f} MB")
        print(f"[PREPROCESS {session_id}]   - DataFrame shape: {df.shape}")
        print(f"[PREPROCESS {session_id}]   - Spike counts shape: {df['spkcnts'].values[0].shape}")
        print(f"[PREPROCESS {session_id}]   - Time points: {len(df['times'].values[0])}")
        print(f"[PREPROCESS {session_id}]   - Save time: {save_time:.1f}s")
    else:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No data processed for this session")
    
    # Clean up
    io.close()
    print(f"[PREPROCESS {session_id}] [OK] NWB file closed")