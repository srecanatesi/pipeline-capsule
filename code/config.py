"""
Configuration file for neural states dimensionality analysis pipeline.

This module centralizes all hyperparameters, constants, and configuration settings
used across the analysis pipeline.
"""

from pathlib import Path
import numpy as np

# ============================================================================
# Data Processing Parameters
# ============================================================================

# Spike count binning
BIN_SIZE = 0.005  # seconds (5 ms) - default bin size for spike counts
BIN_SIZE_SPONTANEOUS = 0.5  # seconds - bin size for spontaneous epochs extraction
NEW_BIN_SIZE = 0.1  # seconds - rebinned size for state analysis

# Spontaneous epoch filtering
MIN_EPOCH_DURATION = 1500.0  # seconds (25 minutes) - minimum duration for spontaneous epochs

# Unit filtering
N_MIN_NEURONS = 35  # minimum number of neurons required for analysis
N_MIN_NEURONS_PREPROCESS = 1  # minimum neurons for preprocessing (less strict)

# Brain region filtering
DEFAULT_REGION = 'VisualCortex'
DEFAULT_AREA = None  # None means all areas
DEFAULT_LAYER = 0  # 0 means all layers

# Regions and areas of interest
REGIONS_OF_INTEREST = ['Thalamus', 'others', 'Hippocampus', 'FrontalCortex', 
                       'VisualCortex', 'Midbrain', 'BasalGanglia']
AREAS_OF_INTEREST = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam', 
                     'LGn', 'LGd', 'CA1', 'CA3', 'DG']

# ============================================================================
# HMM Cross-validation Parameters
# ============================================================================

# State space for cross-validation
HMM_K_MIN = 2  # minimum number of states to test
HMM_K_MAX = 20  # maximum number of states to test

# Cross-validation settings
HMM_N_KFOLD = 5  # number of cross-validation folds

# Iteration parameters
HMM_N_ITERS_XVAL = 1000  # iterations for cross-validation
HMM_N_ITERS_XVAL_PROCESS = 500  # iterations for process_session cross-validation
HMM_N_ITERS_FINAL = 1000  # iterations for final HMM fit
HMM_N_ITERS_FINAL_PROCESS = 500  # iterations for process_session final fit

# Convergence tolerance
HMM_TOLERANCE = 0.01  # default tolerance
HMM_TOLERANCE_XVAL = 1.0  # tolerance for cross-validation
HMM_TOLERANCE_FINAL = 1.#0.001  # tolerance for final fit

# Final fit parameters
HMM_N_FINAL_FIT = 3  # number of final fits to perform (best one selected)

# ============================================================================
# Dry Run Parameters (for testing/debugging with reduced iterations)
# ============================================================================

# Dry run mode flag
DRY_RUN_ENABLED = True  # Set to True to enable dry run mode

# Dry run state space (reduced for faster testing)
DRY_RUN_K_MIN = 2  # minimum number of states to test in dry run
DRY_RUN_K_MAX = 20  # maximum number of states to test in dry run (reduced from 20)

# Dry run cross-validation settings
DRY_RUN_N_KFOLD = 2  # number of cross-validation folds for dry run (reduced from 5)

# Dry run iteration parameters (significantly reduced for speed)
DRY_RUN_N_ITERS_XVAL = 100  # iterations for cross-validation in dry run (reduced from 1000)
DRY_RUN_N_ITERS_XVAL_PROCESS = 100  # iterations for process_session cross-validation in dry run (reduced from 500)
DRY_RUN_N_ITERS_FINAL = 100  # iterations for final HMM fit in dry run (reduced from 1000)
DRY_RUN_N_ITERS_FINAL_PROCESS = 100  # iterations for process_session final fit in dry run (reduced from 500)

# Dry run convergence tolerance (more relaxed for faster convergence)
DRY_RUN_TOLERANCE = 0.1  # default tolerance for dry run (more relaxed)
DRY_RUN_TOLERANCE_XVAL = 1.  # tolerance for cross-validation in dry run
DRY_RUN_TOLERANCE_FINAL = 1.  # tolerance for final fit in dry run (more relaxed from 0.001)

# Dry run final fit parameters
DRY_RUN_N_FINAL_FIT = 1  # number of final fits to perform in dry run (reduced from 3)

# Dry run neuron selection
DRY_RUN_N_NEURONS = 20  # number of neurons to randomly sample in dry run mode (None = use all neurons)

# Dry run knee detection parameters
DRY_RUN_KNEE_LOCATOR_S = 1.0  # sensitivity parameter for knee detection in dry run (more aggressive)

# Knee detection parameters
KNEE_LOCATOR_S = 1.0  # sensitivity parameter for knee detection
KNEE_CURVE = 'concave'
KNEE_DIRECTION = 'increasing'

# ============================================================================
# Helper Functions for Dry Run Parameters
# ============================================================================

def get_hmm_k_min():
    """Get minimum number of states based on dry run mode."""
    return DRY_RUN_K_MIN if DRY_RUN_ENABLED else HMM_K_MIN


def get_hmm_k_max():
    """Get maximum number of states based on dry run mode."""
    return DRY_RUN_K_MAX if DRY_RUN_ENABLED else HMM_K_MAX


def get_hmm_n_kfold():
    """Get number of cross-validation folds based on dry run mode."""
    return DRY_RUN_N_KFOLD if DRY_RUN_ENABLED else HMM_N_KFOLD


def get_hmm_n_iters_xval():
    """Get cross-validation iterations based on dry run mode."""
    return DRY_RUN_N_ITERS_XVAL if DRY_RUN_ENABLED else HMM_N_ITERS_XVAL


def get_hmm_n_iters_xval_process():
    """Get process_session cross-validation iterations based on dry run mode."""
    return DRY_RUN_N_ITERS_XVAL_PROCESS if DRY_RUN_ENABLED else HMM_N_ITERS_XVAL_PROCESS


def get_hmm_n_iters_final():
    """Get final fit iterations based on dry run mode."""
    return DRY_RUN_N_ITERS_FINAL if DRY_RUN_ENABLED else HMM_N_ITERS_FINAL


def get_hmm_n_iters_final_process():
    """Get process_session final fit iterations based on dry run mode."""
    return DRY_RUN_N_ITERS_FINAL_PROCESS if DRY_RUN_ENABLED else HMM_N_ITERS_FINAL_PROCESS


def get_hmm_tolerance_xval_process():
    """Get cross-validation tolerance for process_session based on dry run mode."""
    return DRY_RUN_TOLERANCE_XVAL if DRY_RUN_ENABLED else HMM_TOLERANCE_XVAL


def get_hmm_tolerance_final_process():
    """Get final fit tolerance for process_session based on dry run mode."""
    return DRY_RUN_TOLERANCE_FINAL if DRY_RUN_ENABLED else HMM_TOLERANCE_FINAL


def get_hmm_tolerance():
    """Get default tolerance based on dry run mode."""
    return DRY_RUN_TOLERANCE if DRY_RUN_ENABLED else HMM_TOLERANCE


def get_hmm_tolerance_xval():
    """Get cross-validation tolerance based on dry run mode."""
    return DRY_RUN_TOLERANCE_XVAL if DRY_RUN_ENABLED else HMM_TOLERANCE_XVAL


def get_hmm_tolerance_final():
    """Get final fit tolerance based on dry run mode."""
    return DRY_RUN_TOLERANCE_FINAL if DRY_RUN_ENABLED else HMM_TOLERANCE_FINAL


def get_hmm_n_final_fit():
    """Get number of final fits based on dry run mode."""
    return DRY_RUN_N_FINAL_FIT if DRY_RUN_ENABLED else HMM_N_FINAL_FIT


def get_knee_locator_s():
    """Get knee locator sensitivity based on dry run mode."""
    return DRY_RUN_KNEE_LOCATOR_S if DRY_RUN_ENABLED else KNEE_LOCATOR_S


def get_dry_run_n_neurons():
    """Get number of neurons to sample in dry run mode."""
    return DRY_RUN_N_NEURONS if DRY_RUN_ENABLED else None

# Multiprocessing
HMM_N_PROCESSES = 10  # number of processes for parallel cross-validation

# Data splitting for cross-validation
N_PARTS = 10  # number of parts to split data into for cross-validation

# ============================================================================
# HMM Final Fit Parameters
# ============================================================================

# State analysis parameters
MIN_STATE_CUMULATIVE_DURATION = 30.0  # seconds - minimum cumulative duration for state analysis
N_COMPONENTS_MAX = 30  # maximum number of components for dimensionality analysis

# ============================================================================
# Waveform Classification Parameters
# ============================================================================

WAVEFORM_N_COMPONENTS = 3  # number of Gaussian mixture components
WAVEFORM_RANDOM_STATE = 0
WAVEFORM_COVARIANCE_TYPE = 'full'
WAVEFORM_MIN_PROBABILITY = 0.95  # minimum probability for classification

# ============================================================================
# Reproducibility Parameters
# ============================================================================

# Global random seed for reproducibility
RANDOM_SEED = 42  # Main random seed for all random operations
AUTOGRAD_SEED = 0  # Seed for autograd.numpy.random (used in HMM)
KFOLD_RANDOM_STATE = 42  # Random state for sklearn KFold cross-validation
NEURON_SAMPLING_SEED = 42  # Seed for random neuron sampling in dry run mode

# ============================================================================
# Optogenetic Analysis Parameters
# ============================================================================

OPTO_PULSE_MIN_DURATION = 0.009  # seconds (9 ms)
OPTO_PULSE_MAX_DURATION = 0.02  # seconds (20 ms)
OPTO_TIME_RESOLUTION = 0.0005  # seconds (0.5 ms bins)
OPTO_BIN_EDGES_START = -0.01  # seconds relative to stimulus
OPTO_BIN_EDGES_END = 0.025  # seconds relative to stimulus
OPTO_BASELINE_START = -0.01  # seconds
OPTO_BASELINE_END = -0.002  # seconds
OPTO_EVOKED_START = 0.001  # seconds
OPTO_EVOKED_END = 0.009  # seconds
OPTO_BASELINE_DURATION = 0.008  # seconds
OPTO_RESPONSE_THRESHOLD = 2.0  # fold-change threshold for optogenetic response

# ============================================================================
# Plotting and Visualization Parameters
# ============================================================================

# Color settings
COLOR_NAMES = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
]

# Cool colors for state visualization
COOL_COLORS = np.array([
    [1,1,1,1],
    [141,211,199,255],
    [255,255,179,255],
    [190,186,218,255],
    [251,128,114,255],
    [128,177,211,255],
    [253,180,98,255],
    [179,222,105,255],
    [252,205,229,255],
    [217,217,217,255],
    [188,128,189,255],
    [204,235,197,255],
    [255,237,111,255]
]) / 255

# Event colors
EVENTS_COLORS = np.array([
    [0, 128, 0],
    [255, 211, 0],
    [0, 158, 255],
    [255, 0, 0],
    [255, 128, 0],
    [42, 82, 255]
]) / 256

# Colormap settings
N_COLORS = 256
COLORMAPS_NAMES = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 'gist_rainbow']

# Figure sizes
FIG_SIZE_LARGE = (12, 8)
FIG_SIZE_MEDIUM = (10, 6)

# ============================================================================
# File Paths and Directories
# ============================================================================

# Data directories (will be set dynamically based on environment)
DATA_DIR_ABS = Path('/data')
DATA_DIR_REL = Path(__file__).resolve().parents[1] / 'data'
RESULTS_DIR_ABS = Path('/results')
RESULTS_DIR_REL = Path(__file__).resolve().parents[1] / 'results'

# Subdirectories
NWB_FOLDER_NAME = 'sessions_nwb'
PREPROCESSED_FOLDER_NAME = 'sessions_preprocessed'
HMM_CROSSVAL_FOLDER_NAME = 'sessions_hmm_crossval'
HMM_FIT_FOLDER_NAME = 'sessions_hmm_fit'
LOGS_FOLDER_NAME = 'logs'
RESULTS_FOLDER_NAME = 'results'

# File names
SESSIONS_CSV_NAME = 'sessions.csv'
UNIT_TABLE_CSV_NAME = 'unit_table_all.csv'

# Output file patterns
PREPROCESSED_FILE_PATTERN = 'df_{session_id}.pkl'
HMM_CROSSVAL_FILE_PATTERN = 'hmm_{session_id}.pkl'
HMM_FIT_FILE_PATTERN = 'hmm_analysis.pkl'

# ============================================================================
# Helper Functions for Paths
# ============================================================================

def get_data_dir():
    """Get data directory, preferring relative path if it exists."""
    if DATA_DIR_REL.exists() and DATA_DIR_REL.is_dir():
        return DATA_DIR_REL
    elif DATA_DIR_ABS.exists() and DATA_DIR_ABS.is_dir():
        return DATA_DIR_ABS
    else:
        return DATA_DIR_REL


def get_results_dir():
    """Get results directory, preferring relative path if it exists."""
    if RESULTS_DIR_REL.exists() and RESULTS_DIR_REL.is_dir():
        return RESULTS_DIR_REL
    elif RESULTS_DIR_ABS.exists() and RESULTS_DIR_ABS.is_dir():
        return RESULTS_DIR_ABS
    else:
        return RESULTS_DIR_REL


def get_nwb_folder():
    """Get NWB folder path."""
    return get_data_dir() / NWB_FOLDER_NAME


def get_preprocessed_folder():
    """Get preprocessed data folder path."""
    return get_results_dir() / PREPROCESSED_FOLDER_NAME


def get_hmm_crossval_folder():
    """Get HMM cross-validation results folder path."""
    return get_results_dir() / HMM_CROSSVAL_FOLDER_NAME


def get_hmm_fit_folder():
    """Get HMM fit results folder path."""
    return get_results_dir() / HMM_FIT_FOLDER_NAME


def get_logs_folder():
    """Get logs folder path."""
    return get_results_dir() / LOGS_FOLDER_NAME


def get_results_folder():
    """Get results folder path."""
    return get_results_dir() / RESULTS_FOLDER_NAME


def get_sessions_csv():
    """Get sessions CSV file path."""
    return get_data_dir() / SESSIONS_CSV_NAME


def get_unit_table_csv():
    """Get unit table CSV file path, searching recursively."""
    data_dir = get_data_dir()
    if data_dir.exists():
        candidates = list(data_dir.rglob(UNIT_TABLE_CSV_NAME))
        if candidates:
            return candidates[0]
    return data_dir / UNIT_TABLE_CSV_NAME

