"""
COA/BWO Intensive Search Configuration

This file defines the intensive search parameters for the COA/BWO experiments.
These parameters ensure high coverage of the search space with 3 seeds per dataset.

Configuration includes:
1. COA algorithm parameters for BiLSTM optimization
2. BWO algorithm parameters for SVR and CNN optimization  
3. Model-specific hyperparameter bounds
4. Dataset-specific configurations
5. Search intensity levels (normal vs intensive)
"""

import numpy as np
from typing import Dict, List, Tuple, Any


# ============================================================================
# 1. GENERAL EXPERIMENT CONFIGURATION
# ============================================================================

DATASETS = ["wind", "electricity", "air_quality", "gefcom"]
SEEDS = [42, 123, 456]  # 3 seeds for robust evaluation
SPLIT_RATIO = 0.8  # Unified 80/20 split

# Search intensity levels
INTENSIVE_SEARCH = {
    "population_size": {
        "coa": 35,   # COA population for BiLSTM
        "bwo_svr": 40,  # BWO population for SVR
        "bwo_cnn": 35   # BWO population for CNN
    },
    "max_iterations": {
        "coa": 60,   # COA iterations for BiLSTM
        "bwo_svr": 70,  # BWO iterations for SVR  
        "bwo_cnn": 60   # BWO iterations for CNN
    },
    "num_parallel_runs": 3  # Number of parallel BWO/COA runs
}

NORMAL_SEARCH = {
    "population_size": {
        "coa": 25,
        "bwo_svr": 30,
        "bwo_cnn": 25
    },
    "max_iterations": {
        "coa": 40,
        "bwo_svr": 50,
        "bwo_cnn": 40
    },
    "num_parallel_runs": 1
}

# ============================================================================
# 2. COA ALGORITHM CONFIGURATION (for BiLSTM)
# ============================================================================

COA_CONFIG = {
    # Basic COA parameters
    "nest_abandonment_prob": 0.25,  # Discovery rate of alien eggs
    "step_size_alpha": 0.01,        # Step size scaling factor
    "levy_beta": 1.5,               # Lévy flight parameter (1 < beta <= 3)
    
    # BiLSTM-specific parameter bounds
    "bilstm_bounds": [
        # [hidden_dim, num_layers, dropout, lr, weight_decay, batch_size]
        (32, 256),      # hidden_dim: 32 to 256
        (1, 4),         # num_layers: 1 to 4
        (0.1, 0.5),     # dropout: 0.1 to 0.5
        (1e-4, 1e-2),   # learning rate: 1e-4 to 1e-2
        (1e-6, 1e-3),   # weight decay: 1e-6 to 1e-3
        (16, 128)       # batch_size: 16 to 128
    ],
    
    # Fixed BiLSTM architecture parameters
    "bilstm_fixed": {
        "seq_len": 96,      # Lookback window
        "pred_len": 24,     # Prediction horizon
        "training_epochs_objective": 20,    # Epochs during optimization
        "training_epochs_final": 50,        # Epochs for final model
        "bidirectional": True
    },
    
    # Performance targets
    "performance_targets": {
        "target_r2": 0.85,      # Target R² score
        "target_rmse_reduction": 0.3,  # Target RMSE reduction vs baseline
        "max_training_time_per_run": 3600  # 1 hour max per optimization run
    }
}

# ============================================================================
# 3. BWO ALGORITHM CONFIGURATION (for SVR and CNN)
# ============================================================================

BWO_CONFIG = {
    # Basic BWO parameters
    "procreation_rate": 0.8,    # Rate of procreation
    "cannibalism_rate": 0.3,    # Rate of cannibalism (weak spiders removed)
    "mutation_rate": 0.1,       # Rate of mutation
    
    # SVR-specific parameter bounds (log scale)
    "svr_bounds": [
        # [log10(C), epsilon, log10(gamma)]
        (-2, 3),        # log10(C): 10^-2 to 10^3
        (0.01, 0.5),    # epsilon: 0.01 to 0.5
        (-4, 1)         # log10(gamma): 10^-4 to 10^1
    ],
    
    # SVR fixed parameters
    "svr_fixed": {
        "kernel": "rbf",
        "seq_len": 24,      # Smaller lookback for SVR
        "pred_len": 1,      # Single-step prediction for SVR
        "scale_features": True,
        "scale_target": True
    },
    
    # CNN-specific parameter bounds (normalized [0, 1] range)
    "cnn_bounds": [
        # [filter1_pow, filter2_pow, filter3_pow, kernel1_idx, kernel2_idx, kernel3_idx,
        #  dropout, log10_lr, log10_weight_decay, batch_size_pow, dense_units_pow]
        (0, 3),  # filter1: 16 to 128 (16 * 2^param)
        (0, 3),  # filter2
        (0, 3),  # filter3
        (0, 3),  # kernel1: 3, 5, 7, 9
        (0, 3),  # kernel2
        (0, 3),  # kernel3
        (0, 1),  # dropout: 0.1 to 0.5
        (0, 1),  # log10_lr: -4 to -2
        (0, 1),  # log10_weight_decay: -6 to -3
        (0, 3),  # batch_size_pow: 16 to 128
        (0, 2),  # dense_units_pow: 64 to 256
    ],
    
    # CNN fixed parameters
    "cnn_fixed": {
        "seq_len": 96,
        "pred_len": 24,
        "num_cnn_layers": 3,
        "use_batch_norm": True,
        "use_maxpool": True,
        "training_epochs_objective": 15,
        "training_epochs_final": 30
    }
}

# ============================================================================
# 4. DATASET-SPECIFIC CONFIGURATIONS
# ============================================================================

DATASET_CONFIG = {
    "wind": {
        "file": "论文1/processed_wind.csv",
        "target_col": "power",
        "expected_rows": 121528,
        "train_samples": 97222,  # 80% of 121528
        "test_samples": 24306,   # 20% of 121528
        "features": ["wind_speed", "wind_direction", "temperature", "humidity", 
                    "pressure", "hour_sin", "hour_cos", "day_sin", "day_cos"],
        "notes": "Wind power generation dataset with meteorological features"
    },
    "electricity": {
        "file": "论文1/Load_history.csv",
        "target_col": None,  # Use last column
        "expected_rows": 33000,
        "train_samples": 26400,
        "test_samples": 6600,
        "features": "all columns except target",
        "notes": "Electricity load history dataset"
    },
    "air_quality": {
        "file": "论文1/air_quality_ready.csv",
        "target_col": "PM2.5",
        "expected_rows": 588644,
        "train_samples": 470915,
        "test_samples": 117729,
        "features": ["PM10", "SO2", "NO2", "CO", "O3", "temperature", "pressure",
                    "humidity", "wind_speed", "wind_direction"],
        "notes": "Air quality monitoring dataset with PM2.5 as target"
    },
    "gefcom": {
        "file": "论文1/gefcom_ready.csv",
        "target_col": "load",
        "expected_rows": 38070,
        "train_samples": 30456,
        "test_samples": 7614,
        "features": ["temperature", "humidity", "pressure", "wind_speed"],
        "notes": "GEFCom2012 electricity load forecasting dataset"
    }
}

# ============================================================================
# 5. SEARCH SPACE COVERAGE METRICS
# ============================================================================

SEARCH_COVERAGE = {
    "coa_bilstm": {
        "total_combinations": "~1.2e6",  # Approximate search space size
        "sampling_rate_intensive": "0.002%",  # With 35 population * 60 iterations
        "sampling_rate_normal": "0.001%",
        "key_parameters": ["hidden_dim", "num_layers", "dropout", "lr", "batch_size"],
        "parameter_interactions": ["hidden_dim × num_layers", "dropout × lr"]
    },
    "bwo_svr": {
        "total_combinations": "~2.5e4",
        "sampling_rate_intensive": "0.12%",  # With 40 population * 70 iterations
        "sampling_rate_normal": "0.06%",
        "key_parameters": ["C", "epsilon", "gamma"],
        "parameter_interactions": ["C × gamma", "epsilon × kernel"]
    },
    "bwo_cnn": {
        "total_combinations": "~4.3e6",
        "sampling_rate_intensive": "0.0005%",  # With 35 population * 60 iterations
        "sampling_rate_normal": "0.0002%",
        "key_parameters": ["filters", "kernel_sizes", "dropout", "dense_units"],
        "parameter_interactions": ["filters × kernel_sizes", "dropout × batch_size"]
    }
}

# ============================================================================
# 6. CONVERGENCE AND STOPPING CRITERIA
# ============================================================================

CONVERGENCE_CONFIG = {
    "coa": {
        "stagnation_iterations": 15,  # Stop if no improvement for N iterations
        "fitness_tolerance": 1e-6,    # Minimum improvement to consider
        "early_stopping_patience": 20,
        "min_iterations": 30          # Minimum iterations before checking convergence
    },
    "bwo": {
        "stagnation_iterations": 12,
        "fitness_tolerance": 1e-6,
        "early_stopping_patience": 15,
        "min_iterations": 25
    },
    "global": {
        "max_wall_time_hours": 24,    # Maximum total experiment time
        "checkpoint_frequency": 5,    # Save checkpoint every N iterations
        "resume_from_checkpoint": True
    }
}

# ============================================================================
# 7. OUTPUT AND RESULT COLLECTION
# ============================================================================

OUTPUT_CONFIG = {
    "result_files": {
        "coa_bilstm": {
            "detailed": "coa_bilstm_all_results.json",
            "summary": "coa_bilstm_summary.csv",
            "search_trace": "coa_bilstm_search_trace.csv",
            "best_configs": "coa_bilstm_best_configs.csv"
        },
        "bwo_svr": {
            "detailed": "bwo_svr_all_results.json",
            "summary": "bwo_svr_summary.csv",
            "search_trace": "bwo_svr_search_trace.csv",
            "best_configs": "bwo_svr_best_configs.csv"
        },
        "bwo_cnn": {
            "detailed": "bwo_cnn_all_results.json",
            "summary": "bwo_cnn_summary.csv",
            "search_trace": "bwo_cnn_search_trace.csv",
            "best_configs": "bwo_cnn_best_configs.csv"
        },
        "combined": {
            "all_models_clean": "all_models_clean_80_20.csv",
            "coa_bwo_search_trace": "coa_bwo_search_trace.csv",
            "coa_bwo_best_configs": "coa_bwo_best_configs.csv",
            "robustness_results": "robustness_80_20.csv",
            "dispatch_results": ["dispatch_rolling_80_20.csv", "dispatch_rts24_80_20.csv"],
            "split_manifest": "split_manifest_80_20_unified.json"
        }
    },
    
    "metrics_to_collect": [
        "R2", "RMSE", "MAE", "MSE",  # Accuracy metrics
        "training_time", "inference_time",  # Efficiency metrics
        "model_size", "num_parameters",  # Complexity metrics
        "best_fitness", "convergence_iteration",  # Optimization metrics
        "hyperparameters"  # Best found hyperparameters
    ],
    
    "validation_metrics": {
        "min_r2": 0.0,        # Minimum acceptable R²
        "max_rmse_ratio": 2.0,  # Maximum RMSE ratio vs baseline
        "training_time_limit": 7200,  # 2 hours max training time
        "memory_limit_gb": 16  # 16GB memory limit
    }
}

# ============================================================================
# 8. HIGH-INTENSITY SEARCH STRATEGIES
# ============================================================================

INTENSIVE_STRATEGIES = {
    "adaptive_parameter_ranges": {
        "enabled": True,
        "adjustment_factor": 0.2,  # Adjust bounds by ±20% based on intermediate results
        "update_frequency": 10     # Update every 10 iterations
    },
    
    "multi_start_optimization": {
        "enabled": True,
        "num_starts": 3,           # Multiple starting points
        "diversity_factor": 0.5    # Ensure diverse starting populations
    },
    
    "parameter_sensitivity_analysis": {
        "enabled": True,
        "method": "morris",        # Morris method for elementary effects
        "samples_per_parameter": 10,
        "identify_interactions": True
    },
    
    "ensemble_selection": {
        "enabled": True,
        "top_k_solutions": 5,      # Keep top 5 solutions from each run
        "ensemble_method": "weighted_average",
        "weights_optimization": True
    }
}

# ============================================================================
# 9. UTILITY FUNCTIONS
# ============================================================================

def get_search_config(intensive: bool = True) -> Dict[str, Any]:
    """Get search configuration based on intensity level."""
    return INTENSIVE_SEARCH if intensive else NORMAL_SEARCH

def get_coa_parameters(intensive: bool = True) -> Dict[str, Any]:
    """Get COA parameters for BiLSTM optimization."""
    search_config = get_search_config(intensive)
    params = COA_CONFIG.copy()
    params.update({
        "population_size": search_config["population_size"]["coa"],
        "max_iterations": search_config["max_iterations"]["coa"]
    })
    return params

def get_bwo_svr_parameters(intensive: bool = True) -> Dict[str, Any]:
    """Get BWO parameters for SVR optimization."""
    search_config = get_search_config(intensive)
    params = BWO_CONFIG.copy()
    params.update({
        "population_size": search_config["population_size"]["bwo_svr"],
        "max_iterations": search_config["max_iterations"]["bwo_svr"]
    })
    return params

def get_bwo_cnn_parameters(intensive: bool = True) -> Dict[str, Any]:
    """Get BWO parameters for CNN optimization."""
    search_config = get_search_config(intensive)
    params = BWO_CONFIG.copy()
    params.update({
        "population_size": search_config["population_size"]["bwo_cnn"],
        "max_iterations": search_config["max_iterations"]["bwo_cnn"]
    })
    return params

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]

def estimate_search_space_coverage(model_type: str, intensive: bool = True) -> Dict[str, str]:
    """Estimate search space coverage for a given model type."""
    if model_type not in SEARCH_COVERAGE:
        raise ValueError(f"Unknown model type: {model_type}")
    
    coverage = SEARCH_COVERAGE[model_type].copy()
    search_config = get_search_config(intensive)
    
    if model_type == "coa_bilstm":
        samples = search_config["population_size"]["coa"] * search_config["max_iterations"]["coa"]
    elif model_type == "bwo_svr":
        samples = search_config["population_size"]["bwo_svr"] * search_config["max_iterations"]["bwo_svr"]
    elif model_type == "bwo_cnn":
        samples = search_config["population_size"]["bwo_cnn"] * search_config["max_iterations"]["bwo_cnn"]
    
    coverage["actual_samples"] = samples
    coverage["intensity_level"] = "intensive" if intensive else "normal"
    
    return coverage


# ============================================================================
# 10. MAIN CONFIGURATION EXPORT
# ============================================================================

COA_BWO_INTENSIVE_CONFIG = {
    "datasets": DATASETS,
    "seeds": SEEDS,
    "split_ratio": SPLIT_RATIO,
    "intensive_search": INTENSIVE_SEARCH,
    "normal_search": NORMAL_SEARCH,
    "coa_config": COA_CONFIG,
    "bwo_config": BWO_CONFIG,
    "dataset_config": DATASET_CONFIG,
    "search_coverage": SEARCH_COVERAGE,
    "convergence_config": CONVERGENCE_CONFIG,
    "output_config": OUTPUT_CONFIG,
    "intensive_strategies": INTENSIVE_STRATEGIES
}

if __name__ == "__main__":
    print("COA/BWO Intensive Search Configuration")
    print("=" * 60)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Split ratio: {SPLIT_RATIO}")
    print("\nSearch Intensity Levels:")
    print(f"  Intensive: {INTENSIVE_SEARCH['population_size']} population, {INTENSIVE_SEARCH['max_iterations']} iterations")
    print(f"  Normal: {NORMAL_SEARCH['population_size']} population, {NORMAL_SEARCH['max_iterations']} iterations")
    
    print("\nEstimated Search Space Coverage (Intensive Mode):")
    for model_type in ["coa_bilstm", "bwo_svr", "bwo_cnn"]:
        coverage = estimate_search_space_coverage(model_type, intensive=True)
        print(f"  {model_type}: {coverage['sampling_rate_intensive']} of {coverage['total_combinations']} combinations")
    
    print("\nOutput files will be saved to:")
    for model, files in OUTPUT_CONFIG["result_files"]["combined"].items():
        if isinstance(files, list):
            print(f"  {model}: {', '.join(files)}")
        else:
            print(f"  {model}: {files}")
    
    print("\nConfiguration loaded successfully!")