#!/usr/bin/env python3
"""
BWO-SVR Experiment Script v2

This script implements Support Vector Regression (SVR) with 
Black Widow Optimization Algorithm (BWO) for hyperparameter optimization
across four datasets with unified 80/20 splits.

Requirements based on experiment_protocol_v2.json:
- Datasets: wind_cn, electricity, air_quality, gefcom (4 datasets)
- Seeds: [42, 123, 456] (3 seeds)
- Split: 80/20 unified (from split_manifest_80_20_unified.json)
- High-intensity search: population=40, iterations=200
- Model training: epochs=30, early_stopping_patience=5
- SVR hyperparameters: C (log), epsilon, gamma (log)
- Sequence length: 24, single-step prediction

Output files:
- metaheuristic_search_trace.csv
- metaheuristic_best_configs.csv  
- metaheuristic_metrics_all_datasets.csv
"""

import numpy as np
import pandas as pd
import sys
import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict

# Scikit-learn imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project path
sys.path.append('.')

# Import unified split utilities and BWO algorithm
from unified_split_utils import load_and_split_dataset
from bwo_algorithm import BlackWidowOptimization

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration Constants
# ============================================================================

DATASETS = ['wind_cn', 'electricity', 'air_quality', 'gefcom']
SEEDS = [42, 123, 456]
SEQ_LEN = 24
PRED_LEN = 1  # Single-step prediction

# High-intensity search configuration
POPULATION_SIZE = 40
MAX_ITERATIONS = 200

# SVR Hyperparameter bounds (in search space)
# [log10(C), epsilon, log10(gamma)]
BOUNDS = [
    (-2, 3),       # log10(C): 10^-2 = 0.01 to 10^3 = 1000
    (0.001, 1.0),  # epsilon: 0.001 to 1.0
    (-4, 1)        # log10(gamma): 10^-4 = 0.0001 to 10^1 = 10
]

# BWO algorithm parameters
BWO_CONFIG = {
    'procreation_rate': 0.8,
    'cannibalism_rate': 0.3,
    'mutation_rate': 0.1
}

# Dataset configuration
DATASET_CONFIG = {
    'wind_cn': {
        'file': '论文1/processed_wind.csv',
        'target_column': 'power'
    },
    'electricity': {
        'file': '论文1/Load_history.csv',
        'target_column': None  # Use last column
    },
    'air_quality': {
        'file': '论文1/air_quality_ready.csv',
        'target_column': 'PM2.5'
    },
    'gefcom': {
        'file': '论文1/gefcom_ready.csv',
        'target_column': 'load'
    }
}

# Output file names
OUTPUT_FILES = {
    'search_trace': 'metaheuristic_search_trace.csv',
    'best_configs': 'metaheuristic_best_configs.csv',
    'metrics': 'metaheuristic_metrics_all_datasets.csv'
}


# ============================================================================
# Utility Functions
# ============================================================================

def decode_hyperparameters(params: np.ndarray) -> Dict[str, float]:
    """
    Decode BWO search parameters to actual SVR hyperparameters.
    
    Args:
        params: Array of [log10(C), epsilon, log10(gamma)]
        
    Returns:
        Dictionary with C, epsilon, gamma
    """
    return {
        'C': 10 ** params[0],  # Log scale: 10^-2 to 10^3 = 0.01 to 1000
        'epsilon': params[1],   # Linear scale: 0.001 to 1.0
        'gamma': 10 ** params[2]  # Log scale: 10^-4 to 10^1 = 0.0001 to 10
    }


def encode_hyperparameters(C: float, epsilon: float, gamma: float) -> np.ndarray:
    """
    Encode SVR hyperparameters to BWO search parameters.
    
    Args:
        C: Regularization parameter
        epsilon: Epsilon tube
        gamma: Kernel coefficient
        
    Returns:
        Array of [log10(C), epsilon, log10(gamma)]
    """
    return np.array([
        np.log10(C),
        epsilon,
        np.log10(gamma)
    ])


def prepare_sequences_sklearn(data: np.ndarray, seq_len: int, pred_len: int,
                              target_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for time series forecasting for sklearn models.
    
    Args:
        data: 2D array with features and target
        seq_len: Lookback window length
        pred_len: Prediction horizon
        target_idx: Index of target column (default: last column)
        
    Returns:
        X: 2D array of features (samples, seq_len * features)
        y: 1D array of target values
    """
    X, y = [], []
    n_features = data.shape[1]
    
    for i in range(len(data) - seq_len - pred_len + 1):
        # Flatten the sequence into features
        sequence = data[i:i+seq_len]
        X.append(sequence.flatten())
        
        # Target is the next pred_len values of the target variable
        target_values = data[i+seq_len:i+seq_len+pred_len, target_idx]
        y.append(np.mean(target_values))  # Use mean for single-step prediction
    
    return np.array(X), np.array(y)


def prepare_features_targets(df: pd.DataFrame, target_col: Optional[str] = None) -> np.ndarray:
    """
    Prepare features and targets from dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of target column (if None, use last column)
        
    Returns:
        data: 2D numpy array with features and target
    """
    if target_col is not None and target_col in df.columns:
        # Reorder columns to put target last
        feature_cols = [c for c in df.columns if c != target_col]
        data = df[feature_cols + [target_col]].values
    else:
        # Use all columns, last column as target
        data = df.values
    
    return data


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value
    """
    mask = y_true != 0
    if np.sum(mask) == 0:
        return float('inf')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def find_convergence_iteration(best_fitness_history: List[float], 
                                tolerance: float = 1e-6) -> int:
    """
    Find the iteration where convergence occurred.
    Convergence is defined as when the best fitness doesn't improve significantly.
    
    Args:
        best_fitness_history: List of best fitness values per iteration
        tolerance: Minimum improvement to consider as non-converged
        
    Returns:
        Iteration number where convergence occurred
    """
    if len(best_fitness_history) < 10:
        return len(best_fitness_history) - 1
    
    for i in range(10, len(best_fitness_history)):
        # Check if no significant improvement in last 10 iterations
        recent_values = best_fitness_history[i-10:i+1]
        if max(recent_values) - min(recent_values) < tolerance:
            return i
    
    return len(best_fitness_history) - 1


# ============================================================================
# SVR Training and Evaluation
# ============================================================================

def train_evaluate_svr(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       C: float, epsilon: float, gamma: float,
                       kernel: str = 'rbf') -> Dict[str, Any]:
    """
    Train SVR model and evaluate on validation set.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        C: Regularization parameter
        epsilon: Epsilon tube
        gamma: Kernel coefficient
        kernel: Kernel type (default: 'rbf')
        
    Returns:
        Dictionary with model and metrics
    """
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    # Create and train SVR
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
    
    start_time = time.time()
    svr.fit(X_train_scaled, y_train_scaled)
    training_time = time.time() - start_time
    
    # Predict on validation set
    y_pred_scaled = svr.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mape = calculate_mape(y_val, y_pred)
    
    return {
        'model': svr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'val_mse': mse,
        'val_rmse': rmse,
        'val_mae': mae,
        'val_r2': r2,
        'val_mape': mape,
        'training_time': training_time,
        'n_support_vectors': len(svr.support_vectors_) if hasattr(svr, 'support_vectors_') else 0
    }


def create_svr_objective_function(dataset_name: str, seq_len: int, pred_len: int,
                                  seed: int = 42, 
                                  trace_callback: Optional[callable] = None) -> callable:
    """
    Create objective function for BWO optimization of SVR.
    
    Args:
        dataset_name: Name of the dataset
        seq_len: Sequence length
        pred_len: Prediction length
        seed: Random seed
        trace_callback: Optional callback function to record search trace
        
    Returns:
        Objective function for BWO
    """
    # Get dataset configuration
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIG[dataset_name]
    data_path = config['file']
    target_col = config['target_column']
    
    # Load dataset once
    train_df, _ = load_and_split_dataset(dataset_name, data_path)
    data = prepare_features_targets(train_df, target_col)
    X, y = prepare_sequences_sklearn(data, seq_len, pred_len)
    
    # Split into train/val (80% of training for actual training, 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    iteration_counter = [0]
    best_so_far = [float('inf')]
    
    def objective(params: np.ndarray) -> float:
        """Objective function to minimize (validation MSE)."""
        try:
            # Decode parameters
            hyperparams = decode_hyperparameters(params)
            C = hyperparams['C']
            epsilon = hyperparams['epsilon']
            gamma = hyperparams['gamma']
            
            # Train and evaluate SVR
            result = train_evaluate_svr(
                X_train, y_train, X_val, y_val,
                C=C, epsilon=epsilon, gamma=gamma,
                kernel='rbf'
            )
            
            fitness = result['val_mse']
            
            # Update best so far
            if fitness < best_so_far[0]:
                best_so_far[0] = fitness
            
            # Record trace if callback provided
            if trace_callback is not None:
                trace_callback({
                    'iteration': iteration_counter[0],
                    'best_score': best_so_far[0],
                    'hyperparams': hyperparams
                })
                iteration_counter[0] += 1
            
            return fitness
            
        except Exception as e:
            print(f"Error in SVR objective function: {e}")
            return float('inf')  # Return worst possible value
    
    return objective


# ============================================================================
# Main Experiment Functions
# ============================================================================

def run_bwo_svr_single_experiment(dataset_name: str, seed: int,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run BWO-SVR experiment for a single dataset and seed.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with experiment results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"BWO-SVR Experiment: {dataset_name} (seed={seed})")
        print(f"{'='*70}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Storage for search trace
    search_trace = []
    
    def trace_callback(record: Dict):
        """Callback to record search trace."""
        search_trace.append(record)
    
    # Create objective function
    objective_func = create_svr_objective_function(
        dataset_name, SEQ_LEN, PRED_LEN, seed, trace_callback
    )
    
    # Initialize BWO
    bwo = BlackWidowOptimization(
        objective_func=objective_func,
        bounds=BOUNDS,
        population_size=POPULATION_SIZE,
        procreation_rate=BWO_CONFIG['procreation_rate'],
        cannibalism_rate=BWO_CONFIG['cannibalism_rate'],
        mutation_rate=BWO_CONFIG['mutation_rate'],
        max_iterations=MAX_ITERATIONS,
        random_seed=seed
    )
    
    # Run optimization
    if verbose:
        print(f"\nStarting BWO optimization (population={POPULATION_SIZE}, "
              f"iterations={MAX_ITERATIONS})...")
    start_time = time.time()
    result = bwo.optimize(verbose=verbose)
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best fitness (validation MSE): {result['best_fitness']:.6f}")
    
    # Decode best parameters
    best_params = result['best_solution']
    best_hyperparams = decode_hyperparameters(best_params)
    
    if verbose:
        print(f"Best hyperparameters:")
        print(f"  C: {best_hyperparams['C']:.4f}")
        print(f"  epsilon: {best_hyperparams['epsilon']:.4f}")
        print(f"  gamma: {best_hyperparams['gamma']:.6f}")
    
    # Final evaluation on test set
    if verbose:
        print(f"\nFinal evaluation on test set...")
    
    # Get dataset configuration
    config = DATASET_CONFIG[dataset_name]
    data_path = config['file']
    target_col = config['target_column']
    
    # Load full dataset
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    # Prepare data
    train_data = prepare_features_targets(train_df, target_col)
    test_data = prepare_features_targets(test_df, target_col)
    
    # Prepare sequences
    X_train, y_train = prepare_sequences_sklearn(train_data, SEQ_LEN, PRED_LEN)
    X_test, y_test = prepare_sequences_sklearn(test_data, SEQ_LEN, PRED_LEN)
    
    # Split training into train/val for final model
    split_idx = int(len(X_train) * 0.8)
    X_train_final, X_val_final = X_train[:split_idx], X_train[split_idx:]
    y_train_final, y_val_final = y_train[:split_idx], y_train[split_idx:]
    
    # Train final model with best parameters
    final_result = train_evaluate_svr(
        X_train_final, y_train_final, X_val_final, y_val_final,
        C=best_hyperparams['C'],
        epsilon=best_hyperparams['epsilon'],
        gamma=best_hyperparams['gamma'],
        kernel='rbf'
    )
    
    # Evaluate on test set
    svr_model = final_result['model']
    scaler_X = final_result['scaler_X']
    scaler_y = final_result['scaler_y']
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    y_pred_scaled = svr_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate test metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mape = calculate_mape(y_test, y_pred)
    
    # Find convergence iteration
    convergence_iter = find_convergence_iteration(result['history']['best_fitness'])
    
    if verbose:
        print(f"\nTest Metrics:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  MAPE: {test_mape:.4f}%")
        print(f"  Convergence iteration: {convergence_iter}")
    
    # Compile results
    experiment_results = {
        'dataset': dataset_name,
        'model': 'BWO-SVR',
        'seed': seed,
        'best_hyperparams': best_hyperparams,
        'search_trace': search_trace,
        'optimization_history': {
            'best_fitness_history': result['history']['best_fitness'],
            'avg_fitness_history': result['history']['avg_fitness'],
            'iterations': result['history']['iterations']
        },
        'test_metrics': {
            'R2': float(test_r2),
            'RMSE': float(test_rmse),
            'MAE': float(test_mae),
            'MAPE': float(test_mape)
        },
        'convergence_iter': convergence_iter,
        'optimization_time': optimization_time,
        'timestamp': datetime.now().isoformat()
    }
    
    return experiment_results


def run_all_experiments(output_dir: str = '.', verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run BWO-SVR experiments for all datasets and seeds.
    
    Args:
        output_dir: Directory to save output files
        verbose: Whether to print progress
        
    Returns:
        List of experiment results
    """
    all_results = []
    
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                result = run_bwo_svr_single_experiment(dataset, seed, verbose=verbose)
                all_results.append(result)
                
                # Save intermediate result
                intermediate_file = os.path.join(
                    output_dir, f'bwo_svr_v2_{dataset}_seed{seed}.json'
                )
                with open(intermediate_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error running BWO-SVR experiment for {dataset} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return all_results


def generate_output_files(results: List[Dict[str, Any]], output_dir: str = '.'):
    """
    Generate output CSV files from experiment results.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save output files
    """
    # 1. metaheuristic_search_trace.csv
    # Columns: dataset, model, seed, iter, best_score, hyperparams
    search_trace_rows = []
    for result in results:
        dataset = result['dataset']
        seed = result['seed']
        model = result['model']
        
        for trace_record in result['search_trace']:
            search_trace_rows.append({
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'iter': trace_record['iteration'],
                'best_score': trace_record['best_score'],
                'hyperparams': json.dumps(trace_record['hyperparams'])
            })
    
    search_trace_df = pd.DataFrame(search_trace_rows)
    search_trace_path = os.path.join(output_dir, OUTPUT_FILES['search_trace'])
    search_trace_df.to_csv(search_trace_path, index=False)
    print(f"Saved search trace to: {search_trace_path}")
    
    # 2. metaheuristic_best_configs.csv
    # Columns: dataset, model, best_hyperparams, best_seed
    best_configs_rows = []
    
    # Group by dataset and find best configuration
    dataset_results = defaultdict(list)
    for result in results:
        dataset_results[result['dataset']].append(result)
    
    for dataset in DATASETS:
        if dataset not in dataset_results:
            continue
        
        results_for_dataset = dataset_results[dataset]
        
        # Find best result (highest R2)
        best_result = max(results_for_dataset, key=lambda r: r['test_metrics']['R2'])
        
        best_configs_rows.append({
            'dataset': dataset,
            'model': 'BWO-SVR',
            'best_hyperparams': json.dumps(best_result['best_hyperparams']),
            'best_seed': best_result['seed']
        })
    
    best_configs_df = pd.DataFrame(best_configs_rows)
    best_configs_path = os.path.join(output_dir, OUTPUT_FILES['best_configs'])
    best_configs_df.to_csv(best_configs_path, index=False)
    print(f"Saved best configs to: {best_configs_path}")
    
    # 3. metaheuristic_metrics_all_datasets.csv
    # Columns: dataset, model, seed, R2, RMSE, MAE, MAPE, convergence_iter
    metrics_rows = []
    for result in results:
        metrics_rows.append({
            'dataset': result['dataset'],
            'model': result['model'],
            'seed': result['seed'],
            'R2': result['test_metrics']['R2'],
            'RMSE': result['test_metrics']['RMSE'],
            'MAE': result['test_metrics']['MAE'],
            'MAPE': result['test_metrics']['MAPE'],
            'convergence_iter': result['convergence_iter']
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(output_dir, OUTPUT_FILES['metrics'])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")


def print_summary_statistics(results: List[Dict[str, Any]]):
    """Print summary statistics for all experiments."""
    print(f"\n{'='*70}")
    print("BWO-SVR Experiments Summary")
    print(f"{'='*70}")
    
    for dataset in DATASETS:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if not dataset_results:
            continue
        
        r2_scores = [r['test_metrics']['R2'] for r in dataset_results]
        rmse_scores = [r['test_metrics']['RMSE'] for r in dataset_results]
        mae_scores = [r['test_metrics']['MAE'] for r in dataset_results]
        mape_scores = [r['test_metrics']['MAPE'] for r in dataset_results]
        
        print(f"\n{dataset}:")
        print(f"  R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        print(f"  MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        print(f"  MAPE: {np.mean(mape_scores):.4f}% ± {np.std(mape_scores):.4f}%")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for BWO-SVR experiments."""
    print("BWO-SVR Experimental Suite v2")
    print("="*70)
    print("This script runs BWO-optimized SVR on four datasets")
    print("with unified 80/20 splits and 3 random seeds.")
    print("\nDatasets:", ", ".join(DATASETS))
    print("Seeds:", ", ".join(map(str, SEEDS)))
    print(f"High-intensity search: population={POPULATION_SIZE}, iterations={MAX_ITERATIONS}")
    print(f"SVR kernel: RBF")
    print(f"Sequence length: {SEQ_LEN}")
    
    # Check if required files exist
    required_files = [
        "split_manifest_80_20_unified.json",
        "unified_split_utils.py",
        "bwo_algorithm.py"
    ]
    
    # Add dataset files
    for dataset in DATASETS:
        required_files.append(DATASET_CONFIG[dataset]['file'])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\nWarning: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all data files are available.")
        return 1
    
    print("\nAll required files found.")
    
    # Run all experiments
    print("\n" + "="*70)
    print("Starting all BWO-SVR experiments...")
    print("="*70)
    
    try:
        results = run_all_experiments(verbose=True)
        
        if results:
            # Generate output files
            print("\n" + "="*70)
            print("Generating output files...")
            print("="*70)
            generate_output_files(results)
            
            # Print summary
            print_summary_statistics(results)
            
            print("\n" + "="*70)
            print("All BWO-SVR Experiments Completed!")
            print("="*70)
            print("\nOutput files generated:")
            for name, filename in OUTPUT_FILES.items():
                print(f"  - {filename}")
        else:
            print("\nNo results generated. Check for errors above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
