#!/usr/bin/env python3
"""
BWO-CNN Experiment Script V2

This script implements Convolutional Neural Network (CNN) with 
Black Widow Optimization Algorithm (BWO) for hyperparameter optimization
across four datasets with unified 80/20 splits.

Requirements based on experiment_protocol_v2.json:
- Datasets: wind_cn, electricity, air_quality, gefcom (4 datasets)
- Seeds: [42, 123, 456] (3 seeds)
- Split: 80/20 unified (load from split_manifest_80_20_unified.json)
- High-intensity search: population=40, iterations=200
- Model training during search: epochs=30, early_stopping_patience=5

Output Files:
1. metaheuristic_search_trace.csv - Search trace with columns: dataset, model, seed, iter, best_score, hyperparams
2. metaheuristic_best_configs.csv - Best configs with columns: dataset, model, best_hyperparams, best_seed
3. metaheuristic_metrics_all_datasets.csv - Metrics with columns: dataset, model, seed, R2, RMSE, MAE, MAPE, convergence_iter

CNN Hyperparameter Bounds:
- num_filters_1: 16 to 128
- num_filters_2: 16 to 128
- kernel_size: 3 to 7
- dropout: 0.1 to 0.5
- learning_rate: 1e-4 to 1e-2

Architecture: Conv1D -> MaxPool -> Conv1D -> MaxPool -> Flatten -> Dense -> Output
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import existing modules
from bwo_algorithm import BlackWidowOptimization
from unified_split_utils import load_and_split_dataset

# Configuration
DATASETS = ["wind_cn", "electricity", "air_quality", "gefcom"]
SEEDS = [42, 123, 456]
MODEL_NAME = "BWO-CNN"

# Data paths mapping
DATA_PATHS = {
    "wind_cn": "论文1/processed_wind.csv",
    "electricity": "论文1/Load_history.csv",
    "air_quality": "论文1/air_quality_ready.csv",
    "gefcom": "论文1/gefcom_ready.csv"
}

# Target column mapping
TARGET_COLUMNS = {
    "wind_cn": "power",
    "electricity": "load",
    "air_quality": "PM2.5",
    "gefcom": "load"
}

# Hyperparameter bounds for BWO (continuous values, will be discretized as needed)
# Order: [num_filters_1, num_filters_2, kernel_size, dropout, learning_rate]
HYPERPARAM_BOUNDS = [
    (16, 128),    # num_filters_1
    (16, 128),    # num_filters_2
    (3, 7),       # kernel_size
    (0.1, 0.5),   # dropout
    (1e-4, 1e-2)  # learning_rate
]

# Fixed parameters
SEQ_LEN = 96
PRED_LEN = 24
POPULATION_SIZE = 40
MAX_ITERATIONS = 200
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32  # Fixed batch size for consistency


class BWOCNNCallback:
    """Custom callback to track BWO search progress."""
    
    def __init__(self, dataset: str, seed: int, trace_file: str):
        self.dataset = dataset
        self.seed = seed
        self.trace_file = trace_file
        self.iteration_scores = []
        self.best_score_overall = float('inf')
        self.convergence_iter = 0
        
    def record_iteration(self, iteration: int, best_score: float, hyperparams: Dict):
        """Record search progress for each iteration."""
        self.iteration_scores.append({
            'dataset': self.dataset,
            'model': MODEL_NAME,
            'seed': self.seed,
            'iter': iteration,
            'best_score': best_score,
            'hyperparams': json.dumps(hyperparams)
        })
        
        if best_score < self.best_score_overall:
            self.best_score_overall = best_score
            self.convergence_iter = iteration
            
    def save_trace(self):
        """Append trace to CSV file."""
        if not self.iteration_scores:
            return
            
        df = pd.DataFrame(self.iteration_scores)
        
        # Check if file exists to determine if we need header
        file_exists = os.path.exists(self.trace_file)
        
        # Append to file
        df.to_csv(self.trace_file, mode='a', header=not file_exists, index=False)
        
    def get_convergence_iter(self) -> int:
        """Return the iteration where best convergence was achieved."""
        return self.convergence_iter


def preprocess_electricity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess electricity data from wide format to long format.
    The Load_history.csv has columns: zone_id, year, month, day, h1, h2, ..., h24
    """
    # Melt the hourly columns
    id_vars = ['zone_id', 'year', 'month', 'day']
    hour_cols = [f'h{i}' for i in range(1, 25)]
    
    # Check if the data is already in the expected format
    if 'h1' not in df.columns:
        # Data might already be processed
        return df
    
    # Melt to long format
    melted = df.melt(
        id_vars=id_vars,
        value_vars=hour_cols,
        var_name='hour',
        value_name='load'
    )
    
    # Extract hour number
    melted['hour'] = melted['hour'].str.extract(r'(\d+)').astype(int)
    
    # Clean load values (remove commas if present)
    melted['load'] = melted['load'].astype(str).str.replace(',', '').astype(float)
    
    # Sort by time
    melted = melted.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
    
    # Keep only the load column for time series
    return melted[['load']]


def load_and_prepare_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load dataset and prepare train/test data.
    
    Returns:
        X_train, y_train, X_test, y_test, num_features
    """
    data_path = DATA_PATHS[dataset_name]
    target_col = TARGET_COLUMNS[dataset_name]
    
    # Load data using unified split
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    # Special handling for electricity data
    if dataset_name == "electricity":
        train_df = preprocess_electricity_data(train_df)
        test_df = preprocess_electricity_data(test_df)
    
    # Ensure target column is in the dataframe
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {dataset_name} dataset")
    
    # Reorder columns to put target last (for prepare_sequences)
    other_cols = [c for c in train_df.columns if c != target_col]
    feature_cols = other_cols + [target_col]
    
    train_df = train_df[feature_cols]
    test_df = test_df[feature_cols]
    
    # Get numeric data
    train_data = train_df.select_dtypes(include=[np.number]).values
    test_data = test_df.select_dtypes(include=[np.number]).values
    
    num_features = train_data.shape[1]
    
    # Prepare sequences
    X_train, y_train = prepare_sequences(train_data, SEQ_LEN, PRED_LEN)
    X_test, y_test = prepare_sequences(test_data, SEQ_LEN, PRED_LEN)
    
    return X_train, y_train, X_test, y_test, num_features


def prepare_sequences(data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for time series forecasting.
    
    Args:
        data: Input data array
        seq_len: Sequence length (lookback window)
        pred_len: Prediction length (forecast horizon)
        
    Returns:
        X: Input sequences (samples, seq_len, features)
        y: Target sequences (samples, pred_len)
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len, -1])  # Last column is target
    return np.array(X), np.array(y)


def decode_hyperparameters(params: np.ndarray) -> Dict[str, Any]:
    """
    Decode BWO parameters to CNN hyperparameters.
    
    Args:
        params: Array of parameters from BWO [num_filters_1, num_filters_2, kernel_size, dropout, learning_rate]
        
    Returns:
        Dictionary of decoded hyperparameters
    """
    # Ensure params are within bounds
    num_filters_1 = int(np.clip(params[0], HYPERPARAM_BOUNDS[0][0], HYPERPARAM_BOUNDS[0][1]))
    num_filters_2 = int(np.clip(params[1], HYPERPARAM_BOUNDS[1][0], HYPERPARAM_BOUNDS[1][1]))
    kernel_size = int(np.clip(params[2], HYPERPARAM_BOUNDS[2][0], HYPERPARAM_BOUNDS[2][1]))
    dropout = float(np.clip(params[3], HYPERPARAM_BOUNDS[3][0], HYPERPARAM_BOUNDS[3][1]))
    learning_rate = float(np.clip(params[4], HYPERPARAM_BOUNDS[4][0], HYPERPARAM_BOUNDS[4][1]))
    
    # Ensure kernel_size is odd for 'same' padding compatibility
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return {
        'num_filters_1': num_filters_1,
        'num_filters_2': num_filters_2,
        'kernel_size': kernel_size,
        'dropout': dropout,
        'learning_rate': learning_rate
    }


def build_cnn_model(input_shape: Tuple[int, int], pred_len: int, 
                    hyperparams: Dict[str, Any]) -> tf.keras.Model:
    """
    Build CNN model with specified hyperparameters.
    
    Architecture: Conv1D -> MaxPool -> Conv1D -> MaxPool -> Flatten -> Dense -> Output
    
    Args:
        input_shape: Shape of input (seq_len, num_features)
        pred_len: Prediction length
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Compiled Keras model
    """
    num_filters_1 = hyperparams['num_filters_1']
    num_filters_2 = hyperparams['num_filters_2']
    kernel_size = hyperparams['kernel_size']
    dropout = hyperparams['dropout']
    learning_rate = hyperparams['learning_rate']
    
    model = tf.keras.Sequential([
        # First Conv1D layer
        tf.keras.layers.Conv1D(
            filters=num_filters_1,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            input_shape=input_shape
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(dropout),
        
        # Second Conv1D layer
        tf.keras.layers.Conv1D(
            filters=num_filters_2,
            kernel_size=kernel_size,
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(dropout),
        
        # Flatten and Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(pred_len)
    ])
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with R2, RMSE, MAE, MAPE
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # R²
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # RMSE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    # MAPE
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))) * 100
    
    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape)
    }


def create_objective_function(dataset_name: str, seed: int, 
                               callback: BWOCNNCallback) -> callable:
    """
    Create the objective function for BWO optimization.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        callback: Callback object to record search trace
        
    Returns:
        Objective function that takes BWO params and returns validation loss
    """
    # Load data once and cache
    try:
        X_train_full, y_train_full, X_test, y_test, num_features = load_and_prepare_data(dataset_name)
    except Exception as e:
        print(f"Error loading data for {dataset_name}: {e}")
        raise
    
    # Split training data into train/validation (80%/20%)
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    print(f"  Data loaded: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    def objective(params: np.ndarray) -> float:
        """Objective function to minimize (validation loss)."""
        try:
            # Decode hyperparameters
            hyperparams = decode_hyperparameters(params)
            
            # Set random seeds for reproducibility
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Build model
            model = build_cnn_model(
                input_shape=(SEQ_LEN, num_features),
                pred_len=PRED_LEN,
                hyperparams=hyperparams
            )
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return validation loss (objective to minimize)
            val_loss = min(history.history['val_loss'])
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return val_loss
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return float('inf')
    
    return objective


def run_bwo_cnn_experiment(dataset_name: str, seed: int, 
                            trace_file: str = "metaheuristic_search_trace.csv") -> Dict[str, Any]:
    """
    Run BWO-CNN experiment for a single dataset and seed.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        trace_file: Path to search trace CSV file
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"BWO-CNN Experiment: {dataset_name} (seed={seed})")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create callback for trace recording
    callback = BWOCNNCallback(dataset_name, seed, trace_file)
    
    # Create objective function
    objective_func = create_objective_function(dataset_name, seed, callback)
    
    # Initialize BWO
    print(f"\nInitializing BWO with population={POPULATION_SIZE}, iterations={MAX_ITERATIONS}")
    bwo = BlackWidowOptimization(
        objective_func=objective_func,
        bounds=HYPERPARAM_BOUNDS,
        population_size=POPULATION_SIZE,
        procreation_rate=0.8,
        cannibalism_rate=0.3,
        mutation_rate=0.1,
        max_iterations=MAX_ITERATIONS,
        random_seed=seed
    )
    
    # Custom optimization loop to record trace every iteration
    print(f"\nStarting optimization...")
    best_overall_fitness = float('inf')
    best_hyperparams = None
    
    for iteration in range(MAX_ITERATIONS):
        # Run one iteration
        # Procreation phase
        offspring = bwo._procreate()
        
        # Mutation phase
        offspring = bwo._mutate(offspring)
        
        # Evaluate offspring
        offspring_fitness = np.array([bwo.objective_func(spider) for spider in offspring])
        
        # Combine parents and offspring
        combined_spiders = np.vstack([bwo.spiders, offspring])
        combined_fitness = np.concatenate([bwo.fitness, offspring_fitness])
        
        # Cannibalism phase
        bwo.spiders, bwo.fitness = bwo._cannibalize(combined_spiders, combined_fitness)
        
        # Sort population
        bwo._sort_population()
        
        # Update global best
        if bwo.fitness[0] < bwo.best_fitness:
            bwo.best_fitness = bwo.fitness[0]
            bwo.best_spider = bwo.spiders[0].copy()
        
        # Decode best hyperparameters for this iteration
        current_best_params = decode_hyperparameters(bwo.best_spider)
        
        # Record iteration
        callback.record_iteration(iteration, bwo.best_fitness, current_best_params)
        
        # Print progress every 20 iterations
        if iteration % 20 == 0 or iteration == MAX_ITERATIONS - 1:
            print(f"  Iteration {iteration:3d}: Best Fitness = {bwo.best_fitness:.6f}")
    
    optimization_time = time.time() - start_time
    
    # Save search trace
    callback.save_trace()
    
    # Get best hyperparameters
    best_hyperparams = decode_hyperparameters(bwo.best_spider)
    best_val_loss = bwo.best_fitness
    convergence_iter = callback.get_convergence_iter()
    
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best hyperparameters:")
    print(f"  num_filters_1: {best_hyperparams['num_filters_1']}")
    print(f"  num_filters_2: {best_hyperparams['num_filters_2']}")
    print(f"  kernel_size: {best_hyperparams['kernel_size']}")
    print(f"  dropout: {best_hyperparams['dropout']:.4f}")
    print(f"  learning_rate: {best_hyperparams['learning_rate']:.6f}")
    
    # Final evaluation with best hyperparameters
    print(f"\nFinal evaluation on test set...")
    
    # Reload data for final evaluation
    X_train_full, y_train_full, X_test, y_test, num_features = load_and_prepare_data(dataset_name)
    
    # Build and train final model
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    final_model = build_cnn_model(
        input_shape=(SEQ_LEN, num_features),
        pred_len=PRED_LEN,
        hyperparams=best_hyperparams
    )
    
    # Split training data for final model
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    # Early stopping for final model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train final model
    final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test, verbose=0)
    
    # Calculate metrics
    test_metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\nTest Metrics:")
    print(f"  R²:   {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE:  {test_metrics['MAE']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.4f}%")
    
    # Clean up
    del final_model
    tf.keras.backend.clear_session()
    
    return {
        'dataset': dataset_name,
        'model': MODEL_NAME,
        'seed': seed,
        'best_hyperparams': best_hyperparams,
        'best_val_loss': best_val_loss,
        'convergence_iter': convergence_iter,
        'optimization_time': optimization_time,
        'metrics': test_metrics
    }


def save_best_config(result: Dict[str, Any], config_file: str = "metaheuristic_best_configs.csv"):
    """
    Save best configuration to CSV file.
    
    Args:
        result: Experiment result dictionary
        config_file: Path to config CSV file
    """
    config_data = {
        'dataset': result['dataset'],
        'model': result['model'],
        'best_hyperparams': json.dumps(result['best_hyperparams']),
        'best_seed': result['seed']
    }
    
    df = pd.DataFrame([config_data])
    
    # Check if file exists
    file_exists = os.path.exists(config_file)
    
    # Append to file
    df.to_csv(config_file, mode='a', header=not file_exists, index=False)


def save_metrics(result: Dict[str, Any], metrics_file: str = "metaheuristic_metrics_all_datasets.csv"):
    """
    Save metrics to CSV file.
    
    Args:
        result: Experiment result dictionary
        metrics_file: Path to metrics CSV file
    """
    metrics_data = {
        'dataset': result['dataset'],
        'model': result['model'],
        'seed': result['seed'],
        'R2': result['metrics']['R2'],
        'RMSE': result['metrics']['RMSE'],
        'MAE': result['metrics']['MAE'],
        'MAPE': result['metrics']['MAPE'],
        'convergence_iter': result['convergence_iter']
    }
    
    df = pd.DataFrame([metrics_data])
    
    # Check if file exists
    file_exists = os.path.exists(metrics_file)
    
    # Append to file
    df.to_csv(metrics_file, mode='a', header=not file_exists, index=False)


def run_all_experiments():
    """Run BWO-CNN experiments for all datasets and seeds."""
    print("\n" + "="*70)
    print("BWO-CNN Experiment Suite V2")
    print("="*70)
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    print(f"Population: {POPULATION_SIZE}")
    print(f"Iterations: {MAX_ITERATIONS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Prediction Length: {PRED_LEN}")
    print("="*70)
    
    # Output files
    trace_file = "metaheuristic_search_trace.csv"
    config_file = "metaheuristic_best_configs.csv"
    metrics_file = "metaheuristic_metrics_all_datasets.csv"
    
    all_results = []
    failed_experiments = []
    
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                # Run experiment
                result = run_bwo_cnn_experiment(dataset, seed, trace_file)
                
                # Save results
                save_best_config(result, config_file)
                save_metrics(result, metrics_file)
                
                all_results.append(result)
                
            except Exception as e:
                error_msg = f"Error running experiment for {dataset} (seed={seed}): {e}"
                print(f"\n[ERROR] {error_msg}")
                failed_experiments.append((dataset, seed, str(e)))
                
                # Print traceback
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if all_results:
        print(f"\nCompleted {len(all_results)} experiments successfully:")
        
        # Group by dataset
        for dataset in DATASETS:
            dataset_results = [r for r in all_results if r['dataset'] == dataset]
            if dataset_results:
                print(f"\n{dataset}:")
                for r in dataset_results:
                    print(f"  Seed {r['seed']}: R²={r['metrics']['R2']:.4f}, "
                          f"RMSE={r['metrics']['RMSE']:.4f}, "
                          f"MAE={r['metrics']['MAE']:.4f}")
    
    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}):")
        for dataset, seed, error in failed_experiments:
            print(f"  {dataset} (seed={seed}): {error}")
    
    print("\n" + "="*70)
    print("Output Files:")
    print(f"  1. Search Trace: {trace_file}")
    print(f"  2. Best Configs: {config_file}")
    print(f"  3. Metrics:      {metrics_file}")
    print("="*70)
    
    return all_results


def check_dependencies():
    """Check if all required files exist."""
    print("Checking dependencies...")
    
    required_files = [
        "bwo_algorithm.py",
        "unified_split_utils.py",
        "split_manifest_80_20_unified.json"
    ] + list(DATA_PATHS.values())
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("\n[ERROR] Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("All required files found.")
    return True


if __name__ == "__main__":
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run all experiments
    results = run_all_experiments()
    
    print("\nDone!")
