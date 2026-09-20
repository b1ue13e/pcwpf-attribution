"""
COA-BiLSTM Experiment Script - High Intensity Search
Population=40, Iterations=200, 3 Seeds, 4 Datasets
Output: metaheuristic_search_trace.csv, metaheuristic_best_configs.csv, metaheuristic_metrics_all_datasets.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

sys.path.append('论文1')

from unified_split_utils import load_and_split_dataset
from coa_algorithm import CuckooOptimizationAlgorithm

warnings.filterwarnings('ignore')

# High-intensity search configuration
POPULATION_SIZE = 40
MAX_ITERATIONS = 200
SEEDS = [42, 123, 456]
DATASETS = ['wind_cn', 'electricity', 'air_quality', 'gefcom']

# Model training config during optimization
OPTIMIZATION_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
FINAL_EPOCHS = 50


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM for time series forecasting."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def prepare_sequences(data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, -1])
    return np.array(X), np.array(y)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                lr: float, weight_decay: float, epochs: int, device: torch.device,
                patience: int = 5) -> Dict[str, Any]:
    """Train the BiLSTM model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'epochs_trained': len(train_losses)
    }


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   device: torch.device) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    
    y_true = y_test
    mse = np.mean((predictions - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_true))
    
    ss_res = np.sum((y_true - predictions) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100
    
    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape), 'MSE': float(mse)}


class COASearchTrace:
    """Records COA search trace."""
    def __init__(self, dataset: str, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.trace = []
    
    def record(self, iteration: int, best_fitness: float, best_params: np.ndarray):
        self.trace.append({
            'dataset': self.dataset,
            'model': 'COA-BiLSTM',
            'seed': self.seed,
            'iter': iteration,
            'best_score': best_fitness,
            'hyperparams': json.dumps({
                'hidden_dim': int(best_params[0]),
                'num_layers': int(best_params[1]),
                'dropout': float(best_params[2]),
                'lr': float(best_params[3]),
                'weight_decay': float(best_params[4]),
                'batch_size': int(best_params[5])
            })
        })


def create_objective_function(dataset_name: str, seq_len: int, pred_len: int,
                              device: torch.device, search_trace: COASearchTrace, seed: int = 42):
    """Create objective function for COA optimization."""
    
    def objective(params: np.ndarray) -> float:
        hidden_dim = max(8, int(params[0]))
        num_layers = max(1, int(params[1]))
        dropout = np.clip(params[2], 0.0, 0.8)
        lr = np.clip(params[3], 1e-5, 1e-1)
        weight_decay = np.clip(params[4], 1e-7, 1e-2)
        batch_size = max(8, int(params[5]))
        
        try:
            data_paths = {
                'wind_cn': ('论文1/processed_wind.csv', 'power'),
                'electricity': ('论文1/Load_history.csv', 'load'),
                'air_quality': ('论文1/air_quality_ready.csv', 'PM2.5'),
                'gefcom': ('论文1/gefcom_ready.csv', 'load')
            }
            
            data_path, target_col = data_paths[dataset_name]
            train_df, test_df = load_and_split_dataset(dataset_name, data_path)
            
            data = train_df.values
            if target_col in train_df.columns:
                cols = [c for c in train_df.columns if c != target_col] + [target_col]
                data = train_df[cols].values
            
            X, y = prepare_sequences(data, seq_len, pred_len)
            
            if len(X) < 100:
                return float('inf')
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            input_dim = X.shape[2]
            model = BiLSTMRegressor(input_dim, hidden_dim, num_layers, pred_len, dropout)
            
            training_results = train_model(
                model, train_loader, val_loader, lr, weight_decay,
                epochs=OPTIMIZATION_EPOCHS, device=device, patience=EARLY_STOPPING_PATIENCE
            )
            
            return training_results['best_val_loss']
            
        except Exception as e:
            return float('inf')
    
    return objective


def run_coa_bilstm_experiment(dataset_name: str, seed: int) -> Dict[str, Any]:
    """Run COA-BiLSTM experiment with high-intensity search."""
    print(f"\n{'='*60}")
    print(f"COA-BiLSTM: {dataset_name} (seed={seed})")
    print(f"Population={POPULATION_SIZE}, Iterations={MAX_ITERATIONS}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    seq_len, pred_len = 96, 24
    
    # Parameter bounds
    bounds = [
        (32, 256),      # hidden_dim
        (1, 4),         # num_layers
        (0.1, 0.5),     # dropout
        (1e-4, 1e-2),   # learning rate
        (1e-6, 1e-3),   # weight decay
        (16, 128)       # batch_size
    ]
    
    search_trace = COASearchTrace(dataset_name, seed)
    
    # Custom COA class with trace recording
    class COAWithTrace(CuckooOptimizationAlgorithm):
        def optimize(self, verbose=False):
            for iteration in range(self.max_iter):
                i = np.random.randint(self.pop_size)
                new_nest = self._get_new_cuckoo(i)
                new_fitness = self.objective_func(new_nest)
                
                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness
                
                self._abandon_nests()
                
                current_best_idx = np.argmin(self.fitness)
                current_best_fitness = self.fitness[current_best_idx]
                
                if current_best_fitness < self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_nest = self.nests[current_best_idx].copy()
                
                self.history['best_fitness'].append(self.best_fitness)
                self.history['avg_fitness'].append(np.mean(self.fitness))
                self.history['iterations'].append(iteration)
                
                # Record trace every 5 iterations to save space
                if iteration % 5 == 0 or iteration == self.max_iter - 1:
                    search_trace.record(iteration, self.best_fitness, self.best_nest)
                
                if verbose and iteration % 20 == 0:
                    print(f"  Iter {iteration}: Best = {self.best_fitness:.6f}")
            
            return {
                'best_solution': self.best_nest,
                'best_fitness': self.best_fitness,
                'history': self.history
            }
    
    objective_func = create_objective_function(dataset_name, seq_len, pred_len, device, search_trace, seed)
    
    coa = COAWithTrace(
        objective_func=objective_func,
        bounds=bounds,
        population_size=POPULATION_SIZE,
        nest_abandonment_prob=0.25,
        step_size_alpha=0.01,
        levy_beta=1.5,
        max_iterations=MAX_ITERATIONS,
        random_seed=seed
    )
    
    start_time = time.time()
    result = coa.optimize(verbose=True)
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization completed in {optimization_time:.2f}s")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    
    # Decode best parameters
    best_params = result['best_solution']
    hidden_dim = int(best_params[0])
    num_layers = int(best_params[1])
    dropout = best_params[2]
    lr = best_params[3]
    weight_decay = best_params[4]
    batch_size = int(best_params[5])
    
    # Final evaluation
    data_paths = {
        'wind_cn': ('论文1/processed_wind.csv', 'power'),
        'electricity': ('论文1/Load_history.csv', 'load'),
        'air_quality': ('论文1/air_quality_ready.csv', 'PM2.5'),
        'gefcom': ('论文1/gefcom_ready.csv', 'load')
    }
    
    data_path, target_col = data_paths[dataset_name]
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    train_data = train_df.values
    test_data = test_df.values
    if target_col in train_df.columns:
        cols = [c for c in train_df.columns if c != target_col] + [target_col]
        train_data = train_df[cols].values
        test_data = test_df[cols].values
    
    X_train, y_train = prepare_sequences(train_data, seq_len, pred_len)
    X_test, y_test = prepare_sequences(test_data, seq_len, pred_len)
    
    split_idx = int(len(X_train) * 0.8)
    X_train_final, X_val_final = X_train[:split_idx], X_train[split_idx:]
    y_train_final, y_val_final = y_train[:split_idx], y_train[split_idx:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train_final))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_final), torch.FloatTensor(y_val_final))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[2]
    model = BiLSTMRegressor(input_dim, hidden_dim, num_layers, pred_len, dropout).to(device)
    
    final_training = train_model(model, train_loader, val_loader, lr, weight_decay, 
                                 epochs=FINAL_EPOCHS, device=device, patience=10)
    
    test_metrics = evaluate_model(model, X_test, y_test, device)
    
    # Find convergence iteration
    convergence_iter = len(result['history']['best_fitness'])
    for i, fit in enumerate(result['history']['best_fitness']):
        if i > 10 and abs(fit - result['history']['best_fitness'][-1]) < 1e-6:
            convergence_iter = i
            break
    
    print(f"Test Metrics: R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}")
    
    return {
        'dataset': dataset_name,
        'model': 'COA-BiLSTM',
        'seed': seed,
        'best_params': {
            'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout,
            'lr': lr, 'weight_decay': weight_decay, 'batch_size': batch_size
        },
        'metrics': test_metrics,
        'optimization_time': optimization_time,
        'convergence_iter': convergence_iter,
        'search_trace': search_trace.trace
    }


def run_all_experiments():
    """Run all COA-BiLSTM experiments."""
    all_results = []
    all_traces = []
    
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                result = run_coa_bilstm_experiment(dataset, seed)
                all_results.append(result)
                all_traces.extend(result['search_trace'])
                
                # Save intermediate
                with open(f'coa_bilstm_{dataset}_seed{seed}.json', 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"Error: {dataset} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate output files
    generate_output_files(all_results, all_traces)
    return all_results


def generate_output_files(all_results: List[Dict], all_traces: List[Dict]):
    """Generate required output CSV files."""
    
    # 1. Search trace
    trace_df = pd.DataFrame(all_traces)
    trace_df.to_csv('metaheuristic_search_trace.csv', index=False)
    print(f"\nSaved metaheuristic_search_trace.csv ({len(trace_df)} rows)")
    
    # 2. Best configs
    best_configs = []
    for r in all_results:
        best_configs.append({
            'dataset': r['dataset'],
            'model': r['model'],
            'best_hyperparams': json.dumps(r['best_params']),
            'best_seed': r['seed'],
            'R2': r['metrics']['R2'],
            'RMSE': r['metrics']['RMSE']
        })
    
    # Find best seed per dataset
    best_per_dataset = {}
    for bc in best_configs:
        key = (bc['dataset'], bc['model'])
        if key not in best_per_dataset or bc['R2'] > best_per_dataset[key]['R2']:
            best_per_dataset[key] = bc
    
    configs_df = pd.DataFrame(list(best_per_dataset.values()))
    configs_df.to_csv('metaheuristic_best_configs.csv', index=False)
    print(f"Saved metaheuristic_best_configs.csv ({len(configs_df)} rows)")
    
    # 3. All metrics
    metrics_rows = []
    for r in all_results:
        metrics_rows.append({
            'dataset': r['dataset'],
            'model': r['model'],
            'seed': r['seed'],
            'R2': r['metrics']['R2'],
            'RMSE': r['metrics']['RMSE'],
            'MAE': r['metrics']['MAE'],
            'MAPE': r['metrics']['MAPE'],
            'convergence_iter': r['convergence_iter']
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('metaheuristic_metrics_all_datasets.csv', index=False)
    print(f"Saved metaheuristic_metrics_all_datasets.csv ({len(metrics_df)} rows)")
    
    # Summary
    print("\n" + "="*60)
    print("COA-BiLSTM Experiments Summary")
    print("="*60)
    for dataset in DATASETS:
        ds_results = [r for r in all_results if r['dataset'] == dataset]
        if ds_results:
            r2_scores = [r['metrics']['R2'] for r in ds_results]
            print(f"{dataset}: R2 = {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


if __name__ == "__main__":
    print("COA-BiLSTM High-Intensity Search")
    print(f"Population={POPULATION_SIZE}, Iterations={MAX_ITERATIONS}")
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    run_all_experiments()
