#!/usr/bin/env python3
"""
Comparison charts of operational metrics driven by different forecast models
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set font to avoid Chinese character warnings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """Load comparison results"""
    path = "dispatch_comparison_results.json"
    if not os.path.exists(path):
        print(f"Result file does not exist: {path}")
        return None
    
    with open(path, 'r') as f:
        results = json.load(f)
    return results

def create_bar_chart(results):
    """Create bar chart comparing metrics"""
    models = ["Ultra-LSNT", "DLinear", "Autoformer", "Informer"]
    
    # Extract data
    total_costs = []
    total_reserves = []
    total_curtailments = []
    
    for model in models:
        if model in results:
            total_costs.append(results[model]['total_cost'])
            total_reserves.append(results[model]['total_reserve'])
            total_curtailments.append(results[model]['total_curtailment'])
        else:
            total_costs.append(0)
            total_reserves.append(0)
            total_curtailments.append(0)
    
    # Convert to million yuan and gigawatt-hours for display
    total_costs_million = [c / 1_000_000 for c in total_costs]
    total_reserves_gwh = [r / 1_000_000 for r in total_reserves]  # Convert to GWh
    total_curtailments_gwh = [c / 1_000_000 for c in total_curtailments]
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total cost comparison
    ax = axes[0, 0]
    bars = ax.bar(models, total_costs_million, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Total Cost (Million ¥)')
    ax.set_title('Total Operation Cost Comparison by Model')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, total_costs_million):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Reserve capacity comparison
    ax = axes[0, 1]
    bars = ax.bar(models, total_reserves_gwh, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Reserve Capacity (GWh)')
    ax.set_title('Reserve Capacity Requirement by Model')
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, total_reserves_gwh):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Wind curtailment comparison
    ax = axes[1, 0]
    bars = ax.bar(models, total_curtailments_gwh, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Wind Curtailment (GWh)')
    ax.set_title('Wind Curtailment by Model')
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, total_curtailments_gwh):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Relative performance radar chart (simplified)
    ax = axes[1, 1]
    
    if "Ultra-LSNT" in results:
        baseline = results["Ultra-LSNT"]
        
        # Calculate relative performance
        relative_costs = []
        relative_reserves = []
        relative_curtailments = []
        
        for model in models:
            if model == "Ultra-LSNT":
                relative_costs.append(1.0)
                relative_reserves.append(1.0)
                relative_curtailments.append(1.0)
            elif model in results:
                r = results[model]
                relative_costs.append(r['total_cost'] / baseline['total_cost'])
                relative_reserves.append(r['total_reserve'] / baseline['total_reserve'])
                relative_curtailments.append(r['total_curtailment'] / baseline['total_curtailment'] if baseline['total_curtailment'] > 0 else 0)
            else:
                relative_costs.append(0)
                relative_reserves.append(0)
                relative_curtailments.append(0)
        
        # Draw line chart
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, relative_costs, width, label='Cost (relative)', color='#1f77b4', alpha=0.7)
        ax.bar(x, relative_reserves, width, label='Reserve (relative)', color='#ff7f0e', alpha=0.7)
        ax.bar(x + width, relative_curtailments, width, label='Curtailment (relative)', color='#2ca02c', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Relative Performance (baseline=1.0)')
        ax.set_title('Relative Performance Comparison (Ultra-LSNT as baseline)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal reference line
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('dispatch_comparison_charts.png', dpi=150, bbox_inches='tight')
    print("Chart saved as: dispatch_comparison_charts.png")
    
    # Create detailed analysis charts
    create_detailed_charts(results, models)
    
    plt.show()

def create_detailed_charts(results, models):
    """Create detailed analysis charts"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Cost breakdown comparison
    ax = axes[0]
    
    # Extract cost breakdown data
    cost_categories = ['Fuel Cost', 'Reserve Cost', 'Purchase Cost', 'Storage Cost', 'Startup/Shutdown Cost']
    model_costs = {model: [] for model in models if model in results}
    
    for model in models:
        if model in results:
            r = results[model]
            if 'cost_breakdown' in r:
                cb = r['cost_breakdown']
                # Extract cost components
                fuel = cb.get('fuel_cost', 0)
                reserve = cb.get('reserve_cost', 0)
                purchase = cb.get('purchase_cost', 0)
                storage = cb.get('storage_cost', 0)
                startup = cb.get('startup_cost', 0) + cb.get('shutdown_cost', 0)
                model_costs[model] = [fuel, reserve, purchase, storage, startup]
    
    # Draw stacked bar chart
    bottom = np.zeros(len(models))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, category in enumerate(cost_categories):
        values = []
        for model in models:
            if model in model_costs and len(model_costs[model]) > i:
                values.append(model_costs[model][i])
            else:
                values.append(0)
        
        ax.bar(models, values, bottom=bottom, label=category, color=colors[i])
        bottom += values
    
    ax.set_ylabel('Cost (¥)')
    ax.set_title('Cost Breakdown Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. Wind forecast vs curtailment relationship
    ax = axes[1]
    
    # Extract wind forecast range and curtailment
    wind_ranges = []
    curtailments = []
    
    for model in models:
        if model in results:
            r = results[model]
            wind_range = r.get('wind_mean_range', [0, 0])
            wind_mid = (wind_range[0] + wind_range[1]) / 2
            wind_ranges.append(wind_mid / 1000)  # Convert to MW
            curtailments.append(r['total_curtailment'] / 1000)  # Convert to MWh
    
    scatter = ax.scatter(wind_ranges, curtailments, s=100, c=range(len(models)), cmap='viridis')
    
    # Add model labels
    for i, model in enumerate(models):
        if model in results:
            ax.annotate(model, (wind_ranges[i], curtailments[i]),
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Wind Forecast Mean (MW)')
    ax.set_ylabel('Total Wind Curtailment (MWh)')
    ax.set_title('Wind Forecast Scale vs Curtailment Relationship')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dispatch_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print("Detailed analysis chart saved as: dispatch_detailed_analysis.png")

def main():
    print("Generating dispatch comparison charts...")
    
    results = load_results()
    if results is None:
        print("Unable to load result data")
        return
    
    print(f"Loaded results for {len(results)} models")
    
    # Create charts
    create_bar_chart(results)
    
    # Generate data summary
    print("\nData Summary:")
    models = ["Ultra-LSNT", "DLinear", "Autoformer", "Informer"]
    for model in models:
        if model in results:
            r = results[model]
            print(f"\n{model}:")
            print(f"  Total cost: {r['total_cost']:,.0f} yuan")
            print(f"  Reserve capacity: {r['total_reserve']:,.0f} kWh")
            print(f"  Wind curtailment: {r['total_curtailment']:,.0f} kWh")
            if 'cost_breakdown' in r:
                cb = r['cost_breakdown']
                print(f"  Cost breakdown: fuel {cb.get('fuel_cost', 0):.0f}, reserve {cb.get('reserve_cost', 0):.0f}, storage {cb.get('storage_cost', 0):.0f}")

if __name__ == "__main__":
    main()