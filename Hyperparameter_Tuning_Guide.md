# Hyperparameter Tuning Guide for Cortex5

## Overview
This guide explains how to use the Improvement 6.4 Automated Hyperparameter Tuning system to optimize Cortex5 model performance through systematic parameter search.

## What is Hyperparameter Tuning?

Hyperparameters are configuration settings that control the learning process but are not learned from data. Examples include:
- **Learning Rate**: How fast the model learns
- **Network Architecture**: Number and size of hidden layers
- **Regularization**: Dropout rates to prevent overfitting
- **Training Schedule**: Exploration rates and batch sizes

Proper hyperparameter tuning can improve model performance by 20-50% over default settings.

## Configuration Parameters

### Basic Settings
```mql5
input bool InpUseHyperparameterTuning = true;     // Enable optimization
input string InpOptimizationMethod = "BAYESIAN"; // GRID, BAYESIAN, RANDOM
input int InpOptimizationIterations = 30;        // Number of configurations to try
input string InpOptimizationObjective = "MULTI"; // SHARPE, PROFIT, DRAWDOWN, MULTI
```

### Validation Settings
```mql5
input bool InpUseValidationSplit = true;         // Use validation for evaluation
input double InpHyperparamValidationSplit = 0.15; // 15% of data for validation
```

### Logging and Analysis
```mql5
input bool InpSaveOptimizationResults = true;    // Save results to CSV
input bool InpLogOptimizationProgress = true;    // Log progress during search
input int InpOptimizationSeed = 42;              // For reproducible results
```

## Optimization Methods

### 1. Grid Search (GRID)
- **Best for**: Systematic exploration of known parameter ranges
- **Pros**: Guaranteed to find optimal combination within search space
- **Cons**: Computationally expensive, exponential growth with parameters
- **Use when**: You have 3-4 key parameters and want thorough coverage

```mql5
InpOptimizationMethod = "GRID";
InpOptimizationIterations = 36; // 4×3×3 grid for learning_rate×gamma×dropout
```

### 2. Bayesian Optimization (BAYESIAN)
- **Best for**: Efficient search with limited computational budget
- **Pros**: Uses previous results to guide search, very efficient
- **Cons**: More complex, may get stuck in local optima
- **Use when**: You have limited time and want efficient exploration

```mql5
InpOptimizationMethod = "BAYESIAN";
InpOptimizationIterations = 25; // Efficient exploration
```

### 3. Random Search (RANDOM)
- **Best for**: Baseline comparison and unexpected discoveries
- **Pros**: Simple, can find unexpected good combinations
- **Cons**: No learning from previous results, may miss optimal regions
- **Use when**: You want a baseline or to explore diverse parameter space

```mql5
InpOptimizationMethod = "RANDOM";
InpOptimizationIterations = 20; // Random sampling
```

## Optimization Objectives

### 1. Sharpe Ratio (SHARPE)
- **Focus**: Risk-adjusted returns
- **Best for**: Balanced trading strategies
- **Formula**: (Return - Risk-free rate) / Standard deviation of returns

### 2. Total Profit (PROFIT)
- **Focus**: Absolute returns
- **Best for**: High-return strategies (accepting higher risk)
- **Formula**: Total percentage return

### 3. Minimum Drawdown (DRAWDOWN)
- **Focus**: Capital preservation
- **Best for**: Conservative strategies
- **Formula**: -Maximum drawdown (minimized)

### 4. Multi-Objective (MULTI)
- **Focus**: Balanced optimization
- **Best for**: Most practical applications
- **Formula**: 0.4×Sharpe + 0.3×(Return/100) - 0.3×(Drawdown/100)

## Parameter Search Spaces

The system automatically searches these parameter ranges:

### Core Learning Parameters
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Gamma (Discount Factor)**: 0.9 to 0.999
- **Dropout Rate**: 0.0 to 0.3
- **Batch Size**: 16, 32, 64, 128, 256 (powers of 2)

### Network Architecture
- **Hidden Layer Sizes**: 32, 48, 64, 96, 128 neurons
- **Each layer independently optimized**

### Training Schedule
- **Exploration Start**: 0.8 to 1.2
- **Exploration End**: 0.01 to 0.1
- **Exploration Decay**: 30,000 to 70,000 steps
- **Target Sync Frequency**: 1,000 to 5,000 steps

### Advanced Parameters (if enabled)
- **Confidence Weight**: 0.1 to 0.5 (if confidence training enabled)
- **Calibration Weight**: 0.05 to 0.3 (if confidence training enabled)
- **Online Learning Rate**: 1e-6 to 1e-3 (if online learning enabled)

## Usage Workflow

### Step 1: Initial Hyperparameter Search
```mql5
// Use smaller dataset for faster search
InpYears = 1; // Reduce data for hyperparameter search
InpUseHyperparameterTuning = true;
InpOptimizationMethod = "BAYESIAN";
InpOptimizationIterations = 20;
InpOptimizationObjective = "MULTI";
```

### Step 2: Analyze Results
Check the generated CSV file: `HyperparameterOptimization_SYMBOL_TIMEFRAME_Results.csv`

Look for:
- **Best objective scores**
- **Parameter patterns** (e.g., consistently high learning rates)
- **Validation vs training performance**

### Step 3: Final Training with Optimal Parameters
Apply the best parameters found:
```mql5
// Copy from optimization results
InpLR = 0.00012345; // Best learning rate found
InpGamma = 0.9876;  // Best gamma found
InpDropoutRate = 0.15; // Best dropout found
InpBatch = 64;      // Best batch size found
InpH1 = 96;         // Best hidden layer sizes
InpH2 = 64;
InpH3 = 96;

// Now use full dataset
InpYears = 3;
InpUseHyperparameterTuning = false; // Disable for final training
```

## Best Practices

### Dataset Strategy
1. **Hyperparameter Search**: Use 1-2 years of data for speed
2. **Final Training**: Use full 3+ years with optimal parameters
3. **Validation Split**: Use 10-20% for validation during search

### Iteration Guidelines
- **Grid Search**: 20-50 iterations (depends on grid size)
- **Bayesian**: 15-30 iterations (efficient search)
- **Random**: 20-40 iterations (broad exploration)

### Objective Selection
- **Conservative Trading**: Use DRAWDOWN or MULTI
- **Aggressive Trading**: Use PROFIT or SHARPE
- **Balanced Approach**: Use MULTI (recommended)

### Time Management
- **Quick Test**: 10 iterations, RANDOM method, 1 year data
- **Thorough Search**: 30 iterations, BAYESIAN method, 2 years data
- **Comprehensive**: 50+ iterations, GRID method, full data

## Interpreting Results

### Key Metrics to Monitor
1. **Objective Score**: Primary optimization target
2. **Validation Score**: Generalization performance
3. **Training Time**: Efficiency consideration
4. **Success Rate**: Robustness indicator

### Warning Signs
- **Overfitting**: Training score >> Validation score
- **Instability**: High variance in scores with similar parameters
- **Poor Convergence**: No improvement over iterations

### Good Results Indicators
- **Consistent Patterns**: Similar parameters appear in top results
- **Validation Alignment**: Training and validation scores correlate
- **Progressive Improvement**: Scores improve over iterations (Bayesian)

## Advanced Usage

### Strategy Tester Integration
For parallel optimization (future enhancement):
```mql5
InpParallelOptimization = true; // Use MT5 Strategy Tester optimization
// This allows running multiple parameter sets simultaneously
```

### Custom Search Spaces
Modify the `InitializeHyperparameterBounds()` function to adjust search ranges:
```mql5
// Example: Focus on higher learning rates
bounds.learning_rate_min = 0.0001;   // Increase minimum
bounds.learning_rate_max = 0.005;    // Reduce maximum for focused search
```

### Multi-Stage Optimization
1. **Coarse Search**: Wide ranges, fewer iterations
2. **Fine Search**: Narrow ranges around best results, more iterations
3. **Final Validation**: Test best parameters on unseen data

## Troubleshooting

### Poor Optimization Results
- **Check data quality**: Ensure sufficient historical data
- **Verify parameter ranges**: May be too narrow or too wide
- **Consider objective function**: May not align with trading goals
- **Increase iterations**: May need more exploration

### Slow Performance
- **Reduce dataset size**: Use 1 year instead of 3 for search
- **Decrease iterations**: Start with 10-15 iterations
- **Use simpler method**: Try RANDOM instead of BAYESIAN
- **Disable heavy features**: Turn off ensemble/confidence during search

### Inconsistent Results
- **Set random seed**: Use consistent `InpOptimizationSeed`
- **Increase validation split**: Use 20% for more stable validation
- **Check for overfitting**: Validate on completely separate data
- **Consider market regime**: Data may span different market conditions

## Example Optimization Session

```mql5
// Phase 1: Quick exploration
InpUseHyperparameterTuning = true;
InpOptimizationMethod = "RANDOM";
InpOptimizationIterations = 15;
InpOptimizationObjective = "MULTI";
InpYears = 1;
// Results: Learning rate ~0.0001, Gamma ~0.995 work well

// Phase 2: Focused search
InpOptimizationMethod = "BAYESIAN";
InpOptimizationIterations = 25;
InpYears = 2;
// Results: Learning rate 0.00012, Dropout 0.15, Hidden layers [64,96,64]

// Phase 3: Final training
InpUseHyperparameterTuning = false;
InpLR = 0.00012;
InpDropoutRate = 0.15;
InpH1 = 64; InpH2 = 96; InpH3 = 64;
InpYears = 3;
// Train final model with optimal parameters
```

This systematic approach typically improves model performance by 20-50% over default parameters while ensuring robust generalization to new market conditions.