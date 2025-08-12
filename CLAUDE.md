# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains Cortex5, a MetaTrader 5 algorithmic trading system that uses Double-Dueling DRQN (Deep Recurrent Q-Network) reinforcement learning for forex trading. The system is implemented in MQL5 and consists of three main components for training, trading, and diagnostics.

## File Structure

- `cortex5.mq5` - Main Expert Advisor (EA) for live trading execution
- `Cortextrainingv5.mq5` - Training script using Double-Dueling DRQN with LSTM memory
- `ModelDiagnostic5.mq5` - Model validation and diagnostic utility
- `CortexBacktestWorking.mq5` - Backtesting implementation
- `CLAUDE details.md` - Comprehensive technical documentation (legacy file)

## Development Commands

This is an MQL5 codebase for MetaTrader 5. There are no traditional build/test commands as MQL5 files are compiled within the MetaTrader 5 environment.

### Working with MQL5 Files
- Compile: Use MetaTrader 5 MetaEditor's compile function (F7)
- Deploy: Copy compiled .ex5 files to MT5's Experts/Indicators folders
- Debug: Use MetaTrader 5's Strategy Tester for backtesting

## Key Architecture Concepts

### AI Model Structure
- **Input**: 35-feature state vector (STATE_SIZE=35) including price, volume, technical indicators
- **Network**: 3-layer dense network (64→64→64) with optional LSTM layer (32 hidden units)
- **Architecture**: Double-Dueling DRQN separating state-value V(s) from advantage A(s,a)
- **Output**: 6 trading actions (BUY_STRONG, BUY_WEAK, SELL_STRONG, SELL_WEAK, HOLD, FLAT)
- **Memory**: 8-step LSTM sequences for market regime persistence

### Trading System Features
- Multi-timeframe analysis (M1, M5, H1, H4, D1)
- Dynamic position scaling based on signal strength
- Advanced risk management with ATR-based stops
- Volatility regime detection and adaptation
- Prioritized Experience Replay (PER) for training

### Model File Format
Models are saved as binary .dat files with:
- Magic number validation (0xC0DE0203)
- Symbol/timeframe metadata
- Network architecture parameters
- Feature normalization parameters (min/max scaling)
- All weights, biases, and training checkpoints

## Important Constants

```mql5
#define STATE_SIZE 35    // Number of input features - DO NOT CHANGE
#define ACTIONS 6        // Number of possible trading actions - DO NOT CHANGE
```

These constants are fundamental to the architecture. Changing them requires retraining all models.

## Development Workflow

### Training a New Model
1. Configure `Cortextrainingv5.mq5` parameters:
   - Set symbol (`InpSymbol="AUTO"` or specific pair)
   - Set timeframe (`InpTF=PERIOD_M5`)
   - Set training data range (`InpYears=3`)
   - Configure network architecture (LSTM, Dueling, Double DQN)
2. Run training script in MT5 Strategy Tester
3. Generated model: `DoubleDueling_DRQN_Model.dat`

### Deploying for Live Trading
1. Place trained model in MT5 Files folder
2. Configure `cortex5.mq5` EA parameters
3. **Critical**: Ensure symbol/timeframe match between model and chart
4. Enable appropriate risk management settings

### Model Validation
Use `ModelDiagnostic5.mq5` to:
- Verify model file integrity and compatibility
- Check architecture parameters
- Validate feature normalization ranges
- Get deployment safety recommendations

## Safety and Risk Management

The system includes multiple safety mechanisms:
- Model symbol/timeframe validation prevents mismatched deployments
- Account drawdown limits and position size controls
- Volatility regime detection with automatic position reduction
- Trading session filtering and spread monitoring
- Comprehensive transaction cost modeling in training

## Code Conventions

When modifying the codebase:
- Follow MQL5 naming conventions (PascalCase for functions, camelCase for variables)
- Maintain the 35-feature state vector structure
- Preserve model file format compatibility
- Use consistent error handling patterns with GetLastError()
- Log important events with appropriate severity levels
- Maintain backwards compatibility with existing model files

## Key Configuration Areas

### Network Architecture (`Cortextrainingv5.mq5`)
- `InpH1, InpH2, InpH3`: Hidden layer sizes (default: 64, 64, 64)
- `InpUseLSTM`: Enable LSTM memory (recommended: true)
- `InpUseDuelingNet`: Enable dueling architecture (recommended: true)
- `InpUseDoubleDQN`: Enable Double DQN (recommended: true)

### Risk Management (`cortex5.mq5`)
- `InpAllowPositionScaling`: Dynamic position sizing
- `InpUseVolatilityRegime`: Volatility-based risk adaptation
- `InpUseTrailingStop`: Trailing stop functionality
- `InpATRMultiplier`: Stop loss distance as ATR multiple

## File Safety Note

All files contain sophisticated financial trading algorithms designed for defensive automated trading with extensive risk controls. The code is educational and research-oriented with built-in safety mechanisms.