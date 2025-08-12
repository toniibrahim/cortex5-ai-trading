# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a MetaTrader 5 (MT5) algorithmic trading system called Cortex, which uses Double-Dueling DRQN (Deep Recurrent Q-Network) reinforcement learning for forex trading. The system consists of three main components:

1. **cortex5.mq5** - Expert Advisor (EA) that executes live trades using a trained AI model
2. **Cortextrainingv5.mq5** - Training script that teaches the AI using historical market data  
3. **ModelDiagnostic5.mq5** - Diagnostic tool to analyze and validate trained models

## File Structure

- `cortex5.mq5` - Main trading EA with comprehensive risk management
- `Cortextrainingv5.mq5` - Double-Dueling DRQN training system with Prioritized Experience Replay
- `ModelDiagnostic5.mq5` - Model validation and inspection utility

## Key Architecture Concepts

### AI Model Structure
- **Input**: 35 market features (STATE_SIZE=35) including price, volume, technical indicators, and position state
- **Base Network**: 3-layer dense network with configurable sizes (default: 64→64→64)
- **LSTM Memory**: Optional recurrent layer (32 hidden units, 8-step sequences) for market regime memory
- **Dueling Architecture**: Separates state-value V(s) from advantage A(s,a) for better action selection
- **Output**: 6 trading actions (BUY_STRONG, BUY_WEAK, SELL_STRONG, SELL_WEAK, HOLD, FLAT)
- **Learning**: Double-Dueling DRQN with LSTM memory, Prioritized Experience Replay (PER), and target network stabilization

### Trading Features
- **Multi-timeframe analysis**: Uses M1, M5, H1, H4, D1 data for trend confirmation
- **Position-aware training**: AI learns optimal position sizing and management
- **Dynamic position scaling**: Adjusts position sizes based on signal strength changes
- **Enhanced cost modeling**: Realistic spread, slippage, and commission simulation
- **Comprehensive risk controls**: Drawdown limits, position limits, volatility filters
- **Advanced risk management**: Trailing stops, break-even moves, volatility regime awareness

### Double-Dueling DRQN Architecture

The system uses an advanced reinforcement learning architecture combining three key innovations:

**1. Double DQN (DDQN)**
- Fixes Q-value overestimation bias in standard DQN
- Uses main network for action selection, target network for evaluation
- Results in more stable and accurate Q-value estimates
- Enabled by default (`InpUseDoubleDQN=true`)

**2. Dueling Network Architecture**
- Separates state-value function V(s) from advantage function A(s,a)
- Final Q-values computed as: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
- Better performance in noisy market conditions
- Configurable head sizes (`InpValueHead=32, InpAdvHead=32`)

**3. LSTM Memory Layer**
- Handles partial observability and market regime persistence
- Maintains 8-step sequence memory (`InpSequenceLen=8`)
- 32 hidden units store information about trend/volatility patterns
- Critical for adapting to changing market conditions

**Network Flow:**
```
Input (35 features) → Dense Layer 1 (64) → Dense Layer 2 (64) → Dense Layer 3 (64)
                                   ↓
                    Optional: LSTM Layer (32 hidden units)
                                   ↓
              Dueling Heads: Value Head (1) + Advantage Head (6)
                                   ↓
              Combined Q-values (6 actions) for trading decisions
```

### Model File Format
Models are saved with checkpoint data for incremental training:
- Magic number 0xC0DE0203 (enhanced format) or 0xC0DE0202 (legacy)
- Symbol/timeframe metadata for safety validation  
- Hidden layer dimensions (h1, h2, h3)
- Architecture flags (LSTM enabled, Dueling enabled)
- LSTM parameters (hidden size, sequence length)
- Dueling head sizes (value head, advantage head)
- Feature normalization parameters (min/max scaling to [0,1])
- All network weights and biases (dense layers, LSTM weights, dueling heads)
- Training checkpoint state (epsilon, steps, last trained time)

### Configuration Parameters

**Network Architecture:**
- `InpH1, InpH2, InpH3`: Hidden layer sizes (default: 64, 64, 64)
- `InpUseLSTM`: Enable LSTM memory layer (default: true)
- `InpLSTMSize`: LSTM hidden units (default: 32)
- `InpSequenceLen`: Memory sequence length (default: 8)
- `InpUseDuelingNet`: Enable dueling heads (default: true)
- `InpValueHead, InpAdvHead`: Dueling head sizes (default: 32, 32)

**Training Features:**
- `InpUseDoubleDQN`: Enable Double DQN (default: true)
- `InpUsePER`: Prioritized Experience Replay (default: true)
- `InpTargetSync`: Target network update frequency (default: 100 steps)

## Development Workflow

### Training a New Model
1. Configure training parameters in `Cortextrainingv5.mq5`:
   - Set symbol (`InpSymbol="AUTO"` or specific like `"EURUSD"`)
   - Set timeframe (`InpTF=PERIOD_M5`)
   - Set data range (`InpYears=3` for 3 years of history)
2. Configure Double-Dueling DRQN architecture:
   - Dense layers: `InpH1=64, InpH2=64, InpH3=64`
   - LSTM: `InpUseLSTM=true, InpLSTMSize=32, InpSequenceLen=8`
   - Dueling: `InpUseDuelingNet=true, InpValueHead=32, InpAdvHead=32`
3. Enable advanced features:
   - `InpUseDoubleDQN=true` (recommended for stability)
   - `InpUsePER=true` (prioritized experience replay)
4. Run training script - it will create `DoubleDueling_DRQN_Model.dat`

### Deploying the Model
1. Place trained model file in MT5 `Files` folder
2. Configure EA parameters in `cortex5.mq5`
3. **Critical**: Ensure symbol/timeframe match between model and chart
4. Enable risk management settings for live trading

### Model Validation
Use `ModelDiagnostic5.mq5` to:
- Verify model file format and compatibility
- Check training data freshness
- Validate architecture parameters
- Get deployment recommendations

## Safety Considerations

The EA includes multiple safety mechanisms:
- **Model mismatch protection**: Prevents trading with wrong symbol/timeframe models
- **Risk management**: Account drawdown limits, position size controls
- **Session filtering**: Trading hours and volatility-based pausing
- **Cost awareness**: Realistic transaction cost modeling

## Important Constants

```mql5
#define STATE_SIZE 35    // Number of input features
#define ACTIONS 6        // Number of possible trading actions
```

These constants are fundamental to the architecture and changing them requires retraining all models.

## Configuration Parameters

### EA Parameters (cortex5.mq5)

#### Position Management
- `InpAllowPositionScaling`: Enable/disable dynamic position scaling (default: true)
- `InpUseRiskSizing`: Use ATR-based position sizing vs fixed lots
- `InpRiskPercent`: Risk percentage per trade for position sizing
- `InpATRMultiplier`: ATR multiplier for stop loss calculation
- `InpRRRatio`: Risk-reward ratio for take profit

#### Advanced Risk Controls
- `InpUseTrailingStop`: Enable trailing stop functionality (default: true)
- `InpTrailStartATR`: ATR multiple profit to start trailing (default: 2.0x)
- `InpTrailStopATR`: ATR multiple for trailing stop distance (default: 1.0x)
- `InpUseBreakEven`: Enable break-even stop loss moves (default: true)
- `InpBreakEvenATR`: ATR multiple profit to trigger break-even (default: 1.5x)
- `InpBreakEvenBuffer`: Points buffer beyond break-even (default: 5.0)

#### Volatility Regime Controls
- `InpUseVolatilityRegime`: Enable volatility regime detection (default: true)
- `InpVolRegimeMultiple`: ATR multiple vs median for high volatility (default: 2.5x)
- `InpVolRegimePeriod`: Lookback period for volatility calculation (default: 50)
- `InpVolRegimeSizeReduce`: Position size multiplier during high volatility (default: 0.5)
- `InpVolRegimePauseMin`: Minutes to pause after extreme volatility (default: 15)

### Training Parameters (Cortextrainingv5.mq5)
- `InpTrainPositionScaling`: Train with position scaling behavior (default: true)
- `InpTrainVolatilityRegime`: Train with volatility regime awareness (default: true)
- `InpEpochs`: Number of training passes over the data
- `InpBatch`: Mini-batch size for training
- `InpUsePER`: Enable Prioritized Experience Replay

## Trading Actions

The system now supports 6 distinct trading actions:
- **BUY_STRONG** (0): Open large long position
- **BUY_WEAK** (1): Open small long position  
- **SELL_STRONG** (2): Open large short position
- **SELL_WEAK** (3): Open small short position
- **HOLD** (4): Maintain current position
- **FLAT** (5): Close all positions and stay flat

The new FLAT action provides the AI with explicit control to exit positions without immediately taking opposite positions, improving strategy flexibility and avoiding unnecessary trades.

## Enhanced Position Management

The system now supports dynamic position scaling, allowing the AI to adjust position sizes when signal strength changes:

- **Scale Up**: When moving from weak to strong signal in same direction, EA adds to the position
- **Scale Down**: When moving from strong to weak signal in same direction, EA partially closes the position  
- **Intelligent Costing**: Scaling operations have reduced transaction costs compared to full reversals
- **Training Alignment**: Training script simulates the same scaling behavior for consistency

### Position Scaling Examples
- Currently in 0.5 lot LONG (weak), AI signals BUY_STRONG → EA adds 0.5 lots to reach 1.0 lot position
- Currently in 1.0 lot SHORT (strong), AI signals SELL_WEAK → EA reduces to 0.5 lot position
- Position scaling can be disabled via `InpAllowPositionScaling` parameter

This enhancement allows the system to exploit the distinction between weak and strong signals more effectively, making the strategy more responsive to changing market conditions.

## Advanced Risk Management

The system includes sophisticated risk management features that go beyond basic position controls:

### Trailing Stops and Break-Even
- **Trailing Stops**: Automatically move stop losses to lock in profits as positions become profitable
- **Break-Even Moves**: Move stop loss to entry price + small buffer when position reaches profit threshold
- **ATR-Based Calculation**: Both features use ATR multiples for dynamic adjustment to market volatility

### Volatility Regime Awareness
- **Dynamic Detection**: Continuously monitors current ATR vs historical median to detect high volatility periods
- **Position Size Reduction**: Automatically reduces position sizes during high volatility regimes
- **Trade Suppression**: Temporarily pauses trading during extreme volatility spikes
- **News Event Protection**: Provides additional safety during unexpected market events

### Benefits
- **Profit Protection**: Mechanical trailing stops secure gains even if AI signals don't exit optimally
- **Risk Adaptation**: Position sizing automatically adjusts to changing market conditions
- **Whipsaw Prevention**: High volatility detection prevents trading in unstable conditions
- **Training Alignment**: Training script simulates same risk controls for consistency

## File Safety

All three files contain sophisticated financial trading algorithms. They are designed for defensive use (automated trading with risk controls) and educational purposes. The code includes extensive safety checks and risk management features.

## EA Feature usage
   The EA program uses the same 35-feature state vector as the training script for decision
  making. Here's what information it processes:

  Market Price Data (Features 0-9)

  - Close prices: Current, 1-bar ago, 2-bars ago, 3-bars ago, 4-bars ago (normalized)
  - OHLC ratios: Open/Close, High/Close, Low/Close ratios
  - Price spreads: High-Low spread, Body size (Close-Open)

  Technical Indicators (Features 10-17)

  - Moving averages: 10-period and 50-period MA ratios to current price
  - RSI: 14-period Relative Strength Index (momentum oscillator)
  - ATR Percentile: Average True Range percentile ranking (volatility measure)
  - Bollinger Bands: Position relative to upper/lower bands
  - MACD: Moving Average Convergence Divergence signal
  - Stochastic: %K oscillator value

  Time-Based Features (Features 18)

  - Month half indicator: First half (0.0) vs second half (1.0) of month

  Market Microstructure (Features 19-22)

  - Spread estimate: 15 pips (0.0015) as you configured
  - Volume momentum: Current volume vs 10-bar and 50-bar averages
  - Absolute volume: Scaled volume level

  Price Momentum (Features 23-27)

  - Short-term momentum: 5-bar price momentum
  - Medium-term momentum: 20-bar price momentum
  - Long-term momentum: 50-bar price momentum
  - Momentum acceleration: Rate of change in momentum
  - Momentum consistency: How consistent the momentum direction is

  Multi-Timeframe Context (Features 28-34)

  - M1 trend: 1-minute timeframe trend direction
  - H1 trend: 1-hour timeframe trend direction
  - H4 trend: 4-hour timeframe trend direction
  - Daily trend: Daily timeframe trend direction
  - Cross-timeframe alignment: How aligned trends are across timeframes
  - Higher timeframe momentum: Momentum from higher timeframes
  - Timeframe volatility: Volatility comparison across timeframes

  How It Makes Decisions

  1. Feature Collection: The EA calls BuildStateVector() every tick to collect all 35 features
  2. Neural Network Processing: Feeds the state through the Double-Dueling DRQN:
    - 3 dense layers (64→64→64)
    - Optional LSTM layer (32 units) for sequence memory
    - Dueling heads splitting into Value + Advantage estimates
  3. Action Selection: Chooses from 6 possible actions:
    - 0: BUY_STRONG
    - 1: BUY_WEAK
    - 2: SELL_STRONG
    - 3: SELL_WEAK
    - 4: HOLD
    - 5: FLAT (close positions)
  4. LSTM Memory: Maintains hidden states across decisions to remember market regimes and patterns

  The EA essentially acts as a sophisticated technical analysis system that processes price action, volume,
  multiple technical indicators, and multi-timeframe context to make trading decisions, with the added intelligence
   of LSTM memory for temporal pattern recognition.

#  Potential Enhancement:

  The system could theoretically be enhanced to include sentiment/economic features by:
  - Adding economic calendar parsing
  - Integrating news sentiment APIs
  - Including VIX or currency volatility indices
  - Adding central bank policy indicators

  But currently, it operates as a purely technical trading system that makes decisions based on price action,
  volume, and technical indicators across multiple timeframes.

# Q_VALUES for trading decision:
The cortex5.mq5 EA uses action selection by maximum Q-value - it simply picks the action with the highest Q-value
   among the 6 possible actions. Here's how it works:

  Action Selection Logic:

  // The EA chooses the action with highest Q-value:
  int action = 0;  // Default to BUY_STRONG
  double max_q = q_values[0];
  for(int i = 1; i < ACTIONS; i++){
      if(q_values[i] > max_q){
          max_q = q_values[i];
          action = i;
      }
  }

  Action Mapping:

  - Action 0: BUY_STRONG (aggressive long position)
  - Action 1: BUY_WEAK (conservative long position)
  - Action 2: SELL_STRONG (aggressive short position)
  - Action 3: SELL_WEAK (conservative short position)
  - Action 4: HOLD (maintain current position)
  - Action 5: FLAT (close all positions)

  No Fixed Q-Value Thresholds:

  Important: There are no specific Q-value thresholds that determine actions. The decision is purely relative -
  whichever action has the highest Q-value gets executed, regardless of the absolute values.

  Examples:
  - If Q-values are [0.1, 0.3, -0.2, -0.1, 0.0, -0.5] → Action 1 (BUY_WEAK) chosen
  - If Q-values are [-2.1, -1.8, -0.5, -1.2, -0.8, -0.3] → Action 5 (FLAT) chosen
  - If Q-values are [5.2, 3.1, 2.8, 4.1, 3.5, 1.9] → Action 0 (BUY_STRONG) chosen

  What This Means:

  1. The EA always takes action - it never "does nothing" because it always picks the highest Q-value
  2. Absolute Q-values don't matter - only the relative ranking matters
  3. FLAT positions occur when Action 5 has the highest Q-value compared to all trading actions
  4. Market confidence isn't directly visible - you can't tell how "confident" the EA is from Q-values alone

  Position Sizing Logic:

  The EA uses different lot sizes based on the chosen action:
  - STRONG actions: Larger position sizes (more aggressive)
  - WEAK actions: Smaller position sizes (more conservative)
  - FLAT: Closes all positions regardless of size

  The specific lot sizes depend on the InpLotSize parameter and risk management settings in the EA configuration.
   Complete Q-Value Display:

   - q[0]: BUY_STRONG
   - q[1]: BUY_WEAK
   - q[2]: SELL_STRONG
   - q[3]: SELL_WEAK
   - q[4]: HOLD
   - q[5]: FLAT ← Now visible in logs

   You'll now be able to see the complete picture of the EA's decision-making process, including when the FLAT
   action has a high Q-value that influences the trading decision.

# Detailed System Protocols and Decision Logic

## Training Protocol Specifications

### Data Preparation Protocol
1. **Historical Data Collection**: Training script automatically fetches OHLCV data for specified timeframe and period
2. **Multi-Timeframe Synchronization**: System aligns data across M1, M5, H1, H4, D1 for feature consistency
3. **Data Quality Validation**: Checks for gaps, outliers, and weekend data contamination
4. **Feature Engineering Pipeline**: 35 features calculated in real-time using consistent formulas across training/live
5. **Normalization Strategy**: Min-max scaling to [0,1] range with parameters saved to model file

### Training Loop Architecture
```
For each epoch:
  1. Shuffle experience buffer (if not using PER)
  2. For each mini-batch:
     - Sample experiences (prioritized if PER enabled)
     - Calculate target Q-values using Double DQN
     - Forward pass through main network
     - Calculate TD errors and losses
     - Backward propagation and weight updates
     - Update PER priorities based on TD errors
  3. Decay exploration epsilon
  4. Sync target network every N steps
  5. Save checkpoint with current state
```

### Reinforcement Learning Protocol
- **Exploration Strategy**: Epsilon-greedy with exponential decay (start: 1.0, end: 0.01, decay: 0.995)
- **Experience Replay**: Circular buffer storing (state, action, reward, next_state, done) tuples
- **Prioritized Experience Replay**: TD-error based sampling with importance sampling weights
- **Target Network Updates**: Periodic copying of main network weights for stability
- **Reward Engineering**: Profit-based rewards with transaction cost penalties and volatility adjustments

## Neural Network Processing Pipeline

### Forward Pass Logic
```
1. Input Layer (35 features) → Feature validation and range checking
2. Dense Layer 1 (64 neurons) → ReLU activation → Dropout (0.2)
3. Dense Layer 2 (64 neurons) → ReLU activation → Dropout (0.2)  
4. Dense Layer 3 (64 neurons) → ReLU activation → Optional LSTM input
5. LSTM Layer (optional, 32 hidden) → Sequence processing → Hidden state updates
6. Dueling Split:
   - Value Head (32→1): State value estimation V(s)
   - Advantage Head (32→6): Action advantages A(s,a)
7. Q-Value Combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

### LSTM Memory Management
- **Hidden State Persistence**: Maintains cell state (C) and hidden state (h) between decisions
- **Sequence Buffer**: Rolling 8-step window of recent states for temporal learning
- **Reset Conditions**: Memory cleared on position closure or extreme volatility events
- **Gradient Flow**: Backpropagation through time limited to sequence length

## Decision-Making Protocol

### Q-Value Calculation Logic
The system computes Q-values representing expected future rewards for each action:

```mql5
// Dueling architecture Q-value combination
for(int i = 0; i < ACTIONS; i++) {
    double advantage_mean = 0;
    for(int j = 0; j < ACTIONS; j++) {
        advantage_mean += advantages[j];
    }
    advantage_mean /= ACTIONS;
    
    q_values[i] = state_value + (advantages[i] - advantage_mean);
}
```

### Action Selection Protocol
1. **Q-Value Computation**: Forward pass produces 6 Q-values
2. **Argmax Selection**: Choose action with highest Q-value (no thresholds)
3. **Position State Check**: Verify current position before executing
4. **Risk Validation**: Check account equity, margin, and volatility conditions
5. **Execution Logic**: Route to appropriate position management function

### Position Management Decision Tree
```
Current Position: NONE
├── BUY_STRONG → Open 1.0 lot LONG
├── BUY_WEAK → Open 0.5 lot LONG  
├── SELL_STRONG → Open 1.0 lot SHORT
├── SELL_WEAK → Open 0.5 lot SHORT
├── HOLD → No action (wait for signal)
└── FLAT → No action (already flat)

Current Position: LONG 0.5 lots
├── BUY_STRONG → Add 0.5 lots (scale up to 1.0)
├── BUY_WEAK → Hold current position
├── SELL_STRONG → Close + Open 1.0 lot SHORT
├── SELL_WEAK → Close + Open 0.5 lot SHORT
├── HOLD → Maintain position
└── FLAT → Close position completely

Current Position: LONG 1.0 lots  
├── BUY_STRONG → Hold current position
├── BUY_WEAK → Reduce to 0.5 lots (scale down)
├── SELL_STRONG → Close + Open 1.0 lot SHORT
├── SELL_WEAK → Close + Open 0.5 lot SHORT  
├── HOLD → Maintain position
└── FLAT → Close position completely
```

## Risk Management Protocol

### Volatility Regime Detection
```mql5
// Real-time volatility assessment
double current_atr = iATR(Symbol(), PERIOD_CURRENT, 14, 0);
double historical_median = CalculateATRMedian(InpVolRegimePeriod);
double volatility_ratio = current_atr / historical_median;

if(volatility_ratio > InpVolRegimeMultiple) {
    // High volatility regime detected
    position_size_multiplier = InpVolRegimeSizeReduce; // e.g., 0.5
    trading_paused = true;
    pause_end_time = TimeCurrent() + InpVolRegimePauseMin * 60;
}
```

### Dynamic Stop Loss Calculation
```mql5
// ATR-based stop loss positioning
double atr_value = iATR(Symbol(), PERIOD_CURRENT, 14, 0);
double stop_distance = atr_value * InpATRMultiplier;

// For LONG positions
double stop_loss = entry_price - stop_distance;

// For SHORT positions  
double stop_loss = entry_price + stop_distance;
```

### Trailing Stop Logic
```mql5
// Profit-based trailing activation
double unrealized_profit = (current_price - entry_price) * position_size * point_value;
double atr_profit_threshold = atr_value * InpTrailStartATR;

if(unrealized_profit >= atr_profit_threshold) {
    // Activate trailing stop
    double new_stop = current_price - (atr_value * InpTrailStopATR);
    if(new_stop > current_stop_loss) {
        ModifyStopLoss(new_stop);
    }
}
```

## Model File Protocol

### Save Format Specification
```
Bytes 0-3: Magic number (0xC0DE0203 for enhanced format)
Bytes 4-67: Symbol string (64 chars, null-terminated)
Bytes 68-71: Timeframe (integer)
Bytes 72-75: Layer 1 size (h1)
Bytes 76-79: Layer 2 size (h2) 
Bytes 80-83: Layer 3 size (h3)
Bytes 84: LSTM enabled flag (0/1)
Bytes 85-88: LSTM hidden size
Bytes 89-92: Sequence length
Bytes 93: Dueling network flag (0/1)
Bytes 94-97: Value head size
Bytes 98-101: Advantage head size
Bytes 102-241: Feature min values (35 * 4 bytes)
Bytes 242-381: Feature max values (35 * 4 bytes)
Bytes 382+: Network weights in order:
  - Dense layer 1 weights and biases
  - Dense layer 2 weights and biases
  - Dense layer 3 weights and biases
  - LSTM weights (if enabled)
  - Value head weights and biases
  - Advantage head weights and biases
  - Epsilon value (double)
  - Training steps (long)
  - Last trained timestamp (datetime)
```

### Model Validation Protocol
1. **Magic Number Check**: Verify file format version
2. **Symbol/Timeframe Match**: Ensure model trained on same instrument
3. **Architecture Validation**: Confirm network dimensions match EA settings
4. **Weight Range Check**: Validate weights are within reasonable bounds (-10 to +10)
5. **Freshness Check**: Compare last trained date with current time
6. **Compatibility Check**: Verify STATE_SIZE and ACTIONS constants match

## Feature Engineering Protocol

### Price-Based Features (0-9)
- **Close Price Normalization**: `(close - min_close) / (max_close - min_close)`
- **Historical Close Ratios**: `close[i] / close[0]` for lookback periods 1-4
- **OHLC Ratios**: Body size, wick ratios, and range calculations
- **Price Momentum**: Short-term price velocity measurements

### Technical Indicator Features (10-17)
- **Moving Average Signals**: `(price - ma) / price` for trend detection
- **RSI Normalization**: Direct RSI value / 100 for overbought/oversold
- **ATR Percentile**: Rank current ATR vs 50-period historical distribution
- **Bollinger Position**: `(price - bb_lower) / (bb_upper - bb_lower)`
- **MACD Signal**: Normalized MACD line and signal crossover detection
- **Stochastic %K**: Momentum oscillator for reversal identification

### Multi-Timeframe Features (28-34)
- **Trend Alignment**: Compare price vs MA across M1, H1, H4, D1 timeframes
- **Momentum Consistency**: Correlation of momentum across timeframes
- **Volatility Context**: Compare current TF volatility vs higher TFs
- **Support/Resistance**: Key level identification across timeframes

## Error Handling and Logging Protocol

### Trade Execution Error Handling
```mql5
int attempts = 0;
while(attempts < MAX_RETRIES) {
    if(OrderSend(...) > 0) {
        LogInfo("Trade executed successfully");
        break;
    } else {
        int error = GetLastError();
        LogError("Trade failed: " + ErrorDescription(error));
        attempts++;
        Sleep(1000); // Wait before retry
    }
}
```

### Model Loading Error Recovery
1. **Primary Model Load**: Attempt to load main model file
2. **Backup Model Check**: If primary fails, check for backup model
3. **Architecture Fallback**: Try loading with relaxed architecture validation
4. **Safe Mode Operation**: Disable trading if all model loads fail
5. **User Notification**: Alert user of model loading issues

### Performance Monitoring Protocol
- **Execution Timing**: Log neural network inference times
- **Memory Usage**: Monitor LSTM hidden state memory consumption  
- **Trade Statistics**: Track win rate, average profit, maximum drawdown
- **Model Drift**: Compare recent performance vs training period performance
- **Feature Distribution**: Alert if live features exceed training ranges

## Advanced System Architecture Details

### Memory Management Architecture
The Cortex system employs sophisticated memory management to handle both short-term decision making and long-term pattern recognition:

#### State Buffer Management
```mql5
// Circular buffer for LSTM sequences
class StateBuffer {
    double states[MAX_SEQUENCE_LENGTH][STATE_SIZE];
    int current_index;
    bool buffer_full;
    
    void AddState(double new_state[]) {
        // Copy new state to buffer
        for(int i = 0; i < STATE_SIZE; i++) {
            states[current_index][i] = new_state[i];
        }
        current_index = (current_index + 1) % MAX_SEQUENCE_LENGTH;
        if(current_index == 0) buffer_full = true;
    }
    
    bool GetSequence(double sequence[][STATE_SIZE]) {
        if(!buffer_full && current_index < InpSequenceLen) return false;
        // Return last N states for LSTM processing
        return BuildSequenceArray(sequence);
    }
};
```

#### LSTM Hidden State Persistence
- **Cell State (C_t)**: Long-term memory component persisted between predictions
- **Hidden State (h_t)**: Short-term working memory updated each decision cycle
- **Reset Triggers**: Memory cleared on new session, position closure, or volatility spikes
- **State Validation**: Checks for NaN/infinite values before processing

### Multi-Threading Architecture
The system uses MT5's limited threading capabilities while maintaining thread safety:

#### Timer-Based Processing
```mql5
void OnTimer() {
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(Symbol(), PERIOD_CURRENT, 0);
    
    if(current_bar_time != last_bar_time) {
        // New bar processing
        UpdateTechnicalIndicators();
        BuildMultiTimeframeFeatures();
        last_bar_time = current_bar_time;
    }
    
    // High-frequency decision making
    if(IsNewTick()) {
        ProcessAIDecision();
    }
}
```

#### Asynchronous Data Loading
- **Background Indicator Calculation**: Pre-compute indicators during low-activity periods
- **Multi-Timeframe Buffering**: Maintain synchronized buffers across different timeframes
- **Lazy Loading**: Load historical data only when needed for feature calculation

### Network Architecture Deep Dive

#### Weight Initialization Strategy
```mql5
// Xavier/Glorot initialization for stable training
double InitializeWeight(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return (2.0 * MathRand() / 32767.0 - 1.0) * limit;
}

void InitializeDenseLayer(int input_size, int output_size, double weights[]) {
    for(int i = 0; i < input_size * output_size; i++) {
        weights[i] = InitializeWeight(input_size, output_size);
    }
}
```

#### Activation Function Implementation
```mql5
double ReLU(double x) {
    return MathMax(0.0, x);
}

double LeakyReLU(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double Tanh(double x) {
    return MathTanh(x);  // For LSTM gates
}

double Sigmoid(double x) {
    return 1.0 / (1.0 + MathExp(-x));  // For LSTM gates
}
```

#### LSTM Cell Implementation
```mql5
struct LSTMCell {
    double forget_gate_weights[LSTM_SIZE * (STATE_SIZE + LSTM_SIZE)];
    double input_gate_weights[LSTM_SIZE * (STATE_SIZE + LSTM_SIZE)];
    double candidate_weights[LSTM_SIZE * (STATE_SIZE + LSTM_SIZE)];
    double output_gate_weights[LSTM_SIZE * (STATE_SIZE + LSTM_SIZE)];
    
    void Forward(double input[], double h_prev[], double c_prev[], 
                 double &h_next[], double &c_next[]) {
        // Forget gate: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
        // Input gate: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
        // Candidate: C_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
        // Cell state: C_t = f_t * C_{t-1} + i_t * C_t
        // Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
        // Hidden state: h_t = o_t * tanh(C_t)
    }
};
```

## System Integration Workflows

### Startup Sequence Protocol
1. **System Initialization**
   ```mql5
   int OnInit() {
       // Phase 1: Configuration validation
       if(!ValidateParameters()) return INIT_PARAMETERS_INCORRECT;
       
       // Phase 2: Model loading and validation
       if(!LoadAndValidateModel()) return INIT_FAILED;
       
       // Phase 3: Market data initialization
       InitializeIndicatorBuffers();
       
       // Phase 4: Risk management setup
       InitializeRiskControls();
       
       // Phase 5: State management initialization
       InitializeLSTMStates();
       
       return INIT_SUCCEEDED;
   }
   ```

2. **Model Loading Sequence**
   - File existence verification
   - Magic number validation
   - Architecture compatibility check
   - Weight loading and range validation
   - Normalization parameter loading
   - Checkpoint state restoration

3. **Market Data Synchronization**
   - Multi-timeframe buffer alignment
   - Historical data gap detection
   - Feature calculation initialization
   - Technical indicator warm-up period

### Real-Time Processing Workflow

#### Tick Processing Pipeline
```mql5
void OnTick() {
    // Stage 1: Market data validation
    if(!IsValidTick()) return;
    
    // Stage 2: Feature vector construction
    double state[STATE_SIZE];
    if(!BuildStateVector(state)) {
        LogError("Failed to build state vector");
        return;
    }
    
    // Stage 3: Neural network inference
    double q_values[ACTIONS];
    if(!PredictAction(state, q_values)) {
        LogError("Neural network prediction failed");
        return;
    }
    
    // Stage 4: Action selection and validation
    int selected_action = SelectBestAction(q_values);
    if(!ValidateAction(selected_action)) {
        LogWarning("Action validation failed, using HOLD");
        selected_action = HOLD;
    }
    
    // Stage 5: Risk management filters
    if(!PassesRiskFilters(selected_action)) {
        LogInfo("Action blocked by risk management");
        return;
    }
    
    // Stage 6: Position management execution
    ExecuteAction(selected_action);
}
```

#### Feature Vector Building Process
```mql5
bool BuildStateVector(double &state[]) {
    int feature_index = 0;
    
    // Price features (0-9)
    if(!BuildPriceFeatures(state, feature_index)) return false;
    
    // Technical indicators (10-17)
    if(!BuildTechnicalFeatures(state, feature_index)) return false;
    
    // Time features (18)
    if(!BuildTimeFeatures(state, feature_index)) return false;
    
    // Market microstructure (19-22)
    if(!BuildMicrostructureFeatures(state, feature_index)) return false;
    
    // Momentum features (23-27)
    if(!BuildMomentumFeatures(state, feature_index)) return false;
    
    // Multi-timeframe features (28-34)
    if(!BuildMultiTimeframeFeatures(state, feature_index)) return false;
    
    // Normalize all features to [0,1] range
    return NormalizeFeatures(state);
}
```

### Error Recovery Workflows

#### Network Prediction Failure Recovery
```mql5
bool PredictActionWithFallback(double state[], double q_values[]) {
    // Attempt 1: Standard prediction
    if(PredictAction(state, q_values)) return true;
    
    // Attempt 2: Reset LSTM states and retry
    ResetLSTMStates();
    if(PredictAction(state, q_values)) {
        LogWarning("Recovered from prediction failure by resetting LSTM");
        return true;
    }
    
    // Attempt 3: Use simplified prediction without LSTM
    if(PredictActionNoLSTM(state, q_values)) {
        LogWarning("Using fallback prediction without LSTM");
        return true;
    }
    
    // Final fallback: Use conservative HOLD action
    SetConservativeQValues(q_values);
    LogError("All prediction methods failed, using conservative fallback");
    return false;
}
```

#### Position Management Error Recovery
```mql5
void ExecuteActionWithRetry(int action) {
    int max_attempts = 3;
    int attempt = 0;
    
    while(attempt < max_attempts) {
        if(TryExecuteAction(action)) {
            LogInfo("Action executed successfully on attempt " + (attempt + 1));
            return;
        }
        
        int error_code = GetLastError();
        LogWarning("Action execution failed: " + ErrorDescription(error_code));
        
        // Handle specific error types
        switch(error_code) {
            case ERR_NOT_ENOUGH_MONEY:
                ReducePositionSize();
                break;
            case ERR_INVALID_STOPS:
                RecalculateStopLoss();
                break;
            case ERR_TRADE_CONTEXT_BUSY:
                Sleep(1000);  // Wait for context to become available
                break;
        }
        
        attempt++;
    }
    
    LogError("Failed to execute action after " + max_attempts + " attempts");
}
```

## Performance Optimization Strategies

### Memory Usage Optimization
- **Buffer Recycling**: Reuse arrays instead of frequent allocation/deallocation
- **Lazy Evaluation**: Calculate features only when needed
- **Cache Management**: Store frequently accessed indicator values
- **Memory Pooling**: Pre-allocate fixed-size buffers for predictable memory usage

### CPU Optimization Techniques
- **Vectorized Operations**: Use array operations where possible
- **Indicator Caching**: Cache technical indicator values between ticks
- **Conditional Processing**: Skip expensive calculations when market is closed
- **Batch Processing**: Group similar calculations together

### Network Inference Optimization
```mql5
// Optimized matrix multiplication for dense layers
void OptimizedMatrixMultiply(double input[], double weights[], double bias[], 
                           double output[], int input_size, int output_size) {
    // Unrolled loops for better performance
    for(int i = 0; i < output_size; i++) {
        double sum = bias[i];
        for(int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }
        output[i] = ReLU(sum);  // Apply activation function
    }
}

## Comprehensive Training Protocol Documentation

### Pre-Training Validation Checklist
Before initiating training, the system performs comprehensive validation:

#### Data Integrity Validation
```mql5
bool ValidateTrainingData() {
    // Check data completeness
    if(Bars(InpSymbol, InpTF) < MIN_REQUIRED_BARS) {
        LogError("Insufficient historical data for training");
        return false;
    }
    
    // Validate data quality
    double gap_threshold = 0.1;  // 10% price gap threshold
    for(int i = 1; i < Bars(InpSymbol, InpTF); i++) {
        double price_change = MathAbs(iClose(InpSymbol, InpTF, i-1) - iClose(InpSymbol, InpTF, i)) 
                            / iClose(InpSymbol, InpTF, i);
        if(price_change > gap_threshold) {
            LogWarning("Large price gap detected at bar " + i);
        }
    }
    
    // Weekend data filtering
    FilterWeekendData();
    return true;
}
```

#### Feature Engineering Validation
```mql5
bool ValidateFeatureEngineering() {
    double test_state[STATE_SIZE];
    
    // Test feature calculation at different market conditions
    datetime test_times[] = {
        D'2023.01.15 08:00:00',  // Normal market hours
        D'2023.01.15 00:00:00',  // Off-market hours
        D'2023.12.25 12:00:00'   // Holiday
    };
    
    for(int i = 0; i < ArraySize(test_times); i++) {
        if(!BuildStateVectorAtTime(test_times[i], test_state)) {
            LogError("Feature calculation failed at " + TimeToString(test_times[i]));
            return false;
        }
        
        // Validate feature ranges
        for(int j = 0; j < STATE_SIZE; j++) {
            if(test_state[j] < 0 || test_state[j] > 1) {
                LogError("Feature " + j + " out of range [0,1]: " + test_state[j]);
                return false;
            }
        }
    }
    
    return true;
}
```

### Training Loop Implementation Details

#### Experience Replay Buffer Management
```mql5
class ExperienceReplayBuffer {
    struct Experience {
        double state[STATE_SIZE];
        int action;
        double reward;
        double next_state[STATE_SIZE];
        bool done;
        double td_error;  // For PER
        int timestamp;
    };
    
    Experience buffer[MAX_BUFFER_SIZE];
    int buffer_size;
    int buffer_index;
    double alpha;  // PER priority exponent
    double beta;   // PER importance sampling
    
    void AddExperience(double state[], int action, double reward, 
                      double next_state[], bool done) {
        // Store new experience
        ArrayCopy(buffer[buffer_index].state, state, 0, 0, STATE_SIZE);
        buffer[buffer_index].action = action;
        buffer[buffer_index].reward = reward;
        ArrayCopy(buffer[buffer_index].next_state, next_state, 0, 0, STATE_SIZE);
        buffer[buffer_index].done = done;
        buffer[buffer_index].timestamp = (int)TimeCurrent();
        
        // Calculate initial TD error (will be updated during training)
        buffer[buffer_index].td_error = MathAbs(reward);
        
        buffer_index = (buffer_index + 1) % MAX_BUFFER_SIZE;
        buffer_size = MathMin(buffer_size + 1, MAX_BUFFER_SIZE);
    }
    
    bool SampleBatch(int batch_size, int indices[], double importance_weights[]) {
        if(buffer_size < batch_size) return false;
        
        if(InpUsePER) {
            return SamplePrioritizedBatch(batch_size, indices, importance_weights);
        } else {
            return SampleUniformBatch(batch_size, indices);
        }
    }
};
```

#### Double DQN Target Calculation
```mql5
void CalculateTargetQValues(Experience batch[], double target_q_values[], int batch_size) {
    for(int i = 0; i < batch_size; i++) {
        if(batch[i].done) {
            // Terminal state - only immediate reward
            target_q_values[i] = batch[i].reward;
        } else {
            // Double DQN: use main network for action selection
            double next_q_main[ACTIONS];
            PredictQValues(batch[i].next_state, next_q_main, false); // Use main network
            
            // Find best action according to main network
            int best_action = 0;
            for(int j = 1; j < ACTIONS; j++) {
                if(next_q_main[j] > next_q_main[best_action]) {
                    best_action = j;
                }
            }
            
            // Evaluate best action using target network
            double next_q_target[ACTIONS];
            PredictQValues(batch[i].next_state, next_q_target, true); // Use target network
            
            // Bellman equation with discount factor
            target_q_values[i] = batch[i].reward + GAMMA * next_q_target[best_action];
        }
    }
}
```

### Model Validation and Testing Protocols

#### Walk-Forward Validation
```mql5
struct ValidationResults {
    double total_return;
    double sharpe_ratio;
    double maximum_drawdown;
    double win_rate;
    int total_trades;
    double average_trade_duration;
};

ValidationResults PerformWalkForwardValidation() {
    ValidationResults results = {};
    int validation_periods = 12; // Monthly validation over 1 year
    int period_length = (int)(InpYears * 365.25 * 24 * 60) / validation_periods; // Minutes per period
    
    for(int period = 0; period < validation_periods; period++) {
        datetime start_time = iTime(InpSymbol, InpTF, period_length * (period + 1));
        datetime end_time = iTime(InpSymbol, InpTF, period_length * period);
        
        // Train on data before start_time
        TrainModelUpToTime(start_time);
        
        // Test on data from start_time to end_time
        ValidationResults period_results = BacktestPeriod(start_time, end_time);
        
        // Accumulate results
        results.total_return += period_results.total_return;
        results.total_trades += period_results.total_trades;
        results.maximum_drawdown = MathMax(results.maximum_drawdown, period_results.maximum_drawdown);
    }
    
    // Calculate aggregate metrics
    results.win_rate /= validation_periods;
    results.sharpe_ratio = CalculateOverallSharpe();
    
    return results;
}
```

#### Model Performance Metrics
```mql5
struct ModelMetrics {
    double training_loss;
    double validation_loss;
    double q_value_stability;
    double feature_importance[STATE_SIZE];
    double prediction_confidence;
    double temporal_consistency;
};

ModelMetrics EvaluateModelPerformance() {
    ModelMetrics metrics = {};
    
    // Calculate training loss over recent batch
    metrics.training_loss = CalculateAverageLoss(last_training_batch);
    
    // Validation on holdout data
    metrics.validation_loss = CalculateValidationLoss();
    
    // Q-value stability: measure variance in Q-values for similar states
    metrics.q_value_stability = CalculateQValueStability();
    
    // Feature importance through perturbation analysis
    CalculateFeatureImportance(metrics.feature_importance);
    
    // Prediction confidence based on Q-value spreads
    metrics.prediction_confidence = CalculatePredictionConfidence();
    
    // Temporal consistency: how often consecutive predictions agree
    metrics.temporal_consistency = CalculateTemporalConsistency();
    
    return metrics;
}
```

### Model Diagnostic Procedures

#### Network Weight Analysis
```mql5
void DiagnoseNetworkWeights() {
    // Check for exploding/vanishing gradients
    double layer1_norm = CalculateWeightNorm(dense_layer1_weights);
    double layer2_norm = CalculateWeightNorm(dense_layer2_weights);
    double layer3_norm = CalculateWeightNorm(dense_layer3_weights);
    
    LogInfo("Layer weight norms: L1=" + layer1_norm + " L2=" + layer2_norm + " L3=" + layer3_norm);
    
    if(layer1_norm > 10.0 || layer2_norm > 10.0 || layer3_norm > 10.0) {
        LogWarning("Potential exploding gradients detected");
    }
    
    if(layer1_norm < 0.01 || layer2_norm < 0.01 || layer3_norm < 0.01) {
        LogWarning("Potential vanishing gradients detected");
    }
    
    // Analyze weight distribution
    AnalyzeWeightDistribution();
    
    // Check for dead neurons
    DetectDeadNeurons();
}
```

#### LSTM State Analysis
```mql5
void DiagnoseLSTMStates() {
    // Check for state saturation
    for(int i = 0; i < LSTM_SIZE; i++) {
        if(MathAbs(lstm_hidden_state[i]) > 0.95) {
            LogWarning("LSTM hidden state " + i + " near saturation: " + lstm_hidden_state[i]);
        }
        
        if(MathAbs(lstm_cell_state[i]) > 10.0) {
            LogWarning("LSTM cell state " + i + " potentially unstable: " + lstm_cell_state[i]);
        }
    }
    
    // Analyze state utilization
    CalculateLSTMStateUtilization();
    
    // Check memory retention capability
    TestMemoryRetention();
}
```

### Advanced Training Techniques

#### Curriculum Learning Implementation
```mql5
void ImplementCurriculumLearning() {
    // Start with simple market conditions
    TrainingPhase phases[] = {
        {TRENDING_MARKETS, 1000},     // Phase 1: Clear trends
        {RANGING_MARKETS, 1500},      // Phase 2: Ranging conditions
        {VOLATILE_MARKETS, 2000},     // Phase 3: High volatility
        {MIXED_CONDITIONS, 3000}      // Phase 4: All conditions
    };
    
    for(int phase = 0; phase < ArraySize(phases); phase++) {
        LogInfo("Starting curriculum phase " + (phase + 1) + ": " + phases[phase].description);
        
        // Filter training data for this phase
        FilterDataForPhase(phases[phase].market_type);
        
        // Train for specified number of episodes
        for(int episode = 0; episode < phases[phase].episodes; episode++) {
            TrainEpisode();
            
            // Gradual difficulty increase within phase
            AdjustDifficultyLevel(episode, phases[phase].episodes);
        }
        
        // Validate phase completion
        if(!ValidatePhaseCompletion(phase)) {
            LogWarning("Phase " + (phase + 1) + " validation failed, repeating...");
            phase--; // Repeat current phase
        }
    }
}
```

#### Adaptive Learning Rate Scheduling
```mql5
void UpdateLearningRate(int training_step, double current_loss) {
    static double best_loss = DBL_MAX;
    static int steps_without_improvement = 0;
    static double current_lr = INITIAL_LEARNING_RATE;
    
    // Cosine annealing with warm restarts
    double base_lr = INITIAL_LEARNING_RATE * 0.5 * (1 + MathCos(MathPI * training_step / TOTAL_TRAINING_STEPS));
    
    // Plateau detection and reduction
    if(current_loss < best_loss) {
        best_loss = current_loss;
        steps_without_improvement = 0;
    } else {
        steps_without_improvement++;
    }
    
    if(steps_without_improvement > PATIENCE_STEPS) {
        current_lr *= LR_DECAY_FACTOR;
        steps_without_improvement = 0;
        LogInfo("Learning rate reduced to: " + current_lr);
    }
    
    // Apply final learning rate
    SetLearningRate(MathMin(base_lr, current_lr));
}
```

## Deployment and Production Monitoring

### Model Deployment Checklist
1. **Pre-Deployment Validation**
   - Model file integrity check
   - Architecture compatibility verification
   - Performance benchmarking against baseline
   - Risk management parameter validation
   - Backtesting on recent unseen data

2. **Deployment Process**
   ```mql5
   bool DeployModel() {
       // Backup existing model
       if(!BackupCurrentModel()) {
           LogError("Failed to backup current model");
           return false;
       }
       
       // Load and validate new model
       if(!LoadAndValidateNewModel()) {
           LogError("New model validation failed");
           RestorePreviousModel();
           return false;
       }
       
       // Gradual rollout with safety monitoring
       EnableGradualRollout();
       
       // Start production monitoring
       StartProductionMonitoring();
       
       return true;
   }
   ```

### Real-Time Performance Monitoring
```mql5
struct PerformanceMetrics {
    double daily_pnl;
    double sharpe_ratio_1d;
    double max_drawdown_1d;
    double prediction_accuracy;
    double feature_drift_score;
    double execution_latency_ms;
    int failed_predictions;
    datetime last_updated;
};

void MonitorLivePerformance() {
    static PerformanceMetrics daily_metrics = {};
    static datetime last_reset = TimeCurrent();
    
    // Daily reset
    if(TimeDay(TimeCurrent()) != TimeDay(last_reset)) {
        LogDailyPerformanceReport(daily_metrics);
        ResetDailyMetrics(daily_metrics);
        last_reset = TimeCurrent();
    }
    
    // Real-time monitoring
    UpdatePnLMetrics(daily_metrics);
    CheckFeatureDrift(daily_metrics);
    MonitorExecutionLatency(daily_metrics);
    
    // Alert conditions
    if(daily_metrics.max_drawdown_1d > MAX_ALLOWED_DRAWDOWN) {
        AlertCritical("Daily drawdown exceeded limit: " + daily_metrics.max_drawdown_1d);
        PauseTrading();
    }
    
    if(daily_metrics.feature_drift_score > DRIFT_THRESHOLD) {
        AlertWarning("Feature drift detected, model may need retraining");
    }
}
```

### Important Instruction Reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
```