# Parameter Optimization Guide (Improvement 7.4)

## Overview
This guide explains how to use Improvement 7.4 "Parameter Flexibility" to systematically optimize the Cortex trading strategy. The flexible parameter system provides comprehensive controls for strategy variants, risk management, filtering, and position sizing without requiring code changes.

## What is Parameter Flexibility?

Parameter Flexibility enables:
- **What-If Analysis**: Test different strategy variants without coding
- **Systematic Optimization**: Use MetaTrader 5 Strategy Tester for parameter sweeps
- **Risk Management Control**: Adjust risk parameters for different market conditions
- **Filter Configuration**: Enable/disable and tune various trading filters
- **Position Sizing Options**: Multiple dynamic sizing methods
- **Session Control**: Time-based trading restrictions
- **Robustness Testing**: Built-in optimization and testing features

This transforms the backtester into a flexible optimization tool for professional strategy development.

## Parameter Groups Overview

### 1. Risk Management Parameters
Controls overall risk exposure and protection mechanisms:

```mql5
// === RISK MANAGEMENT PARAMETERS ===
input bool    InpFlexRiskEnabled         = true;   // Enable flexible risk management controls
input double  InpRiskPercentage          = 2.0;    // Account risk percentage per trade (1.0-5.0)
input double  InpStopLossATR             = 2.0;    // Stop loss as ATR multiplier (1.0-4.0)
input double  InpTakeProfitATR           = 3.0;    // Take profit as ATR multiplier (2.0-6.0)
input double  InpTrailingStopATR         = 1.5;    // Trailing stop as ATR multiplier (1.0-3.0)
input bool    InpUseFixedStops           = false;  // Use fixed pip stops instead of ATR
input double  InpFixedStopPips           = 20.0;   // Fixed stop loss in pips (10-50)
input double  InpFixedTakeProfitPips     = 40.0;   // Fixed take profit in pips (20-100)
input double  InpMaxLossPerDay           = 500.0;  // Maximum daily loss limit ($)
input double  InpMaxDrawdownStop         = 20.0;   // Auto-stop at drawdown % (10-30)
input int     InpMaxConsecutiveLosses    = 5;      // Max consecutive losses before halt (3-10)
```

**Key Parameters for Optimization:**
- **InpRiskPercentage**: Primary risk control (1-5%)
- **InpStopLossATR**: Market-adaptive stop distance (1-4 ATR)
- **InpTakeProfitATR**: Risk-reward optimization (2-6 ATR)

### 2. Trading Filter Parameters
Controls signal quality and entry conditions:

```mql5
// === TRADING FILTERS ===
input bool    InpFlexFiltersEnabled      = true;   // Enable flexible trading filters
input bool    InpConfidenceFilterOn      = true;   // Enable confidence-based filtering
input double  InpConfidenceThreshold     = 0.65;   // Minimum confidence to trade (0.5-0.9)
input bool    InpVolatilityFilterOn      = true;   // Enable volatility filtering
input double  InpMinVolatilityATR        = 0.0005; // Minimum ATR to trade (0.0001-0.002)
input double  InpMaxVolatilityATR        = 0.005;  // Maximum ATR to trade (0.003-0.01)
input bool    InpSpreadFilterOn          = true;   // Enable spread filtering
input double  InpMaxSpreadATR            = 0.3;    // Max spread as % of ATR (0.1-0.5)
input bool    InpTrendFilterOn           = false;  // Enable trend alignment filter
input double  InpTrendFilterPeriod       = 50.0;   // Trend filter MA period (20-200)
input bool    InpNewsFilterOn            = false;  // Enable news avoidance filter
input int     InpNewsAvoidMinutes        = 30;     // Minutes to avoid before/after news (15-60)
```

**Key Parameters for Optimization:**
- **InpConfidenceThreshold**: Signal quality gate (0.5-0.9)
- **InpMinVolatilityATR/InpMaxVolatilityATR**: Market condition filtering
- **InpTrendFilterOn**: Trend alignment requirement

### 3. Position Sizing Parameters
Controls dynamic position sizing methods:

```mql5
// === POSITION SIZING ===
input bool    InpFlexSizingEnabled       = true;   // Enable flexible position sizing
input double  InpBaseLotSize             = 0.1;    // Base lot size (0.01-1.0)
input double  InpMaxLotSize              = 0.5;    // Maximum lot size (0.1-2.0)
input bool    InpVolatilityBasedSizing   = true;   // Size based on volatility
input double  InpVolatilitySizeMultiplier = 1.0;   // Volatility sizing multiplier (0.5-2.0)
input bool    InpConfidenceBasedSizing   = true;   // Size based on confidence
input double  InpConfidenceSizeMultiplier = 1.5;   // Confidence sizing multiplier (1.0-3.0)
input bool    InpEquityBasedSizing       = false;  // Size based on account equity
input double  InpEquitySizeMultiplier    = 1.0;    // Equity sizing multiplier (0.5-2.0)
input bool    InpPartialClosingEnabled   = false;  // Enable partial position closing
input double  InpPartialClosePercent     = 50.0;   // Partial close percentage (25-75)
input double  InpPartialCloseProfitLevel = 1.5;    // Profit level for partial close (ATR multiplier)
```

**Key Parameters for Optimization:**
- **InpVolatilityBasedSizing**: Reduces size in high volatility
- **InpConfidenceBasedSizing**: Increases size for high-confidence signals
- **InpVolatilitySizeMultiplier/InpConfidenceSizeMultiplier**: Sizing aggressiveness

### 4. Session and Time Parameters
Controls when trading is allowed:

```mql5
// === SESSION AND TIME FILTERS ===
input bool    InpFlexTimeEnabled         = true;   // Enable flexible time controls
input bool    InpSessionFilterOn         = true;   // Enable session filtering
input int     InpSessionStartHour        = 8;      // Trading session start hour (0-23)
input int     InpSessionEndHour          = 18;     // Trading session end hour (0-23)
input bool    InpMondayTradingOn         = true;   // Enable Monday trading
input bool    InpTuesdayTradingOn        = true;   // Enable Tuesday trading
input bool    InpWednesdayTradingOn      = true;   // Enable Wednesday trading
input bool    InpThursdayTradingOn       = true;   // Enable Thursday trading
input bool    InpFridayTradingOn         = true;   // Enable Friday trading
input bool    InpWeekendTradingOn        = false;  // Enable weekend trading
input int     InpMinTradeInterval        = 5;      // Minimum minutes between trades (1-60)
input int     InpMaxTradesPerHour        = 10;     // Maximum trades per hour (1-20)
input bool    InpAvoidFridayClosing      = true;   // Avoid trading 2 hours before Friday close
input bool    InpAvoidMondayOpening      = false;  // Avoid trading 2 hours after Monday open
```

**Key Parameters for Optimization:**
- **InpSessionStartHour/InpSessionEndHour**: Optimal trading hours
- **Day trading controls**: Day-of-week performance analysis
- **InpMinTradeInterval**: Anti-whipsaw protection

### 5. Signal Quality Parameters
Controls signal validation and confirmation:

```mql5
// === SIGNAL QUALITY CONTROLS ===
input bool    InpFlexSignalEnabled      = true;   // Enable flexible signal controls
input double  InpSignalStrengthMin      = 0.1;    // Minimum signal strength (0.05-0.3)
input double  InpSignalStrengthMax      = 1.0;    // Maximum signal strength (0.8-1.5)
input bool    InpSignalConfirmationOn   = false;  // Require signal confirmation
input int     InpConfirmationBars       = 2;      // Bars for signal confirmation (1-5)
input bool    InpAntiWhipsawOn          = true;   // Enable anti-whipsaw protection
input double  InpMinPriceMovement       = 5.0;    // Minimum price movement in pips (2-20)
input bool    InpMultiTimeframeOn       = false;  // Enable multi-timeframe confirmation
input string  InpMTFTimeframes          = "H1,H4"; // MTF timeframes (comma separated)
input double  InpMTFAgreementLevel      = 0.7;    // MTF agreement level (0.5-1.0)
```

**Key Parameters for Optimization:**
- **InpSignalStrengthMin/Max**: Signal quality range
- **InpSignalConfirmationOn**: Signal persistence requirement
- **InpAntiWhipsawOn**: Rapid reversal protection

### 6. Advanced Feature Parameters
Controls sophisticated market analysis features:

```mql5
// === ADVANCED FEATURES ===
input bool    InpFlexAdvancedEnabled    = true;   // Enable advanced feature controls
input bool    InpMarketRegimeOn         = false;  // Enable market regime detection
input double  InpRegimeVolThreshold     = 1.5;    // Regime volatility threshold (1.2-2.0)
input double  InpRegimeTrendThreshold   = 0.7;    // Regime trend threshold (0.5-0.9)
input bool    InpCorrelationFilterOn    = false;  // Enable correlation filtering
input double  InpMaxCorrelationLevel    = 0.8;    // Maximum correlation level (0.6-0.95)
input bool    InpVolumeFilterOn         = false;  // Enable volume filtering
input double  InpMinVolumeLevel         = 0.5;    // Minimum volume level (0.3-1.0)
input bool    InpMomentumFilterOn       = false;  // Enable momentum filtering
input double  InpMomentumThreshold      = 0.6;    // Momentum threshold (0.4-0.8)
input bool    InpSeasonalityOn          = false;  // Enable seasonality adjustments
input string  InpSeasonalPatterns       = "EUR_MORNING,USD_AFTERNOON"; // Seasonal patterns
```

**Key Parameters for Optimization:**
- **InpMarketRegimeOn**: Adaptive behavior for different market conditions
- **InpCorrelationFilterOn**: Multiple market analysis
- **InpMomentumFilterOn**: Momentum-based filtering

### 7. Optimization Control Parameters
Controls testing and robustness features:

```mql5
// === OPTIMIZATION CONTROLS ===
input bool    InpOptimizationMode       = false;  // Enable optimization mode features
input double  InpRandomSeed             = 12345.0; // Random seed for reproducible tests (1-99999)
input bool    InpDataShuffleOn          = false;  // Enable data shuffling for robustness testing
input double  InpDataShufflePercent     = 5.0;    // Percentage of data to shuffle (1-20)
input bool    InpSpreadRandomization    = false;  // Enable spread randomization
input double  InpSpreadVariationPercent = 10.0;   // Spread variation percentage (5-25)
input bool    InpSlippageSimulation     = false;  // Enable slippage simulation
input double  InpSlippageMaxPips        = 2.0;    // Maximum slippage in pips (0.5-5.0)
input bool    InpCommissionSimulation   = true;   // Enable commission simulation
input double  InpCommissionPerLot       = 7.0;    // Commission per lot in account currency (3-15)
input bool    InpParameterSetSaving     = true;   // Save parameter sets to file
input string  InpParameterSetName       = "Default_Set"; // Parameter set name for saving
```

**Key Parameters for Optimization:**
- **InpOptimizationMode**: Enables robustness testing features
- **InpRandomSeed**: Reproducible optimization runs
- **InpParameterSetSaving**: Track successful parameter combinations

## Optimization Workflow

### Phase 1: Baseline Establishment
1. **Run Default Parameters**: Establish baseline performance
2. **Validate System**: Ensure all features work correctly
3. **Document Baseline**: Record initial metrics for comparison

```mql5
// Recommended baseline settings
InpRiskPercentage = 2.0;
InpStopLossATR = 2.0;
InpTakeProfitATR = 3.0;
InpConfidenceThreshold = 0.65;
InpBaseLotSize = 0.1;
InpMaxLotSize = 0.5;
```

### Phase 2: Risk Management Optimization
Focus on core risk parameters first:

#### Risk Percentage Optimization
```mql5
// MT5 Strategy Tester Optimization Setup
Parameter: InpRiskPercentage
Start: 1.0
Step: 0.25
Stop: 4.0
```

#### Stop Loss Optimization
```mql5
// Optimize stop distance
Parameter: InpStopLossATR
Start: 1.0
Step: 0.2
Stop: 3.5
```

#### Take Profit Optimization
```mql5
// Optimize reward-to-risk ratio
Parameter: InpTakeProfitATR  
Start: 2.0
Step: 0.5
Stop: 5.0
```

### Phase 3: Filter Optimization
Optimize entry quality filters:

#### Confidence Threshold
```mql5
// Find optimal confidence level
Parameter: InpConfidenceThreshold
Start: 0.5
Step: 0.05
Stop: 0.85
```

#### Volatility Range
```mql5
// Optimize volatility window
Parameter: InpMinVolatilityATR
Start: 0.0001
Step: 0.0002
Stop: 0.001

Parameter: InpMaxVolatilityATR
Start: 0.003
Step: 0.001
Stop: 0.008
```

### Phase 4: Position Sizing Optimization
Optimize dynamic sizing methods:

#### Volatility-Based Sizing
```mql5
// Test volatility sensitivity
Parameter: InpVolatilitySizeMultiplier
Start: 0.5
Step: 0.1
Stop: 1.5
```

#### Confidence-Based Sizing  
```mql5
// Test confidence scaling
Parameter: InpConfidenceSizeMultiplier
Start: 1.0
Step: 0.2
Stop: 2.5
```

### Phase 5: Session Optimization
Optimize trading hours and frequency:

#### Session Hours
```mql5
// Find optimal trading window
Parameter: InpSessionStartHour
Start: 6
Step: 1
Stop: 10

Parameter: InpSessionEndHour
Start: 16
Step: 1
Stop: 20
```

#### Trade Frequency
```mql5
// Optimize trade spacing
Parameter: InpMinTradeInterval
Start: 1
Step: 5
Stop: 30

Parameter: InpMaxTradesPerHour
Start: 5
Step: 2
Stop: 15
```

### Phase 6: Advanced Feature Testing
Test sophisticated features:

#### Market Regime Detection
```mql5
// Enable regime detection
InpMarketRegimeOn = true;

// Optimize thresholds
Parameter: InpRegimeVolThreshold
Start: 1.2
Step: 0.1
Stop: 2.0
```

#### Multi-Filter Combinations
```mql5
// Test filter combinations
InpConfidenceFilterOn = true;
InpVolatilityFilterOn = true;
InpTrendFilterOn = true;

// Optimize combined thresholds
```

## Strategy Tester Setup

### Basic Optimization Setup
1. **Open Strategy Tester**: Tools → Strategy Tester
2. **Select Expert**: CortexBacktestWorking
3. **Choose Symbol**: EURUSD (or target symbol)
4. **Set Timeframe**: M5 (or target timeframe)
5. **Select Dates**: Choose optimization period
6. **Set Model**: Every tick (for accuracy)

### Optimization Configuration
1. **Go to Optimization Tab**
2. **Check "Optimization"**
3. **Select Parameters**: Choose parameters to optimize
4. **Set Optimization Criterion**: 
   - **Balance**: For absolute profit
   - **Profit Factor**: For risk-adjusted performance
   - **Sharpe Ratio**: For risk-adjusted returns (if available)
   - **Custom**: Use comprehensive metrics

### Advanced Optimization Settings
```mql5
// Use genetic algorithm for complex optimizations
Optimization Method: Genetic Algorithm
Maximum Passes: 1000-5000 (depending on parameters)
```

### Robustness Testing Setup
```mql5
// Enable robustness testing
InpOptimizationMode = true;
InpDataShuffleOn = true;
InpDataShufflePercent = 10.0;
InpSpreadRandomization = true;
InpSpreadVariationPercent = 15.0;
InpSlippageSimulation = true;
InpSlippageMaxPips = 1.5;
```

## Analysis and Validation

### Parameter Sensitivity Analysis
1. **Single Parameter Sweeps**: Test each parameter individually
2. **Multi-Parameter Optimization**: Test parameter interactions
3. **Sensitivity Testing**: Test small variations around optimal values
4. **Robustness Validation**: Use shuffle and randomization testing

### Performance Metrics Analysis
Use comprehensive metrics (7.2) to evaluate optimization results:

#### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.5)
- **Maximum Drawdown**: Risk measure (target: <15%)
- **Profit Factor**: Trade quality (target: >1.5)
- **Win Rate**: Signal accuracy (target: >45%)

#### Secondary Metrics
- **Sortino Ratio**: Downside risk focus
- **Calmar Ratio**: Return vs maximum drawdown
- **Kelly Criterion**: Optimal position sizing guidance
- **Recovery Factor**: Drawdown recovery ability

### CSV Analysis Integration
Use CSV logging (7.3) for detailed analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load optimization results
trades = pd.read_csv('Cortex_Trades.csv')
metrics = pd.read_csv('Cortex_Metrics.csv')

# Analyze parameter effectiveness
confidence_analysis = trades.groupby(pd.cut(trades['ConfidenceScore'], 10)).agg({
    'PnL': ['mean', 'count'],
    'Duration': 'mean'
})

# Plot parameter sensitivity
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(trades['ConfidenceScore'], trades['PnL'])
plt.title('Confidence vs PnL')
plt.xlabel('Confidence Score')
plt.ylabel('Profit/Loss')
```

## Best Practices

### Optimization Guidelines
1. **Start Simple**: Begin with basic parameters before advanced features
2. **One Group at a Time**: Optimize parameter groups sequentially
3. **Validate Results**: Use out-of-sample testing for validation
4. **Document Everything**: Save parameter sets and results
5. **Avoid Overfitting**: Use walk-forward analysis

### Parameter Selection Strategy
1. **Risk First**: Establish acceptable risk levels before profit optimization
2. **Quality over Quantity**: Fewer high-quality trades beat many poor trades
3. **Market Adaptation**: Different parameters for different market conditions
4. **Robustness Testing**: Parameters should work across different conditions

### Common Pitfalls to Avoid
1. **Over-Optimization**: Too many parameters can lead to curve fitting
2. **Ignoring Transaction Costs**: Always include realistic costs
3. **Single Period Optimization**: Test across multiple time periods
4. **Ignoring Risk**: Don't optimize for return without considering risk
5. **Parameter Correlation**: Understand how parameters interact

### Optimization Schedule
#### Daily Optimization
- Monitor key risk parameters
- Adjust for current market conditions
- Check emergency stops and limits

#### Weekly Optimization  
- Review and adjust filters
- Analyze session performance
- Update position sizing parameters

#### Monthly Optimization
- Comprehensive parameter review
- Multi-parameter optimization
- Strategy robustness testing
- Walk-forward validation

#### Quarterly Optimization
- Full strategy reoptimization
- Market regime analysis
- Parameter stability testing
- Strategy evolution planning

## Integration with Other Improvements

### Comprehensive Metrics (7.2)
- Use 50+ metrics for parameter evaluation
- Focus on risk-adjusted returns
- Monitor drawdown patterns
- Track performance consistency

### CSV Logging (7.3)
- Export detailed parameter impact data
- Analyze trade-by-trade parameter effectiveness
- Create parameter performance reports
- Track optimization history

### EA Synchronization (7.1)
- Ensure parameters work in both backtester and live EA
- Maintain consistent behavior across systems
- Validate parameter changes in live environment
- Use unified risk management logic

## Conclusion

The Parameter Flexibility system transforms strategy optimization from manual code changes to systematic parameter tuning. By providing comprehensive controls over all aspects of strategy behavior, it enables:

- **Professional Strategy Development**: Systematic optimization methodology
- **Risk Management Excellence**: Precise risk control and adaptation
- **Market Condition Adaptation**: Different parameters for different conditions
- **Robustness Validation**: Built-in testing for strategy stability
- **Operational Efficiency**: Quick what-if analysis without coding

This system enables confident strategy deployment through systematic optimization and validation, ensuring robust performance across different market conditions.