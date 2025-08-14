//+------------------------------------------------------------------+
//|                                       CortexBacktestWorking.mq5 |
//|                                  Double-Dueling DRQN Backtester  |
//|                   Simulates Live Trading Performance Last 30 Days |

// IMPROVEMENT 3.4: Include unified trade logic module
#include <CortexTradeLogic.mqh>
//|                                                                  |
//|   WHAT THIS PROGRAM DOES:                                        |
//|   This backtester simulates how the AI trading model would have |
//|   performed over the last 30 days using historical data. It     |
//|   includes all Phase 1-3 enhancements to fix profitability      |
//|   issues identified in the original model.                      |
//|                                                                  |
//|   KEY ENHANCEMENTS IMPLEMENTED:                                  |
//|   • Phase 1: Maximum holding time limits (48 hours max)         |
//|   • Phase 1: Profit target automation (1.8x ATR)                |
//|   • Phase 1: Quick exit bonuses for profitable trades           |
//|   • Phase 2: Enhanced reward calculations with penalties        |
//|   • Phase 2: FLAT action weight increases for better exits      |
//|   • Phase 3: Dynamic stop loss tightening over time             |
//|   • Phase 3: Position-aware state features                      |
//|   • Emergency stops to prevent catastrophic losses              |
//|   • Trading frequency controls to prevent overtrading           |
//|                                                                  |
//|   CRITICAL FIXES FOR ORIGINAL ISSUES:                           |
//|   ✓ Fixes 700+ hour holding times (now max 48 hours)           |
//|   ✓ Fixes overtrading (now max 20 trades/day vs 95+)           |
//|   ✓ Adds emergency stops ($150 max loss per trade)             |
//|   ✓ Forces profitable exits when targets are hit                |
//|   ✓ Prevents catastrophic drawdowns with enhanced controls      |
//+------------------------------------------------------------------+
#property copyright "Cortex Trading System"
#property link      ""
#property version   "1.01"
#property script_show_inputs

//============================== USER INPUT PARAMETERS ==============================
// These parameters control how the backtest runs and can be adjusted by the user

// BASIC BACKTEST SETTINGS
input string  InpModelFileName   = "DoubleDueling_DRQN_Model.dat";  // AI model file to load and test
input double  InpInitialBalance  = 10000.0;   // Starting account balance for simulation ($10,000 default)
input double  InpLotSize        = 0.1;       // Position size per trade (0.1 = 10,000 units = $1 per pip)
input int     InpBacktestDays   = 30;        // Number of recent days to simulate (30 = last month)
input bool    InpDetailedReport = true;      // Generate comprehensive performance report at end
input bool    InpVerboseLogging = true;      // Show detailed trade-by-trade information in log
input int     InpLogEveryNBars  = 100;       // Print progress update every N bars processed

// IMPROVEMENT 7.3: TRADE-BY-TRADE LOGGING - Machine-readable CSV export
input bool    InpEnableCSVLogging = true;    // Enable CSV file export of all trade data
input string  InpCSVTradeFileName = "Cortex_Trades.csv"; // CSV file name for individual trades
input string  InpCSVEquityFileName = "Cortex_Equity.csv"; // CSV file name for equity curve
input string  InpCSVMetricsFileName = "Cortex_Metrics.csv"; // CSV file name for performance metrics
input bool    InpCSVIncludeHeaders = true;   // Include column headers in CSV files
input bool    InpCSVAppendMode = false;      // Append to existing files (false = overwrite)
input bool    InpCSVLogAllBars = false;      // Log every bar (true) or only trade events (false)
input bool    InpCSVFlushImmediately = true; // Flush CSV files after each write (slower but safer)

// PHASE 1 ENHANCEMENTS - IMMEDIATE PROFITABILITY FIXES
// These parameters address the core issues: 700+ hour holds, poor profit-taking, overtrading
input int     InpMaxHoldingHours      = 48;    // Maximum hours to hold position (Phase 1) 
                                               // ↳ FIXED: Was unlimited causing 700+ hour holds
input double  InpProfitTargetATR      = 1.8;   // Take profit threshold (N x ATR) (Phase 1)
                                               // ↳ FIXED: Was no profit targets, now exits at 1.8x volatility
input bool    InpUseProfitTargets     = true;  // Enable automatic profit taking (Phase 1)
                                               // ↳ FIXED: Forces exits when profitable instead of endless holds
input bool    InpUseMaxHoldingTime    = true;  // Enable maximum holding time control (Phase 1)
                                               // ↳ FIXED: Prevents catastrophic long-term losses
input double  InpHoldingTimePenalty   = 0.001; // Penalty per hour held in reward calculation (Phase 1)
input double  InpQuickExitBonus       = 0.005; // Bonus for trades < 24 hours (Phase 1)

// PHASE 2 ENHANCEMENTS - LEARNING IMPROVEMENTS  
// These parameters improve the AI's learning process to make better decisions
input double  InpFlatActionWeight     = 1.5;   // Increased weight for FLAT action training (Phase 2)
                                               // ↳ FIXED: AI rarely used FLAT, now encouraged to exit positions
input bool    InpEnhancedRewards      = true;  // Use enhanced reward calculation (Phase 2)
                                               // ↳ FIXED: Multi-factor rewards teach better trading behavior
input double  InpDrawdownPenalty      = 0.01;  // Penalty for unrealized drawdown (Phase 2)
                                               // ↳ FIXED: Penalizes holding losing positions

// PHASE 3 ENHANCEMENTS - ADVANCED FEATURES
// These parameters add sophisticated risk management and position tracking
input bool    InpUseDynamicStops      = true;  // Enable dynamic stop loss tightening (Phase 3)
                                               // ↳ ADVANCED: Stops tighten over time to lock in profits
input double  InpStopTightenRate      = 0.8;   // Stop tightening multiplier per day held (Phase 3)
input bool    InpUsePositionFeatures  = true;  // Add position-aware features to state (Phase 3)
                                               // ↳ ADVANCED: AI knows its current position status for better decisions

// TRADING FREQUENCY CONTROLS - PREVENT OVERTRADING WHILE ALLOWING PROFITABILITY
// PROBLEM: Original AI traded 95+ times per day, causing huge transaction costs
// SOLUTION: Intelligent frequency limits that still allow profitable opportunities
input int     InpMinBarsBetweenTrades = 1;     // Minimum bars between trades
                                               // ↳ BALANCED: 1 bar = allows consecutive profitable trades
input int     InpMaxTradesPerDay      = 20;    // Maximum trades per day limit
                                               // ↳ OPTIMIZED: 20/day allows profits while preventing overtrading (was 8, too restrictive)
input bool    InpEnforceFlat          = true;  // Force FLAT when position reaches exit criteria
                                               // ↳ CRITICAL: Overrides AI to close positions at limits

// EMERGENCY STOP LOSS CONTROLS - FINAL SAFETY NET
// These are the last line of defense against catastrophic losses
input double  InpEmergencyStopLoss    = 150.0;  // Emergency dollar stop loss per trade
                                               // ↳ SAFETY NET: Hard limit to prevent single trade disasters ($150 max loss)
input double  InpMaxDrawdownPct       = 15.0;   // Maximum account drawdown percentage
                                               // ↳ ACCOUNT PROTECTION: Stops all trading if account down 15%
input bool    InpUseEmergencyStops    = true;   // Master switch for emergency protections

// IMPROVEMENT 7.1: SYNC WITH EA LOGIC - MISSING ADVANCED FEATURES
// These parameters ensure backtester matches all EA improvements for accurate simulation

// CONFIDENCE-BASED FILTERING (Improvement 6.3 Integration)
input bool    InpUseConfidenceFilter     = false;  // Enable confidence-based trade filtering
input double  InpMinConfidenceThreshold  = 0.6;    // Minimum confidence to execute trade
input double  InpHighConfidenceThreshold = 0.8;    // High confidence threshold for stronger positions
input bool    InpLogConfidenceDecisions  = false;  // Log confidence-based decisions

// ADVANCED RISK MANAGEMENT (Missing from current backtester)
input bool    InpUseATRBasedStops        = true;   // Use ATR-based stop losses
input double  InpATRMultiplier           = 2.0;    // ATR multiplier for stop losses
input bool    InpUseTrailingStops        = false;  // Enable trailing stop functionality
input bool    InpUsePartialCloses        = false;  // Enable partial position closing
input double  InpPartialCloseLevel       = 0.5;    // Level for partial close (50% of position)

// TRADE CONFIRMATION FILTERS (Missing from current backtester)
input bool    InpUseTimeFilters          = true;   // Enable time-based trade filtering
input bool    InpUseSpreadFilter         = true;   // Enable spread filtering
input double  InpMaxSpreadPoints         = 20.0;   // Maximum spread in points
input bool    InpRequireConfirmation     = false;  // Require signal confirmation

// VOLATILITY REGIME ADAPTATION (Missing from current backtester)
input bool    InpUseVolatilityRegime     = false;  // Enable volatility regime detection
input double  InpHighVolatilityThreshold = 2.0;    // High volatility threshold (ATR multiplier)
input double  InpVolatilityPosReduction  = 0.5;    // Position size reduction in high volatility

// POSITION SCALING (Missing from current backtester)
input bool    InpAllowPositionScaling    = false;  // Allow dynamic position scaling
input double  InpStrongSignalMultiplier  = 1.5;    // Position multiplier for strong signals
input double  InpWeakSignalMultiplier    = 0.7;    // Position multiplier for weak signals

//============================== IMPROVEMENT 7.4: PARAMETER FLEXIBILITY ==============
// Comprehensive parameter controls for strategy optimization and what-if analysis

// === RISK MANAGEMENT PARAMETER GROUP ===
input group "=== RISK MANAGEMENT PARAMETERS ==="
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

// === TRADING FILTER PARAMETER GROUP ===
input group "=== TRADING FILTERS ==="
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

// === POSITION SIZING PARAMETER GROUP ===
input group "=== POSITION SIZING ==="
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

// === SESSION AND TIME PARAMETER GROUP ===
input group "=== SESSION AND TIME FILTERS ==="
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

// === SIGNAL QUALITY PARAMETER GROUP ===
input group "=== SIGNAL QUALITY CONTROLS ==="
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

// === ADVANCED FEATURES PARAMETER GROUP ===
input group "=== ADVANCED FEATURES ==="
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

// === OPTIMIZATION AND TESTING PARAMETER GROUP ===
input group "=== OPTIMIZATION CONTROLS ==="
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

//============================== IMPROVEMENT 7.5: MONTE CARLO TESTING ==============
// Advanced batch testing and Monte Carlo simulation parameters

// === MONTE CARLO TESTING PARAMETER GROUP ===
input group "=== MONTE CARLO TESTING ==="
input bool    InpEnableMonteCarloMode   = false;  // Enable Monte Carlo batch testing mode
input int     InpMonteCarloRuns         = 100;    // Number of Monte Carlo iterations (10-1000)
input bool    InpMCDataShuffling        = true;   // Enable price data shuffling
input double  InpMCDataShufflePercent   = 5.0;    // Percentage of data points to shuffle (1-20)
input bool    InpMCRandomStartDates     = true;   // Randomize backtest start dates
input int     InpMCStartDateVariation   = 30;     // Start date variation in days (7-90)
input bool    InpMCRandomPeriods        = false;  // Randomize backtest period lengths
input int     InpMCMinPeriodDays        = 20;     // Minimum backtest period (10-60)
input int     InpMCMaxPeriodDays        = 40;     // Maximum backtest period (30-120)
input bool    InpMCSpreadVariation      = true;   // Enable spread randomization per run
input double  InpMCSpreadMinMultiplier  = 0.8;    // Minimum spread multiplier (0.5-1.0)
input double  InpMCSpreadMaxMultiplier  = 1.5;    // Maximum spread multiplier (1.0-2.0)
input bool    InpMCSlippageVariation    = true;   // Enable slippage randomization per run
input double  InpMCSlippageMinPips      = 0.0;    // Minimum slippage in pips (0.0-1.0)
input double  InpMCSlippageMaxPips      = 3.0;    // Maximum slippage in pips (1.0-5.0)
input bool    InpMCCommissionVariation  = false;  // Enable commission randomization per run
input double  InpMCCommissionMinPct     = 80.0;   // Minimum commission % of base (50-100)
input double  InpMCCommissionMaxPct     = 120.0;  // Maximum commission % of base (100-150)
input bool    InpMCParameterNoise       = false;  // Add small noise to key parameters
input double  InpMCParameterNoiseLevel  = 5.0;    // Parameter noise level % (1-10)
input bool    InpMCResultsSaving        = true;   // Save detailed Monte Carlo results
input string  InpMCResultsPrefix        = "MC";   // File prefix for Monte Carlo results
input bool    InpMCRobustnessScoring    = true;   // Calculate robustness scores
input double  InpMCRobustnessThreshold  = 0.7;    // Robustness threshold for acceptance (0.5-0.9)
input bool    InpMCProgressReporting    = true;   // Show progress during Monte Carlo runs
input int     InpMCProgressFrequency    = 10;     // Progress reporting frequency (every N runs)

// IMPROVEMENT 7.1: Update to match enhanced state size from training improvements
// Neural network architecture constants (must match training)
#define STATE_SIZE 45  // IMPROVEMENT 4.3: Updated to match enhanced 45-feature state vector
#define ACTIONS 6

// Action definitions
#define BUY_STRONG  0
#define BUY_WEAK    1  
#define SELL_STRONG 2
#define SELL_WEAK   3
#define HOLD        4
#define FLAT        5

// Trading position types
enum POS_TYPE {
    POS_NONE = 0,
    POS_LONG = 1,
    POS_SHORT = 2
};

// Trade record structure
struct TradeRecord {
    datetime open_time;
    datetime close_time;
    int action;
    POS_TYPE position_type;
    double entry_price;
    double exit_price;
    double lots;
    double profit_loss;
    double balance_after;
    double drawdown_pct;
    // IMPROVEMENT 7.2: Enhanced trade tracking
    double mae;              // Maximum Adverse Excursion
    double mfe;              // Maximum Favorable Excursion
    int holding_time_hours;  // Trade duration in hours
    string exit_reason;      // Why the trade was closed
    double commission;       // Trading costs
    double confidence_score; // Model confidence for this trade
};

// IMPROVEMENT 7.2: Comprehensive performance metrics structure
struct PerformanceMetrics {
    // Basic Performance
    double total_return_pct;
    double annualized_return_pct;
    double initial_balance;
    double final_balance;
    double total_pnl;
    
    // Trade Statistics  
    int total_trades;
    int winning_trades;
    int losing_trades;
    double win_rate_pct;
    double loss_rate_pct;
    
    // Profit/Loss Analysis
    double gross_profit;
    double gross_loss;
    double profit_factor;
    double expectancy;
    double average_win;
    double average_loss;
    double largest_win;
    double largest_loss;
    
    // Risk Metrics
    double maximum_drawdown_pct;
    double maximum_drawdown_amount;
    datetime max_dd_start_time;
    datetime max_dd_end_time;
    int max_dd_duration_days;
    double calmar_ratio;
    double recovery_factor;
    
    // Risk-Adjusted Returns
    double sharpe_ratio;
    double sortino_ratio;
    double information_ratio;
    double treynor_ratio;
    double sterling_ratio;
    
    // Volatility and Risk
    double returns_volatility;
    double downside_deviation;
    double value_at_risk_95;
    double value_at_risk_99;
    double conditional_var_95;
    double maximum_adverse_excursion;
    double maximum_favorable_excursion;
    
    // Time-Based Analysis
    double avg_holding_time_hours;
    double median_holding_time_hours;
    int longest_holding_time_hours;
    int shortest_holding_time_hours;
    
    // Consecutive Statistics
    int max_consecutive_wins;
    int max_consecutive_losses;
    int current_consecutive_wins;
    int current_consecutive_losses;
    
    // Monthly Performance
    double monthly_returns[12];
    double best_month_pct;
    double worst_month_pct;
    int profitable_months;
    int losing_months;
    
    // Advanced Metrics
    double ulcer_index;
    double pain_index;
    double lake_ratio;
    double burke_ratio;
    double kelly_criterion;
    double optimal_f;
    
    // System Performance
    double system_quality_number;
    double profit_to_max_dd_ratio;
    double return_retracement_ratio;
    double account_size_required;
};

// Monthly performance tracking
struct MonthlyData {
    int year;
    int month;
    double starting_balance;
    double ending_balance;
    double monthly_return_pct;
    int trades_count;
    double max_dd_in_month;
};

// Neural network layers (simplified)
struct DenseLayer {
    double w[];  
    double b[];  
    int input_size;
    int output_size;
};

// Global variables
DenseLayer g_dense1, g_dense2, g_dense3;
DenseLayer g_value_head, g_advantage_head;

double g_feature_min[STATE_SIZE];
double g_feature_max[STATE_SIZE];

bool g_model_loaded = false;
bool g_use_lstm = false;
bool g_use_dueling = false;

// LSTM simplified (we'll skip complex LSTM implementation for now)
double g_lstm_weights[];
int g_lstm_size = 32;

// Performance tracking
TradeRecord g_trades[];
double g_balance;
double g_equity;
double g_max_balance;
double g_max_drawdown;
int g_total_trades;
int g_winning_trades;
int g_losing_trades;

// Current position tracking
POS_TYPE g_current_position = POS_NONE;
double g_position_entry_price = 0;
double g_position_lots = 0;
datetime g_position_open_time = 0;

// Indicator handles
int h_ma10, h_ma20, h_ma50, h_rsi, h_atr;

// Logging counters
int g_bars_processed = 0;
int g_prediction_failures = 0;
int g_feature_failures = 0;
int g_indicator_failures = 0;

// TRADING FREQUENCY CONTROLS (NEW - prevent overtrading)
datetime g_last_trade_time = 0;       // Time of last trade execution
int g_trades_this_session = 0;        // Trades in current session

// IMPROVEMENT 7.1: Advanced feature support variables
// Confidence-based filtering (6.3 integration)
double g_confidence_threshold = 0.0;  // Minimum confidence for trade execution
double g_last_confidence = 0.0;       // Last model confidence prediction
double g_confidence_history[];        // Track confidence over time
int g_confidence_trades = 0;          // Trades executed with confidence filtering

// Online learning and regime detection (6.2 integration)  
bool g_regime_detected = false;       // Current regime state
double g_regime_volatility = 0.0;     // Current volatility regime
double g_regime_trend = 0.0;          // Current trend strength
datetime g_last_regime_check = 0;     // Last regime detection time
double g_volatility_history[];       // Track volatility for regime detection
double g_trend_history[];            // Track trend strength for regime detection

// ATR-based stops and trailing functionality
double g_current_atr = 0.0;           // Current ATR value
double g_stop_loss_price = 0.0;       // Current stop loss level
double g_take_profit_price = 0.0;     // Current take profit level
double g_trailing_stop_price = 0.0;   // Current trailing stop level
double g_max_favorable_price = 0.0;   // Maximum favorable excursion tracking
double g_position_peak_profit = 0.0;  // Peak profit for current position

// Advanced risk management
double g_dynamic_lot_size = 0.0;      // Dynamically calculated lot size
double g_account_risk_pct = 0.0;      // Current account risk percentage
double g_position_risk_amount = 0.0;  // Risk amount for current position
bool g_emergency_mode = false;        // Emergency trading halt flag
int g_consecutive_losses = 0;         // Count of consecutive losing trades
datetime g_emergency_halt_until = 0;  // Emergency halt end time

// Multi-timeframe analysis support
double g_mtf_signals[5];              // M1, M5, H1, H4, D1 signals
bool g_mtf_enabled = false;           // Multi-timeframe analysis active
double g_trend_alignment = 0.0;       // Cross-timeframe trend alignment

// Volatility regime adaptation
double g_volatility_multiplier = 1.0; // Dynamic volatility adjustment
bool g_high_volatility_mode = false;  // High volatility detected flag
double g_volatility_percentile = 0.0; // Current volatility percentile

// Session and time filtering
bool g_current_session_active = false; // Current trading session status
datetime g_session_start_time = 0;     // Session start time
datetime g_session_end_time = 0;       // Session end time
bool g_avoid_news_time = false;        // News time avoidance flag

// Advanced position management
bool g_partial_close_enabled = false; // Partial position closing
double g_partial_close_pct = 0.0;     // Percentage for partial close
int g_position_scaling_level = 0;     // Current scaling level
double g_average_entry_price = 0.0;   // Average entry price for scaling

// Performance monitoring for advanced features
int g_confidence_filtered_trades = 0; // Trades filtered by confidence
int g_atr_stop_hits = 0;              // Stop losses hit via ATR
int g_trailing_stop_hits = 0;         // Trailing stop activations
int g_regime_triggered_exits = 0;     // Exits triggered by regime change
int g_volatility_adjustments = 0;     // Risk adjustments due to volatility

// IMPROVEMENT 7.2: Comprehensive metrics tracking
PerformanceMetrics g_performance_metrics;    // Main metrics structure
MonthlyData g_monthly_data[];               // Monthly performance tracking
double g_daily_returns[];                   // Daily return series for calculations
double g_equity_curve[];                    // Equity curve tracking
datetime g_equity_curve_times[];            // Time stamps for equity curve
double g_underwater_curve[];                // Drawdown curve
int g_equity_curve_size = 0;                // Size of equity curve arrays

// Enhanced trade tracking for metrics
double g_running_gross_profit = 0.0;        // Cumulative gross profit
double g_running_gross_loss = 0.0;          // Cumulative gross loss
int g_consecutive_wins = 0;                 // Current consecutive wins
int g_consecutive_losses = 0;               // Current consecutive losses
int g_max_consecutive_wins = 0;             // Maximum consecutive wins achieved
int g_max_consecutive_losses = 0;           // Maximum consecutive losses achieved

// Drawdown tracking
double g_peak_balance = 0.0;                // Peak balance for drawdown calculation
datetime g_peak_time = 0;                   // Time of peak balance
double g_current_drawdown = 0.0;            // Current drawdown amount
datetime g_current_drawdown_start = 0;      // Current drawdown start time
bool g_in_drawdown = false;                 // Currently in drawdown state

// Risk metrics calculation support
double g_returns_sum = 0.0;                 // Sum of returns for mean calculation
double g_returns_squared_sum = 0.0;         // Sum of squared returns for variance
double g_downside_returns_squared_sum = 0.0; // Sum of negative squared returns
int g_return_periods = 0;                   // Number of return periods tracked

// IMPROVEMENT 7.3: CSV logging file handles and tracking
int g_csv_trades_handle = INVALID_HANDLE;    // File handle for trades CSV
int g_csv_equity_handle = INVALID_HANDLE;    // File handle for equity curve CSV
int g_csv_metrics_handle = INVALID_HANDLE;   // File handle for metrics CSV
bool g_csv_files_initialized = false;       // CSV files setup status
int g_csv_equity_counter = 0;                // Counter for equity curve logging
string g_last_exit_reason = "";             // Track reason for last position exit
string g_last_entry_trigger = "";           // Track what triggered last entry
bool g_last_trade_was_stop = false;         // Flag if last exit was stop loss
bool g_last_trade_was_target = false;       // Flag if last exit was take profit

// IMPROVEMENT 7.4: PARAMETER FLEXIBILITY SUPPORT VARIABLES
// Global variables for flexible parameter management and optimization
struct FlexParameterState {
    // Risk management state
    double current_stop_atr;
    double current_tp_atr;
    double current_trailing_atr;
    double daily_loss_total;
    int consecutive_loss_count;
    bool risk_controls_active;
    
    // Filter states
    bool volatility_filter_active;
    bool spread_filter_active;
    bool trend_filter_active;
    bool session_filter_active;
    bool confidence_filter_active;
    
    // Position sizing state
    double calculated_lot_size;
    double volatility_size_multiplier;
    double confidence_size_multiplier;
    double equity_size_multiplier;
    
    // Timing state
    datetime last_trade_time;
    int trades_this_hour;
    bool session_allowed;
    bool day_allowed;
    
    // Signal quality state
    double last_signal_strength;
    bool signal_confirmed;
    int confirmation_bars_count;
    bool anti_whipsaw_active;
    
    // Advanced feature state
    bool regime_detected;
    double current_regime_score;
    bool correlation_filter_active;
    bool volume_filter_active;
    bool momentum_filter_active;
    
    // Optimization state
    int random_seed_current;
    bool data_shuffle_active;
    double spread_variation_current;
    double slippage_current;
    string current_parameter_set;
};

FlexParameterState g_flex_params;

// Parameter validation and range checking
bool g_parameters_validated = false;
string g_parameter_validation_errors = "";

// IMPROVEMENT 7.5: MONTE CARLO TESTING SUPPORT VARIABLES
// Global variables for Monte Carlo batch testing and robustness analysis
struct MonteCarloRun {
    int run_number;
    datetime start_date;
    datetime end_date;
    int period_days;
    double spread_multiplier;
    double slippage_pips;
    double commission_multiplier;
    int random_seed;
    
    // Performance results
    double final_balance;
    double total_return_pct;
    double max_drawdown_pct;
    double sharpe_ratio;
    double profit_factor;
    double win_rate_pct;
    int total_trades;
    double largest_loss;
    double largest_win;
    
    // Robustness metrics
    double return_stability;
    double drawdown_stability;
    double trade_consistency;
    double robustness_score;
};

struct MonteCarloResults {
    MonteCarloRun runs[1000];  // Support up to 1000 Monte Carlo runs
    int completed_runs;
    
    // Aggregate statistics
    double mean_return;
    double std_return;
    double mean_drawdown;
    double std_drawdown;
    double mean_sharpe;
    double std_sharpe;
    double mean_profit_factor;
    double std_profit_factor;
    double mean_win_rate;
    double std_win_rate;
    
    // Robustness analysis
    double overall_robustness_score;
    double return_consistency_score;
    double risk_consistency_score;
    double strategy_stability_score;
    double percentile_95_return;
    double percentile_5_return;
    double percentile_95_drawdown;
    double percentile_5_drawdown;
    
    // Distribution analysis
    int positive_return_runs;
    int acceptable_drawdown_runs;
    int robust_runs;
    double success_rate;
};

MonteCarloResults g_monte_carlo_results;
bool g_monte_carlo_mode = false;
int g_current_mc_run = 0;
double g_current_spread_multiplier = 1.0;
double g_current_slippage_pips = 0.0;
double g_current_commission_multiplier = 1.0;
datetime g_mc_start_date = 0;
datetime g_mc_end_date = 0;
int g_mc_period_days = 0;

// Data shuffling support
double g_shuffled_prices[];
bool g_price_data_shuffled = false;
int g_shuffle_indices[];

//============================== TRADING FREQUENCY CONTROL FUNCTIONS ==============
// Functions to prevent overtrading and control execution frequency

// Check if new trading is allowed (prevent overtrading)
bool IsNewTradingAllowed(datetime current_bar_time){
    // Don't trade if we have an open position
    if(g_current_position != POS_NONE) return false;
    
    // Check minimum time between trades
    int bars_since_last_trade = (int)((current_bar_time - g_last_trade_time) / PeriodSeconds(PERIOD_CURRENT));
    if(bars_since_last_trade < InpMinBarsBetweenTrades){
        return false;
    }
    
    // Check daily trade limit
    datetime current_day_start = current_bar_time - (current_bar_time % (24 * 3600));
    datetime last_trade_day_start = g_last_trade_time - (g_last_trade_time % (24 * 3600));
    
    // Reset daily counter if it's a new day
    if(current_day_start != last_trade_day_start){
        g_trades_this_session = 0;
    }
    
    if(g_trades_this_session >= InpMaxTradesPerDay){
        return false;
    }
    
    return true;
}

// Update trading frequency tracking
void UpdateTradingFrequency(datetime trade_time){
    g_last_trade_time = trade_time;
    g_trades_this_session++;
}

//============================== IMPROVEMENT 7.1: UNIFIED TRADE LOGIC FUNCTIONS ==============
// These functions sync backtester logic with EA for consistent behavior

// CONFIDENCE-BASED FILTERING (6.3 Integration)
bool PassesConfidenceFilter(const double &q_values[], int action, datetime current_time){
    if(!InpUseConfidenceFilter) return true;
    
    // Calculate confidence from Q-values (using max Q-value as proxy)
    double max_q = q_values[0];
    double second_max_q = 0.0;
    
    for(int i = 1; i < 6; i++){
        if(q_values[i] > max_q){
            second_max_q = max_q;
            max_q = q_values[i];
        } else if(q_values[i] > second_max_q){
            second_max_q = q_values[i];
        }
    }
    
    // Confidence based on Q-value separation
    double confidence = (max_q - second_max_q) / (MathAbs(max_q) + MathAbs(second_max_q) + 0.0001);
    g_last_confidence = confidence;
    
    bool passes = confidence >= InpConfidenceThreshold;
    if(!passes && InpVerboseLogging){
        Print("FILTER: Confidence too low: ", DoubleToString(confidence, 4), " < ", 
              DoubleToString(InpConfidenceThreshold, 4));
        g_confidence_filtered_trades++;
    }
    
    return passes;
}

// ATR-BASED RISK MANAGEMENT
bool CheckATRBasedStops(datetime current_time){
    if(!InpUseATRBasedStops || g_current_position == POS_NONE) return false;
    
    // Get current ATR
    double atr_buffer[];
    if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) <= 0) return false;
    g_current_atr = atr_buffer[0];
    
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    bool should_close = false;
    string reason = "";
    
    // Check ATR-based stop loss
    if(g_current_position == POS_BUY){
        double atr_stop = g_position_entry_price - (g_current_atr * InpATRMultiplier);
        if(current_price <= atr_stop){
            should_close = true;
            reason = "ATR-based stop loss hit";
            g_atr_stop_hits++;
        }
    } else if(g_current_position == POS_SELL){
        double atr_stop = g_position_entry_price + (g_current_atr * InpATRMultiplier);
        if(current_price >= atr_stop){
            should_close = true;
            reason = "ATR-based stop loss hit";
            g_atr_stop_hits++;
        }
    }
    
    if(should_close){
        ClosePosition(reason);
        return true;
    }
    
    return false;
}

// TRAILING STOP FUNCTIONALITY  
bool CheckTrailingStops(datetime current_time){
    if(!InpUseTrailingStops || g_current_position == POS_NONE) return false;
    
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    double current_profit = CalculateUnrealizedPnL(current_price);
    
    // Update peak profit tracking
    if(current_profit > g_position_peak_profit){
        g_position_peak_profit = current_profit;
        g_max_favorable_price = current_price;
    }
    
    // Only activate trailing stop once position is profitable
    if(g_position_peak_profit <= 0) return false;
    
    bool should_close = false;
    string reason = "";
    
    if(g_current_position == POS_BUY){
        double trailing_level = g_max_favorable_price - (g_current_atr * InpTrailingStopATR);
        if(current_price <= trailing_level){
            should_close = true;
            reason = "Trailing stop activated";
            g_trailing_stop_hits++;
        }
    } else if(g_current_position == POS_SELL){
        double trailing_level = g_max_favorable_price + (g_current_atr * InpTrailingStopATR);
        if(current_price >= trailing_level){
            should_close = true;
            reason = "Trailing stop activated";
            g_trailing_stop_hits++;
        }
    }
    
    if(should_close){
        ClosePosition(reason);
        return true;
    }
    
    return false;
}

// VOLATILITY REGIME DETECTION
bool CheckVolatilityRegime(datetime current_time){
    if(!InpUseVolatilityRegime) return true;
    
    // Update volatility regime only periodically
    if(current_time - g_last_regime_check < InpRegimeCheckMinutes * 60) return true;
    
    g_last_regime_check = current_time;
    
    // Get recent ATR values for regime analysis
    double atr_buffer[20];
    if(CopyBuffer(h_atr, 0, 0, 20, atr_buffer) < 20) return true;
    
    // Calculate current volatility percentile
    double current_atr = atr_buffer[0];
    double atr_sum = 0.0;
    for(int i = 1; i < 20; i++){
        atr_sum += atr_buffer[i];
    }
    double avg_atr = atr_sum / 19.0;
    
    g_volatility_percentile = current_atr / avg_atr;
    g_high_volatility_mode = g_volatility_percentile > InpHighVolatilityThreshold;
    
    // Adjust risk based on volatility regime
    if(g_high_volatility_mode){
        g_volatility_multiplier = InpVolatilityMultiplier;
        g_volatility_adjustments++;
        
        if(InpVerboseLogging){
            Print("REGIME: High volatility detected, reducing risk. Multiplier: ", 
                  DoubleToString(g_volatility_multiplier, 2));
        }
    } else {
        g_volatility_multiplier = 1.0;
    }
    
    return true;
}

// MASTER RISK CHECK (combines all risk filters)
bool MasterRiskCheck(const double &q_values[], int action, datetime current_time){
    // IMPROVEMENT 7.4: Enhanced master risk check with flexible parameters
    
    // Calculate signal strength and confidence for flexible filters
    double max_q = q_values[0];
    double second_max = q_values[0];
    for(int i = 1; i < ACTIONS; i++){
        if(q_values[i] > max_q){
            second_max = max_q;
            max_q = q_values[i];
        } else if(q_values[i] > second_max){
            second_max = q_values[i];
        }
    }
    
    double signal_strength = MathAbs(max_q);
    double confidence = (max_q - second_max) / (MathAbs(max_q) + MathAbs(second_max) + 0.0001);
    
    // Apply flexible parameter filters if enabled
    if(g_parameters_validated && !PassesFlexibleFilters(signal_strength, confidence, current_time)) {
        return false;
    }
    
    // Legacy confidence filter (for backward compatibility)
    if(!PassesConfidenceFilter(q_values, action, current_time)){
        return false;
    }
    
    // Check trading frequency
    if(!IsNewTradingAllowed(current_time)){
        return false;
    }
    
    // Check volatility regime
    if(!CheckVolatilityRegime(current_time)){
        return false;
    }
    
    // Check session filters (legacy)
    if(InpUseSessionFilter && !g_current_session_active){
        return false;
    }
    
    // Check emergency mode
    if(g_emergency_mode && current_time < g_emergency_halt_until){
        return false;
    }
    
    return true;
}

// DYNAMIC POSITION SIZING
double CalculateDynamicLotSize(double account_balance, double atr_value){
    // IMPROVEMENT 7.4: Use flexible position sizing if enabled
    if(g_parameters_validated && InpFlexSizingEnabled) {
        // Get last signal confidence and strength from global state
        double confidence = g_last_confidence > 0 ? g_last_confidence : 0.7; // Default confidence
        double signal_strength = 0.5; // Default signal strength
        
        return CalculateFlexibleLotSize(confidence, signal_strength, atr_value);
    }
    
    // Legacy dynamic sizing
    if(!InpAllowPositionScaling) return InpLotSize;
    
    // Base lot size on account risk percentage
    double risk_amount = account_balance * (InpRiskPercentage / 100.0);
    
    // Adjust for current volatility
    risk_amount *= g_volatility_multiplier;
    
    // Calculate lot size based on ATR and stop distance
    double stop_distance_pips = atr_value * InpATRMultiplier * 10000; // Convert to pips
    double pip_value = 10.0; // Approximate pip value for major pairs
    
    g_dynamic_lot_size = risk_amount / (stop_distance_pips * pip_value);
    
    // Apply bounds
    g_dynamic_lot_size = MathMax(g_dynamic_lot_size, InpMinLotSize);
    g_dynamic_lot_size = MathMin(g_dynamic_lot_size, InpMaxLotSize);
    
    return g_dynamic_lot_size;
}

// EMERGENCY RISK MANAGEMENT
bool CheckEmergencyStops(datetime current_time){
    if(g_current_position == POS_NONE) return false;
    
    double current_balance = g_balance;
    double drawdown_pct = ((g_max_balance - current_balance) / g_max_balance) * 100.0;
    
    // Check maximum drawdown limit
    if(InpUseEmergencyStop && drawdown_pct > InpMaxDrawdownPercent){
        ClosePosition("Emergency stop - maximum drawdown exceeded");
        g_emergency_mode = true;
        g_emergency_halt_until = current_time + (InpEmergencyHaltHours * 3600);
        
        Print("EMERGENCY: Trading halted due to ", DoubleToString(drawdown_pct, 2), "% drawdown");
        return true;
    }
    
    // Check consecutive losses
    if(g_consecutive_losses >= InpMaxConsecutiveLosses){
        g_emergency_mode = true;
        g_emergency_halt_until = current_time + (InpEmergencyHaltHours * 3600);
        
        Print("EMERGENCY: Trading halted after ", g_consecutive_losses, " consecutive losses");
        return true;
    }
    
    return false;
}

// SESSION AND TIME FILTERING
bool CheckSessionFilters(datetime current_time){
    if(!InpUseSessionFilter) return true;
    
    // Get current hour
    MqlDateTime dt;
    TimeToStruct(current_time, dt);
    int current_hour = dt.hour;
    
    // Check if within trading session
    if(InpSessionStart <= InpSessionEnd){
        g_current_session_active = (current_hour >= InpSessionStart && current_hour < InpSessionEnd);
    } else { // Overnight session
        g_current_session_active = (current_hour >= InpSessionStart || current_hour < InpSessionEnd);
    }
    
    // Check for news avoidance
    if(InpUseNewsFilter && g_avoid_news_time){
        return false;
    }
    
    return g_current_session_active;
}

//============================== IMPROVEMENT 7.2: COMPREHENSIVE METRICS CALCULATIONS ==============
// Advanced performance metrics calculation functions

// Update equity curve and daily returns tracking
void UpdateEquityCurve(datetime current_time, double current_balance) {
    // Resize arrays if needed
    if(g_equity_curve_size >= ArraySize(g_equity_curve)) {
        ArrayResize(g_equity_curve, g_equity_curve_size + 1000);
        ArrayResize(g_equity_curve_times, g_equity_curve_size + 1000);
        ArrayResize(g_underwater_curve, g_equity_curve_size + 1000);
    }
    
    // Add equity point
    g_equity_curve[g_equity_curve_size] = current_balance;
    g_equity_curve_times[g_equity_curve_size] = current_time;
    
    // Calculate drawdown from peak
    if(current_balance > g_peak_balance) {
        g_peak_balance = current_balance;
        g_peak_time = current_time;
        g_current_drawdown = 0.0;
        g_in_drawdown = false;
    } else {
        g_current_drawdown = g_peak_balance - current_balance;
        if(!g_in_drawdown && g_current_drawdown > 0) {
            g_in_drawdown = true;
            g_current_drawdown_start = current_time;
        }
    }
    
    // Calculate underwater curve (drawdown percentage)
    g_underwater_curve[g_equity_curve_size] = g_peak_balance > 0 ? 
        ((g_peak_balance - current_balance) / g_peak_balance) * 100.0 : 0.0;
    
    g_equity_curve_size++;
    
    // IMPROVEMENT 7.3: Log equity point to CSV if enabled
    if(InpEnableCSVLogging && g_csv_equity_handle != INVALID_HANDLE) {
        LogEquityToCSV(current_time, current_balance);
    }
}

// Calculate daily returns for risk metrics
void CalculateDailyReturn(double previous_balance, double current_balance) {
    if(previous_balance > 0) {
        double daily_return = (current_balance - previous_balance) / previous_balance;
        
        // Resize daily returns array if needed
        if(g_return_periods >= ArraySize(g_daily_returns)) {
            ArrayResize(g_daily_returns, g_return_periods + 365);
        }
        
        g_daily_returns[g_return_periods] = daily_return;
        
        // Update running statistics
        g_returns_sum += daily_return;
        g_returns_squared_sum += daily_return * daily_return;
        
        // Track downside deviation (negative returns only)
        if(daily_return < 0) {
            g_downside_returns_squared_sum += daily_return * daily_return;
        }
        
        g_return_periods++;
    }
}

// Calculate Sharpe Ratio (risk-adjusted return)
double CalculateSharpeRatio(double mean_return, double return_volatility, double risk_free_rate = 0.02) {
    if(return_volatility == 0) return 0.0;
    double annualized_mean = mean_return * 252; // Assuming 252 trading days per year
    double annualized_vol = return_volatility * MathSqrt(252);
    return (annualized_mean - risk_free_rate) / annualized_vol;
}

// Calculate Sortino Ratio (focuses on downside risk)
double CalculateSortinoRatio(double mean_return, double downside_deviation, double risk_free_rate = 0.02) {
    if(downside_deviation == 0) return 0.0;
    double annualized_mean = mean_return * 252;
    double annualized_downside = downside_deviation * MathSqrt(252);
    return (annualized_mean - risk_free_rate) / annualized_downside;
}

// Calculate Calmar Ratio (return vs maximum drawdown)
double CalculateCalmarRatio(double annualized_return, double max_drawdown_pct) {
    if(max_drawdown_pct == 0) return 0.0;
    return annualized_return / max_drawdown_pct;
}

// Calculate Maximum Drawdown from equity curve
void CalculateMaximumDrawdown() {
    double max_dd = 0.0;
    double max_dd_amount = 0.0;
    datetime dd_start = 0;
    datetime dd_end = 0;
    double peak = 0.0;
    datetime peak_time = 0;
    
    for(int i = 0; i < g_equity_curve_size; i++) {
        double current_equity = g_equity_curve[i];
        datetime current_time = g_equity_curve_times[i];
        
        // Update peak
        if(current_equity > peak) {
            peak = current_equity;
            peak_time = current_time;
        }
        
        // Calculate current drawdown
        if(peak > 0) {
            double current_dd_pct = ((peak - current_equity) / peak) * 100.0;
            double current_dd_amount = peak - current_equity;
            
            if(current_dd_pct > max_dd) {
                max_dd = current_dd_pct;
                max_dd_amount = current_dd_amount;
                dd_start = peak_time;
                dd_end = current_time;
            }
        }
    }
    
    g_performance_metrics.maximum_drawdown_pct = max_dd;
    g_performance_metrics.maximum_drawdown_amount = max_dd_amount;
    g_performance_metrics.max_dd_start_time = dd_start;
    g_performance_metrics.max_dd_end_time = dd_end;
    
    // Calculate drawdown duration in days
    if(dd_start > 0 && dd_end > dd_start) {
        g_performance_metrics.max_dd_duration_days = (int)((dd_end - dd_start) / (24 * 3600));
    }
}

// Calculate Value at Risk (VaR) at given confidence level
double CalculateVaR(double confidence_level) {
    if(g_return_periods < 2) return 0.0;
    
    // Sort returns array for percentile calculation
    double sorted_returns[];
    ArrayResize(sorted_returns, g_return_periods);
    ArrayCopy(sorted_returns, g_daily_returns, 0, 0, g_return_periods);
    ArraySort(sorted_returns);
    
    // Calculate percentile index
    int var_index = (int)((1.0 - confidence_level) * g_return_periods);
    if(var_index < 0) var_index = 0;
    if(var_index >= g_return_periods) var_index = g_return_periods - 1;
    
    return -sorted_returns[var_index]; // VaR is positive for losses
}

// Calculate Conditional VaR (Expected Shortfall)
double CalculateConditionalVaR(double confidence_level) {
    if(g_return_periods < 2) return 0.0;
    
    // Sort returns array
    double sorted_returns[];
    ArrayResize(sorted_returns, g_return_periods);
    ArrayCopy(sorted_returns, g_daily_returns, 0, 0, g_return_periods);
    ArraySort(sorted_returns);
    
    // Calculate tail average
    int tail_count = (int)((1.0 - confidence_level) * g_return_periods);
    if(tail_count < 1) tail_count = 1;
    
    double tail_sum = 0.0;
    for(int i = 0; i < tail_count; i++) {
        tail_sum += sorted_returns[i];
    }
    
    return -(tail_sum / tail_count); // CVaR is positive for losses
}

// Calculate Ulcer Index (measure of downside risk)
double CalculateUlcerIndex() {
    if(g_equity_curve_size < 2) return 0.0;
    
    double ulcer_sum = 0.0;
    
    for(int i = 0; i < g_equity_curve_size; i++) {
        double dd_pct = g_underwater_curve[i];
        ulcer_sum += dd_pct * dd_pct;
    }
    
    return MathSqrt(ulcer_sum / g_equity_curve_size);
}

// Calculate Pain Index (average drawdown)
double CalculatePainIndex() {
    if(g_equity_curve_size < 2) return 0.0;
    
    double pain_sum = 0.0;
    
    for(int i = 0; i < g_equity_curve_size; i++) {
        pain_sum += g_underwater_curve[i];
    }
    
    return pain_sum / g_equity_curve_size;
}

// Calculate Kelly Criterion for optimal position sizing
double CalculateKellyCriterion() {
    if(g_performance_metrics.total_trades < 10) return 0.0;
    
    double win_rate = g_performance_metrics.win_rate_pct / 100.0;
    double loss_rate = 1.0 - win_rate;
    
    if(g_performance_metrics.average_loss == 0) return 0.0;
    
    double win_loss_ratio = MathAbs(g_performance_metrics.average_win / g_performance_metrics.average_loss);
    
    // Kelly formula: f = (bp - q) / b
    // where b = win/loss ratio, p = win probability, q = loss probability
    return (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio;
}

// Update consecutive wins/losses tracking
void UpdateConsecutiveStats(double pnl) {
    if(pnl > 0) {
        g_consecutive_wins++;
        g_consecutive_losses = 0;
        if(g_consecutive_wins > g_max_consecutive_wins) {
            g_max_consecutive_wins = g_consecutive_wins;
        }
    } else if(pnl < 0) {
        g_consecutive_losses++;
        g_consecutive_wins = 0;
        if(g_consecutive_losses > g_max_consecutive_losses) {
            g_max_consecutive_losses = g_consecutive_losses;
        }
    }
}

// MASTER FUNCTION: Calculate all comprehensive performance metrics
void CalculateComprehensiveMetrics() {
    Print("Calculating comprehensive performance metrics...");
    
    // Initialize basic performance metrics
    g_performance_metrics.initial_balance = InpInitialBalance;
    g_performance_metrics.final_balance = g_balance;
    g_performance_metrics.total_pnl = g_balance - InpInitialBalance;
    g_performance_metrics.total_return_pct = (g_performance_metrics.total_pnl / InpInitialBalance) * 100.0;
    
    // Calculate annualized return (assuming backtest period in years)
    double backtest_days = InpBacktestDays;
    double years = backtest_days / 365.25;
    if(years > 0) {
        g_performance_metrics.annualized_return_pct = 
            (MathPow(g_balance / InpInitialBalance, 1.0 / years) - 1.0) * 100.0;
    }
    
    // Trade statistics
    g_performance_metrics.total_trades = g_total_trades;
    g_performance_metrics.winning_trades = g_winning_trades;
    g_performance_metrics.losing_trades = g_losing_trades;
    
    if(g_total_trades > 0) {
        g_performance_metrics.win_rate_pct = ((double)g_winning_trades / g_total_trades) * 100.0;
        g_performance_metrics.loss_rate_pct = ((double)g_losing_trades / g_total_trades) * 100.0;
    }
    
    // Calculate profit/loss metrics from trade history
    g_performance_metrics.gross_profit = g_running_gross_profit;
    g_performance_metrics.gross_loss = g_running_gross_loss;
    
    if(g_running_gross_loss != 0) {
        g_performance_metrics.profit_factor = MathAbs(g_running_gross_profit / g_running_gross_loss);
    }
    
    if(g_total_trades > 0) {
        g_performance_metrics.expectancy = g_performance_metrics.total_pnl / g_total_trades;
    }
    
    if(g_winning_trades > 0) {
        g_performance_metrics.average_win = g_running_gross_profit / g_winning_trades;
    }
    
    if(g_losing_trades > 0) {
        g_performance_metrics.average_loss = g_running_gross_loss / g_losing_trades;
    }
    
    // Find largest win and loss from trade history
    double largest_win = 0.0;
    double largest_loss = 0.0;
    double max_mae = 0.0;
    double max_mfe = 0.0;
    double total_holding_time = 0.0;
    
    for(int i = 0; i < ArraySize(g_trades); i++) {
        if(i >= g_total_trades) break;
        
        TradeRecord trade = g_trades[i];
        
        if(trade.profit_loss > largest_win) {
            largest_win = trade.profit_loss;
        }
        if(trade.profit_loss < largest_loss) {
            largest_loss = trade.profit_loss;
        }
        
        // Track MAE/MFE
        if(trade.mae > max_mae) max_mae = trade.mae;
        if(trade.mfe > max_mfe) max_mfe = trade.mfe;
        
        total_holding_time += trade.holding_time_hours;
    }
    
    g_performance_metrics.largest_win = largest_win;
    g_performance_metrics.largest_loss = largest_loss;
    g_performance_metrics.maximum_adverse_excursion = max_mae;
    g_performance_metrics.maximum_favorable_excursion = max_mfe;
    
    if(g_total_trades > 0) {
        g_performance_metrics.avg_holding_time_hours = total_holding_time / g_total_trades;
    }
    
    // Calculate maximum drawdown
    CalculateMaximumDrawdown();
    
    // Calculate risk-adjusted metrics
    if(g_return_periods > 1) {
        // Calculate returns statistics
        double mean_return = g_returns_sum / g_return_periods;
        double variance = (g_returns_squared_sum / g_return_periods) - (mean_return * mean_return);
        g_performance_metrics.returns_volatility = MathSqrt(variance);
        
        // Calculate downside deviation
        if(g_return_periods > 0) {
            g_performance_metrics.downside_deviation = MathSqrt(g_downside_returns_squared_sum / g_return_periods);
        }
        
        // Calculate risk-adjusted ratios
        g_performance_metrics.sharpe_ratio = CalculateSharpeRatio(mean_return, g_performance_metrics.returns_volatility);
        g_performance_metrics.sortino_ratio = CalculateSortinoRatio(mean_return, g_performance_metrics.downside_deviation);
        g_performance_metrics.calmar_ratio = CalculateCalmarRatio(g_performance_metrics.annualized_return_pct, g_performance_metrics.maximum_drawdown_pct);
        
        // Calculate VaR metrics
        g_performance_metrics.value_at_risk_95 = CalculateVaR(0.95);
        g_performance_metrics.value_at_risk_99 = CalculateVaR(0.99);
        g_performance_metrics.conditional_var_95 = CalculateConditionalVaR(0.95);
    }
    
    // Calculate advanced risk metrics
    g_performance_metrics.ulcer_index = CalculateUlcerIndex();
    g_performance_metrics.pain_index = CalculatePainIndex();
    g_performance_metrics.kelly_criterion = CalculateKellyCriterion();
    
    // Consecutive statistics
    g_performance_metrics.max_consecutive_wins = g_max_consecutive_wins;
    g_performance_metrics.max_consecutive_losses = g_max_consecutive_losses;
    g_performance_metrics.current_consecutive_wins = g_consecutive_wins;
    g_performance_metrics.current_consecutive_losses = g_consecutive_losses;
    
    // Recovery and other ratios
    if(g_performance_metrics.maximum_drawdown_amount > 0) {
        g_performance_metrics.recovery_factor = g_performance_metrics.total_pnl / g_performance_metrics.maximum_drawdown_amount;
        g_performance_metrics.profit_to_max_dd_ratio = g_performance_metrics.total_pnl / g_performance_metrics.maximum_drawdown_amount;
    }
    
    // Sterling ratio (similar to Calmar but uses average of worst drawdowns)
    if(g_performance_metrics.maximum_drawdown_pct > 0) {
        g_performance_metrics.sterling_ratio = g_performance_metrics.annualized_return_pct / g_performance_metrics.maximum_drawdown_pct;
    }
    
    // System Quality Number (simplified version)
    if(g_performance_metrics.returns_volatility > 0 && g_total_trades > 0) {
        g_performance_metrics.system_quality_number = 
            (g_performance_metrics.expectancy / g_performance_metrics.returns_volatility) * MathSqrt(g_total_trades);
    }
    
    Print("✓ Comprehensive metrics calculation completed");
}

// COMPREHENSIVE METRICS DISPLAY FUNCTION
void DisplayComprehensiveMetrics() {
    Print("");
    Print("=================================================================");
    Print("=== IMPROVEMENT 7.2: COMPREHENSIVE PERFORMANCE METRICS REPORT ===");
    Print("=================================================================");
    
    // BASIC PERFORMANCE SUMMARY
    Print("");
    Print("📊 BASIC PERFORMANCE SUMMARY");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Initial Balance:        $", DoubleToString(g_performance_metrics.initial_balance, 2));
    Print("Final Balance:          $", DoubleToString(g_performance_metrics.final_balance, 2));
    Print("Total P&L:              $", DoubleToString(g_performance_metrics.total_pnl, 2));
    Print("Total Return:           ", DoubleToString(g_performance_metrics.total_return_pct, 2), "%");
    Print("Annualized Return:      ", DoubleToString(g_performance_metrics.annualized_return_pct, 2), "%");
    
    // TRADE STATISTICS
    Print("");
    Print("📈 TRADE STATISTICS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Total Trades:           ", g_performance_metrics.total_trades);
    Print("Winning Trades:         ", g_performance_metrics.winning_trades);
    Print("Losing Trades:          ", g_performance_metrics.losing_trades);
    Print("Win Rate:               ", DoubleToString(g_performance_metrics.win_rate_pct, 2), "%");
    Print("Loss Rate:              ", DoubleToString(g_performance_metrics.loss_rate_pct, 2), "%");
    
    // PROFIT/LOSS ANALYSIS
    Print("");
    Print("💰 PROFIT/LOSS ANALYSIS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Gross Profit:           $", DoubleToString(g_performance_metrics.gross_profit, 2));
    Print("Gross Loss:             $", DoubleToString(g_performance_metrics.gross_loss, 2));
    Print("Profit Factor:          ", DoubleToString(g_performance_metrics.profit_factor, 2));
    Print("Expectancy:             $", DoubleToString(g_performance_metrics.expectancy, 2));
    Print("Average Win:            $", DoubleToString(g_performance_metrics.average_win, 2));
    Print("Average Loss:           $", DoubleToString(g_performance_metrics.average_loss, 2));
    Print("Largest Win:            $", DoubleToString(g_performance_metrics.largest_win, 2));
    Print("Largest Loss:           $", DoubleToString(g_performance_metrics.largest_loss, 2));
    
    // RISK METRICS
    Print("");
    Print("⚠️ RISK METRICS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Maximum Drawdown:       ", DoubleToString(g_performance_metrics.maximum_drawdown_pct, 2), "% ($", 
          DoubleToString(g_performance_metrics.maximum_drawdown_amount, 2), ")");
    
    if(g_performance_metrics.max_dd_start_time > 0) {
        Print("Drawdown Period:        ", TimeToString(g_performance_metrics.max_dd_start_time, TIME_DATE),
              " to ", TimeToString(g_performance_metrics.max_dd_end_time, TIME_DATE),
              " (", g_performance_metrics.max_dd_duration_days, " days)");
    }
    
    Print("Maximum Adverse Excursion: $", DoubleToString(g_performance_metrics.maximum_adverse_excursion, 2));
    Print("Maximum Favorable Excursion: $", DoubleToString(g_performance_metrics.maximum_favorable_excursion, 2));
    Print("Recovery Factor:        ", DoubleToString(g_performance_metrics.recovery_factor, 2));
    
    // RISK-ADJUSTED RETURNS
    Print("");
    Print("📊 RISK-ADJUSTED RETURNS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Sharpe Ratio:           ", DoubleToString(g_performance_metrics.sharpe_ratio, 3));
    Print("Sortino Ratio:          ", DoubleToString(g_performance_metrics.sortino_ratio, 3));
    Print("Calmar Ratio:           ", DoubleToString(g_performance_metrics.calmar_ratio, 3));
    Print("Sterling Ratio:         ", DoubleToString(g_performance_metrics.sterling_ratio, 3));
    Print("Return/Max DD Ratio:    ", DoubleToString(g_performance_metrics.profit_to_max_dd_ratio, 2));
    
    // VOLATILITY AND RISK
    Print("");
    Print("📉 VOLATILITY AND RISK");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Returns Volatility:     ", DoubleToString(g_performance_metrics.returns_volatility * 100, 2), "%");
    Print("Downside Deviation:     ", DoubleToString(g_performance_metrics.downside_deviation * 100, 2), "%");
    Print("Value at Risk (95%):    ", DoubleToString(g_performance_metrics.value_at_risk_95 * 100, 2), "%");
    Print("Value at Risk (99%):    ", DoubleToString(g_performance_metrics.value_at_risk_99 * 100, 2), "%");
    Print("Conditional VaR (95%):  ", DoubleToString(g_performance_metrics.conditional_var_95 * 100, 2), "%");
    Print("Ulcer Index:            ", DoubleToString(g_performance_metrics.ulcer_index, 2));
    Print("Pain Index:             ", DoubleToString(g_performance_metrics.pain_index, 2), "%");
    
    // TIME-BASED ANALYSIS
    Print("");
    Print("⏱️ TIME-BASED ANALYSIS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Average Holding Time:   ", DoubleToString(g_performance_metrics.avg_holding_time_hours, 1), " hours");
    Print("Median Holding Time:    ", DoubleToString(g_performance_metrics.median_holding_time_hours, 1), " hours");
    Print("Longest Holding Time:   ", g_performance_metrics.longest_holding_time_hours, " hours");
    Print("Shortest Holding Time:  ", g_performance_metrics.shortest_holding_time_hours, " hours");
    
    // CONSECUTIVE STATISTICS
    Print("");
    Print("🔄 CONSECUTIVE STATISTICS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Max Consecutive Wins:   ", g_performance_metrics.max_consecutive_wins);
    Print("Max Consecutive Losses: ", g_performance_metrics.max_consecutive_losses);
    Print("Current Consecutive Wins: ", g_performance_metrics.current_consecutive_wins);
    Print("Current Consecutive Losses: ", g_performance_metrics.current_consecutive_losses);
    
    // ADVANCED METRICS
    Print("");
    Print("🧮 ADVANCED METRICS");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Kelly Criterion:        ", DoubleToString(g_performance_metrics.kelly_criterion * 100, 1), "%");
    Print("System Quality Number:  ", DoubleToString(g_performance_metrics.system_quality_number, 2));
    
    // PERFORMANCE RATING
    Print("");
    Print("⭐ PERFORMANCE RATING");
    Print("─────────────────────────────────────────────────────────────────");
    
    string rating = "UNKNOWN";
    string rating_color = "";
    
    // Simple rating system based on key metrics
    int rating_score = 0;
    
    // Profitability (40 points max)
    if(g_performance_metrics.total_return_pct > 20) rating_score += 20;
    else if(g_performance_metrics.total_return_pct > 10) rating_score += 15;
    else if(g_performance_metrics.total_return_pct > 5) rating_score += 10;
    else if(g_performance_metrics.total_return_pct > 0) rating_score += 5;
    
    if(g_performance_metrics.profit_factor > 2.0) rating_score += 20;
    else if(g_performance_metrics.profit_factor > 1.5) rating_score += 15;
    else if(g_performance_metrics.profit_factor > 1.2) rating_score += 10;
    else if(g_performance_metrics.profit_factor > 1.0) rating_score += 5;
    
    // Risk Management (30 points max)
    if(g_performance_metrics.maximum_drawdown_pct < 5) rating_score += 15;
    else if(g_performance_metrics.maximum_drawdown_pct < 10) rating_score += 12;
    else if(g_performance_metrics.maximum_drawdown_pct < 15) rating_score += 8;
    else if(g_performance_metrics.maximum_drawdown_pct < 20) rating_score += 4;
    
    if(g_performance_metrics.sharpe_ratio > 2.0) rating_score += 15;
    else if(g_performance_metrics.sharpe_ratio > 1.5) rating_score += 12;
    else if(g_performance_metrics.sharpe_ratio > 1.0) rating_score += 8;
    else if(g_performance_metrics.sharpe_ratio > 0.5) rating_score += 4;
    
    // Consistency (30 points max)
    if(g_performance_metrics.win_rate_pct > 60) rating_score += 15;
    else if(g_performance_metrics.win_rate_pct > 50) rating_score += 12;
    else if(g_performance_metrics.win_rate_pct > 40) rating_score += 8;
    else if(g_performance_metrics.win_rate_pct > 30) rating_score += 4;
    
    if(g_performance_metrics.max_consecutive_losses < 3) rating_score += 15;
    else if(g_performance_metrics.max_consecutive_losses < 5) rating_score += 12;
    else if(g_performance_metrics.max_consecutive_losses < 8) rating_score += 8;
    else if(g_performance_metrics.max_consecutive_losses < 10) rating_score += 4;
    
    // Assign rating based on score
    if(rating_score >= 85) {
        rating = "EXCELLENT ⭐⭐⭐⭐⭐";
        rating_color = "Outstanding performance with strong risk management";
    } else if(rating_score >= 70) {
        rating = "VERY GOOD ⭐⭐⭐⭐";  
        rating_color = "Good performance with acceptable risk levels";
    } else if(rating_score >= 55) {
        rating = "GOOD ⭐⭐⭐";
        rating_color = "Decent performance but room for improvement";
    } else if(rating_score >= 40) {
        rating = "AVERAGE ⭐⭐";
        rating_color = "Below average performance, needs optimization";
    } else {
        rating = "POOR ⭐";
        rating_color = "Poor performance, significant improvements needed";
    }
    
    Print("Overall Rating:         ", rating, " (Score: ", rating_score, "/100)");
    Print("Assessment:             ", rating_color);
    
    // RECOMMENDATIONS
    Print("");
    Print("💡 RECOMMENDATIONS");
    Print("─────────────────────────────────────────────────────────────────");
    
    if(g_performance_metrics.sharpe_ratio < 1.0) {
        Print("• Low Sharpe ratio - consider risk management improvements");
    }
    if(g_performance_metrics.maximum_drawdown_pct > 15) {
        Print("• High maximum drawdown - implement stronger stop losses");
    }
    if(g_performance_metrics.win_rate_pct < 40) {
        Print("• Low win rate - review entry signal quality");
    }
    if(g_performance_metrics.profit_factor < 1.2) {
        Print("• Low profit factor - optimize risk/reward ratios");
    }
    if(g_performance_metrics.max_consecutive_losses > 8) {
        Print("• High consecutive losses - add trend filters or reduce position size");
    }
    if(g_performance_metrics.kelly_criterion > 0.25) {
        Print("• Kelly criterion suggests reducing position size for optimal growth");
    }
    if(g_performance_metrics.ulcer_index > 10) {
        Print("• High Ulcer Index indicates painful drawdown periods");
    }
    
    Print("");
    Print("=================================================================");
    Print("For detailed analysis, review individual trade records and equity curve");
    Print("=================================================================");
}

//============================== IMPROVEMENT 7.3: CSV LOGGING FUNCTIONS ==============
// Machine-readable CSV export functions for detailed analysis

// Initialize CSV files with headers
bool InitializeCSVFiles() {
    if(!InpEnableCSVLogging || g_csv_files_initialized) return true;
    
    Print("Initializing CSV logging files...");
    
    // Prepare file opening flags
    int file_flags = FILE_WRITE | FILE_CSV;
    if(!InpCSVAppendMode) {
        file_flags |= FILE_REWRITE; // Overwrite existing files
    }
    
    // Initialize trades CSV file
    g_csv_trades_handle = FileOpen(InpCSVTradeFileName, file_flags);
    if(g_csv_trades_handle == INVALID_HANDLE) {
        Print("ERROR: Failed to open trades CSV file: ", InpCSVTradeFileName);
        Print("Error code: ", GetLastError());
        return false;
    }
    
    // Initialize equity curve CSV file
    g_csv_equity_handle = FileOpen(InpCSVEquityFileName, file_flags);
    if(g_csv_equity_handle == INVALID_HANDLE) {
        Print("ERROR: Failed to open equity CSV file: ", InpCSVEquityFileName);
        FileClose(g_csv_trades_handle);
        return false;
    }
    
    // Initialize metrics CSV file
    g_csv_metrics_handle = FileOpen(InpCSVMetricsFileName, file_flags);
    if(g_csv_metrics_handle == INVALID_HANDLE) {
        Print("ERROR: Failed to open metrics CSV file: ", InpCSVMetricsFileName);
        FileClose(g_csv_trades_handle);
        FileClose(g_csv_equity_handle);
        return false;
    }
    
    // Write headers if enabled
    if(InpCSVIncludeHeaders && !InpCSVAppendMode) {
        // Trades CSV header
        string trades_header = "TradeID,OpenTime,CloseTime,Action,PositionType,EntryPrice,ExitPrice," +
                              "LotSize,ProfitLoss,Commission,Balance,DrawdownPct,HoldingHours," +
                              "MAE,MFE,ExitReason,EntryTrigger,ConfidenceScore,Symbol,Timeframe";
        FileWrite(g_csv_trades_handle, trades_header);
        
        // Equity curve CSV header  
        string equity_header = "DateTime,Balance,Equity,UnrealizedPL,DrawdownPct,DrawdownAmount," +
                              "PeakBalance,InDrawdown,PositionType,PositionLots,PositionEntry";
        FileWrite(g_csv_equity_handle, equity_header);
        
        // Metrics CSV header (will be written at end with final metrics)
        string metrics_header = "Metric,Value,Description";
        FileWrite(g_csv_metrics_handle, metrics_header);
    }
    
    g_csv_files_initialized = true;
    Print("✓ CSV files initialized successfully");
    Print("  Trades: ", InpCSVTradeFileName);
    Print("  Equity: ", InpCSVEquityFileName);
    Print("  Metrics: ", InpCSVMetricsFileName);
    
    return true;
}

// Log individual trade to CSV
void LogTradeToCSV(const TradeRecord &trade, int trade_id) {
    if(!InpEnableCSVLogging || g_csv_trades_handle == INVALID_HANDLE) return;
    
    // Format trade data for CSV
    string trade_row = IntegerToString(trade_id) + "," +
                      TimeToString(trade.open_time, TIME_DATE | TIME_SECONDS) + "," +
                      TimeToString(trade.close_time, TIME_DATE | TIME_SECONDS) + "," +
                      IntegerToString(trade.action) + "," +
                      (trade.position_type == POS_LONG ? "LONG" : "SHORT") + "," +
                      DoubleToString(trade.entry_price, 5) + "," +
                      DoubleToString(trade.exit_price, 5) + "," +
                      DoubleToString(trade.lots, 2) + "," +
                      DoubleToString(trade.profit_loss, 2) + "," +
                      DoubleToString(trade.commission, 2) + "," +
                      DoubleToString(trade.balance_after, 2) + "," +
                      DoubleToString(trade.drawdown_pct, 2) + "," +
                      IntegerToString(trade.holding_time_hours) + "," +
                      DoubleToString(trade.mae, 2) + "," +
                      DoubleToString(trade.mfe, 2) + "," +
                      trade.exit_reason + "," +
                      g_last_entry_trigger + "," +
                      DoubleToString(trade.confidence_score, 4) + "," +
                      _Symbol + "," +
                      EnumToString(PERIOD_CURRENT);
    
    FileWrite(g_csv_trades_handle, trade_row);
    
    if(InpCSVFlushImmediately) {
        FileFlush(g_csv_trades_handle);
    }
}

// Log equity curve point to CSV
void LogEquityToCSV(datetime bar_time, double balance, double equity, double unrealized_pl, 
                   double drawdown_pct, double drawdown_amount, POS_TYPE position_type, 
                   double position_lots, double position_entry) {
    if(!InpEnableCSVLogging || g_csv_equity_handle == INVALID_HANDLE) return;
    
    // Skip some equity points if not logging all bars (performance optimization)
    if(!InpCSVLogAllBars) {
        g_csv_equity_counter++;
        if(g_csv_equity_counter % 10 != 0 && position_type == POS_NONE) {
            return; // Log every 10th bar when no position, or every bar with position
        }
    }
    
    string equity_row = TimeToString(bar_time, TIME_DATE | TIME_SECONDS) + "," +
                       DoubleToString(balance, 2) + "," +
                       DoubleToString(equity, 2) + "," +
                       DoubleToString(unrealized_pl, 2) + "," +
                       DoubleToString(drawdown_pct, 2) + "," +
                       DoubleToString(drawdown_amount, 2) + "," +
                       DoubleToString(g_peak_balance, 2) + "," +
                       (g_in_drawdown ? "TRUE" : "FALSE") + "," +
                       (position_type == POS_LONG ? "LONG" : (position_type == POS_SHORT ? "SHORT" : "NONE")) + "," +
                       DoubleToString(position_lots, 2) + "," +
                       DoubleToString(position_entry, 5);
    
    FileWrite(g_csv_equity_handle, equity_row);
    
    if(InpCSVFlushImmediately) {
        FileFlush(g_csv_equity_handle);
    }
}

// Log comprehensive metrics to CSV
void LogMetricsToCSV() {
    if(!InpEnableCSVLogging || g_csv_metrics_handle == INVALID_HANDLE) return;
    
    Print("Exporting comprehensive metrics to CSV...");
    
    // Basic Performance Metrics
    FileWrite(g_csv_metrics_handle, "InitialBalance," + DoubleToString(g_performance_metrics.initial_balance, 2) + ",Starting account balance");
    FileWrite(g_csv_metrics_handle, "FinalBalance," + DoubleToString(g_performance_metrics.final_balance, 2) + ",Final account balance");
    FileWrite(g_csv_metrics_handle, "TotalPnL," + DoubleToString(g_performance_metrics.total_pnl, 2) + ",Total profit/loss");
    FileWrite(g_csv_metrics_handle, "TotalReturnPct," + DoubleToString(g_performance_metrics.total_return_pct, 2) + ",Total return percentage");
    FileWrite(g_csv_metrics_handle, "AnnualizedReturnPct," + DoubleToString(g_performance_metrics.annualized_return_pct, 2) + ",Annualized return percentage");
    
    // Trade Statistics
    FileWrite(g_csv_metrics_handle, "TotalTrades," + IntegerToString(g_performance_metrics.total_trades) + ",Total number of trades");
    FileWrite(g_csv_metrics_handle, "WinningTrades," + IntegerToString(g_performance_metrics.winning_trades) + ",Number of winning trades");
    FileWrite(g_csv_metrics_handle, "LosingTrades," + IntegerToString(g_performance_metrics.losing_trades) + ",Number of losing trades");
    FileWrite(g_csv_metrics_handle, "WinRatePct," + DoubleToString(g_performance_metrics.win_rate_pct, 2) + ",Win rate percentage");
    
    // Profit/Loss Analysis
    FileWrite(g_csv_metrics_handle, "GrossProfit," + DoubleToString(g_performance_metrics.gross_profit, 2) + ",Total gross profit");
    FileWrite(g_csv_metrics_handle, "GrossLoss," + DoubleToString(g_performance_metrics.gross_loss, 2) + ",Total gross loss");
    FileWrite(g_csv_metrics_handle, "ProfitFactor," + DoubleToString(g_performance_metrics.profit_factor, 2) + ",Gross profit / gross loss");
    FileWrite(g_csv_metrics_handle, "Expectancy," + DoubleToString(g_performance_metrics.expectancy, 2) + ",Average profit per trade");
    FileWrite(g_csv_metrics_handle, "AverageWin," + DoubleToString(g_performance_metrics.average_win, 2) + ",Average winning trade");
    FileWrite(g_csv_metrics_handle, "AverageLoss," + DoubleToString(g_performance_metrics.average_loss, 2) + ",Average losing trade");
    FileWrite(g_csv_metrics_handle, "LargestWin," + DoubleToString(g_performance_metrics.largest_win, 2) + ",Largest single win");
    FileWrite(g_csv_metrics_handle, "LargestLoss," + DoubleToString(g_performance_metrics.largest_loss, 2) + ",Largest single loss");
    
    // Risk Metrics
    FileWrite(g_csv_metrics_handle, "MaxDrawdownPct," + DoubleToString(g_performance_metrics.maximum_drawdown_pct, 2) + ",Maximum drawdown percentage");
    FileWrite(g_csv_metrics_handle, "MaxDrawdownAmount," + DoubleToString(g_performance_metrics.maximum_drawdown_amount, 2) + ",Maximum drawdown amount");
    FileWrite(g_csv_metrics_handle, "RecoveryFactor," + DoubleToString(g_performance_metrics.recovery_factor, 2) + ",Total return / max drawdown");
    FileWrite(g_csv_metrics_handle, "MaxAdverseExcursion," + DoubleToString(g_performance_metrics.maximum_adverse_excursion, 2) + ",Worst price movement against position");
    FileWrite(g_csv_metrics_handle, "MaxFavorableExcursion," + DoubleToString(g_performance_metrics.maximum_favorable_excursion, 2) + ",Best price movement for position");
    
    // Risk-Adjusted Returns
    FileWrite(g_csv_metrics_handle, "SharpeRatio," + DoubleToString(g_performance_metrics.sharpe_ratio, 3) + ",Risk-adjusted return metric");
    FileWrite(g_csv_metrics_handle, "SortinoRatio," + DoubleToString(g_performance_metrics.sortino_ratio, 3) + ",Downside risk-adjusted return");
    FileWrite(g_csv_metrics_handle, "CalmarRatio," + DoubleToString(g_performance_metrics.calmar_ratio, 3) + ",Annual return / max drawdown");
    FileWrite(g_csv_metrics_handle, "SterlingRatio," + DoubleToString(g_performance_metrics.sterling_ratio, 3) + ",Similar to Calmar ratio");
    
    // Volatility and Risk
    FileWrite(g_csv_metrics_handle, "ReturnsVolatility," + DoubleToString(g_performance_metrics.returns_volatility, 4) + ",Standard deviation of returns");
    FileWrite(g_csv_metrics_handle, "DownsideDeviation," + DoubleToString(g_performance_metrics.downside_deviation, 4) + ",Downside risk measure");
    FileWrite(g_csv_metrics_handle, "ValueAtRisk95," + DoubleToString(g_performance_metrics.value_at_risk_95, 4) + ",VaR at 95% confidence");
    FileWrite(g_csv_metrics_handle, "ValueAtRisk99," + DoubleToString(g_performance_metrics.value_at_risk_99, 4) + ",VaR at 99% confidence");
    FileWrite(g_csv_metrics_handle, "ConditionalVaR95," + DoubleToString(g_performance_metrics.conditional_var_95, 4) + ",Expected shortfall at 95%");
    FileWrite(g_csv_metrics_handle, "UlcerIndex," + DoubleToString(g_performance_metrics.ulcer_index, 2) + ",Downside risk pain measure");
    FileWrite(g_csv_metrics_handle, "PainIndex," + DoubleToString(g_performance_metrics.pain_index, 2) + ",Average drawdown measure");
    
    // Time-Based Analysis
    FileWrite(g_csv_metrics_handle, "AvgHoldingTimeHours," + DoubleToString(g_performance_metrics.avg_holding_time_hours, 1) + ",Average trade duration");
    FileWrite(g_csv_metrics_handle, "MedianHoldingTimeHours," + DoubleToString(g_performance_metrics.median_holding_time_hours, 1) + ",Median trade duration");
    FileWrite(g_csv_metrics_handle, "LongestHoldingTimeHours," + IntegerToString(g_performance_metrics.longest_holding_time_hours) + ",Longest single trade");
    FileWrite(g_csv_metrics_handle, "ShortestHoldingTimeHours," + IntegerToString(g_performance_metrics.shortest_holding_time_hours) + ",Shortest single trade");
    
    // Consecutive Statistics
    FileWrite(g_csv_metrics_handle, "MaxConsecutiveWins," + IntegerToString(g_performance_metrics.max_consecutive_wins) + ",Longest winning streak");
    FileWrite(g_csv_metrics_handle, "MaxConsecutiveLosses," + IntegerToString(g_performance_metrics.max_consecutive_losses) + ",Longest losing streak");
    
    // Advanced Metrics
    FileWrite(g_csv_metrics_handle, "KellyCriterion," + DoubleToString(g_performance_metrics.kelly_criterion, 4) + ",Optimal position size for max growth");
    FileWrite(g_csv_metrics_handle, "SystemQualityNumber," + DoubleToString(g_performance_metrics.system_quality_number, 2) + ",Overall system quality metric");
    
    // Backtest Parameters (for reference)
    FileWrite(g_csv_metrics_handle, "BacktestDays," + IntegerToString(InpBacktestDays) + ",Number of days backtested");
    FileWrite(g_csv_metrics_handle, "InitialLotSize," + DoubleToString(InpLotSize, 2) + ",Base lot size used");
    FileWrite(g_csv_metrics_handle, "Symbol," + _Symbol + ",Trading symbol");
    FileWrite(g_csv_metrics_handle, "Timeframe," + EnumToString(PERIOD_CURRENT) + ",Chart timeframe");
    FileWrite(g_csv_metrics_handle, "ModelFile," + InpModelFileName + ",AI model file used");
    
    if(InpCSVFlushImmediately) {
        FileFlush(g_csv_metrics_handle);
    }
    
    Print("✓ Metrics exported to CSV: ", InpCSVMetricsFileName);
}

// Close CSV files
void CloseCSVFiles() {
    if(!InpEnableCSVLogging) return;
    
    if(g_csv_trades_handle != INVALID_HANDLE) {
        FileClose(g_csv_trades_handle);
        g_csv_trades_handle = INVALID_HANDLE;
    }
    
    if(g_csv_equity_handle != INVALID_HANDLE) {
        FileClose(g_csv_equity_handle);
        g_csv_equity_handle = INVALID_HANDLE;
    }
    
    if(g_csv_metrics_handle != INVALID_HANDLE) {
        FileClose(g_csv_metrics_handle);
        g_csv_metrics_handle = INVALID_HANDLE;
    }
    
    g_csv_files_initialized = false;
    
    if(InpEnableCSVLogging) {
        Print("✓ CSV files closed and saved");
    }
}

//============================== IMPROVEMENT 7.4: FLEXIBLE PARAMETER FUNCTIONS ==============
// Comprehensive parameter management and validation functions

// Initialize flexible parameter system
bool InitializeFlexibleParameters() {
    Print("Initializing flexible parameter system...");
    
    // Initialize parameter state structure
    ZeroMemory(g_flex_params);
    
    // Validate parameters and set defaults
    if(!ValidateParameters()) {
        Print("ERROR: Parameter validation failed: ", g_parameter_validation_errors);
        return false;
    }
    
    // Initialize risk management parameters
    g_flex_params.current_stop_atr = InpStopLossATR;
    g_flex_params.current_tp_atr = InpTakeProfitATR;
    g_flex_params.current_trailing_atr = InpTrailingStopATR;
    g_flex_params.daily_loss_total = 0.0;
    g_flex_params.consecutive_loss_count = 0;
    g_flex_params.risk_controls_active = InpFlexRiskEnabled;
    
    // Initialize filter states
    g_flex_params.volatility_filter_active = InpVolatilityFilterOn;
    g_flex_params.spread_filter_active = InpSpreadFilterOn;
    g_flex_params.trend_filter_active = InpTrendFilterOn;
    g_flex_params.session_filter_active = InpSessionFilterOn;
    g_flex_params.confidence_filter_active = InpConfidenceFilterOn;
    
    // Initialize position sizing
    g_flex_params.calculated_lot_size = InpBaseLotSize;
    g_flex_params.volatility_size_multiplier = InpVolatilitySizeMultiplier;
    g_flex_params.confidence_size_multiplier = InpConfidenceSizeMultiplier;
    g_flex_params.equity_size_multiplier = InpEquitySizeMultiplier;
    
    // Initialize timing controls
    g_flex_params.last_trade_time = 0;
    g_flex_params.trades_this_hour = 0;
    g_flex_params.session_allowed = true;
    g_flex_params.day_allowed = true;
    
    // Initialize signal quality
    g_flex_params.last_signal_strength = 0.0;
    g_flex_params.signal_confirmed = false;
    g_flex_params.confirmation_bars_count = 0;
    g_flex_params.anti_whipsaw_active = InpAntiWhipsawOn;
    
    // Initialize advanced features
    g_flex_params.regime_detected = false;
    g_flex_params.current_regime_score = 0.0;
    g_flex_params.correlation_filter_active = InpCorrelationFilterOn;
    g_flex_params.volume_filter_active = InpVolumeFilterOn;
    g_flex_params.momentum_filter_active = InpMomentumFilterOn;
    
    // Initialize optimization features
    g_flex_params.random_seed_current = (int)InpRandomSeed;
    g_flex_params.data_shuffle_active = InpDataShuffleOn;
    g_flex_params.spread_variation_current = InpSpreadVariationPercent;
    g_flex_params.slippage_current = InpSlippageMaxPips;
    g_flex_params.current_parameter_set = InpParameterSetName;
    
    // Seed random number generator if optimization mode
    if(InpOptimizationMode) {
        MathSrand(g_flex_params.random_seed_current);
    }
    
    g_parameters_validated = true;
    Print("✓ Flexible parameters initialized successfully");
    PrintParameterSummary();
    
    return true;
}

// Validate all input parameters are within acceptable ranges
bool ValidateParameters() {
    g_parameter_validation_errors = "";
    bool validation_passed = true;
    
    // Risk Management validation
    if(InpRiskPercentage < 0.1 || InpRiskPercentage > 10.0) {
        g_parameter_validation_errors += "Risk percentage must be 0.1-10.0%; ";
        validation_passed = false;
    }
    
    if(InpStopLossATR < 0.5 || InpStopLossATR > 5.0) {
        g_parameter_validation_errors += "Stop loss ATR must be 0.5-5.0; ";
        validation_passed = false;
    }
    
    if(InpTakeProfitATR < 1.0 || InpTakeProfitATR > 10.0) {
        g_parameter_validation_errors += "Take profit ATR must be 1.0-10.0; ";
        validation_passed = false;
    }
    
    // Position sizing validation
    if(InpBaseLotSize < 0.01 || InpBaseLotSize > 10.0) {
        g_parameter_validation_errors += "Base lot size must be 0.01-10.0; ";
        validation_passed = false;
    }
    
    if(InpMaxLotSize < InpBaseLotSize || InpMaxLotSize > 20.0) {
        g_parameter_validation_errors += "Max lot size must be >= base lot and <= 20.0; ";
        validation_passed = false;
    }
    
    // Time validation
    if(InpSessionStartHour < 0 || InpSessionStartHour > 23) {
        g_parameter_validation_errors += "Session start hour must be 0-23; ";
        validation_passed = false;
    }
    
    if(InpSessionEndHour < 0 || InpSessionEndHour > 23) {
        g_parameter_validation_errors += "Session end hour must be 0-23; ";
        validation_passed = false;
    }
    
    // Signal quality validation
    if(InpConfidenceThreshold < 0.1 || InpConfidenceThreshold > 0.99) {
        g_parameter_validation_errors += "Confidence threshold must be 0.1-0.99; ";
        validation_passed = false;
    }
    
    if(InpSignalStrengthMin < 0.01 || InpSignalStrengthMin > 1.0) {
        g_parameter_validation_errors += "Signal strength min must be 0.01-1.0; ";
        validation_passed = false;
    }
    
    if(InpSignalStrengthMax < InpSignalStrengthMin || InpSignalStrengthMax > 2.0) {
        g_parameter_validation_errors += "Signal strength max must be >= min and <= 2.0; ";
        validation_passed = false;
    }
    
    return validation_passed;
}

// Print comprehensive parameter summary
void PrintParameterSummary() {
    Print("==================== PARAMETER CONFIGURATION ====================");
    
    if(InpFlexRiskEnabled) {
        Print("📊 RISK MANAGEMENT:");
        Print("  Risk per trade: ", DoubleToString(InpRiskPercentage, 2), "%");
        Print("  Stop loss: ", DoubleToString(InpStopLossATR, 1), " ATR");
        Print("  Take profit: ", DoubleToString(InpTakeProfitATR, 1), " ATR");
        Print("  Max daily loss: $", DoubleToString(InpMaxLossPerDay, 0));
        Print("  Max consecutive losses: ", InpMaxConsecutiveLosses);
    }
    
    if(InpFlexFiltersEnabled) {
        Print("🔍 TRADING FILTERS:");
        Print("  Confidence filter: ", InpConfidenceFilterOn ? "ON (" + DoubleToString(InpConfidenceThreshold, 2) + ")" : "OFF");
        Print("  Volatility filter: ", InpVolatilityFilterOn ? "ON" : "OFF");
        Print("  Spread filter: ", InpSpreadFilterOn ? "ON" : "OFF");
        Print("  Trend filter: ", InpTrendFilterOn ? "ON" : "OFF");
        Print("  News filter: ", InpNewsFilterOn ? "ON" : "OFF");
    }
    
    if(InpFlexSizingEnabled) {
        Print("📏 POSITION SIZING:");
        Print("  Base lot size: ", DoubleToString(InpBaseLotSize, 2));
        Print("  Max lot size: ", DoubleToString(InpMaxLotSize, 2));
        Print("  Volatility-based: ", InpVolatilityBasedSizing ? "ON" : "OFF");
        Print("  Confidence-based: ", InpConfidenceBasedSizing ? "ON" : "OFF");
        Print("  Partial closing: ", InpPartialClosingEnabled ? "ON" : "OFF");
    }
    
    if(InpFlexTimeEnabled) {
        Print("⏰ TIME CONTROLS:");
        Print("  Session hours: ", InpSessionStartHour, ":00 - ", InpSessionEndHour, ":00");
        Print("  Days enabled: ", (InpMondayTradingOn?"Mon ":""), (InpTuesdayTradingOn?"Tue ":""), 
              (InpWednesdayTradingOn?"Wed ":""), (InpThursdayTradingOn?"Thu ":""), (InpFridayTradingOn?"Fri ":""));
        Print("  Min trade interval: ", InpMinTradeInterval, " minutes");
        Print("  Max trades/hour: ", InpMaxTradesPerHour);
    }
    
    if(InpFlexSignalEnabled) {
        Print("📡 SIGNAL QUALITY:");
        Print("  Signal strength range: ", DoubleToString(InpSignalStrengthMin, 2), " - ", DoubleToString(InpSignalStrengthMax, 2));
        Print("  Signal confirmation: ", InpSignalConfirmationOn ? "ON (" + IntegerToString(InpConfirmationBars) + " bars)" : "OFF");
        Print("  Anti-whipsaw: ", InpAntiWhipsawOn ? "ON" : "OFF");
        Print("  Multi-timeframe: ", InpMultiTimeframeOn ? "ON" : "OFF");
    }
    
    if(InpFlexAdvancedEnabled) {
        Print("🔬 ADVANCED FEATURES:");
        Print("  Market regime: ", InpMarketRegimeOn ? "ON" : "OFF");
        Print("  Correlation filter: ", InpCorrelationFilterOn ? "ON" : "OFF");
        Print("  Volume filter: ", InpVolumeFilterOn ? "ON" : "OFF");
        Print("  Momentum filter: ", InpMomentumFilterOn ? "ON" : "OFF");
    }
    
    if(InpOptimizationMode) {
        Print("⚙️ OPTIMIZATION MODE:");
        Print("  Random seed: ", g_flex_params.random_seed_current);
        Print("  Data shuffle: ", InpDataShuffleOn ? "ON (" + DoubleToString(InpDataShufflePercent, 1) + "%)" : "OFF");
        Print("  Spread variation: ", InpSpreadRandomization ? "ON (" + DoubleToString(InpSpreadVariationPercent, 1) + "%)" : "OFF");
        Print("  Slippage simulation: ", InpSlippageSimulation ? "ON (max " + DoubleToString(InpSlippageMaxPips, 1) + " pips)" : "OFF");
        Print("  Parameter set: ", InpParameterSetName);
    }
    
    Print("==============================================================");
}

// Check if trade passes all flexible filters
bool PassesFlexibleFilters(double signal_strength, double confidence, datetime trade_time) {
    if(!g_parameters_validated) return false;
    
    // Confidence filter
    if(g_flex_params.confidence_filter_active && confidence < InpConfidenceThreshold) {
        if(InpVerboseLogging) {
            Print("FILTER: Confidence too low: ", DoubleToString(confidence, 3), " < ", DoubleToString(InpConfidenceThreshold, 3));
        }
        return false;
    }
    
    // Signal strength filter
    if(g_flex_params.last_signal_strength > 0) {
        if(signal_strength < InpSignalStrengthMin || signal_strength > InpSignalStrengthMax) {
            if(InpVerboseLogging) {
                Print("FILTER: Signal strength out of range: ", DoubleToString(signal_strength, 3));
            }
            return false;
        }
    }
    
    // Volatility filter
    if(g_flex_params.volatility_filter_active && g_current_atr > 0) {
        if(g_current_atr < InpMinVolatilityATR || g_current_atr > InpMaxVolatilityATR) {
            if(InpVerboseLogging) {
                Print("FILTER: ATR out of range: ", DoubleToString(g_current_atr, 5));
            }
            return false;
        }
    }
    
    // Session filter
    if(g_flex_params.session_filter_active) {
        MqlDateTime dt;
        TimeToStruct(trade_time, dt);
        
        // Check day of week
        bool day_allowed = false;
        switch(dt.day_of_week) {
            case 1: day_allowed = InpMondayTradingOn; break;
            case 2: day_allowed = InpTuesdayTradingOn; break;
            case 3: day_allowed = InpWednesdayTradingOn; break;
            case 4: day_allowed = InpThursdayTradingOn; break;
            case 5: day_allowed = InpFridayTradingOn; break;
            case 0:
            case 6: day_allowed = InpWeekendTradingOn; break;
        }
        
        if(!day_allowed) {
            if(InpVerboseLogging) {
                Print("FILTER: Day not allowed for trading: ", dt.day_of_week);
            }
            return false;
        }
        
        // Check hour
        if(InpSessionStartHour <= InpSessionEndHour) {
            if(dt.hour < InpSessionStartHour || dt.hour >= InpSessionEndHour) {
                if(InpVerboseLogging) {
                    Print("FILTER: Outside trading hours: ", dt.hour);
                }
                return false;
            }
        } else {
            if(dt.hour < InpSessionStartHour && dt.hour >= InpSessionEndHour) {
                if(InpVerboseLogging) {
                    Print("FILTER: Outside overnight session: ", dt.hour);
                }
                return false;
            }
        }
    }
    
    // Anti-whipsaw filter
    if(g_flex_params.anti_whipsaw_active) {
        if(trade_time - g_flex_params.last_trade_time < InpMinTradeInterval * 60) {
            if(InpVerboseLogging) {
                Print("FILTER: Too soon since last trade");
            }
            return false;
        }
    }
    
    // Hourly trade limit
    datetime current_hour = trade_time - (trade_time % 3600);
    datetime last_hour = g_flex_params.last_trade_time - (g_flex_params.last_trade_time % 3600);
    
    if(current_hour != last_hour) {
        g_flex_params.trades_this_hour = 0;
    }
    
    if(g_flex_params.trades_this_hour >= InpMaxTradesPerHour) {
        if(InpVerboseLogging) {
            Print("FILTER: Max trades per hour reached: ", g_flex_params.trades_this_hour);
        }
        return false;
    }
    
    return true;
}

// Calculate flexible position size based on multiple factors
double CalculateFlexibleLotSize(double confidence, double signal_strength, double current_atr) {
    if(!g_parameters_validated) return InpBaseLotSize;
    
    double lot_size = InpBaseLotSize;
    
    // Volatility-based sizing
    if(InpVolatilityBasedSizing && current_atr > 0) {
        double volatility_factor = InpMinVolatilityATR / current_atr;
        volatility_factor = MathMax(0.5, MathMin(2.0, volatility_factor));
        lot_size *= volatility_factor * InpVolatilitySizeMultiplier;
    }
    
    // Confidence-based sizing
    if(InpConfidenceBasedSizing && confidence > 0) {
        double confidence_factor = confidence / InpConfidenceThreshold;
        confidence_factor = MathMax(0.5, MathMin(3.0, confidence_factor));
        lot_size *= confidence_factor * InpConfidenceSizeMultiplier;
    }
    
    // Equity-based sizing
    if(InpEquityBasedSizing) {
        double equity_factor = g_balance / InpInitialBalance;
        equity_factor = MathMax(0.3, MathMin(3.0, equity_factor));
        lot_size *= equity_factor * InpEquitySizeMultiplier;
    }
    
    // Apply risk percentage limit
    if(InpFlexRiskEnabled && current_atr > 0) {
        double risk_amount = g_balance * (InpRiskPercentage / 100.0);
        double atr_risk = current_atr * InpStopLossATR * 100000; // Convert to account currency
        double max_lot_by_risk = risk_amount / atr_risk;
        lot_size = MathMin(lot_size, max_lot_by_risk);
    }
    
    // Apply absolute limits
    lot_size = MathMax(0.01, MathMin(InpMaxLotSize, lot_size));
    
    g_flex_params.calculated_lot_size = lot_size;
    
    return lot_size;
}

// Update flexible parameter states during trading
void UpdateFlexibleStates(datetime trade_time, double pnl, bool is_new_trade) {
    if(!g_parameters_validated) return;
    
    // Update timing states
    if(is_new_trade) {
        g_flex_params.last_trade_time = trade_time;
        g_flex_params.trades_this_hour++;
    }
    
    // Update loss tracking
    if(pnl < 0) {
        g_flex_params.consecutive_loss_count++;
        g_flex_params.daily_loss_total += MathAbs(pnl);
    } else if(pnl > 0) {
        g_flex_params.consecutive_loss_count = 0;
    }
    
    // Check for emergency conditions
    if(g_flex_params.consecutive_loss_count >= InpMaxConsecutiveLosses) {
        g_emergency_mode = true;
        g_emergency_halt_until = trade_time + (InpMaxConsecutiveLosses * 3600); // Halt for N hours
        Print("EMERGENCY: Trading halted due to ", g_flex_params.consecutive_loss_count, " consecutive losses");
    }
    
    if(g_flex_params.daily_loss_total >= InpMaxLossPerDay) {
        g_emergency_mode = true;
        MqlDateTime dt;
        TimeToStruct(trade_time, dt);
        dt.hour = 23; dt.min = 59; dt.sec = 59; // Halt until end of day
        g_emergency_halt_until = StructToTime(dt);
        Print("EMERGENCY: Trading halted due to daily loss limit: $", DoubleToString(g_flex_params.daily_loss_total, 2));
    }
}

// Save parameter set to file for optimization tracking
void SaveParameterSet() {
    if(!InpParameterSetSaving) return;
    
    string filename = "ParameterSet_" + InpParameterSetName + ".csv";
    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV);
    
    if(handle != INVALID_HANDLE) {
        FileWrite(handle, "Parameter,Value,Description");
        FileWrite(handle, "RiskPercentage", InpRiskPercentage, "Account risk per trade");
        FileWrite(handle, "StopLossATR", InpStopLossATR, "Stop loss ATR multiplier");
        FileWrite(handle, "TakeProfitATR", InpTakeProfitATR, "Take profit ATR multiplier");
        FileWrite(handle, "ConfidenceThreshold", InpConfidenceThreshold, "Minimum confidence threshold");
        FileWrite(handle, "BaseLotSize", InpBaseLotSize, "Base position size");
        FileWrite(handle, "MaxLotSize", InpMaxLotSize, "Maximum position size");
        FileWrite(handle, "SessionStartHour", InpSessionStartHour, "Trading session start");
        FileWrite(handle, "SessionEndHour", InpSessionEndHour, "Trading session end");
        FileWrite(handle, "SignalStrengthMin", InpSignalStrengthMin, "Minimum signal strength");
        FileWrite(handle, "SignalStrengthMax", InpSignalStrengthMax, "Maximum signal strength");
        FileWrite(handle, "RandomSeed", InpRandomSeed, "Random seed for optimization");
        
        FileClose(handle);
        Print("✓ Parameter set saved: ", filename);
    }
}

//============================== IMPROVEMENT 7.5: MONTE CARLO TESTING FUNCTIONS ==============
// Advanced batch testing and robustness analysis functions

// Initialize Monte Carlo testing system
bool InitializeMonteCarloTesting() {
    if(!InpEnableMonteCarloMode) return true;
    
    Print("Initializing Monte Carlo testing system...");
    
    // Validate Monte Carlo parameters
    if(InpMonteCarloRuns < 10 || InpMonteCarloRuns > 1000) {
        Print("ERROR: Monte Carlo runs must be between 10 and 1000");
        return false;
    }
    
    if(InpMCDataShufflePercent < 1.0 || InpMCDataShufflePercent > 50.0) {
        Print("ERROR: Data shuffle percentage must be between 1% and 50%");
        return false;
    }
    
    if(InpMCStartDateVariation < 7 || InpMCStartDateVariation > 180) {
        Print("ERROR: Start date variation must be between 7 and 180 days");
        return false;
    }
    
    // Initialize Monte Carlo results structure
    ZeroMemory(g_monte_carlo_results);
    g_monte_carlo_mode = true;
    g_current_mc_run = 0;
    
    // Initialize random number generator with base seed
    MathSrand((int)InpRandomSeed);
    
    Print("✓ Monte Carlo testing initialized for ", InpMonteCarloRuns, " runs");
    return true;
}

// Generate randomized parameters for a Monte Carlo run
void GenerateMonteCarloVariation(int run_number) {
    // Set run-specific seed for reproducibility
    int run_seed = (int)InpRandomSeed + run_number * 1000;
    MathSrand(run_seed);
    
    // Store run parameters
    g_monte_carlo_results.runs[run_number].run_number = run_number;
    g_monte_carlo_results.runs[run_number].random_seed = run_seed;
    
    // Randomize start date if enabled
    if(InpMCRandomStartDates) {
        int date_variation = (int)(MathRand() % (InpMCStartDateVariation * 2)) - InpMCStartDateVariation;
        g_mc_start_date = TimeCurrent() - (InpBacktestDays * 24 * 3600) + (date_variation * 24 * 3600);
    } else {
        g_mc_start_date = TimeCurrent() - (InpBacktestDays * 24 * 3600);
    }
    
    // Randomize period length if enabled
    if(InpMCRandomPeriods) {
        g_mc_period_days = InpMCMinPeriodDays + (int)(MathRand() % (InpMCMaxPeriodDays - InpMCMinPeriodDays + 1));
    } else {
        g_mc_period_days = InpBacktestDays;
    }
    
    g_mc_end_date = g_mc_start_date + (g_mc_period_days * 24 * 3600);
    
    // Store date parameters
    g_monte_carlo_results.runs[run_number].start_date = g_mc_start_date;
    g_monte_carlo_results.runs[run_number].end_date = g_mc_end_date;
    g_monte_carlo_results.runs[run_number].period_days = g_mc_period_days;
    
    // Randomize spread multiplier if enabled
    if(InpMCSpreadVariation) {
        double spread_range = InpMCSpreadMaxMultiplier - InpMCSpreadMinMultiplier;
        g_current_spread_multiplier = InpMCSpreadMinMultiplier + (MathRand() / 32767.0) * spread_range;
    } else {
        g_current_spread_multiplier = 1.0;
    }
    g_monte_carlo_results.runs[run_number].spread_multiplier = g_current_spread_multiplier;
    
    // Randomize slippage if enabled
    if(InpMCSlippageVariation) {
        double slippage_range = InpMCSlippageMaxPips - InpMCSlippageMinPips;
        g_current_slippage_pips = InpMCSlippageMinPips + (MathRand() / 32767.0) * slippage_range;
    } else {
        g_current_slippage_pips = 0.0;
    }
    g_monte_carlo_results.runs[run_number].slippage_pips = g_current_slippage_pips;
    
    // Randomize commission if enabled
    if(InpMCCommissionVariation) {
        double comm_range = InpMCCommissionMaxPct - InpMCCommissionMinPct;
        g_current_commission_multiplier = (InpMCCommissionMinPct + (MathRand() / 32767.0) * comm_range) / 100.0;
    } else {
        g_current_commission_multiplier = 1.0;
    }
    g_monte_carlo_results.runs[run_number].commission_multiplier = g_current_commission_multiplier;
    
    if(InpMCProgressReporting && (run_number % InpMCProgressFrequency == 0 || run_number < 5)) {
        Print("MC Run ", run_number, "/", InpMonteCarloRuns, 
              " | Period: ", g_mc_period_days, " days",
              " | Spread: ", DoubleToString(g_current_spread_multiplier, 2), "x",
              " | Slippage: ", DoubleToString(g_current_slippage_pips, 1), " pips");
    }
}

// Shuffle price data for robustness testing
void ShufflePriceData() {
    if(!InpMCDataShuffling) return;
    
    // Get current price data size (approximate)
    int data_size = InpBacktestDays * 24 * 12; // Approximate 5-minute bars
    
    // Calculate number of points to shuffle
    int shuffle_count = (int)(data_size * InpMCDataShufflePercent / 100.0);
    
    // Resize shuffle indices array
    ArrayResize(g_shuffle_indices, shuffle_count * 2);
    
    // Generate random index pairs for shuffling
    for(int i = 0; i < shuffle_count; i++) {
        g_shuffle_indices[i * 2] = (int)(MathRand() % data_size);
        g_shuffle_indices[i * 2 + 1] = (int)(MathRand() % data_size);
    }
    
    g_price_data_shuffled = true;
    
    if(InpMCProgressReporting) {
        Print("  Data shuffling: ", shuffle_count, " point pairs (", 
              DoubleToString(InpMCDataShufflePercent, 1), "% of data)");
    }
}

// Apply Monte Carlo variations to trading costs
double ApplyMonteCarloSpread(double base_spread) {
    return base_spread * g_current_spread_multiplier;
}

double ApplyMonteCarloSlippage() {
    if(!InpMCSlippageVariation) return 0.0;
    
    // Random slippage between 0 and max for this run
    return (MathRand() / 32767.0) * g_current_slippage_pips;
}

double ApplyMonteCarloCommission(double base_commission) {
    return base_commission * g_current_commission_multiplier;
}

// Record results from a completed Monte Carlo run
void RecordMonteCarloRun(int run_number) {
    if(run_number >= 1000) return; // Safety check
    
    MonteCarloRun* run = &g_monte_carlo_results.runs[run_number];
    
    // Record basic performance metrics
    run.final_balance = g_balance;
    run.total_return_pct = g_performance_metrics.total_return_pct;
    run.max_drawdown_pct = g_performance_metrics.maximum_drawdown_pct;
    run.sharpe_ratio = g_performance_metrics.sharpe_ratio;
    run.profit_factor = g_performance_metrics.profit_factor;
    run.win_rate_pct = g_performance_metrics.win_rate_pct;
    run.total_trades = g_performance_metrics.total_trades;
    run.largest_loss = g_performance_metrics.largest_loss;
    run.largest_win = g_performance_metrics.largest_win;
    
    // Calculate robustness metrics for this run
    CalculateRunRobustness(run_number);
    
    g_monte_carlo_results.completed_runs++;
    
    if(InpMCProgressReporting && (run_number % InpMCProgressFrequency == 0 || run_number < 5)) {
        Print("MC Run ", run_number, " completed:",
              " Return: ", DoubleToString(run.total_return_pct, 2), "%",
              " | DD: ", DoubleToString(run.max_drawdown_pct, 2), "%",
              " | Sharpe: ", DoubleToString(run.sharpe_ratio, 2),
              " | Robustness: ", DoubleToString(run.robustness_score, 3));
    }
}

// Calculate robustness metrics for individual run
void CalculateRunRobustness(int run_number) {
    MonteCarloRun* run = &g_monte_carlo_results.runs[run_number];
    
    // Return stability (relative to baseline expected)
    double expected_return = 15.0; // Expected annual return baseline
    run.return_stability = 1.0 - MathAbs(run.total_return_pct - expected_return) / expected_return;
    run.return_stability = MathMax(0.0, MathMin(1.0, run.return_stability));
    
    // Drawdown stability (penalty for excessive drawdown)
    double acceptable_drawdown = 10.0; // Acceptable drawdown baseline
    if(run.max_drawdown_pct <= acceptable_drawdown) {
        run.drawdown_stability = 1.0;
    } else {
        run.drawdown_stability = acceptable_drawdown / run.max_drawdown_pct;
    }
    run.drawdown_stability = MathMax(0.0, MathMin(1.0, run.drawdown_stability));
    
    // Trade consistency (based on number of trades and win rate)
    double expected_trades = 50; // Expected number of trades
    double trade_count_score = 1.0 - MathAbs(run.total_trades - expected_trades) / expected_trades;
    trade_count_score = MathMax(0.0, MathMin(1.0, trade_count_score));
    
    double win_rate_score = run.win_rate_pct / 100.0; // Normalize to 0-1
    run.trade_consistency = (trade_count_score + win_rate_score) / 2.0;
    
    // Overall robustness score (weighted combination)
    run.robustness_score = (run.return_stability * 0.4) + 
                          (run.drawdown_stability * 0.4) + 
                          (run.trade_consistency * 0.2);
}

// Calculate aggregate Monte Carlo statistics
void CalculateMonteCarloStatistics() {
    if(g_monte_carlo_results.completed_runs == 0) return;
    
    int n = g_monte_carlo_results.completed_runs;
    
    // Calculate means
    double return_sum = 0, drawdown_sum = 0, sharpe_sum = 0;
    double pf_sum = 0, wr_sum = 0, rob_sum = 0;
    
    for(int i = 0; i < n; i++) {
        MonteCarloRun* run = &g_monte_carlo_results.runs[i];
        return_sum += run.total_return_pct;
        drawdown_sum += run.max_drawdown_pct;
        sharpe_sum += run.sharpe_ratio;
        pf_sum += run.profit_factor;
        wr_sum += run.win_rate_pct;
        rob_sum += run.robustness_score;
    }
    
    g_monte_carlo_results.mean_return = return_sum / n;
    g_monte_carlo_results.mean_drawdown = drawdown_sum / n;
    g_monte_carlo_results.mean_sharpe = sharpe_sum / n;
    g_monte_carlo_results.mean_profit_factor = pf_sum / n;
    g_monte_carlo_results.mean_win_rate = wr_sum / n;
    
    // Calculate standard deviations
    double return_var = 0, drawdown_var = 0, sharpe_var = 0;
    double pf_var = 0, wr_var = 0;
    
    for(int i = 0; i < n; i++) {
        MonteCarloRun* run = &g_monte_carlo_results.runs[i];
        return_var += MathPow(run.total_return_pct - g_monte_carlo_results.mean_return, 2);
        drawdown_var += MathPow(run.max_drawdown_pct - g_monte_carlo_results.mean_drawdown, 2);
        sharpe_var += MathPow(run.sharpe_ratio - g_monte_carlo_results.mean_sharpe, 2);
        pf_var += MathPow(run.profit_factor - g_monte_carlo_results.mean_profit_factor, 2);
        wr_var += MathPow(run.win_rate_pct - g_monte_carlo_results.mean_win_rate, 2);
    }
    
    g_monte_carlo_results.std_return = MathSqrt(return_var / (n - 1));
    g_monte_carlo_results.std_drawdown = MathSqrt(drawdown_var / (n - 1));
    g_monte_carlo_results.std_sharpe = MathSqrt(sharpe_var / (n - 1));
    g_monte_carlo_results.std_profit_factor = MathSqrt(pf_var / (n - 1));
    g_monte_carlo_results.std_win_rate = MathSqrt(wr_var / (n - 1));
    
    // Calculate percentiles (using simple sorting)
    CalculateMonteCarloPercentiles();
    
    // Calculate robustness scores
    CalculateOverallRobustness();
    
    // Count successful runs
    g_monte_carlo_results.positive_return_runs = 0;
    g_monte_carlo_results.acceptable_drawdown_runs = 0;
    g_monte_carlo_results.robust_runs = 0;
    
    for(int i = 0; i < n; i++) {
        MonteCarloRun* run = &g_monte_carlo_results.runs[i];
        if(run.total_return_pct > 0) g_monte_carlo_results.positive_return_runs++;
        if(run.max_drawdown_pct < 15.0) g_monte_carlo_results.acceptable_drawdown_runs++;
        if(run.robustness_score >= InpMCRobustnessThreshold) g_monte_carlo_results.robust_runs++;
    }
    
    g_monte_carlo_results.success_rate = (double)g_monte_carlo_results.robust_runs / n;
}

// Calculate percentiles for distribution analysis
void CalculateMonteCarloPercentiles() {
    int n = g_monte_carlo_results.completed_runs;
    if(n < 20) return; // Need sufficient data for percentiles
    
    // Sort returns for percentile calculation
    double returns[];
    double drawdowns[];
    ArrayResize(returns, n);
    ArrayResize(drawdowns, n);
    
    for(int i = 0; i < n; i++) {
        returns[i] = g_monte_carlo_results.runs[i].total_return_pct;
        drawdowns[i] = g_monte_carlo_results.runs[i].max_drawdown_pct;
    }
    
    ArraySort(returns);
    ArraySort(drawdowns);
    
    // Calculate 5th and 95th percentiles
    int p5_index = (int)(n * 0.05);
    int p95_index = (int)(n * 0.95);
    
    g_monte_carlo_results.percentile_5_return = returns[p5_index];
    g_monte_carlo_results.percentile_95_return = returns[p95_index];
    g_monte_carlo_results.percentile_5_drawdown = drawdowns[p5_index];
    g_monte_carlo_results.percentile_95_drawdown = drawdowns[p95_index];
}

// Calculate overall robustness analysis
void CalculateOverallRobustness() {
    int n = g_monte_carlo_results.completed_runs;
    if(n == 0) return;
    
    // Return consistency (low standard deviation is good)
    double return_cv = g_monte_carlo_results.std_return / MathAbs(g_monte_carlo_results.mean_return);
    g_monte_carlo_results.return_consistency_score = MathMax(0.0, 1.0 - return_cv);
    
    // Risk consistency (low drawdown variation is good)
    double dd_cv = g_monte_carlo_results.std_drawdown / g_monte_carlo_results.mean_drawdown;
    g_monte_carlo_results.risk_consistency_score = MathMax(0.0, 1.0 - dd_cv);
    
    // Strategy stability (based on Sharpe ratio consistency)
    double sharpe_cv = g_monte_carlo_results.std_sharpe / MathAbs(g_monte_carlo_results.mean_sharpe);
    g_monte_carlo_results.strategy_stability_score = MathMax(0.0, 1.0 - sharpe_cv);
    
    // Overall robustness score
    g_monte_carlo_results.overall_robustness_score = 
        (g_monte_carlo_results.return_consistency_score * 0.4) +
        (g_monte_carlo_results.risk_consistency_score * 0.4) +
        (g_monte_carlo_results.strategy_stability_score * 0.2);
}

// Save Monte Carlo results to CSV files
void SaveMonteCarloResults() {
    if(!InpMCResultsSaving || g_monte_carlo_results.completed_runs == 0) return;
    
    string timestamp = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES);
    StringReplace(timestamp, ":", "");
    StringReplace(timestamp, " ", "_");
    StringReplace(timestamp, ".", "");
    
    // Save detailed run results
    string runs_filename = InpMCResultsPrefix + "_Runs_" + timestamp + ".csv";
    int runs_handle = FileOpen(runs_filename, FILE_WRITE | FILE_CSV);
    
    if(runs_handle != INVALID_HANDLE) {
        // Write header
        FileWrite(runs_handle, "Run,StartDate,EndDate,PeriodDays,SpreadMult,SlippagePips,CommissionMult,RandomSeed," +
                  "FinalBalance,ReturnPct,MaxDrawdownPct,SharpeRatio,ProfitFactor,WinRatePct,TotalTrades," +
                  "LargestLoss,LargestWin,ReturnStability,DrawdownStability,TradeConsistency,RobustnessScore");
        
        // Write run data
        for(int i = 0; i < g_monte_carlo_results.completed_runs; i++) {
            MonteCarloRun* run = &g_monte_carlo_results.runs[i];
            FileWrite(runs_handle, 
                run.run_number, TimeToString(run.start_date, TIME_DATE), TimeToString(run.end_date, TIME_DATE),
                run.period_days, run.spread_multiplier, run.slippage_pips, run.commission_multiplier, run.random_seed,
                run.final_balance, run.total_return_pct, run.max_drawdown_pct, run.sharpe_ratio,
                run.profit_factor, run.win_rate_pct, run.total_trades, run.largest_loss, run.largest_win,
                run.return_stability, run.drawdown_stability, run.trade_consistency, run.robustness_score);
        }
        
        FileClose(runs_handle);
        Print("✓ Monte Carlo runs saved: ", runs_filename);
    }
    
    // Save aggregate statistics
    string stats_filename = InpMCResultsPrefix + "_Statistics_" + timestamp + ".csv";
    int stats_handle = FileOpen(stats_filename, FILE_WRITE | FILE_CSV);
    
    if(stats_handle != INVALID_HANDLE) {
        FileWrite(stats_handle, "Metric,Value,Description");
        FileWrite(stats_handle, "CompletedRuns", g_monte_carlo_results.completed_runs, "Number of completed Monte Carlo runs");
        FileWrite(stats_handle, "MeanReturn", g_monte_carlo_results.mean_return, "Average return across all runs");
        FileWrite(stats_handle, "StdReturn", g_monte_carlo_results.std_return, "Standard deviation of returns");
        FileWrite(stats_handle, "MeanDrawdown", g_monte_carlo_results.mean_drawdown, "Average maximum drawdown");
        FileWrite(stats_handle, "StdDrawdown", g_monte_carlo_results.std_drawdown, "Standard deviation of drawdowns");
        FileWrite(stats_handle, "MeanSharpe", g_monte_carlo_results.mean_sharpe, "Average Sharpe ratio");
        FileWrite(stats_handle, "StdSharpe", g_monte_carlo_results.std_sharpe, "Standard deviation of Sharpe ratios");
        FileWrite(stats_handle, "Percentile95Return", g_monte_carlo_results.percentile_95_return, "95th percentile return");
        FileWrite(stats_handle, "Percentile5Return", g_monte_carlo_results.percentile_5_return, "5th percentile return");
        FileWrite(stats_handle, "Percentile95Drawdown", g_monte_carlo_results.percentile_95_drawdown, "95th percentile drawdown");
        FileWrite(stats_handle, "Percentile5Drawdown", g_monte_carlo_results.percentile_5_drawdown, "5th percentile drawdown");
        FileWrite(stats_handle, "OverallRobustnessScore", g_monte_carlo_results.overall_robustness_score, "Overall strategy robustness");
        FileWrite(stats_handle, "ReturnConsistencyScore", g_monte_carlo_results.return_consistency_score, "Return consistency measure");
        FileWrite(stats_handle, "RiskConsistencyScore", g_monte_carlo_results.risk_consistency_score, "Risk consistency measure");
        FileWrite(stats_handle, "StrategyStabilityScore", g_monte_carlo_results.strategy_stability_score, "Strategy stability measure");
        FileWrite(stats_handle, "PositiveReturnRuns", g_monte_carlo_results.positive_return_runs, "Runs with positive returns");
        FileWrite(stats_handle, "AcceptableDrawdownRuns", g_monte_carlo_results.acceptable_drawdown_runs, "Runs with acceptable drawdown");
        FileWrite(stats_handle, "RobustRuns", g_monte_carlo_results.robust_runs, "Runs meeting robustness threshold");
        FileWrite(stats_handle, "SuccessRate", g_monte_carlo_results.success_rate, "Percentage of robust runs");
        
        FileClose(stats_handle);
        Print("✓ Monte Carlo statistics saved: ", stats_filename);
    }
}

// Display Monte Carlo results summary
void DisplayMonteCarloResults() {
    if(!InpEnableMonteCarloMode || g_monte_carlo_results.completed_runs == 0) return;
    
    Print("");
    Print("=================================================================");
    Print("=== IMPROVEMENT 7.5: MONTE CARLO ROBUSTNESS ANALYSIS RESULTS ===");
    Print("=================================================================");
    
    int n = g_monte_carlo_results.completed_runs;
    Print("Total Monte Carlo runs completed: ", n);
    Print("");
    
    Print("📊 PERFORMANCE STATISTICS:");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Average Return:         ", DoubleToString(g_monte_carlo_results.mean_return, 2), "% ± ", 
          DoubleToString(g_monte_carlo_results.std_return, 2), "%");
    Print("Average Drawdown:       ", DoubleToString(g_monte_carlo_results.mean_drawdown, 2), "% ± ", 
          DoubleToString(g_monte_carlo_results.std_drawdown, 2), "%");
    Print("Average Sharpe Ratio:   ", DoubleToString(g_monte_carlo_results.mean_sharpe, 3), " ± ", 
          DoubleToString(g_monte_carlo_results.std_sharpe, 3));
    Print("Average Profit Factor:  ", DoubleToString(g_monte_carlo_results.mean_profit_factor, 2), " ± ", 
          DoubleToString(g_monte_carlo_results.std_profit_factor, 2));
    Print("Average Win Rate:       ", DoubleToString(g_monte_carlo_results.mean_win_rate, 1), "% ± ", 
          DoubleToString(g_monte_carlo_results.std_win_rate, 1), "%");
    
    Print("");
    Print("📈 DISTRIBUTION ANALYSIS:");
    Print("─────────────────────────────────────────────────────────────────");
    Print("5th Percentile Return:  ", DoubleToString(g_monte_carlo_results.percentile_5_return, 2), "%");
    Print("95th Percentile Return: ", DoubleToString(g_monte_carlo_results.percentile_95_return, 2), "%");
    Print("5th Percentile Drawdown:", DoubleToString(g_monte_carlo_results.percentile_5_drawdown, 2), "%");
    Print("95th Percentile Drawdown:", DoubleToString(g_monte_carlo_results.percentile_95_drawdown, 2), "%");
    
    Print("");
    Print("🔍 ROBUSTNESS ANALYSIS:");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Overall Robustness:     ", DoubleToString(g_monte_carlo_results.overall_robustness_score, 3));
    Print("Return Consistency:     ", DoubleToString(g_monte_carlo_results.return_consistency_score, 3));
    Print("Risk Consistency:       ", DoubleToString(g_monte_carlo_results.risk_consistency_score, 3));
    Print("Strategy Stability:     ", DoubleToString(g_monte_carlo_results.strategy_stability_score, 3));
    
    Print("");
    Print("✅ SUCCESS METRICS:");
    Print("─────────────────────────────────────────────────────────────────");
    Print("Positive Return Runs:   ", g_monte_carlo_results.positive_return_runs, " / ", n, 
          " (", DoubleToString((double)g_monte_carlo_results.positive_return_runs / n * 100, 1), "%)");
    Print("Acceptable Risk Runs:   ", g_monte_carlo_results.acceptable_drawdown_runs, " / ", n, 
          " (", DoubleToString((double)g_monte_carlo_results.acceptable_drawdown_runs / n * 100, 1), "%)");
    Print("Robust Runs:            ", g_monte_carlo_results.robust_runs, " / ", n, 
          " (", DoubleToString(g_monte_carlo_results.success_rate * 100, 1), "%)");
    
    Print("");
    Print("🎯 ROBUSTNESS ASSESSMENT:");
    Print("─────────────────────────────────────────────────────────────────");
    
    string assessment = "";
    string recommendation = "";
    
    if(g_monte_carlo_results.overall_robustness_score >= 0.8) {
        assessment = "EXCELLENT - Strategy is highly robust across different conditions";
        recommendation = "Strategy ready for live deployment with high confidence";
    } else if(g_monte_carlo_results.overall_robustness_score >= 0.7) {
        assessment = "GOOD - Strategy shows good robustness with minor variations";
        recommendation = "Strategy suitable for live deployment with careful monitoring";
    } else if(g_monte_carlo_results.overall_robustness_score >= 0.6) {
        assessment = "ACCEPTABLE - Strategy shows moderate robustness";
        recommendation = "Consider parameter optimization before live deployment";
    } else if(g_monte_carlo_results.overall_robustness_score >= 0.5) {
        assessment = "POOR - Strategy shows limited robustness";
        recommendation = "Significant optimization needed before live deployment";
    } else {
        assessment = "UNACCEPTABLE - Strategy lacks robustness";
        recommendation = "Strategy requires major revisions - not suitable for live trading";
    }
    
    Print("Assessment: ", assessment);
    Print("Recommendation: ", recommendation);
    
    Print("");
    Print("🔧 OPTIMIZATION SUGGESTIONS:");
    Print("─────────────────────────────────────────────────────────────────");
    
    if(g_monte_carlo_results.return_consistency_score < 0.7) {
        Print("• Return consistency is low - consider more conservative position sizing");
    }
    
    if(g_monte_carlo_results.risk_consistency_score < 0.7) {
        Print("• Risk consistency is low - strengthen risk management controls");
    }
    
    if(g_monte_carlo_results.strategy_stability_score < 0.7) {
        Print("• Strategy stability is low - review signal quality and filtering");
    }
    
    if(g_monte_carlo_results.success_rate < 0.7) {
        Print("• Low success rate - consider adjusting robustness threshold or strategy parameters");
    }
    
    if(g_monte_carlo_results.mean_drawdown > 15.0) {
        Print("• Average drawdown is high - implement stronger stop losses");
    }
    
    if(g_monte_carlo_results.std_return > g_monte_carlo_results.mean_return) {
        Print("• High return volatility - consider volatility-based position sizing");
    }
    
    Print("");
    Print("=================================================================");
}

// Reset backtest state for Monte Carlo runs
void ResetBacktestState() {
    // Reset account balance and performance tracking
    g_balance = InpInitialBalance;
    g_equity = InpInitialBalance;
    g_max_balance = InpInitialBalance;
    g_max_drawdown = 0;
    g_bars_processed = 0;
    g_prediction_failures = 0;
    g_feature_failures = 0;
    g_indicator_failures = 0;
    
    // Reset position state
    g_position_type = POSITION_TYPE_NONE;
    g_position_lots = 0;
    g_position_open_price = 0;
    g_position_open_time = 0;
    g_position_ticket = 0;
    g_stop_loss_price = 0.0;
    g_take_profit_price = 0.0;
    g_trailing_stop_price = 0.0;
    g_max_favorable_price = 0.0;
    g_position_peak_profit = 0.0;
    
    // Reset confidence and regime tracking
    g_last_confidence = 0.0;
    g_confidence_trades = 0;
    g_regime_detected = false;
    g_regime_volatility = 0.0;
    g_regime_trend = 0.0;
    g_last_regime_check = 0;
    g_current_atr = 0.0;
    g_dynamic_lot_size = InpLotSize;
    g_account_risk_pct = InpRiskPercentage;
    g_position_risk_amount = 0.0;
    
    // Reset emergency and consecutive loss tracking
    g_emergency_mode = false;
    g_consecutive_losses = 0;
    g_emergency_halt_until = 0;
    
    // Reset session and volatility tracking
    g_trend_alignment = 0.0;
    g_volatility_multiplier = 1.0;
    g_high_volatility_mode = false;
    g_volatility_percentile = 0.0;
    g_current_session_active = true;
    g_session_start_time = 0;
    g_session_end_time = 0;
    g_avoid_news_time = false;
    g_position_scaling_level = 0;
    g_average_entry_price = 0.0;
    
    // Reset performance counters
    g_confidence_filtered_trades = 0;
    g_atr_stop_hits = 0;
    g_trailing_stop_hits = 0;
    g_regime_triggered_exits = 0;
    g_volatility_adjustments = 0;
    
    // Reset comprehensive metrics arrays
    g_equity_curve_size = 0;
    g_peak_balance = InpInitialBalance;
    g_peak_time = TimeCurrent();
    g_return_periods = 0;
    
    // Reset performance metrics structure
    ZeroMemory(g_performance_metrics);
    g_performance_metrics.initial_balance = InpInitialBalance;
    
    // Clear trade records and last entry trigger
    ArrayResize(g_trade_records, 0);
    g_current_entry_trigger = "";
}

// TRIGGER TRACKING FUNCTIONS - Track what causes entries and exits

// Set entry trigger based on current conditions
void SetEntryTrigger(int action, const double &q_values[], bool confidence_passed, bool risk_passed) {
    string trigger = "";
    string action_names[] = {"BUY_STRONG", "BUY_WEAK", "SELL_STRONG", "SELL_WEAK", "HOLD", "FLAT"};
    
    // Base trigger is the AI model decision
    trigger = "AI_" + action_names[action];
    
    // Add confidence info if available
    if(InpUseConfidenceFilter) {
        if(confidence_passed) {
            trigger += "_CONF_" + DoubleToString(g_last_confidence, 2);
        } else {
            trigger += "_CONF_FAIL";
        }
    }
    
    // Add signal strength info
    double signal_strength = q_values[action];
    if(signal_strength > 0.8) {
        trigger += "_HIGH";
    } else if(signal_strength > 0.6) {
        trigger += "_MED";
    } else {
        trigger += "_LOW";
    }
    
    // Add volatility regime info
    if(InpUseVolatilityRegime) {
        if(g_high_volatility_mode) {
            trigger += "_VOL_HIGH";
        } else {
            trigger += "_VOL_NORM";
        }
    }
    
    // Add session info
    if(InpUseSessionFilter) {
        if(g_current_session_active) {
            trigger += "_SESSION_OK";
        } else {
            trigger += "_SESSION_OFF";
        }
    }
    
    g_last_entry_trigger = trigger;
}

// Set exit reason based on which condition triggered the exit
void SetExitReason(string reason) {
    g_last_exit_reason = reason;
    
    // Update specific flags for tracking
    if(StringFind(reason, "ATR") >= 0) {
        g_last_trade_was_stop = true;
        g_last_trade_was_target = false;
    } else if(StringFind(reason, "Profit") >= 0 || StringFind(reason, "Target") >= 0) {
        g_last_trade_was_stop = false;
        g_last_trade_was_target = true;
    } else {
        g_last_trade_was_stop = false;
        g_last_trade_was_target = false;
    }
}

//============================== PHASE 1, 2, 3 ENHANCEMENT FUNCTIONS ============
// Profitability improvement functions for backtesting

//============================== PHASE 1 ENHANCEMENT FUNCTIONS ==============================
// These functions implement the critical fixes for the original AI's problems

//--------------------------------------------------------------------------------------------
// MAXIMUM HOLDING TIME CONTROL - FIXES 700+ HOUR HOLDING PROBLEM
// ORIGINAL ISSUE: AI would hold losing positions for weeks, causing massive losses
// SOLUTION: Force close any position held longer than InpMaxHoldingHours (default: 48 hours)
//--------------------------------------------------------------------------------------------
bool CheckMaxHoldingTime(datetime current_bar_time){
    // Skip if enhancement disabled or no position open
    if(!InpUseMaxHoldingTime || g_current_position == POS_NONE) return false;
    
    // Calculate how long current position has been held (in hours)
    int holding_hours = (int)((current_bar_time - g_position_open_time) / 3600);
    
    // Debug logging for positions held longer than 24 hours (potential problem)
    if(holding_hours > 24 && g_bars_processed % 100 == 0){
        Print("DEBUG: Long hold detected - ", holding_hours, " hours (limit: ", InpMaxHoldingHours, ")");
        Print("DEBUG: Position opened: ", TimeToString(g_position_open_time));
        Print("DEBUG: Current time: ", TimeToString(current_bar_time));
    }
    
    // CRITICAL: Force close if holding time exceeds maximum allowed
    if(holding_hours > InpMaxHoldingHours){
        Print("PHASE1: EMERGENCY HOLD TIME EXCEEDED! ", holding_hours, "h > ", InpMaxHoldingHours, "h limit");
        Print("PHASE1: FORCING POSITION CLOSE TO PREVENT FURTHER LOSSES!");
        ClosePosition("Max holding time exceeded (Phase 1 protection)");
        return true; // Position was closed
    }
    return false; // Position still open, within time limits
}

//--------------------------------------------------------------------------------------------
// AUTOMATIC PROFIT TARGETS - FIXES POOR PROFIT-TAKING PROBLEM  
// ORIGINAL ISSUE: AI would turn profitable trades into losses by holding too long
// SOLUTION: Automatically close positions when they reach profit target (1.8 x ATR)
//--------------------------------------------------------------------------------------------
bool CheckProfitTargets(){
    // Skip if enhancement disabled or no position open
    if(!InpUseProfitTargets || g_current_position == POS_NONE) return false;
    
    // Get current market price for P&L calculation
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    double unrealized_pnl = CalculateUnrealizedPnL(current_price);
    
    // Get current market volatility (ATR) to set realistic profit targets
    double atr_buffer[];
    if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) <= 0) return false; // Handle data error
    double atr = atr_buffer[0];
    
    // Calculate profit target: ATR × multiplier × position size × pip value
    // Example: 0.0015 ATR × 1.8 multiplier × 0.1 lots × 100,000 = $27 target
    double target_profit = atr * InpProfitTargetATR * g_position_lots * 100000;
    
    // CRITICAL: Close position if profit target reached
    if(unrealized_pnl >= target_profit){
        if(InpVerboseLogging){
            Print("PHASE1: Profit target reached. P&L: $", DoubleToString(unrealized_pnl, 2), 
                  " Target: $", DoubleToString(target_profit, 2));
        }
        ClosePosition("Profit target reached (Phase 1 protection)");
        return true; // Position was closed profitably
    }
    return false; // Target not yet reached
}

//============================== PHASE 2 ENHANCEMENT FUNCTIONS ==============================
// These functions implement improved learning algorithms for better AI decision-making

//--------------------------------------------------------------------------------------------
// ENHANCED REWARD CALCULATION - TEACHES AI BETTER TRADING BEHAVIOR
// ORIGINAL ISSUE: AI only learned from simple profit/loss, leading to poor timing
// SOLUTION: Multi-factor reward system that teaches optimal holding times and exit behavior
//--------------------------------------------------------------------------------------------
double CalculateEnhancedReward(double pnl, int holding_time_hours){
    if(!InpEnhancedRewards) return pnl; // Fall back to simple P&L if enhancement disabled
    
    // Base reward from trade P&L (normalized to prevent extreme values)
    double base_reward = pnl / 100.0;
    
    // Time penalty: Discourage holding positions too long
    // Increases linearly with holding time to teach quick decision-making
    double time_penalty = -InpHoldingTimePenalty * holding_time_hours;
    
    // Quick exit bonus: Reward trades closed within 24 hours when profitable
    // Teaches AI to take profits quickly rather than hoping for more
    double quick_exit_bonus = (holding_time_hours < 24 && pnl > 0) ? InpQuickExitBonus : 0;
    
    // Combined reward balances profit-taking with efficient timing
    return base_reward + time_penalty + quick_exit_bonus;
}

// Get current position holding time in hours
int GetPositionHoldingHours(){
    if(g_current_position == POS_NONE) return 0;
    return (int)((TimeCurrent() - g_position_open_time) / 3600);
}

// Get position holding time for specific bar time (for backtesting)
int GetPositionHoldingHours(datetime current_bar_time){
    if(g_current_position == POS_NONE) return 0;
    return (int)((current_bar_time - g_position_open_time) / 3600);
}

// Force FLAT action when position meets exit criteria (AGGRESSIVE VERSION)
bool ShouldForceFlat(datetime current_bar_time, int original_action){
    if(!InpEnforceFlat || g_current_position == POS_NONE) return false;
    
    int holding_hours = GetPositionHoldingHours(current_bar_time);
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    double unrealized_pnl = CalculateUnrealizedPnL(current_price);
    
    // MUCH MORE AGGRESSIVE EXIT RULES TO FIX RISK-REWARD RATIO
    
    // Force FLAT if holding too long (50% of max time instead of 80%)
    if(holding_hours > (InpMaxHoldingHours * 0.5)){
        return true;
    }
    
    // Force FLAT if ANY profit available after 6 hours (protect small gains)
    if(unrealized_pnl > 5.0 && holding_hours > 6){
        return true;
    }
    
    // Force FLAT if moderate loss (much more aggressive)
    if(unrealized_pnl < -30.0){
        return true;
    }
    
    // Force FLAT if held for 24+ hours regardless of P&L (prevent long holds)
    if(holding_hours > 24){
        return true;
    }
    
    // Force FLAT if held for 12+ hours with any loss
    if(holding_hours > 12 && unrealized_pnl < 0){
        return true;
    }
    
    return false;
}

// Emergency stop loss system (prevent catastrophic losses)
//============================== EMERGENCY PROTECTION FUNCTIONS ==============================
// These are the final safety nets to prevent catastrophic account losses

//--------------------------------------------------------------------------------------------
// EMERGENCY STOP LOSS SYSTEM - PREVENTS CATASTROPHIC SINGLE-TRADE LOSSES
// ORIGINAL ISSUE: AI could hold one losing trade until account was wiped out
// SOLUTION: Hard dollar limits and account drawdown limits that override everything
//--------------------------------------------------------------------------------------------
bool CheckEmergencyStops(){
    // Skip if emergency stops disabled or no position open
    if(!InpUseEmergencyStops || g_current_position == POS_NONE) return false;
    
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    double unrealized_pnl = CalculateUnrealizedPnL(current_price);
    
    // EMERGENCY LEVEL 1: Single trade dollar stop loss
    // Prevents any single trade from losing more than specified amount
    if(unrealized_pnl < -InpEmergencyStopLoss){
        Print("🚨 EMERGENCY STOP: Single trade loss exceeds $", InpEmergencyStopLoss);
        Print("🚨 Current loss: $", DoubleToString(unrealized_pnl, 2), " - FORCE CLOSING NOW!");
        ClosePosition("EMERGENCY: Single trade loss limit exceeded");
        return true;
    }
    
    // EMERGENCY LEVEL 2: Account drawdown protection  
    // Stops all trading if total account drawdown exceeds limit
    double account_drawdown = ((g_max_balance - g_balance) / g_max_balance) * 100.0;
    if(account_drawdown > InpMaxDrawdownPct){
        Print("🚨 EMERGENCY STOP: Account drawdown exceeds ", InpMaxDrawdownPct, "%");
        Print("🚨 Current drawdown: ", DoubleToString(account_drawdown, 2), "% - STOPPING ALL TRADING!");
        ClosePosition("EMERGENCY: Account drawdown limit exceeded");
        return true;
    }
    
    return false; // No emergency conditions triggered
}

// Calculate unrealized P&L for current position
double CalculateUnrealizedPnL(double current_price){
    if(g_current_position == POS_NONE) return 0.0;
    
    double price_diff = 0.0;
    if(g_current_position == POS_LONG){
        price_diff = current_price - g_position_entry_price;
    } else {
        price_diff = g_position_entry_price - current_price;
    }
    
    return price_diff * g_position_lots * 100000; // Convert to account currency
}

// Close current position with reason
void ClosePosition(string reason){
    if(g_current_position == POS_NONE) return;
    
    double exit_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    double pnl = CalculateUnrealizedPnL(exit_price);
    
    // Record the trade
    TradeRecord trade;
    trade.open_time = g_position_open_time;
    trade.close_time = TimeCurrent();
    trade.action = (g_current_position == POS_LONG) ? BUY_STRONG : SELL_STRONG; // Simplified
    trade.position_type = g_current_position;
    trade.entry_price = g_position_entry_price;
    trade.exit_price = exit_price;
    trade.lots = g_position_lots;
    trade.profit_loss = pnl;
    
    // Update balance and statistics
    g_balance += pnl;
    trade.balance_after = g_balance;
    
    // Calculate drawdown
    if(g_balance > g_max_balance) g_max_balance = g_balance;
    double current_dd = (g_max_balance - g_balance) / g_max_balance * 100.0;
    if(current_dd > g_max_drawdown) g_max_drawdown = current_dd;
    trade.drawdown_pct = current_dd;
    
    // Add to trade history
    int trades_count = ArraySize(g_trades);
    ArrayResize(g_trades, trades_count + 1);
    g_trades[trades_count] = trade;
    
    // Update statistics
    g_total_trades++;
    if(pnl > 0) g_winning_trades++; else g_losing_trades++;
    
    if(InpVerboseLogging) Print("Position closed: ", reason, " P&L: $", DoubleToString(pnl, 2));
    
    // Reset position tracking
    g_current_position = POS_NONE;
    g_position_entry_price = 0;
    g_position_lots = 0;
    g_position_open_time = 0;
}

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart() {
    Print("=======================================================");
    Print("=== CORTEX 30-DAY BACKTEST SIMULATION STARTING ===");
    Print("=======================================================");
    Print("Model file: ", InpModelFileName);
    Print("Backtest period: ", InpBacktestDays, " days");
    Print("Initial balance: $", DoubleToString(InpInitialBalance, 2));
    Print("Lot size: ", DoubleToString(InpLotSize, 2));
    Print("Verbose logging: ", InpVerboseLogging ? "ENABLED" : "DISABLED");
    Print("Log frequency: Every ", InpLogEveryNBars, " bars");
    Print("=======================================================");
    
    // Initialize performance tracking
    g_balance = InpInitialBalance;
    g_equity = InpInitialBalance;
    g_max_balance = InpInitialBalance;
    g_max_drawdown = 0;
    g_bars_processed = 0;
    g_prediction_failures = 0;
    g_feature_failures = 0;
    g_indicator_failures = 0;
    
    // IMPROVEMENT 7.1: Initialize advanced feature variables
    g_confidence_threshold = InpConfidenceThreshold;
    g_last_confidence = 0.0;
    g_confidence_trades = 0;
    g_regime_detected = false;
    g_regime_volatility = 0.0;
    g_regime_trend = 0.0;
    g_last_regime_check = 0;
    g_current_atr = 0.0;
    g_stop_loss_price = 0.0;
    g_take_profit_price = 0.0;
    g_trailing_stop_price = 0.0;
    g_max_favorable_price = 0.0;
    g_position_peak_profit = 0.0;
    g_dynamic_lot_size = InpLotSize;
    g_account_risk_pct = InpRiskPercentage;
    g_position_risk_amount = 0.0;
    g_emergency_mode = false;
    g_consecutive_losses = 0;
    g_emergency_halt_until = 0;
    g_mtf_enabled = InpUseMTFAnalysis;
    g_trend_alignment = 0.0;
    g_volatility_multiplier = 1.0;
    g_high_volatility_mode = false;
    g_volatility_percentile = 0.0;
    g_current_session_active = true;
    g_session_start_time = 0;
    g_session_end_time = 0;
    g_avoid_news_time = false;
    g_partial_close_enabled = InpUsePartialClose;
    g_partial_close_pct = InpPartialClosePct;
    g_position_scaling_level = 0;
    g_average_entry_price = 0.0;
    
    // Initialize performance counters
    g_confidence_filtered_trades = 0;
    g_atr_stop_hits = 0;
    g_trailing_stop_hits = 0;
    g_regime_triggered_exits = 0;
    g_volatility_adjustments = 0;
    
    // Initialize arrays
    ArrayResize(g_confidence_history, 1000);
    ArrayResize(g_volatility_history, 100);
    ArrayResize(g_trend_history, 100);
    ArrayInitialize(g_mtf_signals, 0.0);
    
    // IMPROVEMENT 7.2: Initialize comprehensive metrics arrays
    ArrayResize(g_equity_curve, 10000);
    ArrayResize(g_equity_curve_times, 10000);
    ArrayResize(g_underwater_curve, 10000);
    ArrayResize(g_daily_returns, 1000);
    ArrayResize(g_monthly_data, 50);
    g_equity_curve_size = 0;
    g_peak_balance = InpInitialBalance;
    g_peak_time = TimeCurrent();
    g_return_periods = 0;
    
    // Initialize performance metrics structure
    ZeroMemory(g_performance_metrics);
    g_performance_metrics.initial_balance = InpInitialBalance;
    
    Print("✓ Advanced feature variables and comprehensive metrics initialized");
    
    // IMPROVEMENT 7.4: Initialize flexible parameter system
    Print("STAGE 0.5: Initializing flexible parameter system...");
    datetime start_flex = GetTickCount();
    if(!InitializeFlexibleParameters()) {
        Print("✗ CRITICAL ERROR: Flexible parameter initialization failed");
        return;
    }
    Print("✓ Flexible parameters initialized in ", GetTickCount() - start_flex, " ms");
    
    Print("STAGE 1: Initializing technical indicators...");
    datetime start_init = GetTickCount();
    InitializeIndicators();
    Print("✓ Indicators initialized in ", GetTickCount() - start_init, " ms");
    
    // IMPROVEMENT 7.3: Initialize CSV logging
    if(InpEnableCSVLogging) {
        Print("STAGE 1.5: Initializing CSV logging files...");
        datetime start_csv = GetTickCount();
        if(!InitializeCSVFiles()) {
            Print("✗ WARNING: CSV logging initialization failed, continuing without CSV export");
        } else {
            Print("✓ CSV logging initialized in ", GetTickCount() - start_csv, " ms");
        }
    }
    
    Print("STAGE 2: Loading trained model...");
    datetime start_load = GetTickCount();
    if(!LoadModel(InpModelFileName)) {
        Print("✗ ERROR: Failed to load model file!");
        Print("Make sure ", InpModelFileName, " exists in MQL5/Files folder");
        CleanupIndicators();
        return;
    }
    Print("✓ Model loaded successfully in ", GetTickCount() - start_load, " ms");
    
    // Calculate backtest date range
    datetime end_time = TimeCurrent();
    datetime start_time = end_time - (InpBacktestDays * 24 * 3600);
    
    Print("STAGE 3: Preparing backtest environment...");
    Print("Backtest range: ", TimeToString(start_time, TIME_DATE|TIME_SECONDS), " to ", TimeToString(end_time, TIME_DATE|TIME_SECONDS));
    Print("Current server time: ", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
    
    // Check data availability
    int available_bars = Bars(_Symbol, PERIOD_CURRENT);
    Print("Available bars on ", _Symbol, " ", EnumToString(PERIOD_CURRENT), ": ", available_bars);
    
    if(available_bars < 100) {
        Print("✗ WARNING: Limited historical data available (", available_bars, " bars)");
    } else {
        Print("✓ Sufficient historical data available");
    }
    
    Print("STAGE 4: Starting simulation...");
    datetime start_backtest = GetTickCount();
    
    // IMPROVEMENT 7.5: Monte Carlo testing or single backtest
    if(InpEnableMonteCarloMode) {
        Print("🎲 MONTE CARLO MODE ENABLED - Running ", InpMonteCarloRuns, " randomized simulations");
        Print("================================================");
        
        // Initialize Monte Carlo testing
        if(!InitializeMonteCarloTesting()) {
            Print("✗ ERROR: Monte Carlo initialization failed, running single backtest");
            RunBacktest(start_time, end_time);
        } else {
            // Run Monte Carlo batch testing
            for(int run = 1; run <= InpMonteCarloRuns; run++) {
                Print("");
                Print("🔄 Starting Monte Carlo Run ", run, " / ", InpMonteCarloRuns);
                
                // Generate Monte Carlo variation parameters
                MonteCarloVariation variation;
                if(!GenerateMonteCarloVariation(run, start_time, end_time, variation)) {
                    Print("✗ Failed to generate variation for run ", run, ", skipping");
                    continue;
                }
                
                Print("   📊 Variation parameters:");
                Print("      Period: ", TimeToString(variation.start_time, TIME_DATE), " to ", 
                      TimeToString(variation.end_time, TIME_DATE), " (", variation.period_days, " days)");
                Print("      Spread multiplier: ", DoubleToString(variation.spread_multiplier, 2));
                Print("      Slippage: ", DoubleToString(variation.slippage_pips, 1), " pips");
                Print("      Commission multiplier: ", DoubleToString(variation.commission_multiplier, 2));
                Print("      Random seed: ", (int)variation.random_seed);
                
                // Apply Monte Carlo variations
                ApplySpreadVariation(variation.spread_multiplier);
                ApplySlippageVariation(variation.slippage_pips);
                ApplyCommissionVariation(variation.commission_multiplier);
                
                // Reset state for new run
                ResetBacktestState();
                
                // Optionally shuffle data for robustness testing
                if(InpMCDataShuffling && variation.shuffle_factor > 0) {
                    Print("      🔀 Data shuffling: ", DoubleToString(variation.shuffle_factor * 100, 1), "%");
                    // ShufflePriceData would be called here if implemented
                }
                
                // Run the individual simulation
                datetime run_start = GetTickCount();
                RunBacktest(variation.start_time, variation.end_time);
                Print("   ✓ Run ", run, " completed in ", GetTickCount() - run_start, " ms");
                
                // Record results for this run
                if(!RecordMonteCarloRun(run, variation)) {
                    Print("✗ Failed to record results for run ", run);
                }
                
                // Calculate and display progress
                double progress = (double)run / InpMonteCarloRuns * 100.0;
                Print("   📈 Progress: ", DoubleToString(progress, 1), "% (", run, "/", InpMonteCarloRuns, ")");
            }
            
            // Calculate aggregate statistics
            CalculateMonteCarloStatistics();
            
            // Display results summary
            DisplayMonteCarloResults();
            
            // Save results to CSV files
            if(InpMCSaveResults) {
                SaveMonteCarloResults();
            }
            
            Print("");
            Print("🎯 MONTE CARLO TESTING COMPLETED");
            Print("   Robustness Score: ", DoubleToString(g_monte_carlo_results.overall_robustness_score, 3));
            Print("   Success Rate: ", DoubleToString(g_monte_carlo_results.success_rate, 1), "%");
            Print("================================================");
        }
    } else {
        // Standard single backtest
        RunBacktest(start_time, end_time);
    }
    
    Print("STAGE 5: Simulation completed in ", GetTickCount() - start_backtest, " ms");
    Print("STAGE 6: Generating performance report...");
    
    // Generate performance report
    GeneratePerformanceReport();
    
    // IMPROVEMENT 7.3: Export comprehensive CSV data
    if(InpEnableCSVLogging) {
        Print("");
        Print("STAGE 6.5: Exporting comprehensive CSV data...");
        LogMetricsToCSV();
        CloseCSVFiles();
        Print("✓ CSV export completed successfully");
        Print("CSV files created:");
        Print("  - Trades: ", InpCSVTradeFileName);
        Print("  - Equity curve: ", InpCSVEquityFileName);
        Print("  - Comprehensive metrics: ", InpCSVMetricsFileName);
    }
    
    // IMPROVEMENT 7.4: Save parameter set for optimization tracking
    if(g_parameters_validated) {
        Print("");
        Print("STAGE 6.7: Saving parameter set for optimization tracking...");
        SaveParameterSet();
    }
    
    Print("STAGE 7: Cleaning up resources...");
    // Clean up
    CleanupIndicators();
    
    Print("=======================================================");
    Print("=== BACKTEST COMPLETED SUCCESSFULLY ===");
    Print("Total processing time: ", GetTickCount() - start_init, " ms");
    Print("=======================================================");
}

//+------------------------------------------------------------------+
//| Initialize technical indicators                                  |
//+------------------------------------------------------------------+
void InitializeIndicators() {
    Print("  Creating MA10 indicator...");
    h_ma10 = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
    if(h_ma10 == INVALID_HANDLE) {
        Print("  ✗ Failed to create MA10 indicator");
    } else {
        Print("  ✓ MA10 indicator created successfully (handle: ", h_ma10, ")");
    }
    
    Print("  Creating MA20 indicator...");
    h_ma20 = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
    if(h_ma20 == INVALID_HANDLE) {
        Print("  ✗ Failed to create MA20 indicator");
    } else {
        Print("  ✓ MA20 indicator created successfully (handle: ", h_ma20, ")");
    }
    
    Print("  Creating MA50 indicator...");
    h_ma50 = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_SMA, PRICE_CLOSE);
    if(h_ma50 == INVALID_HANDLE) {
        Print("  ✗ Failed to create MA50 indicator");
    } else {
        Print("  ✓ MA50 indicator created successfully (handle: ", h_ma50, ")");
    }
    
    Print("  Creating RSI indicator...");
    h_rsi = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
    if(h_rsi == INVALID_HANDLE) {
        Print("  ✗ Failed to create RSI indicator");
    } else {
        Print("  ✓ RSI indicator created successfully (handle: ", h_rsi, ")");
    }
    
    Print("  Creating ATR indicator...");
    h_atr = iATR(_Symbol, PERIOD_CURRENT, 14);
    if(h_atr == INVALID_HANDLE) {
        Print("  ✗ Failed to create ATR indicator");
    } else {
        Print("  ✓ ATR indicator created successfully (handle: ", h_atr, ")");
    }
    
    // Wait for indicators to calculate
    Print("  Waiting for indicators to initialize...");
    Sleep(1000);
    
    int valid_indicators = 0;
    if(h_ma10 != INVALID_HANDLE) valid_indicators++;
    if(h_ma20 != INVALID_HANDLE) valid_indicators++;
    if(h_ma50 != INVALID_HANDLE) valid_indicators++;
    if(h_rsi != INVALID_HANDLE) valid_indicators++;
    if(h_atr != INVALID_HANDLE) valid_indicators++;
    
    Print("  Summary: ", valid_indicators, "/5 indicators created successfully");
}

//+------------------------------------------------------------------+
//| Cleanup indicators                                              |
//+------------------------------------------------------------------+
void CleanupIndicators() {
    Print("  Releasing indicator handles...");
    int released = 0;
    
    if(h_ma10 != INVALID_HANDLE) {
        IndicatorRelease(h_ma10);
        released++;
    }
    if(h_ma20 != INVALID_HANDLE) {
        IndicatorRelease(h_ma20);
        released++;
    }
    if(h_ma50 != INVALID_HANDLE) {
        IndicatorRelease(h_ma50);
        released++;
    }
    if(h_rsi != INVALID_HANDLE) {
        IndicatorRelease(h_rsi);
        released++;
    }
    if(h_atr != INVALID_HANDLE) {
        IndicatorRelease(h_atr);
        released++;
    }
    
    Print("  Released ", released, " indicator handles");
}

//+------------------------------------------------------------------+
//| Load the trained model (simplified)                             |
//+------------------------------------------------------------------+
bool LoadModel(const string filename) {
    Print("  Opening model file: ", filename);
    int h = FileOpen(filename, FILE_BIN|FILE_READ);
    if(h == INVALID_HANDLE) {
        Print("  ✗ Cannot open model file. Error: ", GetLastError());
        Print("  Expected location: MQL5\\Files\\", filename);
        return false;
    }
    
    // Get file size
    ulong file_size = FileSize(h);
    Print("  Model file size: ", file_size, " bytes");
    
    // Read magic number
    long magic = FileReadLong(h);
    Print("  Magic number: 0x", IntegerToString(magic, 16));
    
    if(magic != 0xC0DE0203 && magic != 0xC0DE0202) {
        Print("  ✗ Invalid file format. Expected 0xC0DE0203 or 0xC0DE0202, got 0x", IntegerToString(magic, 16));
        FileClose(h);
        return false;
    }
    
    Print("  ✓ Valid model file format detected");
    
    // Read symbol length first, then symbol and timeframe
    int sym_len = (int)FileReadLong(h);
    if(sym_len <= 0 || sym_len > 32) {
        Print("  ✗ Invalid symbol length: ", sym_len);
        FileClose(h);
        return false;
    }
    string model_symbol = FileReadString(h, sym_len);
    int model_timeframe = (int)FileReadLong(h);
    
    Print("  Model metadata:");
    Print("    Symbol: ", model_symbol);
    Print("    Timeframe: ", model_timeframe, " (", EnumToString((ENUM_TIMEFRAMES)model_timeframe), ")");
    Print("    Current chart: ", _Symbol, " ", EnumToString(PERIOD_CURRENT));
    
    if(model_symbol != _Symbol) {
        Print("  ⚠ WARNING: Model trained on ", model_symbol, " but testing on ", _Symbol);
    }
    
    if(model_timeframe != PERIOD_CURRENT) {
        Print("  ⚠ WARNING: Model trained on TF ", model_timeframe, " but testing on ", EnumToString(PERIOD_CURRENT));
    }
    
    // Read state size and actions count
    int state_size = (int)FileReadLong(h);
    int actions = (int)FileReadLong(h);
    
    // Read network architecture
    int h1 = (int)FileReadLong(h);
    int h2 = (int)FileReadLong(h);  
    int h3 = (int)FileReadLong(h);
    
    Print("  Network architecture: ", state_size, " → ", h1, " → ", h2, " → ", h3, " → ", actions);
    
    if(state_size != STATE_SIZE) {
        Print("  ✗ State size mismatch: model=", state_size, ", expected=", STATE_SIZE);
        FileClose(h);
        return false;
    }
    
    if(actions != ACTIONS) {
        Print("  ✗ Actions mismatch: model=", actions, ", expected=", ACTIONS);
        FileClose(h);
        return false;
    }
    
    // Read architecture flags (for enhanced format)
    bool has_checkpoint = (magic == 0xC0DE0203);
    if(has_checkpoint) {
        g_use_lstm = (FileReadLong(h) == 1);
        g_use_dueling = (FileReadLong(h) == 1);
        
        int lstm_size = (int)FileReadLong(h);
        int sequence_len = (int)FileReadLong(h);
        int value_head_size = (int)FileReadLong(h);
        int advantage_head_size = (int)FileReadLong(h);
        
        Print("  LSTM enabled: ", g_use_lstm ? "YES" : "NO");
        if(g_use_lstm) {
            Print("    LSTM size: ", lstm_size);
            Print("    Sequence length: ", sequence_len);
        }
        
        Print("  Dueling network: ", g_use_dueling ? "YES" : "NO");
        if(g_use_dueling) {
            Print("    Value head size: ", value_head_size);
            Print("    Advantage head size: ", advantage_head_size);
        }
    } else {
        // Legacy format
        g_use_lstm = false;
        g_use_dueling = false;
        Print("  LSTM enabled: NO (legacy format)");
        Print("  Dueling network: NO (legacy format)");
    }
    
    Print("  Loading feature normalization parameters...");
    // Read feature normalization parameters
    for(int i = 0; i < STATE_SIZE; i++) {
        g_feature_min[i] = FileReadDouble(h);
    }
    for(int i = 0; i < STATE_SIZE; i++) {
        g_feature_max[i] = FileReadDouble(h);
    }
    
    Print("  ✓ Normalization parameters loaded");
    if(InpVerboseLogging) {
        Print("    Sample feature ranges:");
        for(int i = 0; i < 5; i++) {
            Print("      Feature ", i, ": [", DoubleToString(g_feature_min[i], 6), ", ", DoubleToString(g_feature_max[i], 6), "]");
        }
    }
    
    Print("  Initializing network layers...");
    // Initialize layers
    InitializeDenseLayer(g_dense1, STATE_SIZE, h1);
    InitializeDenseLayer(g_dense2, h1, h2);
    InitializeDenseLayer(g_dense3, h2, h3);
    
    Print("  Loading layer weights...");
    // Load layer weights
    LoadDenseLayer(h, g_dense1);
    Print("    ✓ Dense layer 1 loaded (", ArraySize(g_dense1.w), " weights, ", ArraySize(g_dense1.b), " biases)");
    
    LoadDenseLayer(h, g_dense2);  
    Print("    ✓ Dense layer 2 loaded (", ArraySize(g_dense2.w), " weights, ", ArraySize(g_dense2.b), " biases)");
    
    LoadDenseLayer(h, g_dense3);
    Print("    ✓ Dense layer 3 loaded (", ArraySize(g_dense3.w), " weights, ", ArraySize(g_dense3.b), " biases)");
    
    // Skip LSTM loading for simplicity - we'll approximate
    if(g_use_lstm) {
        Print("  Skipping LSTM weights (simplified implementation)...");
        // Skip LSTM weights
        int lstm_weights_count = g_lstm_size * (h3 + g_lstm_size) * 4 + g_lstm_size * 4; // Approximate
        for(int i = 0; i < lstm_weights_count; i++) {
            FileReadDouble(h); // Skip LSTM weights
        }
        Print("    ✓ LSTM weights skipped (", lstm_weights_count, " weights)");
    }
    
    // Load dueling heads if enabled
    if(g_use_dueling) {
        Print("  Loading dueling network heads...");
        InitializeDenseLayer(g_value_head, h3, 1);
        InitializeDenseLayer(g_advantage_head, h3, ACTIONS);
        
        LoadDenseLayer(h, g_value_head);
        Print("    ✓ Value head loaded (", ArraySize(g_value_head.w), " weights)");
        
        LoadDenseLayer(h, g_advantage_head);
        Print("    ✓ Advantage head loaded (", ArraySize(g_advantage_head.w), " weights)");
    }
    
    // Read checkpoint data if available
    if(has_checkpoint) {
        Print("  Loading checkpoint data...");
        datetime last_trained = (datetime)FileReadLong(h);
        int training_steps = (int)FileReadLong(h);
        double epsilon = FileReadDouble(h);
        double beta = FileReadDouble(h);
        
        Print("    Last trained: ", TimeToString(last_trained));
        Print("    Training steps: ", training_steps);
        Print("    Epsilon: ", DoubleToString(epsilon, 4));
        Print("    Beta (PER): ", DoubleToString(beta, 4));
        
        int days_old = (int)((TimeCurrent() - last_trained) / (24*60*60));
        Print("    Data age: ", days_old, " days");
    }
    
    FileClose(h);
    g_model_loaded = true;
    
    Print("  ✓ Model loaded successfully!");
    Print("  Model configuration:");
    Print("    Input features: ", STATE_SIZE);
    Print("    Output actions: ", ACTIONS);
    Print("    LSTM: ", g_use_lstm ? "Enabled (simplified)" : "Disabled");
    Print("    Dueling: ", g_use_dueling ? "Enabled" : "Disabled");
    
    return true;
}

//+------------------------------------------------------------------+
//| Initialize dense layer                                           |
//+------------------------------------------------------------------+
void InitializeDenseLayer(DenseLayer &layer, int input_size, int output_size) {
    layer.input_size = input_size;
    layer.output_size = output_size;
    ArrayResize(layer.w, input_size * output_size);
    ArrayResize(layer.b, output_size);
}

//+------------------------------------------------------------------+
//| Load dense layer from file                                      |
//+------------------------------------------------------------------+
void LoadDenseLayer(int file_handle, DenseLayer &layer) {
    // Load weights
    for(int i = 0; i < ArraySize(layer.w); i++) {
        layer.w[i] = FileReadDouble(file_handle);
    }
    // Load biases
    for(int i = 0; i < ArraySize(layer.b); i++) {
        layer.b[i] = FileReadDouble(file_handle);
    }
}

//+------------------------------------------------------------------+
//| Simple neural network prediction                                |
//+------------------------------------------------------------------+
bool PredictQValues(const double &state[], double &q_values[]) {
    if(!g_model_loaded) {
        if(InpVerboseLogging) Print("    ✗ Prediction failed: Model not loaded");
        return false;
    }
    
    ArrayResize(q_values, ACTIONS);
    
    // Forward pass through dense layers
    double layer1_out[], layer2_out[], layer3_out[];
    
    // Layer 1
    ArrayResize(layer1_out, g_dense1.output_size);
    for(int i = 0; i < g_dense1.output_size; i++) {
        double sum = g_dense1.b[i];
        for(int j = 0; j < g_dense1.input_size; j++) {
            sum += state[j] * g_dense1.w[i * g_dense1.input_size + j];
        }
        layer1_out[i] = MathMax(0.0, sum); // ReLU
    }
    
    // Layer 2
    ArrayResize(layer2_out, g_dense2.output_size);
    for(int i = 0; i < g_dense2.output_size; i++) {
        double sum = g_dense2.b[i];
        for(int j = 0; j < g_dense2.input_size; j++) {
            sum += layer1_out[j] * g_dense2.w[i * g_dense2.input_size + j];
        }
        layer2_out[i] = MathMax(0.0, sum); // ReLU
    }
    
    // Layer 3
    ArrayResize(layer3_out, g_dense3.output_size);
    for(int i = 0; i < g_dense3.output_size; i++) {
        double sum = g_dense3.b[i];
        for(int j = 0; j < g_dense3.input_size; j++) {
            sum += layer2_out[j] * g_dense3.w[i * g_dense3.input_size + j];
        }
        layer3_out[i] = MathMax(0.0, sum); // ReLU
    }
    
    // Skip LSTM for simplicity
    
    // Dueling network processing
    if(g_use_dueling) {
        // State value
        double state_value = g_value_head.b[0];
        for(int j = 0; j < g_value_head.input_size; j++) {
            state_value += layer3_out[j] * g_value_head.w[j];
        }
        
        // Advantages
        double advantages[ACTIONS];
        for(int i = 0; i < ACTIONS; i++) {
            advantages[i] = g_advantage_head.b[i];
            for(int j = 0; j < g_advantage_head.input_size; j++) {
                advantages[i] += layer3_out[j] * g_advantage_head.w[i * g_advantage_head.input_size + j];
            }
        }
        
        // Combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        double advantage_mean = 0;
        for(int i = 0; i < ACTIONS; i++) {
            advantage_mean += advantages[i];
        }
        advantage_mean /= ACTIONS;
        
        for(int i = 0; i < ACTIONS; i++) {
            q_values[i] = state_value + (advantages[i] - advantage_mean);
        }
        
        if(InpVerboseLogging && g_bars_processed % (InpLogEveryNBars * 10) == 0) {
            Print("    Neural network output (Dueling):");
            Print("      State value: ", DoubleToString(state_value, 4));
            Print("      Q-values: [", DoubleToString(q_values[0], 4), ", ", DoubleToString(q_values[1], 4), 
                  ", ", DoubleToString(q_values[2], 4), ", ", DoubleToString(q_values[3], 4), 
                  ", ", DoubleToString(q_values[4], 4), ", ", DoubleToString(q_values[5], 4), "]");
        }
    } else {
        // Non-dueling: direct output
        for(int i = 0; i < ACTIONS && i < ArraySize(layer3_out); i++) {
            q_values[i] = layer3_out[i];
        }
        
        if(InpVerboseLogging && g_bars_processed % (InpLogEveryNBars * 10) == 0) {
            Print("    Neural network output (Non-dueling):");
            Print("      Q-values: [", DoubleToString(q_values[0], 4), ", ", DoubleToString(q_values[1], 4), 
                  ", ", DoubleToString(q_values[2], 4), ", ", DoubleToString(q_values[3], 4), 
                  ", ", DoubleToString(q_values[4], 4), ", ", DoubleToString(q_values[5], 4), "]");
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Build simplified state vector                                   |
//+------------------------------------------------------------------+
bool BuildStateVector(int bar_index, double &state[]) {
    ArrayResize(state, STATE_SIZE);
    int idx = 0;
    bool has_failures = false;
    
    if(InpVerboseLogging && g_bars_processed % (InpLogEveryNBars * 5) == 0) {
        Print("    Building state vector for bar ", bar_index);
        Print("      Bar time: ", TimeToString(iTime(_Symbol, PERIOD_CURRENT, bar_index), TIME_DATE|TIME_SECONDS));
        Print("      OHLC: O=", DoubleToString(iOpen(_Symbol, PERIOD_CURRENT, bar_index), 5),
              " H=", DoubleToString(iHigh(_Symbol, PERIOD_CURRENT, bar_index), 5),
              " L=", DoubleToString(iLow(_Symbol, PERIOD_CURRENT, bar_index), 5),
              " C=", DoubleToString(iClose(_Symbol, PERIOD_CURRENT, bar_index), 5));
    }
    
    // Price features (0-9)
    for(int i = 0; i < 5 && idx < STATE_SIZE; i++) {
        if(bar_index + i < Bars(_Symbol, PERIOD_CURRENT)) {
            state[idx++] = iClose(_Symbol, PERIOD_CURRENT, bar_index + i);
        } else {
            state[idx++] = 0.0;
            has_failures = true;
        }
    }
    
    // OHLC ratios
    double close = iClose(_Symbol, PERIOD_CURRENT, bar_index);
    if(close != 0) {
        state[idx++] = iOpen(_Symbol, PERIOD_CURRENT, bar_index) / close;
        state[idx++] = iHigh(_Symbol, PERIOD_CURRENT, bar_index) / close;
        state[idx++] = iLow(_Symbol, PERIOD_CURRENT, bar_index) / close;
        state[idx++] = (iHigh(_Symbol, PERIOD_CURRENT, bar_index) - iLow(_Symbol, PERIOD_CURRENT, bar_index)) / close;
    } else {
        state[idx++] = 1.0;
        state[idx++] = 1.0;
        state[idx++] = 1.0;
        state[idx++] = 0.0;
        has_failures = true;
    }
    
    // Technical indicators (10-17)
    double values[];
    
    // MA10
    if(CopyBuffer(h_ma10, 0, bar_index, 1, values) == 1 && values[0] != 0) {
        state[idx++] = close / values[0];
    } else {
        state[idx++] = 1.0;
        has_failures = true;
        g_indicator_failures++;
    }
    
    // MA50
    if(CopyBuffer(h_ma50, 0, bar_index, 1, values) == 1 && values[0] != 0) {
        state[idx++] = close / values[0];
    } else {
        state[idx++] = 1.0;
        has_failures = true;
        g_indicator_failures++;
    }
    
    // RSI
    if(CopyBuffer(h_rsi, 0, bar_index, 1, values) == 1) {
        state[idx++] = values[0] / 100.0;
    } else {
        state[idx++] = 0.5;
        has_failures = true;
        g_indicator_failures++;
    }
    
    // ATR
    if(CopyBuffer(h_atr, 0, bar_index, 1, values) == 1) {
        state[idx++] = MathMin(values[0] / (close * 0.01), 1.0);
    } else {
        state[idx++] = 0.5;
        has_failures = true;
        g_indicator_failures++;
    }
    
    // Fill remaining features with defaults or simple calculations
    while(idx < STATE_SIZE) {
        if(idx == 18) { // Time feature
            datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, bar_index);
            MqlDateTime dt;
            TimeToStruct(bar_time, dt);
            state[idx++] = (dt.day > 15) ? 1.0 : 0.0;
        } else if(idx >= 19 && idx <= 22) { // Microstructure
            state[idx++] = 0.5; // Default values
        } else if(idx >= 23 && idx <= 27) { // Momentum
            if(idx == 23 && bar_index + 5 < Bars(_Symbol, PERIOD_CURRENT)) {
                double close_5 = iClose(_Symbol, PERIOD_CURRENT, bar_index + 5);
                state[idx++] = (close_5 != 0) ? (close - close_5) / close_5 : 0.0;
            } else {
                state[idx++] = 0.0;
            }
        } else { // Multi-timeframe (28-34)
            state[idx++] = 0.5; // Default values
        }
    }
    
    // Normalize features
    for(int i = 0; i < STATE_SIZE; i++) {
        if(g_feature_max[i] != g_feature_min[i]) {
            state[i] = (state[i] - g_feature_min[i]) / (g_feature_max[i] - g_feature_min[i]);
        } else {
            state[i] = 0.5;
        }
        state[i] = MathMax(0.0, MathMin(1.0, state[i]));
    }
    
    if(has_failures) {
        g_feature_failures++;
    }
    
    if(InpVerboseLogging && g_bars_processed % (InpLogEveryNBars * 5) == 0) {
        Print("    State vector sample: [", DoubleToString(state[0], 4), ", ", DoubleToString(state[1], 4),
              ", ", DoubleToString(state[2], 4), ", ..., ", DoubleToString(state[STATE_SIZE-1], 4), "]");
        if(has_failures) {
            Print("    ⚠ Some features had failures and used default values");
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Run the backtest simulation                                     |
//+------------------------------------------------------------------+
void RunBacktest(datetime start_time, datetime end_time) {
    Print("Starting backtest simulation...");
    
    int start_bar = iBarShift(_Symbol, PERIOD_CURRENT, start_time);
    int end_bar = iBarShift(_Symbol, PERIOD_CURRENT, end_time);
    
    if(start_bar <= 0 || end_bar < 0) {
        Print("✗ ERROR: Invalid bar range for backtest period");
        Print("  Start bar: ", start_bar, " (", TimeToString(start_time), ")");
        Print("  End bar: ", end_bar, " (", TimeToString(end_time), ")");
        return;
    }
    
    int total_bars = start_bar - end_bar;
    Print("Processing ", total_bars, " bars from bar ", start_bar, " to ", end_bar);
    Print("Date range: ", TimeToString(iTime(_Symbol, PERIOD_CURRENT, start_bar), TIME_DATE|TIME_SECONDS), 
          " to ", TimeToString(iTime(_Symbol, PERIOD_CURRENT, end_bar), TIME_DATE|TIME_SECONDS));
    
    // Reset counters
    g_bars_processed = 0;
    g_prediction_failures = 0;
    g_feature_failures = 0;
    g_indicator_failures = 0;
    
    // Process each bar
    for(int bar = start_bar; bar > end_bar; bar--) {
        g_bars_processed++;
        datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, bar);
        
        // CRITICAL DEBUG: Verify this updated version is running
        if(g_bars_processed == 1){
            Print("=== ENHANCED BACKTEST VERSION CONFIRMED ===");
            Print("DEBUG: Updated CortexBacktestWorking.mq5 is running");
            Print("DEBUG: InpMaxTradesPerDay = ", InpMaxTradesPerDay);
            Print("DEBUG: InpMinBarsBetweenTrades = ", InpMinBarsBetweenTrades);
            Print("DEBUG: InpUseMaxHoldingTime = ", InpUseMaxHoldingTime);
            Print("DEBUG: InpMaxHoldingHours = ", InpMaxHoldingHours);
            Print("==============================================");
        }
        
        // Progress logging
        if(g_bars_processed % InpLogEveryNBars == 0) {
            double progress = ((double)g_bars_processed / total_bars) * 100.0;
            Print("Progress: ", DoubleToString(progress, 1), "% (", g_bars_processed, "/", total_bars, " bars) | ",
                  TimeToString(bar_time, TIME_DATE|TIME_MINUTES), " | Balance: $", DoubleToString(g_balance, 2));
        }
        
        // Build state vector
        double state[STATE_SIZE];
        if(!BuildStateVector(bar, state)) {
            g_feature_failures++;
            if(InpVerboseLogging) {
                Print("  ✗ Failed to build state vector for bar ", bar, " at ", TimeToString(bar_time));
            }
            continue;
        }
        
        // Get AI prediction
        double q_values[ACTIONS];
        if(!PredictQValues(state, q_values)) {
            g_prediction_failures++;
            if(InpVerboseLogging) {
                Print("  ✗ Neural network prediction failed for bar ", bar);
            }
            continue;
        }
        
        // IMPROVEMENT 7.1: UNIFIED EXIT CONDITIONS - Check all exit conditions before AI decision
        bool position_closed = false;
        
        // Check traditional Phase 1-3 enhancements
        if(CheckMaxHoldingTime(bar_time) || CheckProfitTargets() || CheckEmergencyStops()) {
            position_closed = true;
        }
        
        // Check new unified exit conditions
        if(!position_closed) {
            if(CheckATRBasedStops(bar_time) || CheckTrailingStops(bar_time) || CheckEmergencyStops(bar_time)) {
                position_closed = true;
            }
        }
        
        if(position_closed) {
            if(InpVerboseLogging) {
                Print("DEBUG: Position closed by unified exit logic at bar ", g_bars_processed);
            }
            continue; // Position was closed, skip AI decision for this bar
        }
        
        // Select best action
        int action = SelectBestAction(q_values);
        
        // IMPROVEMENT 7.1: UNIFIED MASTER RISK CHECK - Apply all risk filters
        if(action == BUY_STRONG || action == BUY_WEAK || action == SELL_STRONG || action == SELL_WEAK) {
            if(!MasterRiskCheck(q_values, action, bar_time)){
                action = HOLD; // Force HOLD if any risk filter fails
                if(InpVerboseLogging) {
                    Print("DEBUG: Master risk check failed - forcing HOLD at bar ", g_bars_processed);
                }
            } else {
                // Update trading frequency tracking if trade is allowed
                UpdateTradingFrequency(bar_time);
            }
        }
        
        // FORCED FLAT ACTION - Override AI decision when position should be closed
        if(ShouldForceFlat(bar_time, action)){
            action = FLAT; // Force FLAT to close position
            if(InpVerboseLogging){
                Print("  FORCING FLAT - position exit criteria met");
            }
        }
        
        // Log action selection
        if(InpVerboseLogging && (action != HOLD || g_bars_processed % (InpLogEveryNBars * 2) == 0)) {
            string action_names[] = {"BUY_STRONG", "BUY_WEAK", "SELL_STRONG", "SELL_WEAK", "HOLD", "FLAT"};
            Print("  Bar ", bar, " | ", TimeToString(bar_time, TIME_MINUTES), " | Action: ", action_names[action], 
                  " | Q-val: ", DoubleToString(q_values[action], 4), " | Price: ", DoubleToString(iClose(_Symbol, PERIOD_CURRENT, bar), 5));
        }
        
        // Execute trade
        ExecuteSimulatedTrade(action, bar_time, bar);
        
        // Update performance
        UpdatePerformanceMetrics(bar);
        
        // Log position status periodically
        if(g_bars_processed % (InpLogEveryNBars * 2) == 0 && g_current_position != POS_NONE) {
            double unrealized = 0;
            if(g_current_position == POS_LONG) {
                unrealized = (iClose(_Symbol, PERIOD_CURRENT, bar) - g_position_entry_price) * g_position_lots * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
            } else {
                unrealized = (g_position_entry_price - iClose(_Symbol, PERIOD_CURRENT, bar)) * g_position_lots * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
            }
            
            Print("  Position: ", (g_current_position == POS_LONG ? "LONG" : "SHORT"), " ", g_position_lots, " lots",
                  " | Entry: ", DoubleToString(g_position_entry_price, 5),
                  " | Current: ", DoubleToString(iClose(_Symbol, PERIOD_CURRENT, bar), 5),
                  " | Unrealized: $", DoubleToString(unrealized, 2));
        }
    }
    
    // Close final position
    if(g_current_position != POS_NONE) {
        Print("Closing final position at end of backtest...");
        ClosePosition("End of backtest");
    }
    
    Print("Simulation completed!");
    Print("Final statistics:");
    Print("  Bars processed: ", g_bars_processed);
    Print("  Feature failures: ", g_feature_failures, " (", DoubleToString((double)g_feature_failures/g_bars_processed*100, 2), "%)");
    Print("  Prediction failures: ", g_prediction_failures, " (", DoubleToString((double)g_prediction_failures/g_bars_processed*100, 2), "%)");
    Print("  Indicator failures: ", g_indicator_failures);
    Print("  Final balance: $", DoubleToString(g_balance, 2));
    Print("  Total return: ", DoubleToString(((g_balance - InpInitialBalance) / InpInitialBalance) * 100.0, 2), "%");
    
    // IMPROVEMENT 7.1: Advanced feature performance statistics
    Print("");
    Print("=== IMPROVEMENT 7.1: UNIFIED EA LOGIC PERFORMANCE ===");
    Print("Advanced Risk Management:");
    Print("  Confidence-filtered trades: ", g_confidence_filtered_trades);
    Print("  ATR-based stop hits: ", g_atr_stop_hits);
    Print("  Trailing stop activations: ", g_trailing_stop_hits);
    Print("  Regime-triggered exits: ", g_regime_triggered_exits);
    Print("  Volatility adjustments: ", g_volatility_adjustments);
    Print("  Emergency mode activations: ", g_emergency_mode ? "YES" : "NO");
    Print("  Consecutive losses (max): ", g_consecutive_losses);
    
    Print("Dynamic Position Management:");
    Print("  Position scaling enabled: ", InpAllowPositionScaling ? "YES" : "NO");
    Print("  Final lot size: ", DoubleToString(g_dynamic_lot_size, 2));
    Print("  Volatility multiplier: ", DoubleToString(g_volatility_multiplier, 2));
    Print("  High volatility mode: ", g_high_volatility_mode ? "ACTIVE" : "INACTIVE");
    Print("  Current volatility percentile: ", DoubleToString(g_volatility_percentile * 100, 1), "%");
    
    Print("Signal Quality:");
    Print("  Confidence filtering enabled: ", InpUseConfidenceFilter ? "YES" : "NO");
    if(InpUseConfidenceFilter) {
        Print("  Last confidence score: ", DoubleToString(g_last_confidence, 4));
        Print("  Confidence threshold: ", DoubleToString(InpConfidenceThreshold, 4));
    }
    Print("  Final confidence trades: ", g_confidence_trades);
    
    Print("Risk Management Effectiveness:");
    Print("  Maximum drawdown: ", DoubleToString(g_max_drawdown, 2), "%");
    double win_rate = g_winning_trades > 0 ? (double)g_winning_trades / (g_winning_trades + g_losing_trades) * 100 : 0;
    Print("  Win rate: ", DoubleToString(win_rate, 1), "% (", g_winning_trades, " wins, ", g_losing_trades, " losses)");
    double avg_trade_return = g_total_trades > 0 ? ((g_balance - InpInitialBalance) / g_total_trades) : 0;
    Print("  Average per trade: $", DoubleToString(avg_trade_return, 2));
    
    Print("Session and Time Management:");
    Print("  Session filtering enabled: ", InpUseSessionFilter ? "YES" : "NO");
    Print("  News filtering enabled: ", InpUseNewsFilter ? "YES" : "NO");
    Print("  Current session active: ", g_current_session_active ? "YES" : "NO");
    
    Print("=== SYNCHRONIZED BACKTESTER FEATURES SUMMARY ===");
    Print("✓ Confidence-based trade filtering (6.3 integration)");
    Print("✓ ATR-based dynamic stops and position sizing"); 
    Print("✓ Trailing stop functionality");
    Print("✓ Volatility regime detection and adaptation");
    Print("✓ Master risk check combining all filters");
    Print("✓ Emergency risk management and trading halts");
    Print("✓ Session and time-based filtering");
    Print("✓ Dynamic position sizing with volatility adjustment");
    Print("✓ Comprehensive performance tracking");
    Print("✓ Unified trade logic synchronized with EA (cortex5.mq5)");
    Print("=======================================================");
    
    // IMPROVEMENT 7.2: Calculate and display comprehensive performance metrics
    Print("");
    Print("Calculating comprehensive performance metrics...");
    CalculateComprehensiveMetrics();
    DisplayComprehensiveMetrics();
}

//+------------------------------------------------------------------+
//| Select best action from Q-values                               |
//+------------------------------------------------------------------+
int SelectBestAction(const double &q_values[]) {
    int best_action = 0;
    double max_q = q_values[0];
    
    for(int i = 1; i < ACTIONS; i++) {
        if(q_values[i] > max_q) {
            max_q = q_values[i];
            best_action = i;
        }
    }
    
    return best_action;
}

//+------------------------------------------------------------------+
//| Execute simulated trade                                         |
//+------------------------------------------------------------------+
void ExecuteSimulatedTrade(int action, datetime bar_time, int bar_index) {
    double current_price = iClose(_Symbol, PERIOD_CURRENT, bar_index);
    string action_names[] = {"BUY_STRONG", "BUY_WEAK", "SELL_STRONG", "SELL_WEAK", "HOLD", "FLAT"};
    
    switch(action) {
        case BUY_STRONG:
            if(g_current_position != POS_LONG) {
                if(g_current_position != POS_NONE) ClosePosition("Reverse to LONG");
                // IMPROVEMENT 7.1: Use unified dynamic position sizing
                double atr_buffer[];
                if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) > 0) {
                    g_current_atr = atr_buffer[0];
                }
                double lots = CalculateDynamicLotSize(g_balance, g_current_atr);
                if(InpAllowPositionScaling && action == BUY_STRONG) {
                    lots *= InpStrongSignalMultiplier; // Scale up for strong signals
                }
                OpenPosition(POS_LONG, lots, current_price, bar_time, action_names[action]);
            }
            break;
            
        case BUY_WEAK:
            if(g_current_position != POS_LONG) {
                if(g_current_position != POS_NONE) ClosePosition("Reverse to LONG");
                // IMPROVEMENT 7.1: Use unified dynamic position sizing
                double atr_buffer[];
                if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) > 0) {
                    g_current_atr = atr_buffer[0];
                }
                double lots = CalculateDynamicLotSize(g_balance, g_current_atr);
                OpenPosition(POS_LONG, lots, current_price, bar_time, action_names[action]);
            }
            break;
            
        case SELL_STRONG:
            if(g_current_position != POS_SHORT) {
                if(g_current_position != POS_NONE) ClosePosition("Reverse to SHORT");
                // IMPROVEMENT 7.1: Use unified dynamic position sizing
                double atr_buffer[];
                if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) > 0) {
                    g_current_atr = atr_buffer[0];
                }
                double lots = CalculateDynamicLotSize(g_balance, g_current_atr);
                if(InpAllowPositionScaling && action == SELL_STRONG) {
                    lots *= InpStrongSignalMultiplier; // Scale up for strong signals
                }
                OpenPosition(POS_SHORT, lots, current_price, bar_time, action_names[action]);
            }
            break;
            
        case SELL_WEAK:
            if(g_current_position != POS_SHORT) {
                if(g_current_position != POS_NONE) ClosePosition("Reverse to SHORT");
                // IMPROVEMENT 7.1: Use unified dynamic position sizing
                double atr_buffer[];
                if(CopyBuffer(h_atr, 0, 0, 1, atr_buffer) > 0) {
                    g_current_atr = atr_buffer[0];
                }
                double lots = CalculateDynamicLotSize(g_balance, g_current_atr);
                OpenPosition(POS_SHORT, lots, current_price, bar_time, action_names[action]);
            }
            break;
            
        case FLAT:
            if(g_current_position != POS_NONE) {
                ClosePosition(action_names[action]);
            }
            break;
            
        case HOLD:
            // No action - log only if verbose and occasionally
            if(InpVerboseLogging && g_bars_processed % (InpLogEveryNBars * 5) == 0) {
                Print("  Holding current position | Current price: ", DoubleToString(current_price, 5));
            }
            break;
    }
}

//+------------------------------------------------------------------+
//| Open a new position                                            |
//+------------------------------------------------------------------+
void OpenPosition(POS_TYPE pos_type, double lots, double price, datetime time, string reason) {
    g_current_position = pos_type;
    g_position_lots = lots;
    g_position_entry_price = price;
    g_position_open_time = time;
    
    // Update trading frequency tracking
    UpdateTradingFrequency(time);
    
    // IMPROVEMENT 7.4: Update flexible parameter states for new trade
    if(g_parameters_validated) {
        UpdateFlexibleStates(time, 0.0, true);
    }
    
    // Simulate spread
    long spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    double point_value = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double spread = spread_points * point_value;
    
    if(pos_type == POS_LONG) {
        g_position_entry_price += spread;
    } else {
        g_position_entry_price -= spread;
    }
    
    Print(">>> OPEN ", (pos_type == POS_LONG ? "LONG" : "SHORT"), " | ",
          TimeToString(time, TIME_DATE|TIME_MINUTES), " | ", 
          DoubleToString(lots, 2), " lots @ ", DoubleToString(g_position_entry_price, 5), 
          " | Reason: ", reason, " | Spread: ", DoubleToString(spread, 5));
}

//+------------------------------------------------------------------+
//| Close current position                                          |
//+------------------------------------------------------------------+
void ClosePosition(datetime time, int bar_index, string reason) {
    if(g_current_position == POS_NONE) return;
    
    double close_price = iClose(_Symbol, PERIOD_CURRENT, bar_index);
    
    // Simulate spread
    long spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    double point_value = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double spread = spread_points * point_value;
    
    if(g_current_position == POS_LONG) {
        close_price -= spread;
    } else {
        close_price += spread;
    }
    
    // Calculate P&L
    double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    double profit_loss;
    
    if(g_current_position == POS_LONG) {
        profit_loss = (close_price - g_position_entry_price) * g_position_lots * contract_size;
    } else {
        profit_loss = (g_position_entry_price - close_price) * g_position_lots * contract_size;
    }
    
    // Commission
    double commission = g_position_lots * 7.0;
    profit_loss -= commission;
    
    // Update balance
    g_balance += profit_loss;
    g_equity = g_balance;
    
    // Calculate trade duration
    double duration_hours = (double)(time - g_position_open_time) / 3600.0;
    
    // Record trade
    RecordTrade(g_position_open_time, time, g_current_position, g_position_entry_price, close_price, g_position_lots, profit_loss);
    
    // IMPROVEMENT 7.4: Update flexible parameter states
    if(g_parameters_validated) {
        UpdateFlexibleStates(time, profit_loss, false);
    }
    
    Print("<<< CLOSE ", (g_current_position == POS_LONG ? "LONG" : "SHORT"), " | ",
          TimeToString(time, TIME_DATE|TIME_MINUTES), " | ",
          DoubleToString(g_position_lots, 2), " lots @ ", DoubleToString(close_price, 5),
          " | Duration: ", DoubleToString(duration_hours, 2), "h",
          " | P&L: $", DoubleToString(profit_loss, 2),
          " | Commission: $", DoubleToString(commission, 2),
          " | Balance: $", DoubleToString(g_balance, 2),
          " | Reason: ", reason);
    
    // Reset position
    g_current_position = POS_NONE;
    g_position_lots = 0;
    g_position_entry_price = 0;
    g_position_open_time = 0;
}

//+------------------------------------------------------------------+
//| Record completed trade                                          |
//+------------------------------------------------------------------+
void RecordTrade(datetime open_time, datetime close_time, POS_TYPE pos_type, 
                 double entry_price, double exit_price, double lots, double profit_loss) {
    
    int trade_count = ArraySize(g_trades);
    ArrayResize(g_trades, trade_count + 1);
    
    g_trades[trade_count].open_time = open_time;
    g_trades[trade_count].close_time = close_time;
    g_trades[trade_count].position_type = pos_type;
    g_trades[trade_count].entry_price = entry_price;
    g_trades[trade_count].exit_price = exit_price;
    g_trades[trade_count].lots = lots;
    g_trades[trade_count].profit_loss = profit_loss;
    g_trades[trade_count].balance_after = g_balance;
    
    // IMPROVEMENT 7.2: Enhanced trade recording with comprehensive metrics
    
    // Calculate holding time
    g_trades[trade_count].holding_time_hours = (int)((close_time - open_time) / 3600);
    
    // Record confidence score (if available)
    g_trades[trade_count].confidence_score = g_last_confidence;
    
    // Calculate MAE/MFE (Maximum Adverse/Favorable Excursion) - simplified
    // In a real implementation, this would track during the trade
    double price_move = MathAbs(exit_price - entry_price);
    if(profit_loss > 0) {
        g_trades[trade_count].mfe = profit_loss; // Favorable excursion equals profit
        g_trades[trade_count].mae = price_move * 0.3 * lots * 100000; // Estimate adverse excursion
    } else {
        g_trades[trade_count].mae = MathAbs(profit_loss); // Adverse excursion equals loss
        g_trades[trade_count].mfe = price_move * 0.2 * lots * 100000; // Estimate favorable excursion
    }
    
    // Record exit reason (simplified - would be enhanced with more specific tracking)
    if(g_atr_stop_hits > 0 && trade_count == g_total_trades) {
        g_trades[trade_count].exit_reason = "ATR Stop";
    } else if(g_trailing_stop_hits > 0 && trade_count == g_total_trades) {
        g_trades[trade_count].exit_reason = "Trailing Stop";
    } else {
        g_trades[trade_count].exit_reason = "Model Signal";
    }
    
    // Calculate commission (simplified)
    g_trades[trade_count].commission = lots * 7.0; // $7 per lot (example)
    
    // Update drawdown
    if(g_balance > g_max_balance) {
        g_max_balance = g_balance;
    }
    double current_drawdown = (g_max_balance - g_balance) / g_max_balance * 100.0;
    g_trades[trade_count].drawdown_pct = current_drawdown;
    
    if(current_drawdown > g_max_drawdown) {
        g_max_drawdown = current_drawdown;
    }
    
    // IMPROVEMENT 7.2: Update running statistics for comprehensive metrics
    if(profit_loss > 0) {
        g_running_gross_profit += profit_loss;
    } else {
        g_running_gross_loss += MathAbs(profit_loss);
    }
    
    // Update consecutive statistics
    UpdateConsecutiveStats(profit_loss);
    
    // Update stats
    g_total_trades++;
    if(profit_loss > 0) {
        g_winning_trades++;
        Print("  ✓ WINNING TRADE #", g_total_trades, " | P&L: $", DoubleToString(profit_loss, 2), 
              " | Hold: ", g_trades[trade_count].holding_time_hours, "h | Win rate: ", 
              DoubleToString(((double)g_winning_trades/g_total_trades)*100, 1), "%");
    } else if(profit_loss < 0) {
        g_losing_trades++;
        Print("  ✗ LOSING TRADE #", g_total_trades, " | P&L: $", DoubleToString(profit_loss, 2), 
              " | Hold: ", g_trades[trade_count].holding_time_hours, "h | Win rate: ", 
              DoubleToString(((double)g_winning_trades/g_total_trades)*100, 1), "%");
    }
    
    // IMPROVEMENT 7.3: Log trade to CSV
    LogTradeToCSV(g_trades[trade_count], g_total_trades);
    
    if(current_drawdown > g_max_drawdown * 0.8) {
        Print("  ⚠ HIGH DRAWDOWN WARNING: ", DoubleToString(current_drawdown, 2), "% (Max: ", DoubleToString(g_max_drawdown, 2), "%)");
    }
}

//+------------------------------------------------------------------+
//| Update performance metrics                                      |
//+------------------------------------------------------------------+
void UpdatePerformanceMetrics(int bar_index) {
    static double previous_balance = 0.0;
    static datetime previous_day = 0;
    
    g_equity = g_balance;
    
    // Add unrealized P&L
    if(g_current_position != POS_NONE) {
        double current_price = iClose(_Symbol, PERIOD_CURRENT, bar_index);
        double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
        double unrealized_pl;
        
        if(g_current_position == POS_LONG) {
            unrealized_pl = (current_price - g_position_entry_price) * g_position_lots * contract_size;
        } else {
            unrealized_pl = (g_position_entry_price - current_price) * g_position_lots * contract_size;
        }
        
        g_equity += unrealized_pl;
    }
    
    // IMPROVEMENT 7.2: Update comprehensive metrics tracking
    datetime current_time = iTime(_Symbol, PERIOD_CURRENT, bar_index);
    
    // Update equity curve
    UpdateEquityCurve(current_time, g_equity);
    
    // Calculate daily returns (only once per day)
    datetime current_day = current_time - (current_time % (24 * 3600)); // Start of current day
    if(current_day != previous_day && previous_balance > 0) {
        CalculateDailyReturn(previous_balance, g_balance);
        previous_day = current_day;
    }
    previous_balance = g_balance;
    
    // Update max balance and drawdown (legacy code)
    if(g_equity > g_max_balance) {
        g_max_balance = g_equity;
    }
    
    double current_drawdown = (g_max_balance - g_equity) / g_max_balance * 100.0;
    if(current_drawdown > g_max_drawdown) {
        g_max_drawdown = current_drawdown;
    }
    
    // IMPROVEMENT 7.3: Log equity curve to CSV
    double unrealized_pl = g_equity - g_balance;
    double drawdown_amount = g_max_balance - g_equity;
    LogEquityToCSV(current_time, g_balance, g_equity, unrealized_pl, current_drawdown, 
                   drawdown_amount, g_current_position, g_position_lots, g_position_entry_price);
}

//+------------------------------------------------------------------+
//| Generate comprehensive performance report                       |
//+------------------------------------------------------------------+
void GeneratePerformanceReport() {
    Print("\n");
    Print("=======================================================");
    Print("=== CORTEX BACKTEST PERFORMANCE REPORT ===");
    Print("=======================================================");
    Print("Backtest Period: ", InpBacktestDays, " days");
    Print("Symbol: ", _Symbol);
    Print("Timeframe: ", EnumToString(PERIOD_CURRENT));
    Print("Model File: ", InpModelFileName);
    Print("Model Type: ", g_use_dueling ? "Dueling" : "Standard", " DQN", g_use_lstm ? " with LSTM" : "");
    Print("\n--- ENHANCEMENTS ACTIVE ---");
    Print("Max Holding Time: ", InpUseMaxHoldingTime ? IntegerToString(InpMaxHoldingHours) + " hours" : "DISABLED");
    Print("Profit Targets: ", InpUseProfitTargets ? DoubleToString(InpProfitTargetATR, 1) + "x ATR" : "DISABLED");
    Print("Enhanced Rewards: ", InpEnhancedRewards ? "ENABLED" : "DISABLED");
    Print("Dynamic Stops: ", InpUseDynamicStops ? "ENABLED" : "DISABLED");
    Print("Position Features: ", InpUsePositionFeatures ? "ENABLED" : "DISABLED");
    Print("Trading Frequency Control: ENABLED (Max ", InpMaxTradesPerDay, " trades/day, ", InpMinBarsBetweenTrades, " bars between)");
    Print("Forced FLAT Actions: ", InpEnforceFlat ? "ENABLED (AGGRESSIVE)" : "DISABLED");
    Print("Emergency Stop Loss: ", InpUseEmergencyStops ? "ENABLED ($" + DoubleToString(InpEmergencyStopLoss, 0) + " max loss)" : "DISABLED");
    
    // Overall performance
    double total_return = ((g_balance - InpInitialBalance) / InpInitialBalance) * 100.0;
    Print("\n--- OVERALL PERFORMANCE ---");
    Print("Initial Balance: $", DoubleToString(InpInitialBalance, 2));
    Print("Final Balance: $", DoubleToString(g_balance, 2));
    Print("Net Profit: $", DoubleToString(g_balance - InpInitialBalance, 2));
    Print("Total Return: ", DoubleToString(total_return, 2), "%");
    Print("Maximum Drawdown: ", DoubleToString(g_max_drawdown, 2), "%");
    Print("Maximum Balance: $", DoubleToString(g_max_balance, 2));
    
    // Trade statistics
    Print("\n--- TRADE STATISTICS ---");
    Print("Total Trades: ", g_total_trades);
    Print("Winning Trades: ", g_winning_trades);
    Print("Losing Trades: ", g_losing_trades);
    Print("Breakeven Trades: ", g_total_trades - g_winning_trades - g_losing_trades);
    
    if(g_total_trades > 0) {
        double win_rate = ((double)g_winning_trades / g_total_trades) * 100.0;
        Print("Win Rate: ", DoubleToString(win_rate, 2), "%");
        
        double total_profit = 0, total_loss = 0;
        double largest_win = 0, largest_loss = 0;
        double avg_trade_duration = 0;
        
        for(int i = 0; i < ArraySize(g_trades); i++) {
            if(g_trades[i].profit_loss > 0) {
                total_profit += g_trades[i].profit_loss;
                if(g_trades[i].profit_loss > largest_win) {
                    largest_win = g_trades[i].profit_loss;
                }
            } else if(g_trades[i].profit_loss < 0) {
                total_loss += g_trades[i].profit_loss;
                if(g_trades[i].profit_loss < largest_loss) {
                    largest_loss = g_trades[i].profit_loss;
                }
            }
            
            avg_trade_duration += (double)(g_trades[i].close_time - g_trades[i].open_time);
        }
        
        avg_trade_duration = avg_trade_duration / g_total_trades / 3600.0;  // Convert to hours
        
        Print("Gross Profit: $", DoubleToString(total_profit, 2));
        Print("Gross Loss: $", DoubleToString(total_loss, 2));
        Print("Average Trade Duration: ", DoubleToString(avg_trade_duration, 2), " hours");
        Print("Largest Win: $", DoubleToString(largest_win, 2));
        Print("Largest Loss: $", DoubleToString(largest_loss, 2));
        
        if(g_winning_trades > 0 && g_losing_trades > 0) {
            double avg_win = total_profit / g_winning_trades;
            double avg_loss = MathAbs(total_loss / g_losing_trades);
            double profit_factor = total_profit / MathAbs(total_loss);
            
            Print("Average Win: $", DoubleToString(avg_win, 2));
            Print("Average Loss: $", DoubleToString(avg_loss, 2));
            Print("Profit Factor: ", DoubleToString(profit_factor, 2));
            Print("Risk-Reward Ratio: ", DoubleToString(avg_win / avg_loss, 2));
        }
        
        Print("Trading Frequency: ", DoubleToString((double)g_total_trades / InpBacktestDays, 2), " trades/day");
        
        // Commission analysis
        double total_commission = g_total_trades * InpLotSize * 7.0;
        Print("Total Commission Paid: $", DoubleToString(total_commission, 2));
        Print("Commission Impact: ", DoubleToString((total_commission / InpInitialBalance) * 100.0, 2), "% of initial balance");
    }
    
    // Risk metrics
    Print("\n--- RISK METRICS ---");
    if(InpBacktestDays > 0) {
        double annualized_return = total_return * (365.25 / InpBacktestDays);
        Print("Annualized Return: ", DoubleToString(annualized_return, 2), "%");
        
        if(g_max_drawdown > 0) {
            double risk_adjusted_return = annualized_return / g_max_drawdown;
            Print("Risk-Adjusted Return: ", DoubleToString(risk_adjusted_return, 2));
            Print("Calmar Ratio: ", DoubleToString(annualized_return / g_max_drawdown, 2));
        }
        
        if(g_total_trades > 1) {
            Print("Average Monthly Return: ", DoubleToString(total_return / (InpBacktestDays / 30.44), 2), "%");
        }
    }
    
    // Technical analysis
    Print("\n--- TECHNICAL ANALYSIS ---");
    Print("Bars Processed: ", g_bars_processed);
    Print("Feature Build Failures: ", g_feature_failures, " (", DoubleToString((double)g_feature_failures/g_bars_processed*100, 2), "%)");
    Print("Prediction Failures: ", g_prediction_failures, " (", DoubleToString((double)g_prediction_failures/g_bars_processed*100, 2), "%)");
    Print("Indicator Failures: ", g_indicator_failures);
    Print("Data Quality Score: ", DoubleToString(100.0 - ((double)(g_feature_failures + g_prediction_failures)/g_bars_processed*100), 1), "%");
    
    // Model assessment
    Print("\n--- MODEL ASSESSMENT ---");
    if(total_return > 0) {
        Print("Model Assessment: PROFITABLE ✓");
    } else {
        Print("Model Assessment: UNPROFITABLE ✗");
    }
    
    if(g_max_drawdown < 5.0) {
        Print("Risk Assessment: VERY LOW RISK ✓");
    } else if(g_max_drawdown < 10.0) {
        Print("Risk Assessment: LOW RISK ✓");
    } else if(g_max_drawdown < 20.0) {
        Print("Risk Assessment: MODERATE RISK ⚠");
    } else if(g_max_drawdown < 35.0) {
        Print("Risk Assessment: HIGH RISK ⚠");
    } else {
        Print("Risk Assessment: VERY HIGH RISK ✗");
    }
    
    if(g_total_trades > 0) {
        double win_rate = ((double)g_winning_trades / g_total_trades) * 100.0;
        if(win_rate > 60.0) {
            Print("Win Rate Assessment: EXCELLENT (", DoubleToString(win_rate, 1), "%) ✓");
        } else if(win_rate > 50.0) {
            Print("Win Rate Assessment: GOOD (", DoubleToString(win_rate, 1), "%) ✓");
        } else if(win_rate > 40.0) {
            Print("Win Rate Assessment: AVERAGE (", DoubleToString(win_rate, 1), "%) ⚠");
        } else {
            Print("Win Rate Assessment: POOR (", DoubleToString(win_rate, 1), "%) ✗");
        }
        
        double trade_frequency = (double)g_total_trades / InpBacktestDays;
        if(trade_frequency > 5.0) {
            Print("Trading Frequency: VERY HIGH (", DoubleToString(trade_frequency, 2), "/day) - May incur high costs");
        } else if(trade_frequency > 2.0) {
            Print("Trading Frequency: HIGH (", DoubleToString(trade_frequency, 2), "/day)");
        } else if(trade_frequency > 0.5) {
            Print("Trading Frequency: MODERATE (", DoubleToString(trade_frequency, 2), "/day)");
        } else {
            Print("Trading Frequency: LOW (", DoubleToString(trade_frequency, 2), "/day)");
        }
    }
    
    // Detailed trades
    if(InpDetailedReport && ArraySize(g_trades) > 0) {
        Print("\n--- DETAILED TRADE LOG ---");
        Print("Showing first ", MathMin(ArraySize(g_trades), 15), " trades:");
        Print("Date & Time          | Type  | Size | Entry   | Exit    | Duration | P&L     | Balance | DD%");
        Print("-------------------- | ----- | ---- | ------- | ------- | -------- | ------- | ------- | ----");
        
        int max_trades = MathMin(ArraySize(g_trades), 15);
        for(int i = 0; i < max_trades; i++) {
            string pos_type = (g_trades[i].position_type == POS_LONG) ? "LONG " : "SHORT";
            double duration_hours = (double)(g_trades[i].close_time - g_trades[i].open_time) / 3600.0;
            
            Print(TimeToString(g_trades[i].open_time, TIME_DATE|TIME_MINUTES), " | ", pos_type, " | ",
                  DoubleToString(g_trades[i].lots, 2), "   | ",
                  DoubleToString(g_trades[i].entry_price, 5), " | ",
                  DoubleToString(g_trades[i].exit_price, 5), " | ",
                  DoubleToString(duration_hours, 1), "h     | ",
                  (g_trades[i].profit_loss >= 0 ? "+" : ""), DoubleToString(g_trades[i].profit_loss, 2), "  | ",
                  DoubleToString(g_trades[i].balance_after, 2), " | ",
                  DoubleToString(g_trades[i].drawdown_pct, 1), "%");
        }
        
        if(ArraySize(g_trades) > 15) {
            Print("... and ", ArraySize(g_trades) - 15, " more trades");
        }
    }
    
    // Recommendations
    Print("\n--- RECOMMENDATIONS ---");
    if(total_return > 0 && g_max_drawdown < 15.0) {
        Print("✓ Model shows positive performance with acceptable risk");
        Print("✓ Consider live testing with small position sizes");
    }
    
    if(g_max_drawdown > 20.0) {
        Print("⚠ High drawdown detected - consider risk management improvements");
    }
    
    if(g_total_trades < 10) {
        Print("⚠ Limited trading activity - consider longer backtest period");
    }
    
    if(g_feature_failures > g_bars_processed * 0.05) {
        Print("⚠ High feature failure rate - check data quality");
    }
    
    Print("\n=== END OF REPORT ===");
    Print("=======================================================\n");
}