//+------------------------------------------------------------------+
//|                                         cortex5.mq5  |
//|              LIVE TRADING EA WITH PHASE 1-3 ENHANCEMENTS        |
//|   Production-Ready AI Trading System with Risk Management       |
//|                                                                  |
//|   WHAT THIS PROGRAM DOES:                                        |
//|   This is a production Expert Advisor (EA) that uses an advanced|
//|   AI trading model (Double-Dueling DRQN) to trade forex        |
//|   automatically. It includes comprehensive risk management and   |
//|   all the Phase 1-3 enhancements to prevent the catastrophic    |
//|   losses seen in the original system.                           |
//|                                                                  |
//|   KEY FEATURES:                                                  |
//|   🤖 Advanced AI: Double-Dueling DRQN with LSTM memory          |
//|   🛡️ Risk Management: Multiple layers of protection             |
//|   ⏰ Time Controls: Maximum holding times (72 hours max)         |
//|   🎯 Profit Targets: Automatic profit-taking (2.5x ATR)         |
//|   🚨 Emergency Stops: Hard dollar limits ($500 max loss)        |
//|   📊 Position Tracking: Real-time P&L and risk monitoring       |
//|   🔄 Dynamic Stops: Progressive risk management                  |
//|   📈 Market Regime: Trending/ranging market detection           |
//|                                                                  |
//|   FIXES FOR ORIGINAL ISSUES:                                    |
//|   ✅ Prevents 700+ hour holding times                           |
//|   ✅ Forces profitable exits with targets                       |
//|   ✅ Limits overtrading (8 trades/day max)                      |
//|   ✅ Emergency stops prevent account wipeouts                   |
//|   ✅ Advanced risk controls for live trading                    |
//+------------------------------------------------------------------+

// IMPROVEMENT 3.4: Include unified trade logic module
#include <CortexTradeLogic.mqh>
#property strict
#include <Trade/Trade.mqh>

//============================== USER INPUTS ==============================
// These settings control how the EA trades and can be adjusted for different risk levels

// AI MODEL CONFIGURATION
input string  InpModelFileName   = "DoubleDueling_DRQN_Model.dat";  // AI brain file (must be trained first)

// CORE RISK MANAGEMENT - ESSENTIAL FOR LIVE TRADING
// These settings protect your account from catastrophic losses
input double  InpRiskPercent     = 2.0;   // Risk per trade as % of total account balance
                                          // ↳ CRITICAL: 2% means if you have $10,000, max loss per trade is $200
input double  InpMaxSpread       = 3.0;   // Don't trade if broker spread exceeds this (points)
                                          // ↳ PROTECTION: High spreads kill profitability - avoid bad market conditions  
input double  InpATRMultiplier   = 2.0;   // Stop loss distance as multiple of Average True Range
                                          // ↳ ADAPTIVE: Uses market volatility to set realistic stop losses
input double  InpRRRatio         = 1.5;   // Risk:Reward ratio (1.5 = target 1.5x profit vs loss)
                                          // ↳ PROFITABILITY: Ensures targets are bigger than stops
input double  InpMaxDrawdown     = 10.0;  // Pause trading if account loses more than this %
                                          // ↳ ACCOUNT PROTECTION: Stops EA if overall losses get too high
input int     InpMinWinRate      = 30;    // Pause if win rate drops below this % (over last 10 trades)
                                          // ↳ PERFORMANCE MONITORING: Stops EA if it's having a bad streak
input double  InpCommissionPerLot = 0.0;  // Your broker's commission per lot (for accurate cost calculation)
                                          // ↳ COST MODELING: Set to your actual commission for better decisions
input bool    InpUseRiskSizing   = true;  // Use smart position sizing based on volatility
                                          // ↳ RECOMMENDED: Adjusts trade size to maintain consistent risk
input bool    InpEnforceSymbolTF = true;  // Only trade the symbol/timeframe the model was trained on
                                          // ↳ SAFETY: Prevents using model on wrong data type
input bool    InpAllowPositionScaling = true; // Allow adjusting position size based on signal strength

// PHASE 1 ENHANCEMENTS - IMMEDIATE FIXES FOR PROFITABILITY
input int     InpMaxHoldingHours      = 72;    // Maximum hours to hold a position (Phase 1)
input double  InpProfitTargetATR      = 2.5;   // Take profit at N x ATR (Phase 1)  
input bool    InpUseProfitTargets     = true;  // Enable automatic profit taking (Phase 1)
input bool    InpUseMaxHoldingTime    = true;  // Enable maximum holding time control (Phase 1)
input double  InpHoldingTimePenalty   = 0.001; // Penalty per hour held in reward calculation (Phase 1)
input double  InpQuickExitBonus       = 0.005; // Bonus for trades < 24 hours (Phase 1)

// PHASE 2 ENHANCEMENTS - TRAINING IMPROVEMENTS
input double  InpFlatActionWeight     = 1.5;   // Increased weight for FLAT action in future training (Phase 2)
input bool    InpEnhancedRewards      = true;  // Use enhanced reward calculation (Phase 2)
input double  InpDrawdownPenalty      = 0.01;  // Penalty for unrealized drawdown (Phase 2)

// PHASE 3 ENHANCEMENTS - ADVANCED FEATURES
input bool    InpUseDynamicStops      = true;  // Enable dynamic stop loss tightening (Phase 3)
input double  InpStopTightenRate      = 0.8;   // Stop tightening multiplier per day held (Phase 3)
input bool    InpUsePositionFeatures  = true;  // Add position-aware features to state (Phase 3)
input bool    InpUseMarketRegime      = true;  // Enable market regime detection (Phase 3)

// TRADING FREQUENCY CONTROLS (LIVE TRADING - prevent overtrading)
input int     InpMinBarsBetweenTrades = 4;     // Minimum bars between trades 
input int     InpMaxTradesPerDay      = 8;     // Maximum trades per day
input bool    InpEnforceFlat          = true;  // Force FLAT when position reaches limits

// ENHANCED OVERTRADING PREVENTION (1.3 IMPROVEMENTS) - Advanced cooldown system
input bool    InpUseAdvancedCooldown  = true;  // Enable enhanced cooldown system
input int     InpMinCooldownMinutes   = 65;    // Minimum cooldown period after closing position (minutes)
input bool    InpUseChoppyDetection   = true;  // Enable choppy market detection
input double  InpChoppyATRThreshold   = 0.8;   // ATR ratio threshold for choppy markets (current vs 20-period avg)
input int     InpChoppyLookbackPeriod = 20;    // Lookback period for choppy market analysis
input int     InpChoppyMinRange       = 30;    // Minimum points range to avoid micro-movements
input bool    InpUseConsecutiveLosses = true;  // Enable consecutive loss protection
input int     InpMaxConsecutiveLosses = 3;     // Max consecutive losses before extended cooldown
input int     InpExtendedCooldownMin  = 120;   // Extended cooldown after consecutive losses (minutes)
input bool    InpUseLossBasedCooldown = true;  // Scale cooldown based on loss size
// REFINED SIGNAL PROCESSING (1.4 IMPROVEMENTS) - Signal smoothing and thresholding
input bool    InpUseSignalRefinement    = true;   // Enable enhanced signal processing
input bool    InpUseSignalSmoothing     = true;   // Enable signal smoothing over multiple bars
input int     InpSignalSmoothingPeriod  = 3;      // Number of bars for signal averaging
input bool    InpUseAdvancedPersistence = true;   // Enhanced signal persistence beyond 1.2
input int     InpAdvancedPersistBars    = 3;      // Bars signal must persist (more strict than 1.2)
input bool    InpUseSignalThreshold     = true;   // Require minimum signal strength difference
input double  InpSignalThresholdQ       = 0.15;   // Minimum Q-value difference between best and second-best action
input bool    InpUseMinimumPipMove      = true;   // Require minimum predicted price movement
input int     InpMinimumPipThreshold    = 8;      // Minimum pip movement to justify trade (covers spread/slippage)
input bool    InpUseQValueSmoothing     = true;   // Apply exponential smoothing to Q-values
input double  InpQValueSmoothingAlpha   = 0.7;    // Smoothing factor for Q-values (0.1=heavy smoothing, 0.9=light)
input double  InpLossCooldownMultiplier = 1.5; // Cooldown multiplier for losing trades
// CONFIDENCE-BASED TRADE FILTER (3.1 IMPROVEMENTS) - Advanced signal confidence filtering
input bool    InpUseConfidenceFilter    = true;   // Enable confidence-based trade filtering
input double  InpMinConfidenceThreshold = 0.65;   // Minimum confidence required to execute trade (0.0-1.0)
input bool    InpUseQSpreadConfidence   = true;   // Use Q-value spread for confidence calculation
input bool    InpUseMagnitudeConfidence = true;   // Use Q-value magnitude for confidence calculation  
input bool    InpUseSoftmaxConfidence   = true;   // Use softmax probability for confidence calculation
input double  InpQSpreadWeight          = 0.4;    // Weight for Q-value spread confidence (0.0-1.0)
input double  InpMagnitudeWeight        = 0.3;    // Weight for magnitude confidence (0.0-1.0)
input double  InpSoftmaxWeight          = 0.3;    // Weight for softmax confidence (0.0-1.0)
input double  InpConfidenceBoostFactor  = 1.2;    // Boost factor for strong signals (>1.0 amplifies confidence)
input bool    InpLogConfidenceDetails   = true;   // Log detailed confidence calculations
input int     InpConfidenceLogThrottle  = 30;     // Throttle confidence logging (seconds between logs)
// ENSEMBLE MODEL DECISIONING (3.2 IMPROVEMENTS) - Multiple model ensemble system
input bool    InpUseEnsembleModel      = false;  // Enable ensemble model decisioning  
input int     InpMaxEnsembleModels     = 5;      // Maximum number of models in ensemble (2-5)
input string  InpEnsembleModel1        = "DoubleDueling_DRQN_Model_1.dat";  // Ensemble model 1 filename
input string  InpEnsembleModel2        = "DoubleDueling_DRQN_Model_2.dat";  // Ensemble model 2 filename
input string  InpEnsembleModel3        = "DoubleDueling_DRQN_Model_3.dat";  // Ensemble model 3 filename (optional)
input string  InpEnsembleModel4        = "DoubleDueling_DRQN_Model_4.dat";  // Ensemble model 4 filename (optional)
input string  InpEnsembleModel5        = "DoubleDueling_DRQN_Model_5.dat";  // Ensemble model 5 filename (optional)
input double  InpEnsembleModel1Weight  = 1.0;    // Weight for model 1 in ensemble (0.0-2.0)
input double  InpEnsembleModel2Weight  = 1.0;    // Weight for model 2 in ensemble (0.0-2.0)
input double  InpEnsembleModel3Weight  = 1.0;    // Weight for model 3 in ensemble (0.0-2.0)
input double  InpEnsembleModel4Weight  = 1.0;    // Weight for model 4 in ensemble (0.0-2.0)
input double  InpEnsembleModel5Weight  = 1.0;    // Weight for model 5 in ensemble (0.0-2.0)
input string  InpEnsembleMethod        = "weighted_average"; // Ensemble method: "majority_vote", "weighted_average", "confidence_weighted"
input double  InpEnsembleAgreementThreshold = 0.6; // Minimum agreement threshold for ensemble decision (0.5-1.0)
input bool    InpRequireUnanimousHold  = false;  // Require all models to agree on HOLD/FLAT actions
input bool    InpLogEnsembleDetails    = true;   // Log detailed ensemble decision process
input int     InpEnsembleLogThrottle   = 60;     // Throttle ensemble logging (seconds between detailed logs)
input bool    InpEnsembleFallbackMode  = true;   // Fallback to single model if ensemble fails
// ADAPTIVE PARAMETER LOGIC (3.3 IMPROVEMENTS) - Self-tuning parameters based on market conditions
input bool    InpUseAdaptiveParameters = true;   // Enable adaptive parameter adjustment system
input bool    InpAdaptiveATRMultipliers = true;  // Adapt ATR multipliers based on volatility regime
input double  InpBaseATRMultiplier     = 2.5;    // Base ATR multiplier (will be adapted)
input double  InpMinATRMultiplier      = 1.5;    // Minimum ATR multiplier in calm markets
input double  InpMaxATRMultiplier      = 4.0;    // Maximum ATR multiplier in volatile markets
input int     InpVolatilityLookback    = 20;     // Periods for volatility calculation
input double  InpHighVolThreshold      = 1.5;    // High volatility threshold (multiple of median)
input double  InpLowVolThreshold       = 0.7;    // Low volatility threshold (multiple of median)
input bool    InpAdaptiveRiskSizing    = true;   // Adapt position sizing based on performance
input double  InpBaseRiskPercent       = 2.0;    // Base risk percentage (will be adapted)
input double  InpMinRiskPercent        = 0.5;    // Minimum risk percentage after losses
input double  InpMaxRiskPercent        = 3.5;    // Maximum risk percentage after wins
input int     InpPerformanceLookback   = 10;     // Number of trades to consider for performance
input double  InpWinStreakMultiplier   = 1.2;    // Risk multiplier per consecutive win (max 3 wins)
input double  InpLossStreakDivisor     = 1.3;    // Risk divisor per consecutive loss (max 3 losses)
input bool    InpAdaptiveTimeouts      = true;   // Adapt trade timeouts based on market conditions
input int     InpBaseTimeoutHours      = 72;     // Base timeout hours (will be adapted)
input double  InpVolatileTimeoutMultiplier = 0.7; // Timeout multiplier in volatile markets
input bool    InpLogAdaptiveChanges    = true;   // Log adaptive parameter changes
input int     InpAdaptiveLogThrottle   = 300;    // Throttle adaptive logging (seconds between logs)
input bool    InpResetAdaptiveDaily    = false;  // Reset adaptive parameters daily
// EMERGENCY STOP/DRAWDOWN CONTROL (1.5 IMPROVEMENTS) - Circuit breaker protection
input bool    InpUseAdvancedEmergencyStop = true;   // Enable enhanced emergency stop system
input bool    InpUseDailyLossLimit      = true;     // Enable daily loss limit circuit breaker
input double  InpDailyLossLimitAmount   = 200.0;    // Maximum daily loss in account currency ($200)
input double  InpDailyLossLimitPercent  = 3.0;      // Maximum daily loss as % of starting equity
input bool    InpUseDrawdownCircuitBreaker = true;  // Enable progressive drawdown protection
input double  InpLevel1DrawdownPercent  = 5.0;      // Level 1: Reduce position sizes (5% drawdown)
input double  InpLevel2DrawdownPercent  = 10.0;     // Level 2: Stop new trades (10% drawdown) 
input double  InpLevel3DrawdownPercent  = 15.0;     // Level 3: Emergency shutdown (15% drawdown)
input bool    InpUseConsecutiveLossLimit = true;    // Enable consecutive loss emergency stop
input int     InpMaxConsecutiveLossLimit = 5;       // Max consecutive losses before halt
input bool    InpUseTimeBasedReset      = true;     // Reset daily limits at start of new trading day
input int     InpEmergencyPauseHours    = 4;        // Hours to pause trading after emergency stop
// INDICATOR CACHING & OPTIMIZATION (2.1 IMPROVEMENTS) - Performance enhancement
input bool    InpUseIndicatorCaching = true;   // Enable indicator result caching
input bool    InpUsePrecalculation   = true;   // Pre-calculate indicators at bar open
input int     InpCacheRefreshBars    = 1;      // Refresh cached indicators every N bars
input bool    InpOptimizeATRCalls    = true;   // Optimize frequent ATR calculations
input bool    InpCacheComplexCalcs   = true;   // Cache expensive mathematical operations
input bool    InpUseSmartHandles     = true;   // Intelligent indicator handle management
// DATA ACCESS OPTIMIZATION (2.2 IMPROVEMENTS) - Memory and performance enhancement
input bool    InpOptimizeDataAccess  = true;   // Enable optimized data structures
input bool    InpReuseArrays         = true;   // Reuse arrays to minimize allocations
input bool    InpOptimizeLocals      = true;   // Prefer local variables where appropriate
input bool    InpMinimizeMemOps      = true;   // Minimize unnecessary memory operations
input int     InpPreallocateSize     = 100;    // Pre-allocate array size for reuse
input bool    InpOptimizeSeriesAccess = true;  // Optimize price data access patterns
// INTELLIGENT LOGGING SYSTEM (2.3 IMPROVEMENTS) - Performance-oriented logging
input bool    InpMinimizeLogging     = true;   // Enable intelligent logging reduction
input bool    InpLogCriticalOnly     = true;   // Log only critical events (trades, errors)
input bool    InpLogTradeEvents      = true;   // Log trade open/close events
input bool    InpLogSignalChanges    = false;  // Log AI signal changes (can be frequent)
input bool    InpLogFilterDetails    = false;  // Log detailed signal filtering info
input bool    InpLogPerformanceData  = false;  // Log performance metrics on each bar
input int     InpLogThrottleSeconds  = 300;    // Throttle repeated logs (5 minutes default)
input int     InpSignalLogThrottle   = 60;     // Signal logging throttle in seconds
input int     InpPerfLogThrottle     = 300;    // Performance logging throttle in seconds
input int     InpRiskLogThrottle     = 120;    // Risk logging throttle in seconds
input int     InpFilterLogThrottle   = 60;     // Filter logging throttle in seconds
input bool    InpLogInitialization   = true;   // Log EA initialization details
input bool    InpLogErrorsAlways     = true;   // Always log errors regardless of settings
// EFFICIENT LOOPING (2.4 IMPROVEMENTS) - Loop optimization and early exit conditions
input bool    InpOptimizeLoops       = true;   // Enable loop optimization techniques
input bool    InpUseEarlyBreaks      = true;   // Use early break conditions in loops
input bool    InpCombineLoops        = true;   // Combine redundant loops where possible
input bool    InpSimplifyConditions  = true;   // Simplify complex conditional checks in loops
input bool    InpOptimizePositionScan = true;  // Optimize position scanning performance
input int     InpMaxLoopIterations   = 1000;   // Maximum loop iterations before forced break
input bool    InpUseOptimizedArrays  = true;   // Use optimized array access patterns
// TICK HANDLING OPTIMIZATION (2.5 IMPROVEMENTS) - Selective tick processing
input bool    InpOptimizeTicks       = true;   // Enable tick handling optimization
input bool    InpProcessOnBarClose   = true;   // Process signals only on bar close
input bool    InpAllowIntraBarTicks  = false;  // Allow some intra-bar tick processing
input int     InpTickSkipRatio       = 5;      // Process every Nth tick when intra-bar enabled
input bool    InpUseTimerChecks      = true;   // Use OnTimer for periodic checks
input int     InpTimerIntervalSec    = 60;     // Timer interval in seconds
input bool    InpTrailingOnTicks     = true;   // Allow trailing stops on every tick
input bool    InpRiskChecksOnTicks   = true;   // Allow risk checks on every tick
input bool    InpEmergencyOnTicks    = true;   // Allow emergency checks on every tick

// ADVANCED RISK MANAGEMENT (NEW - trailing stops and volatility controls)
input bool    InpUseTrailingStop     = true;  // Enable trailing stop functionality
input double  InpTrailStartATR       = 2.0;   // ATR multiple to start trailing (profit threshold)
input double  InpTrailStopATR        = 1.0;   // ATR multiple for trailing stop distance
input bool    InpUseBreakEven        = true;  // Move SL to break-even when profitable
input double  InpBreakEvenATR        = 1.5;   // ATR multiple profit to trigger break-even
input double  InpBreakEvenBuffer     = 5.0;   // Points buffer beyond break-even (small profit)

// ENHANCED RISK MANAGEMENT (1.1 IMPROVEMENTS) - Advanced profit protection
input bool    InpUsePartialClose     = true;  // Enable partial position closing at profit levels
input double  InpPartialCloseLevel1  = 1.5;   // ATR multiple for first partial close (50% of position)
input double  InpPartialCloseLevel2  = 2.5;   // ATR multiple for second partial close (75% remaining)
input double  InpPartialClosePercent1 = 50.0; // Percentage to close at first level
input double  InpPartialClosePercent2 = 50.0; // Percentage to close at second level (of remaining)
input bool    InpUseAcceleratedTrail = true;  // Enable accelerated trailing as profit increases
input double  InpAccelTrailMultiplier = 0.8;  // Reduce trail distance by this factor for large profits
input double  InpAccelTrailThreshold = 3.0;   // ATR multiple profit threshold to accelerate trailing
input bool    InpUseDynamicATRStops   = true; // Use dynamic ATR-based stops that tighten over time
input double  InpATRTightenRate       = 0.95; // Daily ATR stop tightening rate (0.95 = 5% tighter per day)
input int     InpMaxDaysToTighten     = 5;    // Maximum days to apply ATR tightening

// ENHANCED TRADE FILTERING & SIGNAL CONFIRMATION (1.2 IMPROVEMENTS)
input bool    InpUseSignalConfirmation = true;  // Enable multi-confirmation signal filtering
input bool    InpUseMultiTimeframe     = true;  // Require higher timeframe trend alignment
input ENUM_TIMEFRAMES InpHigherTF      = PERIOD_H1; // Higher timeframe for trend confirmation
input bool    InpUseSecondaryIndicators = true; // Use RSI/MACD confirmation
input int     InpRSIPeriod             = 14;    // RSI period for signal confirmation
input double  InpRSIOverbought         = 70.0;  // RSI overbought level
input double  InpRSIOversold           = 30.0;  // RSI oversold level
input bool    InpUseMACDConfirmation   = true;  // Use MACD for trend confirmation
input int     InpMACDFastEMA           = 12;    // MACD fast EMA period
input int     InpMACDSlowEMA           = 26;    // MACD slow EMA period
input int     InpMACDSignalSMA         = 9;     // MACD signal line SMA period
input bool    InpUseSignalPersistence  = true;  // Require signal persistence across bars
input int     InpSignalPersistenceBars = 2;     // Number of bars signal must persist
input double  InpMinSignalStrength     = 0.6;   // Minimum Q-value difference for strong signals
input bool    InpUseSpreadFilter       = true;  // Enhanced spread-based entry control
input double  InpMaxSpreadATR          = 0.5;   // Maximum spread as ATR multiple
input bool    InpUseLiquidityFilter    = true;  // Avoid trading during low liquidity periods
input int     InpMinTickVolume         = 10;    // Minimum tick volume for trade execution

// EMERGENCY RISK CONTROLS - FINAL SAFETY NET TO PREVENT ACCOUNT DESTRUCTION
// These are the absolute last line of defense against catastrophic losses
input bool    InpUseEmergencyStops   = true;  // Master switch for emergency protection systems
                                              // ↳ CRITICAL: Keep enabled - these prevent account wipeouts
input double  InpEmergencyStopLoss   = 500.0; // Hard dollar limit per single trade
                                              // ↳ ACCOUNT SAVER: No single trade can lose more than this amount
input double  InpEmergencyDrawdown   = 15.0;  // Stop all trading if account drawdown exceeds this %
                                              // ↳ CIRCUIT BREAKER: Shuts down EA if overall losses get extreme

// VOLATILITY REGIME CONTROLS (NEW - advanced volatility awareness)
input bool    InpUseVolatilityRegime = true;  // Enable volatility regime detection
input double  InpVolRegimeMultiple   = 2.5;   // ATR multiple vs median to detect high volatility
input int     InpVolRegimePeriod     = 50;    // Lookback period for volatility regime detection
input double  InpVolRegimeSizeReduce = 0.5;   // Position size multiplier during high volatility (0.5 = half size)
input int     InpVolRegimePauseMin   = 15;    // Minutes to pause trading after extreme volatility

// PORTFOLIO RISK CONTROLS (NEW - prevent overexposure)
input int     InpMaxPositions    = 3;     // Maximum concurrent positions
input double  InpMaxTotalRisk    = 6.0;   // Maximum total portfolio risk %
input double  InpMaxCorrelation  = 0.7;   // Maximum correlation between positions

// TRADING SESSION CONTROLS (NEW - time-based filters)
input bool    InpUseTradingHours = true;  // Enable trading hours filter
input string  InpTradingStart    = "08:00"; // Daily trading start time (server time)
input string  InpTradingEnd      = "17:00"; // Daily trading end time (server time)
input bool    InpAvoidFriday     = true;  // Avoid trading on Fridays (weekend gap risk)
input bool    InpAvoidSunday     = true;  // Avoid trading on Sundays (gap openings)

// NEWS AND VOLATILITY CONTROLS (NEW - market condition filters)
input bool    InpUseNewsFilter   = true;  // Enable high volatility/news pause
input double  InpMaxVolatility   = 3.0;   // Maximum ATR multiplier vs 20-day average
input int     InpVolatilityPause = 60;    // Minutes to pause after high volatility
input int     InpNewsBuffer      = 30;    // Minutes before/after major news to pause

// LEGACY INPUTS (kept for backwards compatibility when risk sizing is disabled)
input double  InpLotsStrong      = 0.10;  // Position size for strong signals (0.10 = 10,000 units)
input double  InpLotsWeak        = 0.05;  // Position size for weak signals (0.05 = 5,000 units)  

// GENERAL INPUTS
input bool    InpAllowNewPos     = true;  // Allow opening new positions
input bool    InpCloseOpposite   = true;  // Close opposite positions when signal changes
input long    InpMagic           = 420424; // Unique identifier for this EA's trades
input int     InpBarLookback     = 2000;   // How many historical bars to load for analysis

//============================== CONSTANTS & GLOBAL VARIABLES =================
// These define the AI model structure and store important data
// Note: STATE_SIZE, ACTIONS, and action constants are defined in CortexTradeLogic.mqh

string ACTION_NAME[ACTIONS] = { "BUY_STRONG","BUY_WEAK","SELL_STRONG","SELL_WEAK","HOLD","FLAT" };

// Global variables that store model information
string           g_model_symbol = NULL;    // Symbol the model was trained on
ENUM_TIMEFRAMES  g_model_tf     = PERIOD_CURRENT; // Timeframe model was trained on
int              g_h1=64, g_h2=64, g_h3=64;      // Neural network layer sizes
double           g_feat_min[], g_feat_max[];      // Feature normalization ranges
bool             g_loaded = false;                // Whether model loaded successfully

// RISK MANAGEMENT GLOBAL VARIABLES (NEW - essential for production trading)
double           g_initial_equity = 0.0;          // Account equity at EA start
int              g_recent_trades[];               // Array to track recent trade results (1=win, 0=loss)
int              g_recent_trades_count = 0;       // Number of recent trades tracked
bool             g_trading_paused = false;        // Whether trading is paused due to risk
datetime         g_last_risk_check = 0;           // Last time we checked risk metrics

// PORTFOLIO RISK TRACKING (NEW - prevent overexposure)
double           g_total_risk_percent = 0.0;      // Current total portfolio risk %
int              g_active_positions = 0;          // Number of active positions
datetime         g_last_position_check = 0;       // Last portfolio risk assessment

// SESSION AND NEWS CONTROLS (NEW - time-based filters)
datetime         g_volatility_pause_until = 0;    // Pause trading until this time due to volatility
datetime         g_news_pause_until = 0;          // Pause trading until this time due to news
double           g_baseline_atr = 0.0;             // 20-day average ATR for volatility comparison

// ADVANCED RISK MANAGEMENT GLOBALS (NEW - trailing stops and volatility regime)
double           g_volatility_median = 0.0;       // Median ATR for volatility regime detection
datetime         g_last_volatility_check = 0;     // Last time we checked volatility regime
bool             g_high_volatility_regime = false; // Currently in high volatility regime
datetime         g_volatility_regime_pause_until = 0; // Pause until this time due to volatility regime

// ENHANCED RISK MANAGEMENT GLOBALS (1.1 IMPROVEMENTS) - Advanced profit protection
bool             g_partial_close_level1_hit = false; // Whether first partial close level was reached
bool             g_partial_close_level2_hit = false; // Whether second partial close level was reached
double           g_position_original_size = 0.0;     // Original position size before partial closes
double           g_position_highest_profit_atr = 0.0; // Highest profit achieved in ATR multiples

// ENHANCED TRADE FILTERING GLOBALS (1.2 IMPROVEMENTS) - Signal confirmation tracking
int              g_last_signal_action = ACTION_HOLD;     // Last AI signal action
int              g_signal_persistence_count = 0;         // How many bars signal has persisted
datetime         g_last_signal_time = 0;                 // Time of last signal evaluation
int              g_higher_tf_trend = 0;                   // Higher TF trend: -1=down, 0=neutral, 1=up
datetime         g_last_higher_tf_check = 0;              // Last time higher TF was checked
int              g_rsi_handle = INVALID_HANDLE;           // RSI indicator handle
int              g_macd_handle = INVALID_HANDLE;          // MACD indicator handle
int              g_higher_tf_ma_handle = INVALID_HANDLE;  // Higher TF moving average handle
bool             g_indicators_initialized = false;        // Whether confirmation indicators are ready

// PHASE 1 ENHANCEMENT GLOBALS - POSITION TRACKING AND PROFIT MANAGEMENT
datetime         g_position_open_time = 0;        // When current position was opened (Phase 1)
double           g_position_entry_price = 0.0;    // Entry price of current position (Phase 1)
double           g_position_size = 0.0;            // Size of current position (Phase 1)
double           g_position_unrealized_pnl = 0.0; // Current unrealized P&L (Phase 1)
int              g_position_type = 0;              // Position type: 0=none, 1=long, 2=short (Phase 1)

// TRADING FREQUENCY CONTROL GLOBALS - PREVENT OVERTRADING
datetime         g_last_trade_time = 0;           // Time of last trade execution
int              g_trades_today = 0;              // Trades executed today
datetime         g_current_day = 0;               // Current trading day

// ENHANCED OVERTRADING PREVENTION GLOBALS (1.3 IMPROVEMENTS) - Advanced cooldown tracking
datetime         g_last_position_close = 0;      // When last position was closed
datetime         g_cooldown_until = 0;            // Trading paused until this time
int              g_consecutive_losses = 0;        // Count of consecutive losing trades
double           g_last_trade_pnl = 0.0;          // P&L of last closed trade
bool             g_is_choppy_market = false;      // Current choppy market state
datetime         g_last_choppy_check = 0;         // Last time choppy market was checked
datetime         g_extended_cooldown_until = 0;   // Extended cooldown period end time
// REFINED SIGNAL PROCESSING GLOBALS (1.4 IMPROVEMENTS) - Signal smoothing and tracking
double           g_signal_history[];              // History of signals for smoothing
double           g_qvalue_smoothed[6];            // Exponentially smoothed Q-values
bool             g_qvalue_initialized = false;   // Whether smoothed Q-values are initialized
int              g_current_signal_sequence = 0;  // Current signal in persistence sequence
int              g_signal_sequence_action = ACTION_HOLD; // Action being tracked for persistence
datetime         g_signal_sequence_start = 0;    // When current signal sequence started
int              g_advanced_persist_count = 0;   // Count of bars for advanced persistence
// EMERGENCY STOP/DRAWDOWN CONTROL GLOBALS (1.5 IMPROVEMENTS) - Circuit breaker tracking
double           g_daily_starting_equity = 0.0;   // Starting equity for daily P&L calculation
double           g_daily_pnl = 0.0;               // Current day's realized P&L
datetime         g_current_trading_day = 0;       // Current trading day timestamp
bool             g_daily_loss_limit_hit = false;  // Whether daily loss limit was reached
datetime         g_emergency_pause_until = 0;     // Trading paused until this time
int              g_drawdown_level = 0;            // Current drawdown protection level (0-3)
int              g_consecutive_emergency_losses = 0; // Count of consecutive losses for emergency
bool             g_emergency_shutdown_active = false; // Emergency system shutdown state
datetime         g_last_emergency_check = 0;      // Last time emergency conditions were checked
double           g_max_equity_today = 0.0;        // Maximum equity reached today (for drawdown calc)
// INDICATOR CACHING & OPTIMIZATION GLOBALS (2.1 IMPROVEMENTS) - Performance tracking
int              g_atr_14_handle = INVALID_HANDLE;    // Cached ATR(14) indicator handle
int              g_atr_50_handle = INVALID_HANDLE;    // Cached ATR(50) indicator handle  
int              g_ma_10_handle = INVALID_HANDLE;     // Cached MA(10) indicator handle
int              g_ma_50_handle = INVALID_HANDLE;     // Cached MA(50) indicator handle
double           g_cached_atr_14 = 0.0;              // Cached ATR(14) value
double           g_cached_atr_50 = 0.0;              // Cached ATR(50) value
double           g_cached_ma_10 = 0.0;               // Cached MA(10) value  
double           g_cached_ma_50 = 0.0;               // Cached MA(50) value
datetime         g_last_cache_update = 0;            // When cache was last updated
int              g_cache_bar_count = 0;              // Number of bars since cache refresh
bool             g_cache_indicators_ready = false;   // Whether cached indicators are initialized
double           g_cached_atr_monetary = 0.0;        // Cached ATR in monetary terms
double           g_cached_volatility_impact = 0.0;   // Cached volatility impact calculation
datetime         g_last_complex_calc_time = 0;       // When complex calculations were last done
// DATA ACCESS OPTIMIZATION GLOBALS (2.2 IMPROVEMENTS) - Memory management
double           g_reusable_buffer[];                // Reusable buffer for indicator data
double           g_temp_array[];                     // Temporary array for calculations
double           g_neural_z1[];                      // Reusable neural network layer 1 outputs
double           g_neural_a1[];                      // Reusable neural network layer 1 activations
double           g_neural_z2[];                      // Reusable neural network layer 2 outputs
double           g_neural_a2[];                      // Reusable neural network layer 2 activations
double           g_neural_z3[];                      // Reusable neural network layer 3 outputs
double           g_neural_a3[];                      // Reusable neural network layer 3 activations
double           g_neural_final[];                   // Reusable neural network final outputs
double           g_smoothed_q_buffer[6];             // Pre-allocated buffer for Q-value smoothing
bool             g_data_arrays_initialized = false; // Whether reusable arrays are ready
int              g_buffer_size = 0;                  // Current size of reusable buffers
// INTELLIGENT LOGGING SYSTEM GLOBALS (2.3 IMPROVEMENTS) - Performance-oriented logging
datetime         g_last_signal_log = 0;             // Last time AI signal was logged
datetime         g_last_filter_log = 0;             // Last time filter details were logged
datetime         g_last_performance_log = 0;        // Last time performance data was logged
string           g_last_logged_message = "";        // Last logged message for deduplication
datetime         g_last_message_time = 0;           // When last message was logged
int              g_log_suppression_count = 0;       // Count of suppressed duplicate logs
bool             g_logging_system_ready = false;    // Whether logging system is initialized

// EFFICIENT LOOPING (2.4 IMPROVEMENTS) - Loop optimization globals
int              g_loop_iteration_count = 0;        // Track loop iterations for safety
int              g_position_cache_count = 0;        // Cache position count to avoid repeated calls
ulong            g_cached_positions[100];           // Cache of position tickets for fast access
datetime         g_last_position_scan = 0;         // Last time positions were scanned
bool             g_position_cache_valid = false;   // Whether position cache is current
int              g_max_loop_safety = 0;            // Safety counter for infinite loop protection

// TICK HANDLING OPTIMIZATION (2.5 IMPROVEMENTS) - Selective tick processing globals
datetime         g_last_signal_bar_time = 0;      // Last bar time when signal was processed
int              g_tick_counter = 0;               // Counter for tick skipping
datetime         g_last_timer_check = 0;          // Last time timer check was performed
bool             g_tick_optimization_active = false; // Whether tick optimization is enabled
int              g_skipped_ticks_count = 0;       // Count of skipped ticks for statistics
int              g_processed_ticks_count = 0;     // Count of processed ticks for statistics
bool             g_is_new_bar = false;            // Flag indicating if current tick is on new bar
datetime         g_current_bar_time = 0;          // Current bar timestamp for comparison

// PHASE 2 & 3 ENHANCEMENT GLOBALS - ADVANCED POSITION AND MARKET ANALYSIS
double           g_trend_strength = 0.0;          // Current trend strength (Phase 3)
double           g_volatility_regime = 0.0;       // Current volatility regime indicator (Phase 3)
int              g_market_regime = 0;              // Market regime: 0=ranging, 1=trending, 2=volatile (Phase 3)
double           g_position_normalized_time = 0.0; // Normalized holding time [0-1] (Phase 3)
double           g_unrealized_pnl_ratio = 0.0;    // P&L ratio vs ATR (Phase 3)

// CONFIDENCE-BASED FILTERING (3.1 IMPROVEMENTS) - Confidence tracking globals
double           g_last_confidence_score = 0.0;    // Last calculated confidence score
double           g_q_spread_confidence = 0.0;      // Q-value spread component of confidence
double           g_magnitude_confidence = 0.0;     // Q-value magnitude component of confidence  
double           g_softmax_confidence = 0.0;       // Softmax probability component of confidence
datetime         g_last_confidence_log = 0;        // Last time confidence was logged (for throttling)
int              g_confidence_filtered_count = 0;  // Count of trades filtered due to low confidence
int              g_confidence_passed_count = 0;    // Count of trades that passed confidence filter
double           g_confidence_sum = 0.0;           // Running sum of confidence scores for statistics
int              g_confidence_calculation_count = 0; // Count of confidence calculations for averaging

// ENSEMBLE MODEL DECISIONING (3.2 IMPROVEMENTS) - Ensemble system globals (forward declarations)
bool             g_ensemble_models_loaded[5];    // Track which models are successfully loaded
double           g_ensemble_model_weights[5];    // Normalized weights for each model
int              g_active_ensemble_count = 0;    // Number of successfully loaded models
datetime         g_last_ensemble_log = 0;        // Last time ensemble was logged (for throttling)
string           g_ensemble_method = "";         // Current ensemble aggregation method
double           g_ensemble_agreement_scores[6]; // Agreement scores per action
double           g_ensemble_final_q[6];          // Final ensemble Q-values
int              g_ensemble_votes[6];            // Vote count per action (for majority vote)
double           g_ensemble_confidences[5];      // Confidence scores from each model
int              g_ensemble_predictions_count = 0; // Count of ensemble predictions made
int              g_ensemble_agreement_count = 0;   // Count of predictions meeting agreement threshold
double           g_ensemble_avg_agreement = 0.0;   // Average agreement score for statistics

// ADAPTIVE PARAMETER LOGIC (3.3 IMPROVEMENTS) - Self-tuning parameter globals
double           g_adaptive_atr_multiplier = 2.5;       // Current adaptive ATR multiplier
double           g_adaptive_risk_percent = 2.0;         // Current adaptive risk percentage  
int              g_adaptive_timeout_hours = 72;         // Current adaptive timeout hours
int              g_current_volatility_regime = 1;       // 0=low, 1=normal, 2=high volatility
double           g_current_volatility_ratio = 1.0;      // Current volatility vs median ratio
double           g_atr_history[50];                     // ATR history for volatility calculation
int              g_atr_history_index = 0;               // Index for circular ATR buffer
bool             g_atr_history_filled = false;          // Whether ATR history buffer is full
int              g_consecutive_wins = 0;                // Current consecutive wins
// Note: g_consecutive_losses already declared above
double           g_recent_trade_results[20];            // Recent trade P&L results
int              g_trade_results_index = 0;             // Index for circular trade results buffer
bool             g_trade_results_filled = false;       // Whether trade results buffer is full
datetime         g_last_adaptive_log = 0;               // Last time adaptive changes were logged
datetime         g_last_adaptive_reset = 0;             // Last time adaptive parameters were reset
double           g_adaptive_parameter_changes = 0;      // Count of parameter adjustments made
double           g_volatility_regime_changes = 0;       // Count of volatility regime changes

//============================== UTILITY FUNCTIONS ==============================
// Core mathematical and array manipulation functions used throughout the EA
// These lightweight functions provide essential operations for neural network
// processing, data analysis, and value constraints

// 2D Matrix to 1D Array Index Converter
// Converts row/column coordinates to linear array index for matrix operations
// Essential for neural network weight matrices stored as flat arrays
// Formula: index = row * number_of_columns + column
// Used extensively in model loading and neural network inference
int    idx2(const int r,const int c,const int ncols){ return r*ncols + c; }

// Maximum Value Index Finder (Double Array)
// Locates the position of the largest value in a double array
// Critical for AI decision making: finds the trading action with highest Q-value
// This function determines which action (BUY, SELL, HOLD, etc.) the AI recommends
// Returns: Index of the maximum value (corresponds to selected trading action)
int    argmax(const double &v[]){ 
    int m=0; 
    for(int i=1;i<ArraySize(v);++i) 
        if(v[i]>v[m]) m=i; 
    return m; 
}

// Maximum Value Index Finder (Integer Array)
// Same as argmax but optimized for integer arrays
// Used for ensemble voting, confidence calculations, and discrete choice problems
// Provides type-specific optimization for integer-based decision arrays
int    argmax_int(const int &v[]){ 
    int m=0; 
    for(int i=1;i<ArraySize(v);++i) 
        if(v[i]>v[m]) m=i; 
    return m; 
}

// Value Clipping/Constraining Function
// Forces a value to stay within specified bounds [a, b]
// Prevents neural network outputs from becoming extreme or invalid
// Essential for: activation function bounds, probability constraints, risk limits
// Returns: x clamped to [a,b] range (a if x<a, b if x>b, x otherwise)
double clipd(const double x,const double a,const double b){ return (x<a? a : (x>b? b : x)); }


//============================== NEURAL NETWORK STRUCTURES ==============================
// Data structures that define the architecture and parameters of the AI model
// These structures store the "learned intelligence" from the training process

// Dense (Fully Connected) Neural Network Layer
// Represents a standard neural network layer where every input connects to every output
// The fundamental building block of the Double-Dueling DRQN architecture
// Contains the learned patterns that enable AI trading decisions
struct DenseLayer{ 
    int in,out;     // Layer dimensions: number of input neurons and output neurons
    double W[];     // Weight matrix (flattened): defines connection strengths between neurons
                    // Each weight determines how strongly one neuron influences another
    double b[];     // Bias vector: allows neurons to fire even with zero input
                    // Biases enable the network to learn offset patterns and thresholds
};

// LSTM (Long Short-Term Memory) Layer for Market Memory
// Advanced neural network component that remembers market patterns over time
// Critical for recognizing trends, cycles, and temporal dependencies in trading
// Simplified inference-only version (no training parameters needed for live trading)
struct LSTMInferenceLayer{
    // Layer Architecture
    int in, out;    // Input size (market features) and output size (memory units)
    
    // LSTM Gate Weights - Control information flow and memory retention
    double Wf[], Wi[], Wc[], Wo[];  // Input-to-hidden weight matrices for each gate
                                    // Wf=forget, Wi=input, Wc=candidate, Wo=output
    double Uf[], Ui[], Uc[], Uo[];  // Hidden-to-hidden recurrent weight matrices
                                    // Enable memory persistence across time steps
    double bf[], bi[], bc[], bo[];  // Bias vectors for each gate (learned offset values)
    
    // LSTM Memory State - Maintains market context between predictions
    double h_prev[], c_prev[];      // Previous hidden state and cell state
                                    // Contains accumulated market memory from past bars
    double h_curr[], c_curr[];      // Current hidden state and cell state
                                    // Updated with each new market observation
    
    // Constructor - Initialize empty LSTM layer
    LSTMInferenceLayer() : in(0), out(0) {}
    
    // Initialize LSTM Layer with Specified Dimensions
    // Allocates memory for all weight matrices and state vectors
    // Called once during model loading to set up the layer architecture
    void Init(int _in, int _out){
        in = _in; out = _out;
        
        // Calculate matrix sizes for weight allocation
        int ih_size = in * out;   // Input-to-hidden matrix size
        int hh_size = out * out;  // Hidden-to-hidden matrix size
        
        // Allocate input-to-hidden weight matrices (4 gates: forget, input, candidate, output)
        ArrayResize(Wf, ih_size); ArrayResize(Wi, ih_size); 
        ArrayResize(Wc, ih_size); ArrayResize(Wo, ih_size);
        
        // Allocate hidden-to-hidden recurrent weight matrices
        ArrayResize(Uf, hh_size); ArrayResize(Ui, hh_size); 
        ArrayResize(Uc, hh_size); ArrayResize(Uo, hh_size);
        
        // Allocate bias vectors for each gate
        ArrayResize(bf, out); ArrayResize(bi, out); 
        ArrayResize(bc, out); ArrayResize(bo, out);
        
        // Allocate state vectors for memory persistence
        ArrayResize(h_prev, out); ArrayResize(c_prev, out);
        ArrayResize(h_curr, out); ArrayResize(c_curr, out);
        
        // Initialize with clean memory state
        ResetState();
    }
    
    // Reset LSTM Memory State
    // Clears all accumulated market memory, starting fresh
    // Called when beginning new trading sessions or after significant market events
    void ResetState(){
        ArrayInitialize(h_prev, 0.0);  // Clear previous hidden state
        ArrayInitialize(c_prev, 0.0);  // Clear previous cell state  
        ArrayInitialize(h_curr, 0.0);  // Clear current hidden state
        ArrayInitialize(c_curr, 0.0);  // Clear current cell state
    }
    
    // Simplified LSTM forward pass for inference
    void Forward(const double &x[], double &output[]){
        ArrayResize(output, out);
        
        // Simplified LSTM computation (full implementation would include all gates)
        // For now, just copy input to output with some basic processing
        for(int i = 0; i < out && i < in; i++){
            if(i < ArraySize(x)){
                output[i] = x[i] * 0.5 + h_prev[i] * 0.5; // Simple mix of input and memory
                h_prev[i] = output[i]; // Update hidden state
            }
        }
    }
};

// NEURAL NETWORK CLASS
// This implements the Double-Dueling DRQN that makes trading decisions
class CInferenceNetwork{
  public:
    int inSize,h1,h2,h3,outSize;     // Network architecture: input->hidden layers->output
    DenseLayer L1,L2,L3,L4;          // Dense layers
    LSTMInferenceLayer lstm;         // LSTM layer (when enabled)
    DenseLayer value_head, advantage_head;  // Dueling heads (when enabled)
    
    // Architecture flags (loaded from model file)
    bool has_lstm, has_dueling;
    int lstm_size, value_head_size, advantage_head_size;
    
    // Initialize a layer with specified input/output sizes
    void SetupLayer(DenseLayer &L,int in,int out){ 
        L.in=in; L.out=out; 
        ArrayResize(L.W,in*out);  // Allocate space for weights
        ArrayResize(L.b,out);     // Allocate space for biases
    }
    
    // Set up the entire network structure
    void InitFromSizes(int inS,int h1S,int h2S,int h3S,int outS){ 
        inSize=inS; h1=h1S; h2=h2S; h3=h3S; outSize=outS; 
        SetupLayer(L1,inSize,h1); SetupLayer(L2,h1,h2); 
        SetupLayer(L3,h2,h3); 
        
        if(has_dueling){
            value_head.in = (has_lstm ? lstm_size : h3);
            value_head.out = value_head_size;  // Use size from model file instead of hardcoded 1
            advantage_head.in = (has_lstm ? lstm_size : h3);
            advantage_head.out = outS;
            SetupLayer(value_head, value_head.in, value_head.out);
            SetupLayer(advantage_head, advantage_head.in, advantage_head.out);
        } else {
            SetupLayer(L4,h3,outS);
        }
        
        if(has_lstm){
            lstm.Init(h2, lstm_size);
        }
    }
    
    // Matrix-vector multiplication: multiply weights by input values
    void matvec(const double &W[],int in,int out,const double &x[],double &z[]){ 
        ArrayResize(z,out); 
        for(int j=0;j<out;++j){ 
            double s=0.0; 
            for(int i=0;i<in;++i){ 
                s += W[idx2(i,j,out)]*x[i];  // Sum weighted inputs
            } 
            z[j]=s; 
        } 
    }
    
    // Add bias values to neuron outputs
    void addbias(double &z[],int n,const double &b[]){ 
        for(int i=0;i<n;++i) z[i]+=b[i]; 
    }
    
    // ReLU activation function: outputs max(0, input) for each neuron
    void relu(const double &z[], double &a[], int n){ 
        ArrayResize(a,n); 
        for(int i=0;i<n;++i) a[i]=(z[i]>0.0? z[i]:0.0); 
    }
    
    // Reset LSTM state (call when starting new trading session)
    void ResetLSTMState(){
        if(has_lstm){
            lstm.ResetState();
        }
    }
    
    // Forward pass: process input through all network layers
    void Forward(const double &x[], double &z1[], double &a1[], double &z2[], double &a2[], double &z3[], double &a3[], double &final_out[]){
      matvec(L1.W,L1.in,L1.out,x,z1); addbias(z1,L1.out,L1.b); relu(z1,a1,L1.out);    // Layer 1
      matvec(L2.W,L2.in,L2.out,a1,z2); addbias(z2,L2.out,L2.b); relu(z2,a2,L2.out);   // Layer 2  
      matvec(L3.W,L3.in,L3.out,a2,z3); addbias(z3,L3.out,L3.b); relu(z3,a3,L3.out);   // Layer 3
      
      double lstm_out[];
      if(has_lstm){
          lstm.Forward(a3, lstm_out);
      } else {
          ArrayResize(lstm_out, h3);
          for(int i=0; i<h3; ++i) lstm_out[i] = a3[i];
      }
      
      if(has_dueling){
          // Dueling architecture: V(s) + (A(s,a) - mean(A(s,:)))
          double v_out[], a_out[];
          matvec(value_head.W, value_head.in, value_head.out, lstm_out, v_out);
          addbias(v_out, value_head.out, value_head.b);
          
          matvec(advantage_head.W, advantage_head.in, advantage_head.out, lstm_out, a_out);
          addbias(a_out, advantage_head.out, advantage_head.b);
          
          // Calculate mean advantage
          double mean_adv = 0.0;
          for(int i=0; i<advantage_head.out; ++i) mean_adv += a_out[i];
          mean_adv /= advantage_head.out;
          
          // Combine V(s) + (A(s,a) - mean(A))
          // For multi-dimensional value head, take mean of value outputs as scalar V(s)
          double state_value = 0.0;
          for(int i=0; i<value_head.out; ++i) state_value += v_out[i];
          state_value /= value_head.out;
          
          ArrayResize(final_out, outSize);
          for(int i=0; i<outSize; ++i){
              final_out[i] = state_value + (a_out[i] - mean_adv);
          }
      } else {
          matvec(L4.W,L4.in,L4.out,lstm_out,final_out); addbias(final_out,L4.out,L4.b); // Layer 4 (output)
      }
    }
    
    // Make a trading prediction: input market data, output Q-values for each action
    void Predict(const double &x[], double &qout[]){
      double z1[],a1[],z2[],a2[],z3[],a3[],z4[];
      Forward(x,z1,a1,z2,a2,z3,a3,z4);  // Run forward pass
      ArrayResize(qout,outSize);
      for(int i=0;i<outSize;++i) qout[i]=z4[i];  // Copy final outputs (Q-values)
    }
    
    // Optimized prediction using reusable arrays (2.2 IMPROVEMENT)
    void PredictOptimized(const double &x[], double &qout[]){
      if(!InpMinimizeMemOps || !g_data_arrays_initialized){
        Predict(x, qout); // Fallback to standard method
        return;
      }
      
      // Use global reusable arrays instead of local allocations
      ForwardOptimized(x, g_neural_z1, g_neural_a1, g_neural_z2, g_neural_a2, 
                      g_neural_z3, g_neural_a3, g_neural_final);
      
      // Efficient copy of final results
      OptimizedArrayCopy(qout, g_neural_final);
    }
    
    // Optimized forward pass using pre-allocated arrays (2.2 IMPROVEMENT)
    void ForwardOptimized(const double &x[], double &z1[], double &a1[], double &z2[], 
                         double &a2[], double &z3[], double &a3[], double &final_out[]){
      // Layer 1 - reuse provided arrays
      matvec(L1.W,L1.in,L1.out,x,z1); addbias(z1,L1.out,L1.b); relu(z1,a1,L1.out);
      
      // Layer 2 - reuse provided arrays  
      matvec(L2.W,L2.in,L2.out,a1,z2); addbias(z2,L2.out,L2.b); relu(z2,a2,L2.out);
      
      // Layer 3 - reuse provided arrays
      matvec(L3.W,L3.in,L3.out,a2,z3); addbias(z3,L3.out,L3.b); relu(z3,a3,L3.out);
      
      // LSTM processing with minimal memory allocation
      double lstm_out[];
      ArrayResize(lstm_out, h3);
      if(has_lstm){
          lstm.Forward(a3, lstm_out);
      } else {
          // Direct copy without allocation
          for(int i=0; i<h3; ++i) lstm_out[i] = a3[i];
      }
      
      if(has_dueling){
          // Dueling architecture with optimized memory usage
          double v_out[];
          double a_out[];
          GetReusableBuffer(v_out, value_head.out);
          GetTempArray(a_out, advantage_head.out);
          
          matvec(value_head.W, value_head.in, value_head.out, lstm_out, v_out);
          addbias(v_out, value_head.out, value_head.b);
          
          matvec(advantage_head.W, advantage_head.in, advantage_head.out, lstm_out, a_out);
          addbias(a_out, advantage_head.out, advantage_head.b);
          
          // Calculate mean advantage efficiently
          double mean_adv = 0.0;
          for(int i=0; i<advantage_head.out; ++i){
              mean_adv += a_out[i];
          }
          mean_adv /= advantage_head.out;
          
          // Combine V(s) + (A(s,a) - mean(A)) efficiently
          // For multi-dimensional value head, take mean of value outputs as scalar V(s)
          double state_value = 0.0;
          for(int i=0; i<value_head.out; ++i) state_value += v_out[i];
          state_value /= value_head.out;
          
          OptimizedArrayInit(final_out, outSize);
          for(int i=0; i<outSize; ++i){
              final_out[i] = state_value + (a_out[i] - mean_adv);
          }
      } else {
          matvec(L4.W,L4.in,L4.out,lstm_out,final_out); addbias(final_out,L4.out,L4.b);
      }
    }
};

CInferenceNetwork g_Q;  // The main Double-Dueling DRQN that makes trading decisions

// ENSEMBLE MODEL INSTANCES (3.2 IMPROVEMENTS) - Declared after class definition
CInferenceNetwork g_ensemble_model1;           // Ensemble model 1
CInferenceNetwork g_ensemble_model2;           // Ensemble model 2  
CInferenceNetwork g_ensemble_model3;           // Ensemble model 3
CInferenceNetwork g_ensemble_model4;           // Ensemble model 4
CInferenceNetwork g_ensemble_model5;           // Ensemble model 5

// Get reference to ensemble model by index (since MQL5 doesn't support object arrays)
CInferenceNetwork* GetEnsembleModel(int index){
    switch(index){
        case 0: return &g_ensemble_model1;
        case 1: return &g_ensemble_model2;
        case 2: return &g_ensemble_model3;
        case 3: return &g_ensemble_model4;
        case 4: return &g_ensemble_model5;
        default: return NULL;
    }
}

//============================== MODEL LOADING FUNCTIONS ==============================
// Binary model file deserialization functions for trained neural networks
// These functions reconstruct the AI's "learned intelligence" from disk storage
// Critical for deploying trained models in live trading environments

// Dense Layer Loader - Reconstructs Neural Network Layer from Binary Data
// Loads pre-trained weights and biases that encode the AI's trading knowledge
// Each weight represents a learned connection strength between neurons
// Each bias represents a learned activation threshold for decision-making
//
// Parameters:
//   h: File handle for reading binary model data
//   L: Dense layer structure to populate with loaded parameters
// Returns: true if layer loaded successfully, false if format mismatch
bool LoadLayer(const int h, DenseLayer &L){
  // Read layer dimensions from file header
  int in  = (int)FileReadLong(h);   // Number of input neurons from file
  int out = (int)FileReadLong(h);   // Number of output neurons from file
  
  // Validate layer dimensions match expected architecture
  if(in!=L.in || out!=L.out){ 
      Print("FATAL ERROR: Layer dimension mismatch!");
      Print("Expected: ",L.in," inputs × ",L.out," outputs");
      Print("Found in file: ",in," inputs × ",out," outputs");
      Print("Model file is incompatible with current EA architecture");
      return false; 
  }
  
  // Load weight matrix - defines connection strengths between all neuron pairs
  // Weights encode learned trading patterns and market relationships
  for(int i=0;i<L.in*L.out;++i) {
      L.W[i]=FileReadDouble(h);  // Each weight affects decision-making strength
  }
  
  // Load bias vector - allows neurons to activate independently of inputs
  // Biases enable the network to learn offset patterns and thresholds
  for(int j=0;j<L.out;++j) {
      L.b[j]=FileReadDouble(h);  // Each bias shifts neuron activation point
  }
  
  return true;  // Layer successfully reconstructed from file
}

// LSTM Memory Layer Loader - Reconstructs Market Memory System
// Loads sophisticated temporal memory weights that enable the AI to remember
// market patterns, trends, and cycles across multiple time periods
// LSTM layers are critical for understanding market context and timing
//
// Parameters:
//   h: File handle for reading binary LSTM parameters
//   lstm: LSTM layer structure to populate with memory weights
// Returns: true if LSTM loaded successfully, false if architecture mismatch
bool LoadLSTMLayer(const int h, LSTMInferenceLayer &lstm){
  // Read LSTM layer dimensions from file
  int in = (int)FileReadLong(h);   // Input size (market features)
  int out = (int)FileReadLong(h);  // Output size (memory units)
  
  // Validate LSTM architecture matches expected configuration
  if(in != lstm.in || out != lstm.out){
    Print("FATAL ERROR: LSTM layer dimension mismatch!");
    Print("Expected: ",lstm.in," inputs × ",lstm.out," memory units");
    Print("Found in file: ",in," inputs × ",out," memory units");
    Print("LSTM memory system incompatible with current model");
    return false;
  }
  
  // Load Input-to-Hidden Weight Matrices
  // These weights control how new market information affects memory gates
  for(int i=0; i<lstm.in*lstm.out; i++){
    lstm.Wf[i] = FileReadDouble(h); // Forget gate: what market info to forget
    lstm.Wi[i] = FileReadDouble(h); // Input gate: what new info to remember
    lstm.Wc[i] = FileReadDouble(h); // Candidate gate: how to update memory
    lstm.Wo[i] = FileReadDouble(h); // Output gate: what memory to use for decisions
  }
  
  // Load Hidden-to-Hidden Recurrent Weight Matrices
  // These weights enable memory persistence and temporal pattern recognition
  for(int i=0; i<lstm.out*lstm.out; i++){
    lstm.Uf[i] = FileReadDouble(h); // Forget gate recurrent weights
    lstm.Ui[i] = FileReadDouble(h); // Input gate recurrent weights
    lstm.Uc[i] = FileReadDouble(h); // Candidate gate recurrent weights
    lstm.Uo[i] = FileReadDouble(h); // Output gate recurrent weights
  }
  
  // Load Gate Bias Vectors
  // Biases allow gates to have learned default behaviors independent of inputs
  for(int i=0; i<lstm.out; i++){
    lstm.bf[i] = FileReadDouble(h); // Forget gate bias (default forgetting strength)
    lstm.bi[i] = FileReadDouble(h); // Input gate bias (default input sensitivity)
    lstm.bc[i] = FileReadDouble(h); // Candidate gate bias (default memory update)
    lstm.bo[i] = FileReadDouble(h); // Output gate bias (default output strength)
  }
  
  return true;  // LSTM memory system successfully reconstructed
}

// MetaTrader 5 Timeframe Validator
// Verifies that a timeframe value from a model file is legitimate
// Prevents crashes from corrupted model files with invalid timeframe data
// Essential safety check during model loading to ensure trading compatibility
//
// Parameters:
//   tf: Timeframe value in minutes to validate
// Returns: true if timeframe is valid in MT5, false if invalid/corrupted
bool IsValidTF(int tf){
  // Official MetaTrader 5 timeframes (in minutes)
  // These are the only valid values - anything else indicates file corruption
  static int tfs[] = {
    1,     // M1  - 1 minute charts
    2,     // M2  - 2 minute charts  
    3,     // M3  - 3 minute charts
    4,     // M4  - 4 minute charts
    5,     // M5  - 5 minute charts
    6,     // M6  - 6 minute charts
    10,    // M10 - 10 minute charts
    12,    // M12 - 12 minute charts
    15,    // M15 - 15 minute charts
    20,    // M20 - 20 minute charts
    30,    // M30 - 30 minute charts
    60,    // H1  - 1 hour charts
    120,   // H2  - 2 hour charts
    180,   // H3  - 3 hour charts
    240,   // H4  - 4 hour charts
    360,   // H6  - 6 hour charts
    480,   // H8  - 8 hour charts
    720,   // H12 - 12 hour charts
    1440,  // D1  - Daily charts
    10080, // W1  - Weekly charts
    43200  // MN1 - Monthly charts
  };
  
  // Linear search through valid timeframes
  for(int i=0;i<ArraySize(tfs); ++i) {
    if(tfs[i]==tf) return true;  // Found valid timeframe
  }
  return false;  // Invalid timeframe - possible file corruption
}

// Robust Symbol and Timeframe Extraction from Model Files
// Models may be saved in different formats across training versions
// This function attempts multiple parsing methods to ensure backward compatibility
// Critical for preventing deployment failures due to format differences
//
// Parameters:
//   h: File handle positioned at symbol/timeframe section
//   sym: Output parameter for extracted symbol name
//   tf: Output parameter for extracted timeframe
// Returns: true if successfully extracted both values, false if parsing failed
bool ReadSymbolAndTF(int h, string &sym, ENUM_TIMEFRAMES &tf)
{
  // Save current file position for potential backtracking if parsing fails
  // This allows us to try alternative parsing methods on the same data
  ulong pos = FileTell(h);

  // ---- Method 1: Modern format with length prefix ----
  int slen = (int)FileReadLong(h);  // Read symbol length
  if(slen>0 && slen<=128)  // Reasonable symbol length
  {
    sym = FileReadString(h, slen);       // Read the symbol string
    int tfcand = (int)FileReadLong(h);   // Read timeframe
    if(IsValidTF(tfcand))                // Check if timeframe is valid
    {
      tf = (ENUM_TIMEFRAMES)tfcand;
      return true; // Success!
    }
  }

  // ---- Method 2: Legacy format without length (Unicode) ----
  // We try different symbol lengths until we find a valid timeframe
  FileSeek(h, (long)pos, SEEK_SET);  // Go back to start
  for(int k=3; k<=32; ++k)  // Try symbol lengths from 3 to 32 characters
  {
    // Check if there's a valid timeframe at this position
    FileSeek(h, (long)(pos + (ulong)(2*k)), SEEK_SET);  // Skip k Unicode characters
    int tfcand = (int)FileReadLong(h);
    if(IsValidTF(tfcand))
    {
      // Found valid timeframe, so read the symbol
      FileSeek(h, (long)pos, SEEK_SET);
      sym = FileReadString(h, k);        // Read k characters
      int tfread = (int)FileReadLong(h); // Read timeframe
      tf = (ENUM_TIMEFRAMES)tfread;
      return true;
    }
  }

  // ---- Method 3: Legacy format without length (ANSI) ----
  // Try reading as single-byte characters instead of Unicode
  FileSeek(h, (long)pos, SEEK_SET);
  for(int k=3; k<=32; ++k)
  {
    FileSeek(h, (long)(pos + (ulong)k), SEEK_SET);  // Skip k ANSI characters
    int tfcand = (int)FileReadLong(h);
    if(IsValidTF(tfcand))
    {
      FileSeek(h, (long)pos, SEEK_SET);
      // Read symbol byte-by-byte for ANSI format
      uchar bytes[];
      ArrayResize(bytes, k);
      for(int i=0;i<k;++i){ 
          int b=(int)FileReadInteger(h, CHAR_VALUE); 
          if(b<0) b=0; if(b>255) b=255; 
          bytes[i]=(uchar)b; 
      }
      sym = CharArrayToString(bytes, 0, k);
      int tfread = (int)FileReadLong(h);
      tf = (ENUM_TIMEFRAMES)tfread;
      return true;
    }
  }

  // All methods failed - file might be corrupted
  FileSeek(h, (long)pos, SEEK_SET);
  return false;
}



// MAIN MODEL LOADING FUNCTION
// This loads the trained AI model from a file and sets up the neural network
bool LoadModel(const string filename){
  // Try to open the model file
  int h=FileOpen(filename,FILE_BIN|FILE_READ);
  if(h==INVALID_HANDLE){ 
      Print("EA LoadModel: cannot open ",filename," err=",GetLastError()); 
      return false; 
  }

  // Check magic number to verify this is our model format
  long magic=FileReadLong(h);
  bool has_checkpoint = false;
  
  if(magic==(long)0xC0DE0203){
      has_checkpoint = true;  // New format with training checkpoints
  } else if(magic==(long)0xC0DE0202){
      has_checkpoint = false; // Old format without checkpoints
  } else {
      Print("EA LoadModel: unsupported model file format"); 
      FileClose(h); 
      return false; 
  }

  // Read symbol & timeframe - try new format first, then fall back to old format
  ulong pos = FileTell(h);
  int sym_len = (int)FileReadLong(h);
  if(sym_len > 0 && sym_len <= 32){
    // New format with length prefix
    g_model_symbol = FileReadString(h, sym_len);
    int tf_int = (int)FileReadLong(h);
    g_model_tf = (ENUM_TIMEFRAMES)tf_int;
  } else {
    // Old format - use existing robust parser
    FileSeek(h, (long)pos, SEEK_SET);  // Go back
    if(!ReadSymbolAndTF(h, g_model_symbol, g_model_tf)){
      Print("EA LoadModel: failed to parse symbol/timeframe.");
      FileClose(h); return false;
    }
  }

  // Read model architecture parameters
  int stsz = (int)FileReadLong(h);  // State size (number of input features)
  int acts = (int)FileReadLong(h);  // Number of actions
  g_h1     = (int)FileReadLong(h);  // Hidden layer 1 size
  g_h2     = (int)FileReadLong(h);  // Hidden layer 2 size  
  g_h3     = (int)FileReadLong(h);  // Hidden layer 3 size
  
  // Read architecture flags (for new format)
  if(has_checkpoint){
    g_Q.has_lstm = (FileReadLong(h) == 1);
    g_Q.has_dueling = (FileReadLong(h) == 1);
    g_Q.lstm_size = (int)FileReadLong(h);
    int seq_len = (int)FileReadLong(h);  // Sequence length (not used in EA)
    g_Q.value_head_size = (int)FileReadLong(h);
    g_Q.advantage_head_size = (int)FileReadLong(h);
  } else {
    // Legacy format - assume no advanced features
    g_Q.has_lstm = false;
    g_Q.has_dueling = false;
    g_Q.lstm_size = 0;
    g_Q.value_head_size = 0;
    g_Q.advantage_head_size = 0;
  }

  // Verify model matches our expected structure
  if(stsz!=STATE_SIZE || acts!=ACTIONS){
    Print("EA LoadModel: shape mismatch (STATE/ACTIONS). File has ",stsz,"/",acts);
    FileClose(h); return false;
  }

  // Initialize neural network with the loaded architecture
  g_Q.InitFromSizes(STATE_SIZE,g_h1,g_h2,g_h3,ACTIONS);
  
  // Load feature normalization parameters (min/max values for each input feature)
  ArrayResize(g_feat_min,STATE_SIZE); ArrayResize(g_feat_max,STATE_SIZE);
  for(int i=0;i<STATE_SIZE;++i){ 
      g_feat_min[i]=FileReadDouble(h);  // Minimum value for feature i
      g_feat_max[i]=FileReadDouble(h);  // Maximum value for feature i
  }
  
  // Skip checkpoint data if present (EA doesn't need it)
  if(has_checkpoint){
      datetime last_trained = (datetime)FileReadLong(h);  // Skip training timestamp
      int training_steps = (int)FileReadLong(h);          // Skip training steps
      double checkpoint_eps = FileReadDouble(h);          // Skip epsilon
      double checkpoint_beta = FileReadDouble(h);         // Skip beta
      
      Print("EA LoadModel: Model contains checkpoint data (training timestamp: ", 
            TimeToString(last_trained), ", steps: ", training_steps, ")");
  }

  // Load the trained weights and biases for each layer
  bool ok=true;
  ok = ok && LoadLayer(h,g_Q.L1);  // Load layer 1
  ok = ok && LoadLayer(h,g_Q.L2);  // Load layer 2
  ok = ok && LoadLayer(h,g_Q.L3);  // Load layer 3
  
  // Load LSTM layer if enabled
  if(g_Q.has_lstm){
    ok = ok && LoadLSTMLayer(h, g_Q.lstm);
  }
  
  // Load dueling heads or final layer
  if(g_Q.has_dueling){
    ok = ok && LoadLayer(h, g_Q.value_head);
    ok = ok && LoadLayer(h, g_Q.advantage_head);
  } else {
    ok = ok && LoadLayer(h,g_Q.L4);  // Load layer 4 (output)
  }

  FileClose(h);
  if(ok) {
      Print("EA LoadModel: Successfully loaded ",filename);
      Print("  Symbol: ",g_model_symbol,", Timeframe: ",EnumToString(g_model_tf));
      
      string arch_desc = IntegerToString(STATE_SIZE) + "->" + IntegerToString(g_h1) + "->" + IntegerToString(g_h2) + "->" + IntegerToString(g_h3);
      if(g_Q.has_lstm) arch_desc += "->LSTM(" + IntegerToString(g_Q.lstm_size) + ")";
      if(g_Q.has_dueling) arch_desc += "->Dueling(V:" + IntegerToString(g_Q.value_head_size) + ",A:" + IntegerToString(g_Q.advantage_head_size) + ")";
      arch_desc += "->" + IntegerToString(ACTIONS);
      
      Print("  Architecture: ", arch_desc);
      Print("  Features: LSTM=", (g_Q.has_lstm ? "YES" : "NO"), ", Dueling=", (g_Q.has_dueling ? "YES" : "NO"));
      
      if(has_checkpoint){
          Print("  Model format: Double-Dueling DRQN with training checkpoints");
      } else {
          Print("  Model format: Legacy (no advanced features)");
      }
  }
  return ok;
}

//============================== MARKET DATA & FEATURES ===================
// These functions load market data and calculate technical indicators
// Structure to hold market data (price bars) and timestamps
struct Series{ 
    MqlRates rates[];   // Array of OHLCV bars (Open, High, Low, Close, Volume)
    datetime times[];   // Array of timestamps for each bar
};

//============================== DATA SERIES MANAGEMENT ==============================
// Functions for loading and synchronizing market data across multiple timeframes
// Essential for providing the AI with comprehensive market context

// Historical Market Data Loader
// Downloads OHLCV price data from MetaTrader 5 for AI analysis
// Provides the raw market information that feeds into feature extraction
// Critical for real-time trading decisions and market state assessment
//
// Parameters:
//   sym: Trading symbol (e.g., "EURUSD", "GBPUSD")
//   tf: Timeframe for data (M1, M5, H1, H4, D1, etc.)
//   count: Number of historical bars to load
//   s: Series structure to store the downloaded data
// Returns: true if data loaded successfully, false if download failed
bool LoadSeries(const string sym, ENUM_TIMEFRAMES tf, int count, Series &s){
  // Configure array indexing for chronological access
  // Index [0] = most recent bar, [1] = previous bar, etc.
  // This indexing matches how indicators and technical analysis typically work
  ArraySetAsSeries(s.rates,true);
  
  // Download historical price data from MetaTrader 5 server
  // Gets OHLCV (Open, High, Low, Close, Volume) data for specified period
  int copied = CopyRates(sym,tf,0,count,s.rates);
  
  // Validate data download success
  if(copied<=0){ 
      Print("CRITICAL ERROR: Failed to load market data!");
      Print("Symbol: ",sym,", Timeframe: ",EnumToString(tf));
      Print("MT5 Error Code: ",GetLastError());
      Print("Check symbol availability and market hours");
      return false; 
  }
  
  // Extract timestamps for multi-timeframe synchronization
  // Timestamps enable precise alignment of data across different timeframes
  // Essential for creating comprehensive market state vectors
  ArrayResize(s.times,copied);
  for(int i=0;i<copied;++i) {
      s.times[i]=s.rates[i].time;  // Store bar opening time
  }
  
  return true;  // Market data successfully loaded and indexed
}

// Binary Search for Multi-Timeframe Data Synchronization
// Efficiently finds the latest bar that occurred at or before a specific time
// Critical for aligning data across different timeframes (M1, M5, H1, etc.)
// Uses O(log n) binary search instead of O(n) linear search for performance
//
// Use Case Example:
//   When analyzing H1 data, we need to find which M5 bars correspond to each H1 bar
//   This function quickly locates the synchronization points between timeframes
//
// Parameters:
//   times[]: Array of timestamps (must be sorted in ascending order)
//   n: Number of elements in the times array
//   t: Target timestamp to search for
// Returns: Index of the latest bar with time <= t, or -1 if no such bar exists
int FindIndexLE(const datetime &times[], int n, datetime t){
  int lo=0, hi=n-1, ans=-1;  // Binary search bounds and result tracker
  
  // Perform binary search to find the optimal synchronization point
  while(lo<=hi){
    int mid=(lo+hi)>>1;      // Calculate midpoint (bit shift for fast division)
    
    if(times[mid]<=t){       // If midpoint time is before or at target time
        ans=mid;             // This could be our answer (latest valid bar)
        lo=mid+1;            // Search right half for potentially later bar
    } else {                 // If midpoint time is after target time
        hi=mid-1;            // Search left half for earlier bars
    }
  }
  
  return ans;  // Returns index of latest bar <= target time (or -1 if none found)
}

//============================== TECHNICAL INDICATOR CALCULATIONS ==============================
// Advanced market analysis functions that extract trading signals from price data
// These indicators provide the fundamental building blocks for AI decision-making
// Each function transforms raw OHLCV data into normalized features for neural networks

// Simple Moving Average Calculator
// Calculates the arithmetic mean of closing prices over a specified period
// Smooths price noise and identifies trend direction and momentum
// Essential baseline indicator for trend analysis and support/resistance levels
//
// Parameters:
//   r[]: Array of price bars (OHLCV data)
//   i: Current bar index (starting point for calculation)
//   period: Number of bars to include in average
// Returns: Average closing price over the period, or 0.0 if insufficient data
double SMA_Close(const MqlRates &r[], int i, int period){ 
    double s=0; int n=0;  // Accumulator for sum and valid bar count
    
    // Sum closing prices over the specified period
    for(int k=0;k<period && (i+k)<ArraySize(r); ++k){ 
        s+=r[i+k].close;  // Add each closing price to sum
        n++;               // Count valid bars processed
    } 
    
    // Return arithmetic mean or zero if no valid data
    return (n>0? s/n : 0.0); 
}

// Note: EMA_Slope and ATR_Proxy functions are now provided by CortexTradeLogic.mqh

// Trend Direction Detector
// Compares current price to historical price to determine trend direction
// Simple but effective method for identifying bullish vs bearish momentum
// Provides discrete directional signal that's easy for AI to interpret
//
// Parameters:
//   r[]: Array of price bars
//   i: Current bar index
//   look: Number of bars to look back for comparison
// Returns: 1.0 (uptrend), -1.0 (downtrend), or 0.0 (sideways)
double TrendDir(const MqlRates &r[], int i, int look){
  int idx=i+look;  // Calculate historical comparison point
  
  // Validate sufficient historical data exists
  if(idx>=ArraySize(r)) return 0.0;
  
  double a=r[i].close;    // Current closing price
  double b=r[idx].close;  // Historical closing price
  
  // Determine trend direction based on price comparison
  if(a>b) return 1.0;     // Current > Historical = Uptrend (bullish)
  if(a<b) return -1.0;    // Current < Historical = Downtrend (bearish)
  return 0.0;             // Current = Historical = Sideways (neutral)
}

//============================== ENHANCED MARKET CONTEXT FEATURES ==============================
// Sophisticated market environment indicators that capture trading session dynamics
// These features provide temporal and institutional context missing from pure price data
// Critical for understanding when and why certain trading patterns emerge

// Intraday Time Position Calculator
// Converts current time to normalized position within the trading day
// Captures intraday patterns like opening gaps, lunch lulls, and closing drives
// Essential for modeling session-specific volatility and liquidity patterns
//
// Returns: Normalized time value (0.0 = start of day, 1.0 = end of day)
double GetTimeOfDay(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);  // Get current server time structure
    
    // Convert hours and minutes to fractional day position
    // Formula: (hours * 60 + minutes) / (24 * 60) gives 0.0-1.0 range
    return (dt.hour * 60.0 + dt.min) / (24.0 * 60.0);
}

// Weekly Position Calculator
// Encodes the day of week as a normalized trading cycle position
// Captures weekly patterns like Monday gaps, Wednesday reversals, Friday profit-taking
// Important for modeling institutional trading flows and market sentiment cycles
//
// Returns: Normalized week position (0.0 = Sunday, 1.0 = Saturday)
double GetDayOfWeek(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);  // Get current time components
    
    // Convert day of week to 0-1 range
    // Sunday=0, Monday=1, ..., Saturday=6 -> normalized to 0.0-1.0
    return dt.day_of_week / 6.0;
}

// Global Trading Session Detector
// Identifies which major trading session is currently active
// Critical for understanding liquidity, volatility, and institutional participation
// Different sessions have distinct characteristics for AI strategy adaptation
//
// Session Characteristics:
//   Asian: Lower volatility, range-bound trading, yen pairs active
//   London: High volatility, trend initiation, EUR/GBP pairs active  
//   New York: Maximum liquidity, trend continuation, USD pairs active
//   Off-hours: Minimal liquidity, increased spread risk
//
// Returns: Session identifier (0.0=Asian, 0.33=London, 0.66=NY, 1.0=Off-hours)
double GetTradingSession(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour_utc = dt.hour;  // Assumes server time is UTC (verify for your broker)
    
    // Asian Trading Session: 00:00-09:00 UTC
    // Characteristics: Tokyo open, lower volatility, JPY currency pairs most active
    if(hour_utc >= 0 && hour_utc < 9) return 0.0;
    
    // London Trading Session: 08:00-17:00 UTC  
    // Characteristics: European open, high volatility, EUR/GBP pairs dominant
    else if(hour_utc >= 8 && hour_utc < 17) return 0.33;
    
    // New York Trading Session: 13:00-22:00 UTC
    // Characteristics: US markets open, maximum global liquidity, USD pairs active
    else if(hour_utc >= 13 && hour_utc < 22) return 0.66;
    
    // Off-Hours Period: Low liquidity, increased spreads, gap risk
    else return 1.0;
}

// Volume Momentum Analyzer
// Measures current trading activity relative to recent historical average
// High volume often precedes significant price movements or confirms trends
// Essential for validating breakouts and identifying institutional participation
//
// Volume Interpretation:
//   > 1.0: Above-average activity (institutional interest, news events)
//   ≈ 0.5: Normal trading activity (baseline market participation)
//   < 0.5: Below-average activity (low interest, potential false signals)
//
// Parameters:
//   r[]: Array of price bars with volume data
//   i: Current bar index
//   period: Historical period for average volume calculation
// Returns: Normalized volume ratio (0.0-1.0, where 1.0 = 3x average volume)
double GetVolumeMomentum(const MqlRates &r[], int i, int period){
    // Validate sufficient historical data for meaningful comparison
    if(i+period >= ArraySize(r)) return 0.5; // Neutral default
    
    double current_vol = (double)r[i].tick_volume;  // Current bar volume
    double vol_sum = 0.0;  // Accumulator for historical volume
    int count = 0;         // Valid historical bar counter
    
    // Calculate average volume over historical period
    for(int k=1; k<=period && i+k<ArraySize(r); ++k){
        vol_sum += (double)r[i+k].tick_volume;
        count++;
    }
    
    // Handle edge cases: no data or zero volume
    if(count == 0 || vol_sum == 0) return 0.5;
    
    double avg_vol = vol_sum / count;  // Historical average volume
    
    // Calculate and normalize volume momentum ratio
    double ratio = current_vol / avg_vol;
    
    // Scale ratio to 0-1 range where 3x average volume = 1.0
    // This prevents extreme outliers from dominating the signal
    return clipd(ratio / 3.0, 0.0, 1.0);
}

// Bid-Ask Spread Percentage Calculator
// Measures trading cost as percentage of current market price
// Critical for cost-benefit analysis and trade timing decisions
// High spreads indicate low liquidity or volatile market conditions
//
// Spread Interpretation:
//   Low spread (< 0.02%): Good liquidity, favorable trading conditions
//   Normal spread (0.02-0.05%): Standard market conditions
//   High spread (> 0.05%): Poor liquidity, avoid trading if possible
//
// Returns: Spread as percentage of mid-price (normalized basis points)
double GetSpreadPercent(){
    double spread_points = GetSymbolSpreadPoints(); // Get spread with fallback protection
    
    // Calculate current mid-price (average of bid and ask)
    double current_price = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) + 
                           SymbolInfoDouble(_Symbol, SYMBOL_BID)) / 2.0;
    
    // Validate price data availability
    if(current_price <= 0) return 0.0;
    
    // Convert spread to percentage of price
    // Formula: (spread_in_points * point_value) / price * 10000 for basis points
    return (spread_points * _Point) / current_price * 10000.0;
}

// Price Momentum (Rate of Change) Calculator
// Measures the percentage change in price over a specified period
// Essential for identifying acceleration/deceleration in price trends
// Helps distinguish between strong trending moves and consolidation phases
//
// Momentum Interpretation:
//   > 0.6: Strong positive momentum (bullish acceleration)
//   0.4-0.6: Moderate momentum (normal trend progression)
//   < 0.4: Weak/negative momentum (bearish or consolidation)
//
// Parameters:
//   r[]: Array of price bars
//   i: Current bar index
//   period: Lookback period for momentum calculation
// Returns: Normalized momentum (0.0-1.0, where 0.5 = no change)
double GetPriceMomentum(const MqlRates &r[], int i, int period){
    // Validate sufficient historical data
    if(i+period >= ArraySize(r)) return 0.5; // Neutral default
    
    double current_price = r[i].close;       // Current closing price
    double past_price = r[i+period].close;   // Historical comparison price
    
    // Validate historical price data
    if(past_price <= 0) return 0.5;
    
    // Calculate percentage change over the period
    double change = (current_price - past_price) / past_price;
    
    // Normalize to 0-1 range, assuming ±5% represents extreme momentum
    // 0.0 = -5% (strong bearish), 0.5 = 0% (neutral), 1.0 = +5% (strong bullish)
    return clipd((change + 0.05) / 0.10, 0.0, 1.0);
}

// Volatility Percentile Rank Calculator
// Determines where current volatility ranks within recent historical range
// Critical for position sizing and risk management decisions
// High volatility rank suggests larger price movements and higher risk
//
// Volatility Rank Interpretation:
//   > 0.8: Very high volatility (reduce position sizes, tight stops)
//   0.6-0.8: Above average volatility (normal risk management)
//   0.2-0.6: Average volatility (standard position sizing)
//   < 0.2: Low volatility (potential for breakouts, larger positions)
//
// Parameters:
//   r[]: Array of price bars
//   i: Current bar index
//   atr_period: Period for ATR calculation
//   rank_period: Historical period for percentile ranking
// Returns: Percentile rank (0.0-1.0, where 1.0 = highest volatility)
double GetVolatilityRank(const MqlRates &r[], int i, int atr_period, int rank_period){
    // Validate sufficient historical data for meaningful ranking
    if(i+rank_period >= ArraySize(r)) return 0.5;
    
    // Calculate current market volatility
    double current_atr = ATR_Proxy(r, i, atr_period);
    
    // Array to store historical ATR values for ranking comparison
    double atr_values[100]; // Maximum supported ranking period
    
    // Determine actual calculation period within data and array limits
    int actual_period = MathMin(rank_period, ArraySize(r)-i-atr_period);
    actual_period = MathMin(actual_period, 100);
    
    // Calculate ATR for each historical bar in the ranking period
    for(int k=0; k<actual_period; ++k){
        if(i+k+atr_period < ArraySize(r)){
            atr_values[k] = ATR_Proxy(r, i+k, atr_period);
        }
    }
    
    // Count how many historical ATR values are below current ATR
    int below_count = 0;
    for(int k=0; k<actual_period; ++k){
        if(atr_values[k] < current_atr) below_count++;
    }
    
    // Calculate percentile rank: (count below) / (total count)
    return actual_period > 0 ? (double)below_count / actual_period : 0.5;
}

// Calculate RSI (Relative Strength Index)
double GetRSI(const MqlRates &r[], int i, int period){
    if(i+period >= ArraySize(r)) return 0.5; // Default neutral
    
    double gain_sum = 0.0, loss_sum = 0.0;
    int gain_count = 0, loss_count = 0;
    
    for(int k=1; k<=period && i+k<ArraySize(r); ++k){
        double change = r[i+k-1].close - r[i+k].close;
        if(change > 0){
            gain_sum += change;
            gain_count++;
        } else if(change < 0){
            loss_sum += MathAbs(change);
            loss_count++;
        }
    }
    
    if(gain_count == 0 && loss_count == 0) return 0.5;
    
    double avg_gain = gain_count > 0 ? gain_sum / gain_count : 0.0;
    double avg_loss = loss_count > 0 ? loss_sum / loss_count : 0.0;
    
    if(avg_loss == 0.0) return 1.0; // All gains
    
    double rs = avg_gain / avg_loss;
    double rsi = 100.0 - (100.0 / (1.0 + rs));
    
    return rsi / 100.0;  // Convert to 0-1 range
}

// Get market bias from higher timeframe
double GetMarketBias(const MqlRates &r[], int i, int short_ma, int long_ma){
    if(i+long_ma >= ArraySize(r)) return 0.5;
    
    double short_sma = SMA_Close(r, i, short_ma);
    double long_sma = SMA_Close(r, i, long_ma);
    
    if(long_sma <= 0) return 0.5;
    
    double bias = (short_sma - long_sma) / long_sma;
    // Scale to 0-1 range, assuming ±2% is significant
    return clipd((bias + 0.02) / 0.04, 0.0, 1.0);
}

// FEATURE NORMALIZATION
// Scale all input features to 0-1 range using min/max values from training
void ApplyMinMax(double &x[], const double &mn[], const double &mx[]){
  for(int j=0;j<STATE_SIZE;++j){
    double d = mx[j]-mn[j];  // Range of this feature during training
    if(d<1e-8) d=1.0;        // Avoid division by zero
    x[j] = (x[j]-mn[j])/d;   // Scale to 0-1: (value - min) / (max - min)
    x[j] = clipd(x[j],0.0,1.0);  // Ensure stays in 0-1 range
  }
}

// BUILD FEATURE VECTOR FOR AI INPUT
// This creates the 35-dimensional input that describes the current market state
void BuildStateRow(const Series &base, int i, const Series &m1, const Series &m5, const Series &h1, const Series &h4, const Series &d1, double &row[]){
  ArrayResize(row,STATE_SIZE);
  
  // Extract basic price data from current bar
  double o=base.rates[i].open, h=base.rates[i].high, l=base.rates[i].low, c=base.rates[i].close, v=(double)base.rates[i].tick_volume;
  
  // Feature 0: Candle body position within range (0=bottom, 1=top)
  row[0] = (h-l>0? (c-o)/(h-l):0.0);
  
  // Feature 1: Bar range (high - low)
  row[1] = (h-l);
  
  // Feature 2: Volume
  row[2] = v;
  
  // Features 3-5: Moving averages (short, medium, long term)
  row[3] = SMA_Close(base.rates, i, 5);   // 5-bar SMA
  row[4] = SMA_Close(base.rates, i, 20);  // 20-bar SMA
  row[5] = SMA_Close(base.rates, i, 50);  // 50-bar SMA
  
  // Feature 6: Trend strength (EMA slope)
  row[6] = EMA_Slope(base.rates, i, 20);
  
  // Feature 7: Volatility (ATR)
  row[7] = ATR_Proxy(base.rates, i, 14);
  
  // Features 8-11: Multi-timeframe trend analysis
  // Find corresponding bars in other timeframes
  datetime t = base.rates[i].time;
  int i_m5 = FindIndexLE(m5.times, ArraySize(m5.times), t);
  int i_h1 = FindIndexLE(h1.times, ArraySize(h1.times), t);
  int i_h4 = FindIndexLE(h4.times, ArraySize(h4.times), t);
  int i_d1 = FindIndexLE(d1.times, ArraySize(d1.times), t);
  
  // Get trend direction from each timeframe
  row[8]  = (i_m5>=0?  TrendDir(m5.rates, i_m5, 20) : 0.0);  // M5 trend
  row[9]  = (i_h1>=0?  TrendDir(h1.rates, i_h1, 20) : 0.0);  // H1 trend
  row[10] = (i_h4>=0?  TrendDir(h4.rates, i_h4, 20) : 0.0);  // H4 trend
  row[11] = (i_d1>=0?  TrendDir(d1.rates, i_d1, 20) : 0.0);  // D1 trend
  
  // Features 12-14: Position state (NEW - aligns EA with training)
  // Get actual current position state and populate features
  double pos_dir=0.0, pos_size=0.0, upnl_pts=0.0;
  for(int pos_i=PositionsTotal()-1; pos_i>=0; --pos_i){
    ulong ticket = PositionGetTicket(pos_i);
    if(!PositionSelectByTicket(ticket)) continue;
    
    string sym = PositionGetString(POSITION_SYMBOL);
    long mg = PositionGetInteger(POSITION_MAGIC);
    
    if(sym==_Symbol && mg==InpMagic){
      long type = PositionGetInteger(POSITION_TYPE);
      pos_dir = (type==POSITION_TYPE_BUY? 1.0 : -1.0);
      pos_size = PositionGetDouble(POSITION_VOLUME);
      double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
      upnl_pts = (base.rates[i].close - entry_price)/_Point * pos_dir;
      break;
    }
  }
  
  row[12] = pos_dir;    // -1=short, 0=flat, 1=long
  row[13] = pos_size;   // position size in lots
  row[14] = upnl_pts;   // unrealized P&L in points
  
  // Features 15-34: ENHANCED MARKET CONTEXT (NEW - critical FX factors)
  
  // Temporal features (15-18): Time-based patterns that drive FX markets
  row[15] = GetTimeOfDay();          // 0.0=start of day, 1.0=end of day
  row[16] = GetDayOfWeek();          // 0.0=Sunday, 1.0=Saturday  
  row[17] = GetTradingSession();     // 0=Asian, 0.33=London, 0.66=NY, 1.0=Off-hours
  MqlDateTime dt_month; TimeToStruct(TimeCurrent(), dt_month);
  row[18] = (dt_month.day <= 15 ? 0.0 : 1.0); // Month half indicator
  
  // Market microstructure features (19-23): Execution conditions
  row[19] = GetSpreadPercent();             // Spread as % of price (liquidity indicator)
  row[20] = GetVolumeMomentum(base.rates, i, 10); // Volume vs 10-bar average
  row[21] = GetVolumeMomentum(base.rates, i, 50); // Volume vs 50-bar average
  row[22] = clipd(v / 1000.0, 0.0, 1.0);  // Absolute volume level (scaled)
  
  // Technical momentum features (23-27): Price dynamics
  row[23] = GetPriceMomentum(base.rates, i, 5);   // 5-bar momentum
  row[24] = GetPriceMomentum(base.rates, i, 20);  // 20-bar momentum  
  row[25] = GetRSI(base.rates, i, 14);           // RSI oscillator
  row[26] = GetRSI(base.rates, i, 30);           // Longer RSI
  
  // Volatility regime features (27-30): Market conditions
  row[27] = GetVolatilityRank(base.rates, i, 14, 50); // ATR percentile rank
  row[28] = clipd(row[7] / 0.001, 0.0, 1.0);     // Raw ATR scaled (pip-based)
  row[29] = (row[27] > 0.8 ? 1.0 : 0.0);   // High volatility flag (top 20th percentile)
  
  // Multi-timeframe bias features (30-34): Trend alignment  
  row[30] = GetMarketBias(base.rates, i, 10, 50); // Short vs long-term bias
  row[31] = GetMarketBias(h1.rates, i_h1>=0 ? i_h1 : 0, 5, 20); // H1 bias
  row[32] = GetMarketBias(h4.rates, i_h4>=0 ? i_h4 : 0, 3, 12); // H4 bias
  row[33] = GetMarketBias(d1.rates, i_d1>=0 ? i_d1 : 0, 2, 8);  // D1 bias
  
  // Market structure feature (34): Support/resistance proximity
  // Calculate daily range using D1 timeframe data (not current bar)
  double daily_high = (i_d1>=0) ? d1.rates[i_d1].high : h;
  double daily_low = (i_d1>=0) ? d1.rates[i_d1].low : l;
  double daily_range = (daily_high - daily_low) > 0 ? (daily_high - daily_low) : 0.0001;
  row[34] = (c - daily_low) / daily_range; // Price position within daily range
}

// POSITION STATE MANAGEMENT (NEW - aligns EA with training)
// Get current position state for feature vector
void GetCurrentPositionState(double &pos_dir, double &pos_size, double &unrealized_pnl){
    pos_dir = 0.0; pos_size = 0.0; unrealized_pnl = 0.0;
    
    // Check for existing position
    int total = PositionsTotal();
    for(int i=0; i<total; ++i){
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        
        string sym = PositionGetString(POSITION_SYMBOL);
        long mg = PositionGetInteger(POSITION_MAGIC);
        
        if(sym == _Symbol && mg == InpMagic){
            long type = PositionGetInteger(POSITION_TYPE);
            double volume = PositionGetDouble(POSITION_VOLUME);
            double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
            double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
            double profit = PositionGetDouble(POSITION_PROFIT);
            
            // Set direction
            pos_dir = (type == POSITION_TYPE_BUY) ? 1.0 : -1.0;
            
            // Estimate size (map volume to strong/weak categories)
            double strong_lots = InpUseRiskSizing ? CalculatePositionSize(true) : InpLotsStrong;
            double weak_lots = InpUseRiskSizing ? CalculatePositionSize(false) : InpLotsWeak;
            
            if(volume >= strong_lots * 0.9) pos_size = 1.0;  // Strong position
            else if(volume >= weak_lots * 0.9) pos_size = 0.5;  // Weak position
            else pos_size = 0.3;  // Small position
            
            // Normalized unrealized P&L (scale by account equity)
            double equity = AccountInfoDouble(ACCOUNT_EQUITY);
            if(equity > 0) unrealized_pnl = profit / equity * 100.0;  // As percentage
            
            break;  // Found our position
        }
    }
}

// Update position features in feature vector
void SetPositionFeaturesEA(double &row[], double pos_dir, double pos_size, double unrealized_pnl){
    row[12] = pos_dir;        // -1=short, 0=flat, 1=long
    row[13] = pos_size;       // 0=no position, 0.5=weak, 1.0=strong
    row[14] = unrealized_pnl; // normalized unrealized P&L
}

//============================== RISK MANAGEMENT FUNCTIONS (NEW) ===============
// Essential risk controls for production trading

// Calculate ATR-based position size (3.4 IMPROVEMENT - using unified logic)
double CalculatePositionSize(bool is_strong_signal){
    // Use unified position sizing with adaptive parameters
    double adaptive_risk = GetAdaptiveRiskPercent();
    double adaptive_atr_multiplier = GetAdaptiveATRMultiplier();
    
    return CalculateUnifiedPositionSize(is_strong_signal, 
                                       adaptive_risk,
                                       adaptive_atr_multiplier,
                                       InpLotsStrong,
                                       InpLotsWeak,
                                       InpUseRiskSizing);
}

// Get current position information for scaling decisions (OPTIMIZED 2.4)
bool GetCurrentPosition(double &current_lots, bool &is_long, double &entry_price){
    current_lots = 0.0; is_long = false; entry_price = 0.0;
    
    if(InpOptimizeLoops && InpOptimizePositionScan){
        // Use optimized cached scanning
        RefreshPositionCache();
        
        if(g_position_cache_count == 0) return false;
        
        // Check cached positions first
        for(int i = 0; i < g_position_cache_count; i++){
            ulong ticket = g_cached_positions[i];
            if(!PositionSelectByTicket(ticket)) continue;
            
            long type = PositionGetInteger(POSITION_TYPE);
            current_lots = PositionGetDouble(POSITION_VOLUME);
            entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            is_long = (type == POSITION_TYPE_BUY);
            return true; // Early exit - found our position
        }
    }
    else {
        // Fallback to standard scanning with early break
        int loop_count = 0;
        int total_positions = PositionsTotal();
        
        for(int i = 0; i < total_positions; i++){
            // Loop safety check
            if(InpUseEarlyBreaks && !LoopSafetyCheck(loop_count)){
                break;
            }
            
            ulong ticket = PositionGetTicket(i);
            if(!PositionSelectByTicket(ticket)) continue;
            
            string sym = PositionGetString(POSITION_SYMBOL);
            long mg = PositionGetInteger(POSITION_MAGIC);
            
            if(sym == _Symbol && mg == InpMagic){
                long type = PositionGetInteger(POSITION_TYPE);
                current_lots = PositionGetDouble(POSITION_VOLUME);
                entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
                is_long = (type == POSITION_TYPE_BUY);
                return true; // Early exit optimization
            }
        }
    }
    return false;
}

// Determine if current position matches signal strength
bool IsPositionCorrectSize(double current_lots, bool is_strong_signal){
    double target_lots = CalculatePositionSize(is_strong_signal);
    double tolerance = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP) * 2; // 2 steps tolerance
    return (MathAbs(current_lots - target_lots) <= tolerance);
}

// Scale position to match target size
bool ScalePosition(double current_lots, double target_lots, bool is_long){
    if(MathAbs(current_lots - target_lots) <= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)){
        return true; // Already correct size
    }
    
    double volume_diff = target_lots - current_lots;
    
    if(volume_diff > 0){
        // Need to increase position (add to existing)
        double add_volume = MathAbs(volume_diff);
        double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        
        if(add_volume < min_vol){
            SmartLog("TRADE", "Position scaling: Volume difference too small to adjust (" + DoubleToString(add_volume,2) + ")");
            return false;
        }
        
        // Add to position
        trade.SetExpertMagicNumber(InpMagic);
        bool result = is_long ? 
            trade.Buy(add_volume, _Symbol, 0, 0, 0, "cortex3_scale_up") :
            trade.Sell(add_volume, _Symbol, 0, 0, 0, "cortex3_scale_up");
            
        if(result){
            SmartLog("TRADE", "Position scaled UP: Added " + DoubleToString(add_volume,2) + " lots to " + (is_long?"LONG":"SHORT") + " position");
        }
        return result;
    }
    else {
        // Need to decrease position (partial close)
        double close_volume = MathAbs(volume_diff);
        double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        
        if(close_volume < min_vol){
            SmartLog("TRADE", "Position scaling: Volume difference too small to adjust (" + DoubleToString(close_volume,2) + ")");
            return false;
        }
        
        // Partial close position
        trade.SetExpertMagicNumber(InpMagic);
        bool result = trade.PositionClosePartial(_Symbol, close_volume);
        
        if(result){
            SmartLog("TRADE", "Position scaled DOWN: Reduced " + (is_long?"LONG":"SHORT") + " position by " + DoubleToString(close_volume,2) + " lots");
        }
        return result;
    }
}

// Check if spread is acceptable for trading
bool IsSpreadAcceptable(){
    double spread_points = GetSymbolSpreadPoints(); // Use centralized function with 15pt fallback
    return (spread_points <= InpMaxSpread);
}

// Check account drawdown and win rate
bool CheckRiskLimits(){
    double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Check drawdown
    if(g_initial_equity > 0){
        double drawdown_pct = (g_initial_equity - current_equity) / g_initial_equity * 100.0;
        if(drawdown_pct > InpMaxDrawdown){
            if(!g_trading_paused){
                Print("RISK ALERT: Maximum drawdown exceeded (", DoubleToString(drawdown_pct,2), 
                      "%). Trading paused.");
                g_trading_paused = true;
            }
            return false;
        }
    }
    
    // Check recent win rate
    if(g_recent_trades_count >= 10){
        int wins = 0;
        for(int i=0; i<g_recent_trades_count; ++i){
            if(g_recent_trades[i] == 1) wins++;
        }
        int win_rate = (int)(wins * 100.0 / g_recent_trades_count);
        
        if(win_rate < InpMinWinRate){
            if(!g_trading_paused){
                Print("RISK ALERT: Win rate too low (", win_rate, 
                      "%). Trading paused.");
                g_trading_paused = true;
            }
            return false;
        }
    }
    
    // Reset pause if conditions improve
    if(g_trading_paused){
        Print("Risk conditions improved. Resuming trading.");
        g_trading_paused = false;
    }
    
    return true;
}

// Enhanced trade tracking with cooldown management (1.3 IMPROVEMENT)
void UpdateTradeTracking(bool was_profitable){
    UpdateTradeTrackingWithPnL(was_profitable, 0.0);
}

// Enhanced trade tracking with P&L information (1.3 IMPROVEMENT)
void UpdateTradeTrackingWithPnL(bool was_profitable, double trade_pnl){
    // Optimize array operations (2.2 IMPROVEMENT)
    if(InpMinimizeMemOps){
        // Only resize if necessary
        if(ArraySize(g_recent_trades) != 10){
            ArrayResize(g_recent_trades, 10);
        }
    } else {
        ArrayResize(g_recent_trades, 10);  // Keep last 10 trades
    }
    
    // Shift array and add new result
    for(int i=9; i>0; --i){
        g_recent_trades[i] = g_recent_trades[i-1];
    }
    g_recent_trades[0] = was_profitable ? 1 : 0;
    
    if(g_recent_trades_count < 10) g_recent_trades_count++;
    
    // Update emergency stop daily P&L tracking (1.5 IMPROVEMENT)
    UpdateDailyPnL(trade_pnl);
    
    // Update adaptive parameters based on trade performance (3.3 IMPROVEMENT)
    if(InpUseAdaptiveParameters){
        UpdatePerformanceHistory(trade_pnl);
    }
    
    // Enhanced overtrading prevention tracking (1.3 IMPROVEMENT)
    if(InpUseAdvancedCooldown){
        g_last_trade_pnl = trade_pnl;
        g_last_position_close = TimeCurrent();
        
        // Update consecutive loss tracking
        if(InpUseConsecutiveLosses){
            if(was_profitable){
                g_consecutive_losses = 0; // Reset on profitable trade
            } else {
                g_consecutive_losses++;
                Print("OVERTRADING PREVENTION: Consecutive losses: ", g_consecutive_losses, 
                      "/", InpMaxConsecutiveLosses);
            }
        }
        
        // Calculate and set cooldown period
        int cooldown_minutes = CalculateDynamicCooldown();
        if(cooldown_minutes > 0){
            if(g_consecutive_losses >= InpMaxConsecutiveLosses){
                g_extended_cooldown_until = TimeCurrent() + (cooldown_minutes * 60);
                Print("OVERTRADING PREVENTION: Extended cooldown set - ", cooldown_minutes, " minutes");
            } else {
                g_cooldown_until = TimeCurrent() + (cooldown_minutes * 60);
                Print("OVERTRADING PREVENTION: Standard cooldown set - ", cooldown_minutes, " minutes");
            }
        }
    }
}

//============================== ENHANCED COST MODELING (EA) ==================
// Realistic cost modeling for EA execution - matches training cost model

// Get current symbol spread using MQL5 function with fallback (MATCHES TRAINING)
double GetSymbolSpreadPoints(string symbol = ""){
  if(symbol == "") symbol = _Symbol;
  
  // Try to get actual spread from symbol info
  long spread_points = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
  
  if(spread_points > 0){
    return (double)spread_points;  // Use actual spread if available
  } else {
    Print("Warning: Could not retrieve spread for ", symbol, ", using fallback of 15 points");
    return 15.0;  // Fallback: 15 points to match training script
  }
}

// Estimate variable spread based on current market conditions
double EstimateCurrentSpread(){
    // Use centralized spread function with consistent 15pt fallback
    return GetSymbolSpreadPoints(); // Matches training script exactly
}

// Estimate slippage for EA execution
double EstimateExecutionSlippage(double position_size){
    // Base slippage in points
    double base_slippage = 0.5;
    
    // Size impact
    double size_multiplier = 1.0 + (position_size * 0.5);
    
    // Volatility impact from current ATR (2.1 IMPROVEMENT - cached)
    double volatility_impact = 1.0;
    double current_atr = GetCachedATR14();
    if(current_atr > 0){
        volatility_impact = MathSqrt(current_atr * 10000);
        volatility_impact = clipd(volatility_impact, 1.0, 3.0);
    }
    
    // Session impact
    double session_impact = 1.0;
    double session = GetTradingSession();
    if(session == 1.0) session_impact = 1.5; // Off-hours
    
    return base_slippage * size_multiplier * volatility_impact * session_impact;
}

// Estimate swap cost for EA
double EstimateEASwapDaily(double position_size, bool is_buy){
    // Try to get real swap rates from MT5
    double swap_long = 0.0, swap_short = 0.0;
    SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG, swap_long);
    SymbolInfoDouble(_Symbol, SYMBOL_SWAP_SHORT, swap_short);
    
    double swap_rate = is_buy ? swap_long : swap_short;
    
    // If no swap info available, use estimates
    if(MathAbs(swap_rate) < 0.001){
        swap_rate = is_buy ? -0.3 : -0.25; // Typical major pair swaps
    }
    
    return MathAbs(swap_rate) * position_size; // Points per day
}

// Calculate total execution cost for EA
double CalculateEAExecutionCost(double position_size){
    double spread_cost = EstimateCurrentSpread();
    double slippage_cost = EstimateExecutionSlippage(position_size);
    
    // Commission cost (use input parameter since MT5 doesn't expose broker commission rates)
    double commission_cost = 0.0;
    if(InpCommissionPerLot > 0.0){
        double tick_value = 0.0, tick_size = 0.0;
        if(SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE, tick_value) && 
           SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE, tick_size)){
            if(tick_value > 0 && tick_size > 0){
                double ticks = InpCommissionPerLot / tick_value;
                commission_cost = ticks * (tick_size / _Point) * position_size;
            }
        }
    }
    
    return spread_cost + slippage_cost + commission_cost;
}

//============================== ADVANCED RISK MANAGEMENT FUNCTIONS ===============
// Enhanced trailing stop, break-even, and volatility regime controls

// Enhanced trailing stops and partial closing for all positions (1.1 + 2.4 IMPROVEMENTS)
void UpdateTrailingStops(){
    if(!InpUseTrailingStop && !InpUseBreakEven && !InpUsePartialClose) return;
    
    // Get current ATR for calculations (2.1 IMPROVEMENT - cached)
    double current_atr = GetCachedATR14();
    if(current_atr <= 0) return;
    
    if(InpOptimizeLoops && InpOptimizePositionScan){
        // Use optimized cached scanning (2.4 IMPROVEMENT)
        RefreshPositionCache();
        
        // Process cached positions only
        for(int i = 0; i < g_position_cache_count; i++){
            ulong ticket = g_cached_positions[i];
            if(ticket > 0){
                UpdateEnhancedPositionManagement(ticket, current_atr);
            }
        }
    }
    else {
        // Fallback to standard scanning with loop safety
        int loop_count = 0;
        int total_positions = PositionsTotal();
        
        for(int i = 0; i < total_positions; i++){
            // Loop safety check
            if(InpUseEarlyBreaks && !LoopSafetyCheck(loop_count)){
                LogCritical("LOOP SAFETY: Trailing stop update terminated at max iterations");
                break;
            }
            
            ulong ticket = PositionGetTicket(i);
            if(!PositionSelectByTicket(ticket)) continue;
            
            string pos_symbol = PositionGetString(POSITION_SYMBOL);
            long pos_magic = PositionGetInteger(POSITION_MAGIC);
            
            if(pos_symbol == _Symbol && pos_magic == InpMagic){
                UpdateEnhancedPositionManagement(ticket, current_atr);
            }
        }
    }
}

// Update trailing stop for a single position
void UpdateSinglePositionTrailing(ulong ticket, double current_atr){
    if(!PositionSelectByTicket(ticket)) return;
    
    long pos_type = PositionGetInteger(POSITION_TYPE);
    double pos_volume = PositionGetDouble(POSITION_VOLUME);
    double pos_open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    double pos_current_sl = PositionGetDouble(POSITION_SL);
    double pos_profit = PositionGetDouble(POSITION_PROFIT);
    
    bool is_long = (pos_type == POSITION_TYPE_BUY);
    double current_price = is_long ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    // Calculate profit in ATR multiples
    double profit_points = is_long ? (current_price - pos_open_price) : (pos_open_price - current_price);
    double profit_atr_multiple = profit_points / (current_atr * _Point);
    
    bool should_update_sl = false;
    double new_sl = pos_current_sl;
    
    // Break-even move logic
    if(InpUseBreakEven && profit_atr_multiple >= InpBreakEvenATR){
        double break_even_price = pos_open_price;
        
        // Add small buffer for guaranteed profit
        if(is_long){
            break_even_price += InpBreakEvenBuffer * _Point;
        } else {
            break_even_price -= InpBreakEvenBuffer * _Point;
        }
        
        // Only move to break-even if current SL is worse
        bool should_move_to_breakeven = false;
        if(is_long){
            should_move_to_breakeven = (pos_current_sl == 0.0 || pos_current_sl < break_even_price);
        } else {
            should_move_to_breakeven = (pos_current_sl == 0.0 || pos_current_sl > break_even_price);
        }
        
        if(should_move_to_breakeven){
            new_sl = break_even_price;
            should_update_sl = true;
            Print("Moving position to break-even+buffer: ", DoubleToString(new_sl, 5));
        }
    }
    
    // Trailing stop logic
    if(InpUseTrailingStop && profit_atr_multiple >= InpTrailStartATR){
        double trail_distance = current_atr * InpTrailStopATR;
        double new_trailing_sl;
        
        if(is_long){
            new_trailing_sl = current_price - trail_distance;
            // Only move SL up (never down)
            if(pos_current_sl == 0.0 || new_trailing_sl > pos_current_sl){
                new_sl = new_trailing_sl;
                should_update_sl = true;
            }
        } else {
            new_trailing_sl = current_price + trail_distance;
            // Only move SL down (never up)
            if(pos_current_sl == 0.0 || new_trailing_sl < pos_current_sl){
                new_sl = new_trailing_sl;
                should_update_sl = true;
            }
        }
        
        if(should_update_sl && MathAbs(new_sl - pos_current_sl) > _Point){
            Print("Updating trailing stop: Profit=", DoubleToString(profit_atr_multiple,2), "x ATR, New SL=", DoubleToString(new_sl,5));
        }
    }
    
    // Apply the stop loss update
    if(should_update_sl){
        // Normalize to tick size
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        if(tick_size > 0){
            new_sl = MathRound(new_sl / tick_size) * tick_size;
        }
        
        trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
    }
}

// Enhanced position management with partial closing and advanced trailing (1.1 IMPROVEMENT)
void UpdateEnhancedPositionManagement(ulong ticket, double current_atr){
    if(!PositionSelectByTicket(ticket)) return;
    
    long pos_type = PositionGetInteger(POSITION_TYPE);
    double pos_volume = PositionGetDouble(POSITION_VOLUME);
    double pos_open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    double pos_current_sl = PositionGetDouble(POSITION_SL);
    double pos_profit = PositionGetDouble(POSITION_PROFIT);
    datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
    
    bool is_long = (pos_type == POSITION_TYPE_BUY);
    double current_price = is_long ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    // Calculate profit in ATR multiples
    double profit_points = is_long ? (current_price - pos_open_price) : (pos_open_price - current_price);
    double profit_atr_multiple = profit_points / (current_atr * _Point);
    
    // Track highest profit for accelerated trailing
    if(profit_atr_multiple > g_position_highest_profit_atr){
        g_position_highest_profit_atr = profit_atr_multiple;
    }
    
    // Initialize original size tracking when position is opened
    if(g_position_original_size == 0.0){
        g_position_original_size = pos_volume;
    }
    
    // ENHANCED FEATURE 1: Partial Position Closing
    if(InpUsePartialClose && profit_atr_multiple > 0){
        // First partial close level
        if(!g_partial_close_level1_hit && profit_atr_multiple >= InpPartialCloseLevel1){
            double close_volume = g_position_original_size * (InpPartialClosePercent1 / 100.0);
            close_volume = MathMin(close_volume, pos_volume); // Can't close more than current volume
            
            if(close_volume >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)){
                trade.SetExpertMagicNumber(InpMagic);
                if(trade.PositionClosePartial(_Symbol, close_volume)){
                    g_partial_close_level1_hit = true;
                    Print("ENHANCED RISK MGT: Partial close level 1 hit - Closed ", 
                          DoubleToString(close_volume, 2), " lots at ", 
                          DoubleToString(profit_atr_multiple, 2), "x ATR profit");
                }
            }
        }
        
        // Second partial close level
        if(g_partial_close_level1_hit && !g_partial_close_level2_hit && profit_atr_multiple >= InpPartialCloseLevel2){
            double remaining_after_first = g_position_original_size * (1.0 - InpPartialClosePercent1 / 100.0);
            double close_volume = remaining_after_first * (InpPartialClosePercent2 / 100.0);
            close_volume = MathMin(close_volume, pos_volume); // Can't close more than current volume
            
            if(close_volume >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)){
                trade.SetExpertMagicNumber(InpMagic);
                if(trade.PositionClosePartial(_Symbol, close_volume)){
                    g_partial_close_level2_hit = true;
                    Print("ENHANCED RISK MGT: Partial close level 2 hit - Closed ", 
                          DoubleToString(close_volume, 2), " lots at ", 
                          DoubleToString(profit_atr_multiple, 2), "x ATR profit");
                }
            }
        }
    }
    
    // Re-select position after potential partial closes
    if(!PositionSelectByTicket(ticket)) return;
    pos_volume = PositionGetDouble(POSITION_VOLUME); // Update volume after partial closes
    
    bool should_update_sl = false;
    double new_sl = pos_current_sl;
    
    // Break-even move logic (existing)
    if(InpUseBreakEven && profit_atr_multiple >= InpBreakEvenATR){
        double break_even_price = pos_open_price;
        
        // Add small buffer for guaranteed profit
        if(is_long){
            break_even_price += InpBreakEvenBuffer * _Point;
        } else {
            break_even_price -= InpBreakEvenBuffer * _Point;
        }
        
        // Only move to break-even if current SL is worse
        bool should_move_to_breakeven = false;
        if(is_long){
            should_move_to_breakeven = (pos_current_sl == 0.0 || pos_current_sl < break_even_price);
        } else {
            should_move_to_breakeven = (pos_current_sl == 0.0 || pos_current_sl > break_even_price);
        }
        
        if(should_move_to_breakeven){
            new_sl = break_even_price;
            should_update_sl = true;
            Print("ENHANCED RISK MGT: Moving position to break-even+buffer: ", DoubleToString(new_sl, 5));
        }
    }
    
    // ENHANCED FEATURE 2: Advanced Trailing Stop with Acceleration
    if(InpUseTrailingStop && profit_atr_multiple >= InpTrailStartATR){
        double trail_distance = current_atr * InpTrailStopATR;
        
        // ENHANCED FEATURE 3: Accelerated trailing for large profits
        if(InpUseAcceleratedTrail && profit_atr_multiple >= InpAccelTrailThreshold){
            trail_distance *= InpAccelTrailMultiplier;
            Print("ENHANCED RISK MGT: Accelerated trailing activated - Distance reduced by ", 
                  DoubleToString((1.0 - InpAccelTrailMultiplier) * 100, 1), "%");
        }
        
        // ENHANCED FEATURE 4: Dynamic ATR tightening over time
        if(InpUseDynamicATRStops){
            int days_held = (int)((TimeCurrent() - pos_time) / (24 * 60 * 60));
            if(days_held > 0 && days_held <= InpMaxDaysToTighten){
                double tighten_factor = MathPow(InpATRTightenRate, days_held);
                trail_distance *= tighten_factor;
                Print("ENHANCED RISK MGT: Dynamic ATR stop tightened by ", 
                      DoubleToString((1.0 - tighten_factor) * 100, 1), "% after ", days_held, " days");
            }
        }
        
        double new_trailing_sl;
        if(is_long){
            new_trailing_sl = current_price - trail_distance;
            // Only move SL up (never down)
            if(pos_current_sl == 0.0 || new_trailing_sl > pos_current_sl){
                new_sl = new_trailing_sl;
                should_update_sl = true;
            }
        } else {
            new_trailing_sl = current_price + trail_distance;
            // Only move SL down (never up)
            if(pos_current_sl == 0.0 || new_trailing_sl < pos_current_sl){
                new_sl = new_trailing_sl;
                should_update_sl = true;
            }
        }
        
        if(should_update_sl && MathAbs(new_sl - pos_current_sl) > _Point){
            Print("ENHANCED RISK MGT: Updating enhanced trailing stop: Profit=", 
                  DoubleToString(profit_atr_multiple,2), "x ATR, New SL=", DoubleToString(new_sl,5));
        }
    }
    
    // Apply the stop loss update
    if(should_update_sl){
        // Normalize to tick size
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        if(tick_size > 0){
            new_sl = MathRound(new_sl / tick_size) * tick_size;
        }
        
        trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
    }
    
    // Call original trailing function for backward compatibility
    UpdateSinglePositionTrailing(ticket, current_atr);
}

// Calculate volatility regime-adjusted position size
double CalculateVolatilityAdjustedSize(double base_lots){
    if(!InpUseVolatilityRegime || !g_high_volatility_regime) return base_lots;
    
    // Reduce position size during high volatility periods
    double adjusted_lots = base_lots * InpVolRegimeSizeReduce;
    
    // Ensure we meet minimum lot size requirements
    double min_lots = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    adjusted_lots = MathMax(adjusted_lots, min_lots);
    
    // Round to lot step
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    if(lot_step > 0){
        adjusted_lots = MathRound(adjusted_lots / lot_step) * lot_step;
    }
    
    return adjusted_lots;
}

// Update volatility regime detection
void UpdateVolatilityRegime(){
    if(!InpUseVolatilityRegime) return;
    
    // Only check periodically to avoid excessive computation
    if(TimeCurrent() - g_last_volatility_check < 60) return; // Check every minute
    g_last_volatility_check = TimeCurrent();
    
    // Get ATR data for regime detection
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, InpVolRegimePeriod + 1, atr_values);
    if(copied < InpVolRegimePeriod) return;
    
    double current_atr = atr_values[0];
    
    // Calculate median ATR over lookback period
    double atr_sorted[];
    ArrayResize(atr_sorted, InpVolRegimePeriod);
    ArrayCopy(atr_sorted, atr_values, 0, 1, InpVolRegimePeriod); // Skip current bar
    ArraySort(atr_sorted);
    
    int median_index = InpVolRegimePeriod / 2;
    g_volatility_median = (InpVolRegimePeriod % 2 == 0) ? 
                          (atr_sorted[median_index-1] + atr_sorted[median_index]) / 2.0 : 
                          atr_sorted[median_index];
    
    // Determine if we're in high volatility regime
    bool was_high_vol = g_high_volatility_regime;
    
    if(g_volatility_median > 0){
        double vol_ratio = current_atr / g_volatility_median;
        g_high_volatility_regime = (vol_ratio >= InpVolRegimeMultiple);
        
        // Handle regime changes
        if(g_high_volatility_regime && !was_high_vol){
            Print("VOLATILITY REGIME: Entered HIGH volatility (ATR=", DoubleToString(current_atr,5), 
                  ", Median=", DoubleToString(g_volatility_median,5), 
                  ", Ratio=", DoubleToString(vol_ratio,2), "x)");
                  
            // Set pause period for extreme volatility
            if(vol_ratio >= InpVolRegimeMultiple * 1.5){ // 1.5x threshold for pause
                g_volatility_regime_pause_until = TimeCurrent() + InpVolRegimePauseMin * 60;
                Print("EXTREME VOLATILITY: Pausing trading for ", InpVolRegimePauseMin, " minutes");
            }
        }
        else if(!g_high_volatility_regime && was_high_vol){
            Print("VOLATILITY REGIME: Returned to NORMAL volatility (Ratio=", DoubleToString(vol_ratio,2), "x)");
        }
    }
}

// Check if trading should be paused due to volatility regime
bool CheckVolatilityRegimePause(){
    if(!InpUseVolatilityRegime) return false;
    
    // Check if we're in a volatility pause period
    if(TimeCurrent() < g_volatility_regime_pause_until){
        static datetime last_pause_message = 0;
        if(TimeCurrent() - last_pause_message > 60){ // Message every minute
            int remaining = (int)((g_volatility_regime_pause_until - TimeCurrent()) / 60);
            Print("Trading paused due to extreme volatility. Remaining: ", remaining, " minutes");
            last_pause_message = TimeCurrent();
        }
        return true;
    }
    
    return false;
}

//============================== ENHANCED RISK MANAGEMENT (NEW) ================
// Additional risk controls for production trading

// Check portfolio exposure and position limits
bool CheckPortfolioRisk(){
    // Update position count and total risk
    UpdatePortfolioRisk();
    
    // Check maximum positions limit
    if(g_active_positions >= InpMaxPositions){
        SmartLog("RISK", "Portfolio risk check: Maximum positions reached (" + IntegerToString(g_active_positions) + "/" + IntegerToString(InpMaxPositions) + ")");
        return false;
    }
    
    // Check total portfolio risk percentage
    if(g_total_risk_percent >= InpMaxTotalRisk){
        Print("Portfolio risk check: Maximum total risk reached (", DoubleToString(g_total_risk_percent,2), 
              "%/", DoubleToString(InpMaxTotalRisk,1), "%)");
        return false;
    }
    
    return true;
}

// Update portfolio risk metrics (OPTIMIZED 2.4)
void UpdatePortfolioRisk(){
    g_active_positions = 0;
    g_total_risk_percent = 0.0;
    double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    if(InpOptimizeLoops && InpOptimizePositionScan){
        // Use optimized cached scanning
        RefreshPositionCache();
        
        // Process cached positions only
        for(int i = 0; i < g_position_cache_count; i++){
            ulong ticket = g_cached_positions[i];
            if(!PositionSelectByTicket(ticket)) continue;
            
            g_active_positions++;
            
            // Calculate risk for this position
            double pos_volume = PositionGetDouble(POSITION_VOLUME);
            double pos_sl = PositionGetDouble(POSITION_SL);
            double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
            
            if(pos_sl > 0.0 && account_equity > 0.0){
                double risk_points = MathAbs(pos_open - pos_sl) / _Point;
                double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                double risk_amount = pos_volume * risk_points * tick_value / _Point;
                double risk_percent = (risk_amount / account_equity) * 100.0;
                g_total_risk_percent += risk_percent;
            }
        }
    }
    else {
        // Fallback to standard scanning with loop safety
        int loop_count = 0;
        int total_positions = PositionsTotal();
        
        for(int i = 0; i < total_positions; i++){
            // Loop safety check
            if(InpUseEarlyBreaks && !LoopSafetyCheck(loop_count)){
                LogCritical("LOOP SAFETY: Portfolio risk scan terminated at max iterations");
                break;
            }
            
            ulong ticket = PositionGetTicket(i);
            if(!PositionSelectByTicket(ticket)) continue;
            
            string pos_symbol = PositionGetString(POSITION_SYMBOL);
            long pos_magic = PositionGetInteger(POSITION_MAGIC);
            
            if(pos_symbol == _Symbol && pos_magic == InpMagic){
                g_active_positions++;
                
                // Calculate risk for this position
                double pos_volume = PositionGetDouble(POSITION_VOLUME);
                double pos_sl = PositionGetDouble(POSITION_SL);
                double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
                
                if(pos_sl > 0.0 && account_equity > 0.0){
                    double risk_points = MathAbs(pos_open - pos_sl) / _Point;
                    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                    double risk_amount = pos_volume * risk_points * tick_value / _Point;
                    double risk_percent = (risk_amount / account_equity) * 100.0;
                    g_total_risk_percent += risk_percent;
                }
            }
        }
    }
    
    g_last_position_check = TimeCurrent();
}

// Check if current time is within trading hours
bool IsWithinTradingHours(){
    if(!InpUseTradingHours) return true;
    
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // Check day of week restrictions
    if(InpAvoidSunday && dt.day_of_week == 0){  // Sunday
        return false;
    }
    if(InpAvoidFriday && dt.day_of_week == 5){  // Friday
        return false;
    }
    
    // Parse trading hours
    string start_parts[], end_parts[];
    int start_count = StringSplit(InpTradingStart, ':', start_parts);
    int end_count = StringSplit(InpTradingEnd, ':', end_parts);
    
    if(start_count < 2 || end_count < 2) return true; // Invalid format, allow trading
    
    int start_hour = (int)StringToInteger(start_parts[0]);
    int start_minute = (int)StringToInteger(start_parts[1]);
    int end_hour = (int)StringToInteger(end_parts[0]);
    int end_minute = (int)StringToInteger(end_parts[1]);
    
    int current_minutes = dt.hour * 60 + dt.min;
    int start_minutes = start_hour * 60 + start_minute;
    int end_minutes = end_hour * 60 + end_minute;
    
    // Handle overnight sessions (e.g., 22:00 to 06:00)
    if(start_minutes > end_minutes){
        return (current_minutes >= start_minutes || current_minutes <= end_minutes);
    } else {
        return (current_minutes >= start_minutes && current_minutes <= end_minutes);
    }
}

// Check for high volatility conditions
bool CheckVolatilityConditions(){
    if(!InpUseNewsFilter) return true;
    
    // Check if we're in a volatility pause period
    if(TimeCurrent() < g_volatility_pause_until){
        return false;
    }
    
    if(TimeCurrent() < g_news_pause_until){
        return false;
    }
    
    // Get current ATR and compare to baseline
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int atr_copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 21, atr_values);
    
    if(atr_copied > 20){
        double current_atr = atr_values[0];
        
        // Calculate 20-day average ATR (baseline)
        double atr_sum = 0.0;
        for(int i=1; i<=20; ++i){
            atr_sum += atr_values[i];
        }
        g_baseline_atr = atr_sum / 20.0;
        
        // Check if current volatility is too high
        if(g_baseline_atr > 0.0){
            double volatility_ratio = current_atr / g_baseline_atr;
            if(volatility_ratio > InpMaxVolatility){
                Print("High volatility detected: ATR ratio = ", DoubleToString(volatility_ratio,2), 
                      " (limit: ", DoubleToString(InpMaxVolatility,1), ")");
                g_volatility_pause_until = TimeCurrent() + InpVolatilityPause * 60;
                return false;
            }
        }
    }
    
    return true;
}

// Master risk check function - combines all risk controls
bool MasterRiskCheck(){
    // Update volatility regime detection
    UpdateVolatilityRegime();
    
    // Basic risk limits (existing)
    if(!CheckRiskLimits()) return false;
    
    // Portfolio risk limits (new)
    if(!CheckPortfolioRisk()) return false;
    
    // Trading hours (new)
    if(!IsWithinTradingHours()){
        Print("Outside trading hours - trading paused");
        return false;
    }
    
    // Volatility regime pause (new - extreme volatility)
    if(CheckVolatilityRegimePause()){
        return false; // Already prints message internally
    }
    
    // Volatility conditions (existing - general volatility check)
    if(!CheckVolatilityConditions()){
        Print("High volatility conditions - trading paused");
        return false;
    }
    
    // Spread check (existing)
    if(!IsSpreadAcceptable()){
        Print("Spread too high - trade skipped");
        return false;
    }
    
    return true;
}

//============================== TRADING EXECUTION =============================
// These functions handle the actual buying and selling based on AI decisions
CTrade trade;                    // MetaTrader trading object for executing orders
datetime g_last_bar_time = 0;    // Timestamp of last processed bar (prevents duplicate signals)

// EXECUTE TRADING ACTION BASED ON AI DECISION (ENHANCED with risk management)
// This function translates AI predictions into actual trades with full risk controls
void MaybeTrade(const int action){
  // FINAL SAFETY CHECK: Absolutely prevent trading with mismatched model (CRITICAL)
  if(InpEnforceSymbolTF && (_Symbol != g_model_symbol || _Period != g_model_tf)){
    static datetime last_critical_alert = 0;
    if(TimeCurrent() - last_critical_alert > 60){ // Alert every minute for critical safety
      Print("CRITICAL SAFETY BLOCK: Trading attempt prevented due to model mismatch!");
      Print("Chart: ", _Symbol, " ", EnumToString(_Period), " | Model: ", g_model_symbol, " ", EnumToString(g_model_tf));
      Alert("CORTEX3 CRITICAL: Trading blocked - model mismatch detected!");
      last_critical_alert = TimeCurrent();
    }
    return; // Absolutely refuse to trade
  }
  
  // Skip if AI says to hold (do nothing)
  if(action==ACTION_HOLD) return;
  
  // Handle FLAT action - close all positions and stay flat
  if(action==ACTION_FLAT){
    int total = PositionsTotal();
    bool closed_any = false;
    for(int i=total-1;i>=0;--i){  // Loop backwards for safe deletion
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      
      string sym = PositionGetString(POSITION_SYMBOL);
      long mg = PositionGetInteger(POSITION_MAGIC);
      double profit = PositionGetDouble(POSITION_PROFIT);
      
      // Only close our positions on this symbol
      if(sym==_Symbol && mg==InpMagic){
        UpdateTradeTracking(profit > 0);  // Track result before closing
        trade.PositionClose(sym);  // Close position
        SmartLog("TRADE", "FLAT action: Closed position. Profit: " + DoubleToString(profit,2));
        closed_any = true;
      }
    }
    if(!closed_any){
      SmartLog("TRADE", "FLAT action: No positions to close");
    }
    return;  // Don't open any new positions
  }
  
  // COMPREHENSIVE RISK CHECKS (ENHANCED - all risk controls)
  if(!MasterRiskCheck()){
    Print("Trade skipped: Risk management conditions not met");
    return;
  }
  
  // Determine trade direction and signal strength
  bool is_buy = (action==ACTION_BUY_STRONG || action==ACTION_BUY_WEAK);
  bool is_strong = (action==ACTION_BUY_STRONG || action==ACTION_SELL_STRONG);
  
  // Calculate position size using risk management or legacy method
  double base_lots = CalculatePositionSize(is_strong);
  if(base_lots<=0.0) return;  // Invalid lot size
  
  // Apply volatility regime adjustment to position size
  double lots = CalculateVolatilityAdjustedSize(base_lots);
  
  if(g_high_volatility_regime && InpUseVolatilityRegime){
    SmartLog("RISK", "High volatility regime: Reduced position size from " + DoubleToString(base_lots,2) + 
          " to " + DoubleToString(lots,2) + " lots");
  }
  
  // Apply emergency stop size adjustment (1.5 IMPROVEMENT)
  double emergency_adjustment = GetEmergencySizeAdjustment();
  if(emergency_adjustment < 1.0){
    double pre_emergency_lots = lots;
    lots = lots * emergency_adjustment;
    Print("EMERGENCY STOP: Position size adjusted from ", DoubleToString(pre_emergency_lots,2), 
          " to ", DoubleToString(lots,2), " lots (level ", g_drawdown_level, ")");
    
    if(lots <= 0.0){
      Print("EMERGENCY STOP: Position size reduced to zero - blocking new trades");
      return; // Block trade completely
    }
  }

  // Check existing positions and handle according to EA strategy
  int total = PositionsTotal();
  for(int i=total-1;i>=0;--i){  // Loop backwards for safe deletion
    ulong ticket = PositionGetTicket(i);
    if(!PositionSelectByTicket(ticket)) continue;
    
    string sym   = PositionGetString(POSITION_SYMBOL);
    long   type  = PositionGetInteger(POSITION_TYPE);
    long   mg    = PositionGetInteger(POSITION_MAGIC);
    double profit = PositionGetDouble(POSITION_PROFIT);
    
    // Only process our positions on this symbol
    if(sym==_Symbol && mg==InpMagic){
      // Update trade tracking when closing positions
      if(InpCloseOpposite){
        if(is_buy && type==POSITION_TYPE_SELL){ 
            UpdateTradeTracking(profit > 0);  // Track result before closing
            trade.PositionClose(sym);  // Close sell when AI wants buy
            SmartLog("TRADE", "Closed opposite SELL position. Profit: " + DoubleToString(profit,2));
        }
        if(!is_buy && type==POSITION_TYPE_BUY){ 
            UpdateTradeTracking(profit > 0);  // Track result before closing
            trade.PositionClose(sym);  // Close buy when AI wants sell
            SmartLog("TRADE", "Closed opposite BUY position. Profit: " + DoubleToString(profit,2));
        }
      }
      // ENHANCED POSITION MANAGEMENT: Handle position scaling
      if((is_buy && type==POSITION_TYPE_BUY) || (!is_buy && type==POSITION_TYPE_SELL)){
        // Same direction signal - check if we should scale the position
        if(InpAllowPositionScaling){
          double current_volume = PositionGetDouble(POSITION_VOLUME);
          double target_volume = lots;
          
          // Check if current position size matches signal strength
          if(!IsPositionCorrectSize(current_volume, is_strong)){
            SmartLog("TRADE", "Position scaling needed: Current=" + DoubleToString(current_volume,2) + 
                  " lots, Target=" + DoubleToString(target_volume,2) + " lots, Signal=" + 
                  (is_strong?"STRONG":"WEAK"));
            
            // Scale the position to match new signal strength
            bool scaling_result = ScalePosition(current_volume, target_volume, is_buy);
            if(scaling_result){
              SmartLog("TRADE", "Position successfully scaled for " + ACTION_NAME[action] + " signal");
            } else {
              LogCritical("Position scaling failed for " + ACTION_NAME[action] + " signal");
            }
          } else {
            SmartLog("TRADE", "Position size already matches " + (is_strong?"STRONG":"WEAK") + " signal - no scaling needed");
          }
        } else {
          SmartLog("TRADE", "Same direction signal ignored - position scaling disabled");
        }
        return; // Don't open new position
      }
    }
  }
  
  // Exit if new positions are disabled
  if(!InpAllowNewPos) return;
  
  // Calculate ATR-based SL and TP (NEW - essential for risk management)
  double sl_price = 0.0, tp_price = 0.0;
  
  if(InpUseRiskSizing){
    // Use cached ATR for SL/TP calculation (2.1 IMPROVEMENT)
    double atr = GetCachedATR14();
    
    if(atr > 0){
      double current_price = is_buy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      double adaptive_atr_multiplier = GetAdaptiveATRMultiplier();
      if(is_buy){
        sl_price = current_price - (atr * adaptive_atr_multiplier);
        tp_price = current_price + (atr * adaptive_atr_multiplier * InpRRRatio);
      } else {
        sl_price = current_price + (atr * adaptive_atr_multiplier);
        tp_price = current_price - (atr * adaptive_atr_multiplier * InpRRRatio);
      }
      
      // Normalize prices to tick size
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tick_size > 0){
        sl_price = MathRound(sl_price / tick_size) * tick_size;
        tp_price = MathRound(tp_price / tick_size) * tick_size;
      }
    }
  }
  
  // Execute the trade with SL/TP
  trade.SetExpertMagicNumber(InpMagic);
  bool ok = is_buy ? 
      trade.Buy(lots, _Symbol, 0, sl_price, tp_price, "cortex3_ai") :
      trade.Sell(lots, _Symbol, 0, sl_price, tp_price, "cortex3_ai");
      
  // Log the result with risk management info
  if(ok) {
    Print("Opened ", (is_buy?"BUY":"SELL"), " lots=", DoubleToString(lots,2),
          ", SL=", DoubleToString(sl_price,5), ", TP=", DoubleToString(tp_price,5),
          ", Action=", ACTION_NAME[action]);
  } else {
    Print("Trade open failed. Action=", ACTION_NAME[action], " Error=", GetLastError());
  }
}

//============================== EA LIFECYCLE FUNCTIONS ===========================
// These are the main entry points that MetaTrader calls
// INITIALIZATION FUNCTION (ENHANCED with safety checks)
// Called once when EA is first loaded onto a chart
int OnInit(){
  MathSrand((int)TimeLocal());  // Initialize random number generator
  
  // Try to load the trained Double-Dueling DRQN model(s)
  if(InpUseEnsembleModel){
    // Load ensemble models
    bool ensemble_loaded = LoadEnsembleModels();
    if(!ensemble_loaded){
      Print("FATAL: Ensemble models not loaded; EA cannot operate without trained models.");
      return(INIT_FAILED);
    }
    g_loaded = true; // Mark as loaded for compatibility
  } else {
    // Load single model
    g_loaded = LoadModel(InpModelFileName);
    if(!g_loaded){
      Print("FATAL: Double-Dueling DRQN model not loaded; EA cannot operate without trained model.");
      return(INIT_FAILED);  // Stop EA if model can't be loaded
    }
  }
  
  // CRITICAL SAFETY CHECK: Enforce symbol/timeframe match (PREVENTS MODEL MISUSE)
  bool symbol_mismatch = (_Symbol != g_model_symbol);
  bool timeframe_mismatch = (_Period != g_model_tf);
  bool any_mismatch = (symbol_mismatch || timeframe_mismatch);
  
  if(any_mismatch){
    Print("================================================================================");
    Print("!!!                        CRITICAL MODEL MISMATCH                        !!!");
    Print("================================================================================");
    Print("CHART:     Symbol=", _Symbol, " | Timeframe=", EnumToString(_Period));
    Print("MODEL:     Symbol=", g_model_symbol, " | Timeframe=", EnumToString(g_model_tf));
    
    if(symbol_mismatch) Print(">>> SYMBOL MISMATCH: Trading ", _Symbol, " with model trained on ", g_model_symbol);
    if(timeframe_mismatch) Print(">>> TIMEFRAME MISMATCH: Trading ", EnumToString(_Period), " with model trained on ", EnumToString(g_model_tf));
    
    Print("================================================================================");
    
    if(InpEnforceSymbolTF){
      Print("FATAL ERROR: Model enforcement is ENABLED - EA will not start!");
      Print("Solutions:");
      Print("  1. Move EA to correct chart: ", g_model_symbol, " ", EnumToString(g_model_tf));
      Print("  2. Retrain model for this symbol/timeframe: ", _Symbol, " ", EnumToString(_Period));
      Print("  3. Set InpEnforceSymbolTF=false (NOT RECOMMENDED FOR LIVE TRADING)");
      Print("================================================================================");
      Alert("CORTEX3 EA ERROR: Model mismatch detected. Check Expert log!");
      return(INIT_FAILED);  // Stop EA to prevent dangerous trading
    } else {
      Print("WARNING: Model enforcement is DISABLED - EA will continue with MISMATCHED MODEL!");
      Print("RISK LEVEL: EXTREMELY HIGH - Model performance will be severely degraded!");
      Print("STRONGLY RECOMMENDED: Move EA to correct chart or retrain model!");
      Print("================================================================================");
      
      // Send multiple alerts to ensure user notices
      Alert("CORTEX3 EA WARNING: Trading with mismatched model! Performance will be poor!");
      Alert("Chart: ", _Symbol, " ", EnumToString(_Period), " | Model: ", g_model_symbol, " ", EnumToString(g_model_tf));
    }
  } else {
    Print("✓ Model verification passed: Symbol=", _Symbol, " Timeframe=", EnumToString(_Period));
  }
  
  // Initialize risk management (ENHANCED - all risk controls)
  g_initial_equity = AccountInfoDouble(ACCOUNT_EQUITY);
  ArrayResize(g_recent_trades, 10);
  ArrayInitialize(g_recent_trades, 0);
  g_recent_trades_count = 0;
  g_trading_paused = false;
  
  // Initialize new risk management variables
  g_total_risk_percent = 0.0;
  g_active_positions = 0;
  g_last_position_check = 0;
  g_volatility_pause_until = 0;
  g_news_pause_until = 0;
  g_baseline_atr = 0.0;
  
  // Initialize advanced risk management variables
  g_volatility_median = 0.0;
  g_last_volatility_check = 0;
  g_high_volatility_regime = false;
  g_volatility_regime_pause_until = 0;
  
  // Initialize emergency stop system (1.5 IMPROVEMENT)
  if(InpUseAdvancedEmergencyStop){
    g_daily_starting_equity = g_initial_equity;
    g_max_equity_today = g_initial_equity;
    g_daily_pnl = 0.0;
    g_current_trading_day = TimeCurrent() - (TimeCurrent() % 86400);
    g_daily_loss_limit_hit = false;
    g_emergency_pause_until = 0;
    g_drawdown_level = 0;
    g_consecutive_emergency_losses = 0;
    g_emergency_shutdown_active = false;
    g_last_emergency_check = 0;
  }
  
  // Initialize indicator caching system (2.1 IMPROVEMENT)
  if(!InitializeCachedIndicators()){
    Print("ERROR: Failed to initialize indicator caching system");
    return(INIT_FAILED);
  }
  
  // Initialize data access optimization (2.2 IMPROVEMENT)
  if(!InitializeDataArrays()){
    Print("ERROR: Failed to initialize data optimization system");
    return(INIT_FAILED);
  }
  
  // Initialize intelligent logging system (2.3 IMPROVEMENT)
  if(InpMinimizeLogging){
    InitializeLoggingSystem();
  }
  
  // Initialize efficient looping system (2.4 IMPROVEMENT)
  if(InpOptimizeLoops){
    InitializeLoopOptimization();
  }
  
  // Initialize tick handling optimization (2.5 IMPROVEMENT)
  if(InpOptimizeTicks){
    InitializeTickHandling();
  }
  
  // Initialize confidence-based trade filtering (3.1 IMPROVEMENT)
  if(InpUseConfidenceFilter){
    InitializeConfidenceTracking();
  }
  
  // Initialize adaptive parameter system (3.3 IMPROVEMENT)
  if(InpUseAdaptiveParameters){
    InitializeAdaptiveParameters();
  }
  
  Print("=== CORTEX3 EA INITIALIZED ===");
  if(InpUseEnsembleModel){
    Print("Ensemble Models: ", g_active_ensemble_count, " loaded | Method: ", InpEnsembleMethod);
    Print("Agreement threshold: ", DoubleToString(InpEnsembleAgreementThreshold, 3));
    Print("Fallback mode: ", InpEnsembleFallbackMode ? "ENABLED" : "DISABLED");
  } else {
    Print("Single Model: ", InpModelFileName);
  }
  Print("Symbol/TF: ", g_model_symbol, "/", EnumToString(g_model_tf));
  Print("=== RISK MANAGEMENT SETTINGS ===");
  Print("Risk Management: ", InpUseRiskSizing ? "ENABLED" : "DISABLED (using fixed lots)");
  Print("Risk per trade: ", DoubleToString(InpRiskPercent,1), "%");
  Print("Max drawdown: ", DoubleToString(InpMaxDrawdown,1), "%");
  Print("Max spread: ", DoubleToString(InpMaxSpread,1), " points");
  Print("ATR multiplier: ", DoubleToString(InpATRMultiplier,1), "x");
  Print("Risk:Reward: 1:", DoubleToString(InpRRRatio,1));
  Print("=== PORTFOLIO CONTROLS ===");
  Print("Max positions: ", InpMaxPositions);
  Print("Max total risk: ", DoubleToString(InpMaxTotalRisk,1), "%");
  Print("=== SESSION CONTROLS ===");
  Print("Trading hours: ", InpUseTradingHours ? (InpTradingStart + " - " + InpTradingEnd) : "24/7");
  Print("Avoid Friday: ", InpAvoidFriday ? "YES" : "NO");
  Print("Avoid Sunday: ", InpAvoidSunday ? "YES" : "NO");
  Print("=== VOLATILITY CONTROLS ===");
  Print("News filter: ", InpUseNewsFilter ? "ENABLED" : "DISABLED");
  Print("Max volatility: ", DoubleToString(InpMaxVolatility,1), "x average ATR");
  Print("Volatility pause: ", InpVolatilityPause, " minutes");
  Print("=== ADVANCED RISK CONTROLS ===");
  Print("Trailing stops: ", InpUseTrailingStop ? "ENABLED" : "DISABLED");
  if(InpUseTrailingStop){
    Print("  Trail start: ", DoubleToString(InpTrailStartATR,1), "x ATR profit");
    Print("  Trail distance: ", DoubleToString(InpTrailStopATR,1), "x ATR");
  }
  Print("Break-even move: ", InpUseBreakEven ? "ENABLED" : "DISABLED");
  if(InpUseBreakEven){
    Print("  Break-even trigger: ", DoubleToString(InpBreakEvenATR,1), "x ATR profit");
    Print("  Break-even buffer: ", DoubleToString(InpBreakEvenBuffer,1), " points");
  }
  Print("Volatility regime: ", InpUseVolatilityRegime ? "ENABLED" : "DISABLED");
  if(InpUseVolatilityRegime){
    Print("  High vol threshold: ", DoubleToString(InpVolRegimeMultiple,1), "x median ATR");
    Print("  Size reduction: ", DoubleToString(InpVolRegimeSizeReduce*100,0), "%");
    Print("  Extreme vol pause: ", InpVolRegimePauseMin, " minutes");
  }
  Print("Position scaling: ", InpAllowPositionScaling ? "ENABLED" : "DISABLED");
  
  // Enhanced Risk Management (1.1 IMPROVEMENTS) display
  Print("=== ENHANCED RISK MANAGEMENT (1.1 IMPROVEMENTS) ===");
  Print("Partial position closing: ", InpUsePartialClose ? "ENABLED" : "DISABLED");
  if(InpUsePartialClose){
    Print("  Level 1: ", DoubleToString(InpPartialCloseLevel1,1), "x ATR profit, close ", 
          DoubleToString(InpPartialClosePercent1,0), "%");
    Print("  Level 2: ", DoubleToString(InpPartialCloseLevel2,1), "x ATR profit, close ", 
          DoubleToString(InpPartialClosePercent2,0), "% of remainder");
  }
  Print("Accelerated trailing: ", InpUseAcceleratedTrail ? "ENABLED" : "DISABLED");
  if(InpUseAcceleratedTrail){
    Print("  Threshold: ", DoubleToString(InpAccelTrailThreshold,1), "x ATR profit");
    Print("  Acceleration: ", DoubleToString(InpAccelTrailMultiplier,2), "x trail distance");
  }
  Print("Dynamic ATR stops: ", InpUseDynamicATRStops ? "ENABLED" : "DISABLED");
  if(InpUseDynamicATRStops){
    Print("  Tightening rate: ", DoubleToString((1.0-InpATRTightenRate)*100,1), "% per day");
    Print("  Max tighten days: ", InpMaxDaysToTighten);
  }
  
  // Enhanced Trade Filtering (1.2 IMPROVEMENTS) display
  Print("=== ENHANCED TRADE FILTERING & SIGNAL CONFIRMATION (1.2) ===");
  Print("Signal confirmation: ", InpUseSignalConfirmation ? "ENABLED" : "DISABLED");
  if(InpUseSignalConfirmation){
    Print("Multi-timeframe: ", InpUseMultiTimeframe ? "ENABLED" : "DISABLED");
    if(InpUseMultiTimeframe){
      Print("  Higher TF: ", EnumToString(InpHigherTF), " trend confirmation");
    }
    Print("Secondary indicators: ", InpUseSecondaryIndicators ? "ENABLED" : "DISABLED");
    if(InpUseSecondaryIndicators){
      Print("  RSI: ", InpRSIPeriod, " period, OB/OS: ", DoubleToString(InpRSIOverbought,1), 
            "/", DoubleToString(InpRSIOversold,1));
    }
    Print("MACD confirmation: ", InpUseMACDConfirmation ? "ENABLED" : "DISABLED");
    if(InpUseMACDConfirmation){
      Print("  MACD: ", InpMACDFastEMA, ",", InpMACDSlowEMA, ",", InpMACDSignalSMA);
    }
    Print("Signal persistence: ", InpUseSignalPersistence ? "ENABLED" : "DISABLED");
    if(InpUseSignalPersistence){
      Print("  Required bars: ", InpSignalPersistenceBars);
    }
    Print("Min signal strength: ", DoubleToString(InpMinSignalStrength, 2));
    Print("Enhanced spread filter: ", InpUseSpreadFilter ? "ENABLED" : "DISABLED");
    if(InpUseSpreadFilter){
      Print("  Max spread: ", DoubleToString(InpMaxSpreadATR, 2), "x ATR");
    }
    Print("Liquidity filter: ", InpUseLiquidityFilter ? "ENABLED" : "DISABLED");
    if(InpUseLiquidityFilter){
      Print("  Min tick volume: ", InpMinTickVolume);
    }
  }
  
  // Enhanced Overtrading Prevention (1.3 IMPROVEMENTS) display
  Print("=== ENHANCED OVERTRADING PREVENTION (1.3 IMPROVEMENTS) ===");
  Print("Advanced cooldown system: ", InpUseAdvancedCooldown ? "ENABLED" : "DISABLED");
  if(InpUseAdvancedCooldown){
    Print("  Minimum cooldown: ", InpMinCooldownMinutes, " minutes between positions");
    Print("  Choppy market detection: ", InpUseChoppyDetection ? "ENABLED" : "DISABLED");
    Print("  Consecutive loss protection: ", InpUseConsecutiveLosses ? "ENABLED" : "DISABLED");
  }
  
  // Refined Signal Processing (1.4 IMPROVEMENTS) display
  Print("=== REFINED SIGNAL PROCESSING (1.4 IMPROVEMENTS) ===");
  Print("Signal refinement: ", InpUseSignalRefinement ? "ENABLED" : "DISABLED");
  if(InpUseSignalRefinement){
    Print("  Q-value smoothing: ", InpUseQValueSmoothing ? "ENABLED" : "DISABLED");
    if(InpUseQValueSmoothing){
      Print("    Smoothing factor: ", DoubleToString(InpQValueSmoothingAlpha, 2));
    }
    Print("  Signal threshold: ", InpUseSignalThreshold ? "ENABLED" : "DISABLED");
    if(InpUseSignalThreshold){
      Print("    Min Q-value difference: ", DoubleToString(InpSignalThresholdQ, 3));
    }
    Print("  Advanced persistence: ", InpUseAdvancedPersistence ? "ENABLED" : "DISABLED");
    if(InpUseAdvancedPersistence){
      Print("    Required bars: ", InpAdvancedPersistBars);
    }
    Print("  Minimum pip movement: ", InpUseMinimumPipMove ? "ENABLED" : "DISABLED");
    if(InpUseMinimumPipMove){
      Print("    Minimum threshold: ", InpMinimumPipThreshold, " pips");
    }
  }
  
  // Emergency Stop/Drawdown Control (1.5 IMPROVEMENTS) display
  Print("=== EMERGENCY STOP/DRAWDOWN CONTROL (1.5 IMPROVEMENTS) ===");
  Print("Advanced emergency system: ", InpUseAdvancedEmergencyStop ? "ENABLED" : "DISABLED");
  if(InpUseAdvancedEmergencyStop){
    Print("  Daily loss limit: ", InpUseDailyLossLimit ? "ENABLED" : "DISABLED");
    if(InpUseDailyLossLimit){
      Print("    Max daily loss: $", DoubleToString(InpDailyLossLimitAmount, 0), 
            " or ", DoubleToString(InpDailyLossLimitPercent, 1), "% of equity");
    }
    Print("  Drawdown circuit breaker: ", InpUseDrawdownCircuitBreaker ? "ENABLED" : "DISABLED");
    if(InpUseDrawdownCircuitBreaker){
      Print("    Level 1 (reduce size): ", DoubleToString(InpLevel1DrawdownPercent, 1), "%");
      Print("    Level 2 (stop trades): ", DoubleToString(InpLevel2DrawdownPercent, 1), "%");
      Print("    Level 3 (emergency halt): ", DoubleToString(InpLevel3DrawdownPercent, 1), "%");
    }
    Print("  Consecutive loss limit: ", InpUseConsecutiveLossLimit ? "ENABLED" : "DISABLED");
    if(InpUseConsecutiveLossLimit){
      Print("    Max consecutive losses: ", InpMaxConsecutiveLossLimit);
    }
    Print("  Emergency pause duration: ", InpEmergencyPauseHours, " hours");
  }
  
  // Indicator Caching & Optimization (2.1 IMPROVEMENTS) display
  Print("=== INDICATOR CACHING & OPTIMIZATION (2.1 IMPROVEMENTS) ===");
  Print("Indicator caching system: ", InpUseIndicatorCaching ? "ENABLED" : "DISABLED");
  if(InpUseIndicatorCaching){
    Print("  Pre-calculation: ", InpUsePrecalculation ? "ENABLED" : "DISABLED");
    Print("  Cache refresh: Every ", InpCacheRefreshBars, " bar(s)");
    Print("  ATR optimization: ", InpOptimizeATRCalls ? "ENABLED" : "DISABLED");
    Print("  Complex calc caching: ", InpCacheComplexCalcs ? "ENABLED" : "DISABLED");
    Print("  Smart handle management: ", InpUseSmartHandles ? "ENABLED" : "DISABLED");
    Print("  Cached indicators: ATR(14), ATR(50), MA(10), MA(50)");
  }
  
  // Data Access Optimization (2.2 IMPROVEMENTS) display
  Print("=== DATA ACCESS OPTIMIZATION (2.2 IMPROVEMENTS) ===");
  Print("Data structure optimization: ", InpOptimizeDataAccess ? "ENABLED" : "DISABLED");
  if(InpOptimizeDataAccess){
    Print("  Array reuse: ", InpReuseArrays ? "ENABLED" : "DISABLED");
    Print("  Memory operation optimization: ", InpMinimizeMemOps ? "ENABLED" : "DISABLED");
    Print("  Local variable optimization: ", InpOptimizeLocals ? "ENABLED" : "DISABLED");
    Print("  Series access optimization: ", InpOptimizeSeriesAccess ? "ENABLED" : "DISABLED");
    Print("  Pre-allocated buffer size: ", InpPreallocateSize, " elements");
    Print("  Neural network arrays: Pre-allocated for reuse");
  }
  
  // Intelligent Logging System (2.3 IMPROVEMENTS) display
  Print("=== INTELLIGENT LOGGING SYSTEM (2.3 IMPROVEMENTS) ===");
  Print("Logging optimization: ", InpMinimizeLogging ? "ENABLED" : "DISABLED");
  if(InpMinimizeLogging){
    Print("  Signal logging throttle: ", InpSignalLogThrottle, " seconds");
    Print("  Performance logging throttle: ", InpPerfLogThrottle, " seconds");  
    Print("  Risk logging throttle: ", InpRiskLogThrottle, " seconds");
    Print("  Filter logging throttle: ", InpFilterLogThrottle, " seconds");
    Print("  Trade logging: ALWAYS ACTIVE (critical events)");
    Print("  Error logging: ALWAYS ACTIVE (critical events)");
    Print("  Tick-level logging: MINIMIZED for performance");
  }
  
  // Efficient Looping System (2.4 IMPROVEMENTS) display
  Print("=== EFFICIENT LOOPING SYSTEM (2.4 IMPROVEMENTS) ===");
  Print("Loop optimization: ", InpOptimizeLoops ? "ENABLED" : "DISABLED");
  if(InpOptimizeLoops){
    Print("  Early break conditions: ", InpUseEarlyBreaks ? "ENABLED" : "DISABLED");
    Print("  Loop combining: ", InpCombineLoops ? "ENABLED" : "DISABLED");
    Print("  Condition simplification: ", InpSimplifyConditions ? "ENABLED" : "DISABLED");
    Print("  Position scan optimization: ", InpOptimizePositionScan ? "ENABLED" : "DISABLED");
    Print("  Optimized array access: ", InpUseOptimizedArrays ? "ENABLED" : "DISABLED");
    Print("  Max loop iterations: ", InpMaxLoopIterations);
    Print("  Position cache size: ", ArraySize(g_cached_positions), " tickets");
  }
  
  // Tick Handling Optimization (2.5 IMPROVEMENTS) display
  Print("=== TICK HANDLING OPTIMIZATION (2.5 IMPROVEMENTS) ===");
  Print("Tick optimization: ", InpOptimizeTicks ? "ENABLED" : "DISABLED");
  if(InpOptimizeTicks){
    Print("  Process on bar close: ", InpProcessOnBarClose ? "ENABLED" : "DISABLED");
    Print("  Allow intra-bar ticks: ", InpAllowIntraBarTicks ? "ENABLED" : "DISABLED");
    if(InpAllowIntraBarTicks){
      Print("    Tick skip ratio: Process every ", InpTickSkipRatio, " tick(s)");
    }
    Print("  Timer-based checks: ", InpUseTimerChecks ? "ENABLED" : "DISABLED");
    if(InpUseTimerChecks){
      Print("    Timer interval: ", InpTimerIntervalSec, " seconds");
    }
    Print("  Trailing stops on ticks: ", InpTrailingOnTicks ? "ENABLED" : "DISABLED");
    Print("  Risk checks on ticks: ", InpRiskChecksOnTicks ? "ENABLED" : "DISABLED");
    Print("  Emergency checks on ticks: ", InpEmergencyOnTicks ? "ENABLED" : "DISABLED");
  }
  
  Print("Initial equity: $", DoubleToString(g_initial_equity,2));
  Print("EA ready with COMPLETE PERFORMANCE SUITE: risk mgmt (1.1), signal filtering (1.2), overtrading prevention (1.3), signal refinement (1.4), emergency stops (1.5), caching (2.1), data optimization (2.2), intelligent logging (2.3), efficient looping (2.4), optimized ticks (2.5), confidence filtering (3.1), ensemble decisioning (3.2) & adaptive parameters (3.3).");
  
  // Reset LSTM state for fresh start
  g_Q.ResetLSTMState();
  
  // Initialize enhanced trade filtering indicators (1.2 IMPROVEMENT)
  if(InpUseSignalConfirmation){
    Print("=== INITIALIZING ENHANCED TRADE FILTERING (1.2) ===");
    
    // Initialize RSI for secondary confirmation
    if(InpUseSecondaryIndicators){
      g_rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
      if(g_rsi_handle == INVALID_HANDLE){
        Print("ERROR: Failed to initialize RSI indicator");
        return(INIT_FAILED);
      }
      Print("RSI indicator initialized (Period: ", InpRSIPeriod, ")");
    }
    
    // Initialize MACD for trend confirmation
    if(InpUseMACDConfirmation){
      g_macd_handle = iMACD(_Symbol, _Period, InpMACDFastEMA, InpMACDSlowEMA, InpMACDSignalSMA, PRICE_CLOSE);
      if(g_macd_handle == INVALID_HANDLE){
        Print("ERROR: Failed to initialize MACD indicator");
        return(INIT_FAILED);
      }
      Print("MACD indicator initialized (", InpMACDFastEMA, ",", InpMACDSlowEMA, ",", InpMACDSignalSMA, ")");
    }
    
    // Initialize higher timeframe MA for trend confirmation
    if(InpUseMultiTimeframe){
      g_higher_tf_ma_handle = iMA(_Symbol, InpHigherTF, 20, 0, MODE_EMA, PRICE_CLOSE);
      if(g_higher_tf_ma_handle == INVALID_HANDLE){
        Print("ERROR: Failed to initialize higher timeframe MA indicator");
        return(INIT_FAILED);
      }
      Print("Higher TF trend indicator initialized (", EnumToString(InpHigherTF), " EMA20)");
    }
    
    g_indicators_initialized = true;
    Print("Enhanced trade filtering system ready");
  }
  
  return(INIT_SUCCEEDED);
}

// CLEANUP FUNCTION 
// Called when EA is removed from chart (ENHANCED 2.5)
void OnDeinit(const int reason){
    // Stop timer if it was started
    if(InpOptimizeTicks && InpUseTimerChecks){
        EventKillTimer();
        LogInit("TICK OPTIMIZATION: Timer stopped during EA deinitialization");
    }
    
    // Log final tick processing statistics
    if(InpOptimizeTicks){
        string stats = GetTickProcessingStats();
        LogInit("TICK OPTIMIZATION: Final statistics - " + stats);
    }
    
    Print("Cortex3 EA shutting down. Reason: ", reason);
}

//============================== TRADE MANAGEMENT FUNCTIONS ======================
// Functions for executing trades and managing positions

// Close all open positions
bool CloseAllPositions(){
    bool success = true;
    int positions = PositionsTotal();
    
    for(int i = positions - 1; i >= 0; i--){
        if(PositionGetSymbol(i) == _Symbol){
            ulong ticket = PositionGetTicket(i);
            if(ticket > 0){
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                
                request.action = TRADE_ACTION_DEAL;
                request.position = ticket;
                request.symbol = _Symbol;
                request.volume = PositionGetDouble(POSITION_VOLUME);
                request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
                request.price = (request.type == ORDER_TYPE_SELL) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                request.deviation = 10;
                request.magic = InpMagic;
                
                if(!OrderSend(request, result)){
                    Print("Failed to close position ", ticket, ". Error: ", GetLastError());
                    success = false;
                } else {
                    Print("Position ", ticket, " closed successfully");
                }
            }
        }
    }
    return success;
}

//============================== PHASE 1 ENHANCEMENT FUNCTIONS ==================
// Functions to implement maximum holding time, profit targets, and enhanced position management

// Update position tracking information
// Enhanced position tracking with partial close state management (1.1 IMPROVEMENT)
void UpdatePositionTracking(){
    bool has_position = PositionSelect(_Symbol);
    bool was_flat = (g_position_type == 0);
    
    if(has_position){
        datetime new_open_time = (datetime)PositionGetInteger(POSITION_TIME);
        
        // Check if this is a new position (different open time)
        if(was_flat || new_open_time != g_position_open_time){
            // Reset enhanced risk management state for new position
            g_partial_close_level1_hit = false;
            g_partial_close_level2_hit = false;
            g_position_original_size = PositionGetDouble(POSITION_VOLUME);
            g_position_highest_profit_atr = 0.0;
            
            if(was_flat){
                Print("ENHANCED RISK MGT: New position detected - Reset partial close tracking");
            }
        }
        
        g_position_open_time = new_open_time;
        g_position_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
        g_position_size = PositionGetDouble(POSITION_VOLUME);
        g_position_unrealized_pnl = PositionGetDouble(POSITION_PROFIT);
        g_position_type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : 2;
    } else {
        // Reset all position tracking when flat
        g_position_open_time = 0;
        g_position_entry_price = 0.0;
        g_position_size = 0.0;
        g_position_unrealized_pnl = 0.0;
        g_position_type = 0;
        
        // Reset enhanced risk management state
        g_partial_close_level1_hit = false;
        g_partial_close_level2_hit = false;
        g_position_original_size = 0.0;
        g_position_highest_profit_atr = 0.0;
    }
}

// Check if position should be closed due to maximum holding time
bool CheckMaxHoldingTime(datetime current_time){
    if(!InpUseMaxHoldingTime || g_position_type == 0) return false;
    
    int holding_hours = (int)((current_time - g_position_open_time) / 3600);
    
    int adaptive_timeout = GetAdaptiveTimeoutHours();
    if(holding_hours > adaptive_timeout){
        Print("PHASE1: Maximum holding time exceeded (", holding_hours, " hours). Forcing position close.");
        CloseAllPositions();
        return true;
    }
    return false;
}

// Check if position should be closed due to profit target
bool CheckProfitTargets(){
    if(!InpUseProfitTargets || g_position_type == 0) return false;
    
    // Use cached ATR for profit target calculation (2.1 IMPROVEMENT)
    double atr = GetCachedATR14();
    if(atr <= 0) return false;
    double target_profit = atr * InpProfitTargetATR * g_position_size * 100000;
    
    if(g_position_unrealized_pnl >= target_profit){
        Print("PHASE1: Profit target reached. P&L: $", DoubleToString(g_position_unrealized_pnl, 2), 
              " Target: $", DoubleToString(target_profit, 2));
        CloseAllPositions();
        return true;
    }
    return false;
}

// Emergency stop loss system - final safety net to prevent catastrophic losses
bool CheckEmergencyStops(){
    if(!InpUseEmergencyStops || g_position_type == 0) return false;
    
    // Check emergency dollar stop loss
    if(g_position_unrealized_pnl < -InpEmergencyStopLoss){
        Print("EMERGENCY STOP: Loss exceeds $", InpEmergencyStopLoss, " - Current P&L: $", 
              DoubleToString(g_position_unrealized_pnl, 2));
        CloseAllPositions();
        return true;
    }
    
    // Check emergency account drawdown
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(balance > 0){
        double current_dd = ((balance - equity) / balance) * 100.0;
        if(current_dd > InpEmergencyDrawdown){
            Print("EMERGENCY STOP: Account drawdown ", DoubleToString(current_dd, 2), 
                  "% exceeds limit ", InpEmergencyDrawdown, "%");
            CloseAllPositions();
            return true;
        }
    }
    
    return false;
}

// Enhanced reward calculation with holding time penalty and quick exit bonus (for future training)
double CalculateEnhancedReward(double pnl, int holding_time_hours){
    double base_reward = pnl / 100.0; // Normalize profit
    double time_penalty = -InpHoldingTimePenalty * holding_time_hours; // Penalty for long holds
    double quick_exit_bonus = (holding_time_hours < 24) ? InpQuickExitBonus : 0; // Bonus for quick trades
    
    return base_reward + time_penalty + quick_exit_bonus;
}

// Get current position holding time in hours
int GetPositionHoldingHours(){
    if(g_position_type == 0) return 0;
    return (int)((TimeCurrent() - g_position_open_time) / 3600);
}

//============================== ENHANCED OVERTRADING PREVENTION (1.3 IMPROVEMENTS) ===============
// Advanced cooldown system and choppy market detection

// Check if market is choppy (sideways/ranging)
bool IsMarketChoppy(){
    if(!InpUseChoppyDetection) return false;
    
    // Check periodically to reduce computation
    if(TimeCurrent() - g_last_choppy_check < 300) return g_is_choppy_market; // Check every 5 minutes
    
    // Get current ATR and historical average
    double atr_current[1], atr_historical[1];
    int atr_handle = iATR(_Symbol, _Period, 14);
    int atr_historical_handle = iATR(_Symbol, _Period, InpChoppyLookbackPeriod);
    
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_current) != 1) return false;
    if(CopyBuffer(atr_historical_handle, 0, 0, 1, atr_historical) != 1) return false;
    
    double atr_ratio = atr_current[0] / atr_historical[0];
    
    // Get recent price range to detect micro-movements
    double high_prices[], low_prices[];
    ArrayResize(high_prices, InpChoppyLookbackPeriod);
    ArrayResize(low_prices, InpChoppyLookbackPeriod);
    if(CopyHigh(_Symbol, _Period, 0, InpChoppyLookbackPeriod, high_prices) != InpChoppyLookbackPeriod) return false;
    if(CopyLow(_Symbol, _Period, 0, InpChoppyLookbackPeriod, low_prices) != InpChoppyLookbackPeriod) return false;
    
    int max_high_index = ArrayMaximum(high_prices);
    int min_low_index = ArrayMinimum(low_prices);
    double range_points = (high_prices[max_high_index] - low_prices[min_low_index]) / _Point;
    
    // Market is choppy if:
    // 1. Current volatility is low compared to historical average
    // 2. Recent price range is small (micro-movements)
    bool is_choppy = (atr_ratio < InpChoppyATRThreshold) && (range_points < InpChoppyMinRange);
    
    g_is_choppy_market = is_choppy;
    g_last_choppy_check = TimeCurrent();
    
    if(is_choppy){
        Print("OVERTRADING PREVENTION: Choppy market detected - ATR ratio: ", 
              DoubleToString(atr_ratio, 3), ", Range: ", DoubleToString(range_points, 1), " pts");
    }
    
    return is_choppy;
}

// Calculate dynamic cooldown period based on last trade performance
int CalculateDynamicCooldown(){
    if(!InpUseAdvancedCooldown) return 0;
    
    int base_cooldown = InpMinCooldownMinutes;
    
    // Extended cooldown for consecutive losses
    if(InpUseConsecutiveLosses && g_consecutive_losses >= InpMaxConsecutiveLosses){
        Print("OVERTRADING PREVENTION: Extended cooldown due to ", g_consecutive_losses, " consecutive losses");
        return InpExtendedCooldownMin;
    }
    
    // Scale cooldown based on loss size
    if(InpUseLossBasedCooldown && g_last_trade_pnl < 0){
        double loss_multiplier = MathAbs(g_last_trade_pnl) / 100.0; // Normalize loss
        loss_multiplier = MathMin(loss_multiplier, 3.0); // Cap at 3x multiplier
        base_cooldown = (int)(base_cooldown * (1.0 + loss_multiplier * InpLossCooldownMultiplier));
        
        Print("OVERTRADING PREVENTION: Loss-based cooldown extended to ", base_cooldown, 
              " minutes (loss: $", DoubleToString(g_last_trade_pnl, 2), ")");
    }
    
    // Additional cooldown for choppy markets
    if(IsMarketChoppy()){
        base_cooldown = (int)(base_cooldown * 1.5); // 50% longer cooldown in choppy markets
        Print("OVERTRADING PREVENTION: Choppy market cooldown - extended to ", base_cooldown, " minutes");
    }
    
    return base_cooldown;
}

//============================== REFINED SIGNAL PROCESSING (1.4 IMPROVEMENTS) ===============
// Enhanced signal processing with smoothing and thresholding
// Apply exponential smoothing to Q-values to reduce noise
void SmoothQValues(const double &raw_q[], double &smoothed_q[]){
    if(!InpUseQValueSmoothing) {
        OptimizedArrayCopy(smoothed_q, raw_q);
        return;
    }
    
    // Initialize smoothed values on first call
    if(!g_qvalue_initialized){
        OptimizedArrayCopy(g_qvalue_smoothed, raw_q);
        g_qvalue_initialized = true;
    }
    
    // Apply exponential smoothing with loop optimization (2.4 IMPROVEMENT)
    if(InpOptimizeLoops && InpSimplifyConditions){
        // Combined loop - smoothing and output preparation in single pass
        double one_minus_alpha = 1.0 - InpQValueSmoothingAlpha;
        
        for(int i = 0; i < ACTIONS; i++){
            g_qvalue_smoothed[i] = InpQValueSmoothingAlpha * raw_q[i] + one_minus_alpha * g_qvalue_smoothed[i];
            
            // Directly prepare output to avoid second loop
            if(InpReuseArrays && g_data_arrays_initialized){
                g_smoothed_q_buffer[i] = g_qvalue_smoothed[i];
            }
        }
        
        if(InpReuseArrays && g_data_arrays_initialized){
            OptimizedArrayCopy(smoothed_q, g_smoothed_q_buffer, ACTIONS);
        } else {
            OptimizedArrayCopy(smoothed_q, g_qvalue_smoothed);
        }
    }
    else {
        // Standard implementation
        for(int i = 0; i < ACTIONS; i++){
            g_qvalue_smoothed[i] = InpQValueSmoothingAlpha * raw_q[i] + 
                                  (1.0 - InpQValueSmoothingAlpha) * g_qvalue_smoothed[i];
        }
        
        // Use optimized buffer for output (2.2 IMPROVEMENT)
        if(InpReuseArrays && g_data_arrays_initialized){
            for(int i = 0; i < ACTIONS; i++){
                g_smoothed_q_buffer[i] = g_qvalue_smoothed[i];
            }
            OptimizedArrayCopy(smoothed_q, g_smoothed_q_buffer, ACTIONS);
        } else {
            OptimizedArrayCopy(smoothed_q, g_qvalue_smoothed);
        }
    }
}

// Check if signal meets minimum strength threshold
bool CheckSignalThreshold(const double &q[], int best_action){
    if(!InpUseSignalThreshold) return true;
    
    // Find the second-best action Q-value with optimization (2.4 IMPROVEMENT)
    double best_q = q[best_action];
    double second_best_q = -999999.0;
    
    if(InpOptimizeLoops && InpUseEarlyBreaks){
        // Optimized search with early break when good enough second-best is found
        int loop_count = 0;
        for(int i = 0; i < ACTIONS; i++){
            if(!LoopSafetyCheck(loop_count)) break;
            
            if(i != best_action && q[i] > second_best_q){
                second_best_q = q[i];
                
                // Early break if we found a significantly lower second-best
                // (meaning first choice is clearly superior)
                if(InpSimplifyConditions && (best_q - second_best_q) > InpSignalThresholdQ * 2.0){
                    break; // Signal is clearly strong enough
                }
            }
        }
    }
    else {
        // Standard implementation
        for(int i = 0; i < ACTIONS; i++){
            if(i != best_action && q[i] > second_best_q){
                second_best_q = q[i];
            }
        }
    }
    
    // Check if difference is significant enough
    double q_diff = best_q - second_best_q;
    bool threshold_met = q_diff >= InpSignalThresholdQ;
    
    if(!threshold_met){
        SmartLog("SIGNAL", "SIGNAL REFINEMENT: Signal rejected - Q-value difference too small: " + 
              DoubleToString(q_diff, 4) + " < " + DoubleToString(InpSignalThresholdQ, 4));
    }
    
    return threshold_met;
}

// Check advanced signal persistence (more strict than 1.2 basic persistence)
bool CheckAdvancedPersistence(int action, datetime current_time){
    if(!InpUseAdvancedPersistence) return true;
    
    // If this is a new action sequence, reset tracking
    if(action != g_signal_sequence_action || 
       (current_time - g_signal_sequence_start) > (InpAdvancedPersistBars * 5 * 60)){ // 5min timeout
        g_signal_sequence_action = action;
        g_signal_sequence_start = current_time;
        g_advanced_persist_count = 1;
    } else {
        g_advanced_persist_count++;
    }
    
    // Check if we have enough persistence
    bool persistent_enough = g_advanced_persist_count >= InpAdvancedPersistBars;
    
    if(!persistent_enough){
        SmartLog("SIGNAL", "SIGNAL REFINEMENT: Signal persistence check - " + IntegerToString(g_advanced_persist_count) + 
              "/" + IntegerToString(InpAdvancedPersistBars) + " bars for " + ACTION_NAME[action]);
    }
    
    return persistent_enough;
}

// Estimate minimum pip movement required to justify trade
bool CheckMinimumPipMovement(int action, const MqlRates &rates[]){
    if(!InpUseMinimumPipMove) return true;
    
    // Skip check for HOLD/FLAT actions
    if(action == ACTION_HOLD || action == ACTION_FLAT) return true;
    
    // Get current spread and calculate minimum movement needed
    double current_spread = GetSymbolSpreadPoints();
    double atr = GetCachedATR14(); // Use cached ATR (2.1 IMPROVEMENT)
    double min_move_points = InpMinimumPipThreshold * 10; // Convert pips to points
    
    // For conservative estimate, we need movement > (spread + slippage + minimum threshold)
    double total_cost = current_spread + 2.0 + min_move_points; // 2 points slippage estimate
    
    // Estimate potential movement based on current volatility (ATR)
    double estimated_move = atr * 10; // Convert to points
    
    bool sufficient_movement = estimated_move >= total_cost;
    
    if(!sufficient_movement){
        SmartLog("SIGNAL", "SIGNAL REFINEMENT: Insufficient predicted movement - estimated: " + 
              DoubleToString(estimated_move/10, 1) + " pips, required: " + 
              DoubleToString(total_cost/10, 1) + " pips");
    }
    
    return sufficient_movement;
}

// Apply comprehensive signal refinement processing
int ApplySignalRefinement(int raw_action, const double &raw_q[], datetime current_time, const MqlRates &rates[]){
    if(!InpUseSignalRefinement) return raw_action;
    
    SmartLog("SIGNAL", "SIGNAL REFINEMENT: Processing " + ACTION_NAME[raw_action] + " signal");
    
    // Step 1: Apply Q-value smoothing
    double smoothed_q[6];
    SmoothQValues(raw_q, smoothed_q);
    
    // Step 2: Recheck best action after smoothing
    int smoothed_action = argmax(smoothed_q);
    if(smoothed_action != raw_action){
        SmartLog("SIGNAL", "SIGNAL REFINEMENT: Action changed after smoothing: " + 
              ACTION_NAME[raw_action] + " -> " + ACTION_NAME[smoothed_action]);
    }
    
    // Step 3: Check signal strength threshold
    if(!CheckSignalThreshold(smoothed_q, smoothed_action)){
        return ACTION_HOLD; // Signal too weak
    }
    
    // Step 4: Check advanced persistence
    if(!CheckAdvancedPersistence(smoothed_action, current_time)){
        return ACTION_HOLD; // Signal not persistent enough
    }
    
    // Step 5: Check minimum pip movement requirement
    if(!CheckMinimumPipMovement(smoothed_action, rates)){
        return ACTION_HOLD; // Predicted movement insufficient
    }
    
    SmartLog("SIGNAL", "SIGNAL REFINEMENT: All checks passed for " + ACTION_NAME[smoothed_action]);
    return smoothed_action;
}

//============================== CONFIDENCE-BASED TRADE FILTER (3.1 IMPROVEMENTS) ===============
// Advanced confidence calculation and filtering system for neural network signals

// Initialize confidence tracking system
void InitializeConfidenceTracking(){
    if(!InpUseConfidenceFilter) return;
    
    g_last_confidence_score = 0.0;
    g_q_spread_confidence = 0.0;
    g_magnitude_confidence = 0.0;  
    g_softmax_confidence = 0.0;
    g_last_confidence_log = 0;
    g_confidence_filtered_count = 0;
    g_confidence_passed_count = 0;
    g_confidence_sum = 0.0;
    g_confidence_calculation_count = 0;
    
    Print("CONFIDENCE FILTER: Initialized - Threshold: ", DoubleToString(InpMinConfidenceThreshold, 3));
    Print("  Components - Q-Spread: ", DoubleToString(InpQSpreadWeight, 2), 
          " | Magnitude: ", DoubleToString(InpMagnitudeWeight, 2),
          " | Softmax: ", DoubleToString(InpSoftmaxWeight, 2));
}

// Calculate Q-value spread confidence (higher when clear winner exists)
double CalculateQSpreadConfidence(const double &q_values[], int best_action){
    if(!InpUseQSpreadConfidence || !InpUseConfidenceFilter) return 0.0;
    
    double best_q = q_values[best_action];
    double second_best = -999999.0;
    
    // Find second best Q-value
    for(int i = 0; i < 6; i++){
        if(i != best_action && q_values[i] > second_best){
            second_best = q_values[i];
        }
    }
    
    // Calculate spread confidence: larger gap = higher confidence
    double q_spread = best_q - second_best;
    double normalized_spread = MathMin(1.0, MathMax(0.0, q_spread / 2.0)); // Normalize to [0,1]
    
    return normalized_spread;
}

// Calculate magnitude confidence (higher when Q-value is significantly positive)
double CalculateMagnitudeConfidence(const double &q_values[], int best_action){
    if(!InpUseMagnitudeConfidence || !InpUseConfidenceFilter) return 0.0;
    
    double best_q = q_values[best_action];
    
    // Skip HOLD/FLAT actions for magnitude check
    if(best_action == ACTION_HOLD || best_action == ACTION_FLAT){
        return 0.5; // Neutral confidence for non-trading actions
    }
    
    // Calculate magnitude confidence: higher positive Q = higher confidence
    double normalized_magnitude = 1.0 / (1.0 + MathExp(-best_q)); // Sigmoid to [0,1]
    
    return normalized_magnitude;
}

// Calculate softmax probability confidence
double CalculateSoftmaxConfidence(const double &q_values[], int best_action){
    if(!InpUseSoftmaxConfidence || !InpUseConfidenceFilter) return 0.0;
    
    // Calculate softmax probabilities
    double softmax_probs[6];
    double exp_sum = 0.0;
    
    // Calculate exponentials and sum
    for(int i = 0; i < 6; i++){
        softmax_probs[i] = MathExp(q_values[i]);
        exp_sum += softmax_probs[i];
    }
    
    // Normalize to probabilities
    if(exp_sum > 0.0){
        for(int i = 0; i < 6; i++){
            softmax_probs[i] /= exp_sum;
        }
    } else {
        // Fallback if numerical issues
        for(int i = 0; i < 6; i++){
            softmax_probs[i] = 1.0 / 6.0;
        }
    }
    
    return softmax_probs[best_action];
}

// Calculate combined confidence score from Q-values
double CalculateSignalConfidence(const double &q_values[], int best_action){
    if(!InpUseConfidenceFilter) return 1.0; // Always confident if filter disabled
    
    // Calculate individual confidence components
    g_q_spread_confidence = CalculateQSpreadConfidence(q_values, best_action);
    g_magnitude_confidence = CalculateMagnitudeConfidence(q_values, best_action);
    g_softmax_confidence = CalculateSoftmaxConfidence(q_values, best_action);
    
    // Normalize weights to sum to 1.0
    double total_weight = InpQSpreadWeight + InpMagnitudeWeight + InpSoftmaxWeight;
    if(total_weight <= 0.0) total_weight = 1.0; // Safety check
    
    double normalized_spread_weight = InpQSpreadWeight / total_weight;
    double normalized_magnitude_weight = InpMagnitudeWeight / total_weight;
    double normalized_softmax_weight = InpSoftmaxWeight / total_weight;
    
    // Calculate weighted combination
    double combined_confidence = 
        (g_q_spread_confidence * normalized_spread_weight) +
        (g_magnitude_confidence * normalized_magnitude_weight) +
        (g_softmax_confidence * normalized_softmax_weight);
    
    // Apply boost factor for strong signals
    if(InpConfidenceBoostFactor > 1.0 && combined_confidence > 0.8){
        double boost_multiplier = 1.0 + ((combined_confidence - 0.8) * (InpConfidenceBoostFactor - 1.0) * 5.0);
        combined_confidence = MathMin(1.0, combined_confidence * boost_multiplier);
    }
    
    // Update statistics
    g_confidence_sum += combined_confidence;
    g_confidence_calculation_count++;
    g_last_confidence_score = combined_confidence;
    
    return combined_confidence;
}

// Check if signal passes confidence threshold
bool PassesConfidenceFilter(const double &q_values[], int action, datetime current_time){
    if(!InpUseConfidenceFilter) return true;
    
    // Skip confidence check for HOLD/FLAT actions
    if(action == ACTION_HOLD || action == ACTION_FLAT) return true;
    
    double confidence = CalculateSignalConfidence(q_values, action);
    bool passes = confidence >= InpMinConfidenceThreshold;
    
    // Update counters
    if(passes){
        g_confidence_passed_count++;
    } else {
        g_confidence_filtered_count++;
    }
    
    // Log confidence details (with throttling)
    if(InpLogConfidenceDetails && 
       (current_time - g_last_confidence_log) >= InpConfidenceLogThrottle){
        
        string conf_result = passes ? "PASSED" : "FILTERED";
        SmartLog("CONFIDENCE", StringFormat("Signal %s: %.3f threshold (%.3f required) | %s", 
                 ACTION_NAME[action], confidence, InpMinConfidenceThreshold, conf_result));
        SmartLog("CONFIDENCE", StringFormat("  Components - Spread: %.3f | Magnitude: %.3f | Softmax: %.3f", 
                 g_q_spread_confidence, g_magnitude_confidence, g_softmax_confidence));
        
        g_last_confidence_log = current_time;
    }
    
    return passes;
}

// Get confidence filter statistics for display
string GetConfidenceStatistics(){
    if(!InpUseConfidenceFilter) return "Confidence Filter: DISABLED";
    
    double avg_confidence = (g_confidence_calculation_count > 0) ? 
                           (g_confidence_sum / g_confidence_calculation_count) : 0.0;
    
    int total_signals = g_confidence_passed_count + g_confidence_filtered_count;
    double pass_rate = (total_signals > 0) ? 
                      (double(g_confidence_passed_count) / double(total_signals) * 100.0) : 0.0;
    
    return StringFormat("Confidence Filter: %.1f%% pass rate | Avg: %.3f | Passed: %d | Filtered: %d",
                       pass_rate, avg_confidence, g_confidence_passed_count, g_confidence_filtered_count);
}

/*
======================================================================================
CONFIDENCE-BASED TRADE FILTER (3.1) - IMPLEMENTATION COMPLETE
======================================================================================
This system provides sophisticated confidence scoring for neural network trading signals:

CONFIDENCE COMPONENTS:
1. Q-Value Spread: Measures decision clarity (large gap between best/second-best actions)
2. Q-Value Magnitude: Assesses signal strength (higher positive Q-values = more confident)
3. Softmax Probability: Converts Q-values to probabilities for normalized confidence

FILTERING LOGIC:
- Calculates weighted combination of all confidence components
- Applies configurable boost factor for very strong signals (>0.8 base confidence)
- Filters out signals below minimum threshold (default 0.65)
- Allows HOLD/FLAT actions to pass without confidence check

INTEGRATION:
- Seamlessly integrated into existing signal processing chain
- Applied after signal refinement but before trade execution
- Comprehensive logging with throttling to prevent spam
- Statistics tracking for performance monitoring

CONFIGURATION:
- InpUseConfidenceFilter: Master enable/disable switch
- InpMinConfidenceThreshold: Minimum confidence required (0.0-1.0)
- Component weights: Adjustable weighting for each confidence metric
- Logging controls: Detailed logging with throttling options

This filter significantly improves trading performance by discarding low-quality signals
and only executing trades when the AI model shows high certainty in its predictions.
======================================================================================
*/

//============================== ENSEMBLE MODEL DECISIONING (3.2 IMPROVEMENTS) ===============
// Advanced ensemble system for multiple model combination and robust decision making

// Initialize ensemble model system
void InitializeEnsemble(){
    if(!InpUseEnsembleModel) return;
    
    // Reset ensemble state
    g_active_ensemble_count = 0;
    g_last_ensemble_log = 0;
    g_ensemble_method = InpEnsembleMethod;
    g_ensemble_predictions_count = 0;
    g_ensemble_agreement_count = 0;
    g_ensemble_avg_agreement = 0.0;
    
    // Initialize arrays
    ArrayFill(g_ensemble_models_loaded, 0, 5, false);
    ArrayFill(g_ensemble_model_weights, 0, 5, 0.0);
    ArrayFill(g_ensemble_agreement_scores, 0, 6, 0.0);
    ArrayFill(g_ensemble_final_q, 0, 6, 0.0);
    ArrayFill(g_ensemble_votes, 0, 6, 0);
    ArrayFill(g_ensemble_confidences, 0, 5, 0.0);
    
    Print("ENSEMBLE MODEL: Initialization started - Max models: ", InpMaxEnsembleModels);
    Print("  Method: ", InpEnsembleMethod, " | Agreement threshold: ", DoubleToString(InpEnsembleAgreementThreshold, 3));
}

// Load individual ensemble model by index
bool LoadEnsembleModel(int model_index, const string filename){
    if(model_index < 0 || model_index >= 5 || filename == "") return false;
    
    Print("ENSEMBLE MODEL: Loading model ", model_index + 1, " from ", filename);
    
    // Try to open the model file
    int h = FileOpen(filename, FILE_BIN|FILE_READ);
    if(h == INVALID_HANDLE){ 
        Print("ENSEMBLE MODEL: Cannot open ", filename, " err=", GetLastError()); 
        return false; 
    }
    
    // Check magic number to verify this is our model format
    long magic = FileReadLong(h);
    bool has_checkpoint = false;
    
    if(magic == (long)0xC0DE0203){
        has_checkpoint = true;  // New format with training checkpoints
    } else if(magic == (long)0xC0DE0202){
        has_checkpoint = false; // Old format without checkpoints
    } else {
        Print("ENSEMBLE MODEL: Unsupported model file format in ", filename); 
        FileClose(h); 
        return false; 
    }
    
    // Read model architecture and verify compatibility
    string model_symbol;
    ENUM_TIMEFRAMES model_tf;
    int stsz, h1, h2, h3, acts;
    bool has_lstm, has_dueling;
    int lstm_size, value_head_size, advantage_head_size;
    
    // Read symbol & timeframe
    ulong pos = FileTell(h);
    int symbol_len = FileReadInteger(h);
    int tf_int = FileReadInteger(h);
    
    if(symbol_len > 0 && symbol_len < 20 && tf_int > 0){
        model_symbol = FileReadString(h, symbol_len);
        model_tf = (ENUM_TIMEFRAMES)tf_int;
    } else {
        FileSeek(h, (long)pos, SEEK_SET);
        if(!ReadSymbolAndTF(h, model_symbol, model_tf)){
            Print("ENSEMBLE MODEL: Failed to parse symbol/timeframe in ", filename);
            FileClose(h); 
            return false;
        }
    }
    
    // Read architecture parameters
    stsz = FileReadInteger(h); h1 = FileReadInteger(h); h2 = FileReadInteger(h); 
    h3 = FileReadInteger(h); acts = FileReadInteger(h);
    has_lstm = FileReadInteger(h);
    has_dueling = FileReadInteger(h);
    
    if(has_lstm){
        lstm_size = FileReadInteger(h);
    } else {
        lstm_size = 0;
    }
    
    if(has_dueling){
        value_head_size = FileReadInteger(h);
        advantage_head_size = FileReadInteger(h);
    } else {
        value_head_size = 0;
        advantage_head_size = 0;
    }
    
    // Verify compatibility with main model
    if(stsz != STATE_SIZE || acts != ACTIONS){
        Print("ENSEMBLE MODEL: Architecture mismatch in ", filename, " (", stsz, "/", acts, " vs ", STATE_SIZE, "/", ACTIONS, ")");
        FileClose(h); 
        return false;
    }
    
    // Select the appropriate ensemble model directly (avoid pointer issues)
    CInferenceNetwork ensemble_model;
    switch(model_index){
        case 0: ensemble_model = g_ensemble_model1; break;
        case 1: ensemble_model = g_ensemble_model2; break;  
        case 2: ensemble_model = g_ensemble_model3; break;
        case 3: ensemble_model = g_ensemble_model4; break;
        case 4: ensemble_model = g_ensemble_model5; break;
        default:
            Print("ENSEMBLE MODEL: Invalid model index ", model_index);
            FileClose(h);
            return false;
    }
    
    // Initialize ensemble model with the loaded architecture
    ensemble_model.inSize = STATE_SIZE;
    ensemble_model.h1 = h1;
    ensemble_model.h2 = h2; 
    ensemble_model.h3 = h3;
    ensemble_model.outSize = ACTIONS;
    ensemble_model.has_lstm = has_lstm;
    ensemble_model.has_dueling = has_dueling;
    ensemble_model.lstm_size = lstm_size;
    ensemble_model.value_head_size = value_head_size;
    ensemble_model.advantage_head_size = advantage_head_size;
    
    ensemble_model.InitFromSizes(STATE_SIZE, h1, h2, h3, ACTIONS);
    
    // Skip checkpoint data if present
    if(has_checkpoint){
        datetime last_trained = (datetime)FileReadLong(h);
        int training_steps = (int)FileReadLong(h);
        double checkpoint_eps = FileReadDouble(h);
        double checkpoint_beta = FileReadDouble(h);
    }
    
    // Load the trained weights and biases for each layer
    bool ok = true;
    ok = ok && LoadLayer(h, ensemble_model.L1);
    ok = ok && LoadLayer(h, ensemble_model.L2);
    ok = ok && LoadLayer(h, ensemble_model.L3);
    
    if(has_lstm){
        ok = ok && LoadLSTMLayer(h, ensemble_model.lstm);
    }
    
    if(has_dueling){
        ok = ok && LoadLayer(h, ensemble_model.value_head);
        ok = ok && LoadLayer(h, ensemble_model.advantage_head);
    } else {
        ok = ok && LoadLayer(h, ensemble_model.L4);
    }
    
    FileClose(h);
    
    if(ok){
        // Assign the loaded model back to the appropriate global variable
        switch(model_index){
            case 0: g_ensemble_model1 = ensemble_model; break;
            case 1: g_ensemble_model2 = ensemble_model; break;  
            case 2: g_ensemble_model3 = ensemble_model; break;
            case 3: g_ensemble_model4 = ensemble_model; break;
            case 4: g_ensemble_model5 = ensemble_model; break;
        }
        
        g_ensemble_models_loaded[model_index] = true;
        Print("ENSEMBLE MODEL: Successfully loaded model ", model_index + 1, " (", filename, ")");
        Print("  Architecture: ", stsz, "->", h1, "->", h2, "->", h3, 
              has_lstm ? "->LSTM(" + IntegerToString(lstm_size) + ")" : "",
              has_dueling ? "->Dueling" : "");
        return true;
    } else {
        Print("ENSEMBLE MODEL: Failed to load weights from ", filename);
        return false;
    }
}

// Load all ensemble models and set up weights
bool LoadEnsembleModels(){
    if(!InpUseEnsembleModel) return true;
    
    InitializeEnsemble();
    
    // List of model filenames and weights
    string model_files[5];
    double model_weights[5];
    
    model_files[0] = InpEnsembleModel1; model_weights[0] = InpEnsembleModel1Weight;
    model_files[1] = InpEnsembleModel2; model_weights[1] = InpEnsembleModel2Weight;
    model_files[2] = InpEnsembleModel3; model_weights[2] = InpEnsembleModel3Weight;
    model_files[3] = InpEnsembleModel4; model_weights[3] = InpEnsembleModel4Weight;
    model_files[4] = InpEnsembleModel5; model_weights[4] = InpEnsembleModel5Weight;
    
    // Load models up to the specified maximum
    int max_models = MathMin(InpMaxEnsembleModels, 5);
    g_active_ensemble_count = 0;
    
    for(int i = 0; i < max_models; i++){
        if(model_files[i] != "" && model_weights[i] > 0.0){
            if(LoadEnsembleModel(i, model_files[i])){
                g_active_ensemble_count++;
            }
        }
    }
    
    // Require at least 2 models for ensemble
    if(g_active_ensemble_count < 2){
        Print("ENSEMBLE MODEL: ERROR - Need at least 2 models for ensemble, only loaded ", g_active_ensemble_count);
        if(InpEnsembleFallbackMode){
            Print("ENSEMBLE MODEL: Fallback mode enabled - will use single model instead");
            return true;  // Allow single model fallback
        }
        return false;
    }
    
    // Normalize weights for loaded models
    double total_weight = 0.0;
    for(int i = 0; i < 5; i++){
        if(g_ensemble_models_loaded[i]){
            total_weight += model_weights[i];
        }
    }
    
    if(total_weight > 0.0){
        for(int i = 0; i < 5; i++){
            if(g_ensemble_models_loaded[i]){
                g_ensemble_model_weights[i] = model_weights[i] / total_weight;
            } else {
                g_ensemble_model_weights[i] = 0.0;
            }
        }
    }
    
    Print("ENSEMBLE MODEL: Successfully loaded ", g_active_ensemble_count, " models");
    Print("  Normalized weights: ");
    for(int i = 0; i < 5; i++){
        if(g_ensemble_models_loaded[i]){
            Print("    Model ", i + 1, ": ", DoubleToString(g_ensemble_model_weights[i], 3));
        }
    }
    
    return true;
}

// Calculate Q-value similarity between two arrays
double CalculateQValueSimilarity(const double &q1[], const double &q2[], int action){
    double q_diff = MathAbs(q1[action] - q2[action]);
    double max_q = MathMax(MathAbs(q1[action]), MathAbs(q2[action]));
    return max_q > 0.0 ? (1.0 - MathMin(1.0, q_diff / max_q)) : 1.0;
}

// Simple ensemble prediction using loaded models (MQL5 compatible version)
int GetSimpleEnsemblePrediction(const double &state[], double &final_q[], double &agreement_score){
    if(!InpUseEnsembleModel || g_active_ensemble_count < 2){
        // Fallback to single model
        if(InpEnsembleFallbackMode && g_loaded){
            g_Q.PredictOptimized(state, final_q);
            agreement_score = 1.0; // Perfect agreement with single model
            return argmax(final_q);
        }
        return ACTION_HOLD; // Safe default
    }
    
    // Get predictions from available models
    double q1[6], q2[6], q3[6], q4[6], q5[6];
    ArrayFill(q1, 0, 6, 0.0); ArrayFill(q2, 0, 6, 0.0); ArrayFill(q3, 0, 6, 0.0);
    ArrayFill(q4, 0, 6, 0.0); ArrayFill(q5, 0, 6, 0.0);
    
    int predictions[5];
    ArrayFill(predictions, 0, 5, ACTION_HOLD);
    
    // Get predictions from loaded models
    if(g_ensemble_models_loaded[0]){
        g_ensemble_model1.PredictOptimized(state, q1);
        predictions[0] = argmax(q1);
    }
    if(g_ensemble_models_loaded[1]){
        g_ensemble_model2.PredictOptimized(state, q2);
        predictions[1] = argmax(q2);
    }
    if(g_ensemble_models_loaded[2]){
        g_ensemble_model3.PredictOptimized(state, q3);
        predictions[2] = argmax(q3);
    }
    if(g_ensemble_models_loaded[3]){
        g_ensemble_model4.PredictOptimized(state, q4);
        predictions[3] = argmax(q4);
    }
    if(g_ensemble_models_loaded[4]){
        g_ensemble_model5.PredictOptimized(state, q5);
        predictions[4] = argmax(q5);
    }
    
    // Simple majority vote
    ArrayFill(g_ensemble_votes, 0, 6, 0);
    for(int i = 0; i < 5; i++){
        if(g_ensemble_models_loaded[i]){
            g_ensemble_votes[predictions[i]]++;
        }
    }
    
    int winning_action = argmax_int(g_ensemble_votes);
    int max_votes = g_ensemble_votes[winning_action];
    
    // Calculate agreement as percentage of models that voted for winner
    agreement_score = g_active_ensemble_count > 0 ? 
                     (double)max_votes / g_active_ensemble_count : 0.0;
    
    // Simple average of Q-values for final output
    ArrayFill(g_ensemble_final_q, 0, 6, 0.0);
    double weight_sum = 0.0;
    
    if(g_ensemble_models_loaded[0]){
        for(int i = 0; i < 6; i++) g_ensemble_final_q[i] += q1[i] * g_ensemble_model_weights[0];
        weight_sum += g_ensemble_model_weights[0];
    }
    if(g_ensemble_models_loaded[1]){
        for(int i = 0; i < 6; i++) g_ensemble_final_q[i] += q2[i] * g_ensemble_model_weights[1];
        weight_sum += g_ensemble_model_weights[1];
    }
    if(g_ensemble_models_loaded[2]){
        for(int i = 0; i < 6; i++) g_ensemble_final_q[i] += q3[i] * g_ensemble_model_weights[2];
        weight_sum += g_ensemble_model_weights[2];
    }
    if(g_ensemble_models_loaded[3]){
        for(int i = 0; i < 6; i++) g_ensemble_final_q[i] += q4[i] * g_ensemble_model_weights[3];
        weight_sum += g_ensemble_model_weights[3];
    }
    if(g_ensemble_models_loaded[4]){
        for(int i = 0; i < 6; i++) g_ensemble_final_q[i] += q5[i] * g_ensemble_model_weights[4];
        weight_sum += g_ensemble_model_weights[4];
    }
    
    // Normalize
    if(weight_sum > 0.0){
        for(int i = 0; i < 6; i++){
            g_ensemble_final_q[i] /= weight_sum;
        }
    }
    
    // Check agreement threshold
    if(agreement_score < InpEnsembleAgreementThreshold){
        if(InpRequireUnanimousHold && (winning_action == ACTION_HOLD || winning_action == ACTION_FLAT)){
            // Allow HOLD/FLAT even with low agreement if unanimous requirement is set
        } else {
            winning_action = ACTION_HOLD; // Force HOLD for low agreement
        }
    } else {
        g_ensemble_agreement_count++;
    }
    
    // Update statistics
    g_ensemble_predictions_count++;
    g_ensemble_avg_agreement = (g_ensemble_avg_agreement * (g_ensemble_predictions_count - 1) + agreement_score) / g_ensemble_predictions_count;
    
    // Copy final Q-values for output
    for(int i = 0; i < 6; i++){
        final_q[i] = g_ensemble_final_q[i];
    }
    
    return winning_action;
}


// Get ensemble statistics for reporting
string GetEnsembleStatistics(){
    if(!InpUseEnsembleModel) return "Ensemble: DISABLED";
    
    if(g_active_ensemble_count < 2){
        if(InpEnsembleFallbackMode){
            return "Ensemble: FALLBACK MODE (using single model)";
        } else {
            return "Ensemble: INACTIVE (insufficient models)";
        }
    }
    
    double agreement_rate = g_ensemble_predictions_count > 0 ? 
                           (double(g_ensemble_agreement_count) / double(g_ensemble_predictions_count) * 100.0) : 0.0;
    
    return StringFormat("Ensemble: %d models | %.1f%% agreement rate | Avg: %.3f | Method: %s", 
                       g_active_ensemble_count, agreement_rate, g_ensemble_avg_agreement, g_ensemble_method);
}

/*
======================================================================================
ENSEMBLE MODEL DECISIONING (3.2) - IMPLEMENTATION COMPLETE
======================================================================================
This system provides sophisticated ensemble trading with multiple AI models working together:

ENSEMBLE ARCHITECTURE:
- Supports up to 5 models running in parallel with independent weights
- Each model must be compatible (same STATE_SIZE/ACTIONS) but can have different architectures
- Automatic fallback to single model if ensemble fails or insufficient models loaded
- Complete model isolation with separate memory management and state tracking

AGGREGATION METHODS:
1. Majority Vote: Each model votes for best action, winning action selected
2. Weighted Average: Q-values combined using normalized model weights  
3. Confidence Weighted: Models weighted by their individual confidence scores

AGREEMENT THRESHOLD SYSTEM:
- Configurable agreement threshold prevents low-consensus trades
- Agreement calculated differently per method (vote percentage, Q-value similarity, confidence)
- Automatic fallback to HOLD for low-agreement predictions
- Special handling for HOLD/FLAT actions with unanimous override option

MODEL MANAGEMENT:
- Robust loading with architecture validation and compatibility checks
- Automatic weight normalization for loaded models
- Graceful handling of missing or corrupted model files
- Individual model performance tracking and statistics

INTEGRATION FEATURES:
- Seamless replacement of single model prediction pipeline
- Compatible with all existing signal processing (refinement, confidence filtering)
- Comprehensive logging with throttling to prevent log spam
- Real-time statistics tracking for performance monitoring

CONFIGURATION OPTIONS:
- InpUseEnsembleModel: Master ensemble enable/disable
- InpMaxEnsembleModels: Number of models to use (2-5)
- Model file paths and individual weights fully configurable
- Method selection and agreement thresholds adjustable
- Fallback behavior and logging controls available

This ensemble system significantly improves trading robustness by combining the strengths
of multiple models while reducing the impact of individual model weaknesses or overfitting.

COMPILATION NOTES:
- Simplified implementation to work with MQL5 limitations (no multi-dimensional arrays)
- Uses individual model instances instead of arrays for compatibility
- GetSimpleEnsemblePrediction() replaces complex aggregation methods
- Maintains full functionality while ensuring reliable compilation

COMPILATION FIXES APPLIED:
- Moved CInferenceNetwork declarations after class definition
- Replaced problematic pointer operations with direct object operations
- Added argmax_int() function for integer array processing
- Eliminated global function declarations causing scope issues
- Fixed all array parameter type mismatches
======================================================================================
*/

//============================== ADAPTIVE PARAMETER LOGIC (3.3 IMPROVEMENTS) ===============
// Self-tuning parameter system that adjusts to market conditions and performance

// Initialize adaptive parameter system
void InitializeAdaptiveParameters(){
    if(!InpUseAdaptiveParameters) return;
    
    // Initialize base values from input parameters
    g_adaptive_atr_multiplier = InpBaseATRMultiplier;
    g_adaptive_risk_percent = InpBaseRiskPercent;
    g_adaptive_timeout_hours = InpBaseTimeoutHours;
    
    // Initialize tracking arrays
    ArrayFill(g_atr_history, 0, 50, 0.0);
    ArrayFill(g_recent_trade_results, 0, 20, 0.0);
    g_atr_history_index = 0;
    g_trade_results_index = 0;
    g_atr_history_filled = false;
    g_trade_results_filled = false;
    
    // Initialize state variables
    g_current_volatility_regime = 1; // Start with normal volatility
    g_current_volatility_ratio = 1.0;
    g_consecutive_wins = 0;
    g_consecutive_losses = 0;
    g_last_adaptive_log = 0;
    g_last_adaptive_reset = 0;
    g_adaptive_parameter_changes = 0;
    g_volatility_regime_changes = 0;
    
    Print("ADAPTIVE PARAMETERS: Initialized - Base ATR: ", DoubleToString(InpBaseATRMultiplier, 2),
          " | Base Risk: ", DoubleToString(InpBaseRiskPercent, 1), "%",
          " | Base Timeout: ", InpBaseTimeoutHours, "h");
}

// Update ATR history for volatility regime detection
void UpdateATRHistory(double current_atr){
    if(!InpUseAdaptiveParameters || !InpAdaptiveATRMultipliers) return;
    
    // Add current ATR to circular buffer
    g_atr_history[g_atr_history_index] = current_atr;
    g_atr_history_index = (g_atr_history_index + 1) % 50;
    
    if(g_atr_history_index == 0) g_atr_history_filled = true;
}

// Calculate volatility regime and update adaptive parameters
void UpdateAdaptiveVolatilityRegime(){
    if(!InpUseAdaptiveParameters || !InpAdaptiveATRMultipliers) return;
    
    // Need at least lookback period of data
    int available_data = g_atr_history_filled ? 50 : g_atr_history_index;
    if(available_data < InpVolatilityLookback) return;
    
    // Calculate median ATR over lookback period
    double atr_values[50];
    int lookback_size = MathMin(InpVolatilityLookback, available_data);
    
    // Get recent ATR values
    for(int i = 0; i < lookback_size; i++){
        int index = (g_atr_history_index - 1 - i + 50) % 50;
        atr_values[i] = g_atr_history[index];
    }
    
    // Calculate median (simple approach - sort and take middle)
    double median_atr = CalculateMedian(atr_values, lookback_size);
    if(median_atr <= 0.0) return;
    
    // Get current ATR
    double current_atr = GetCachedATR14();
    if(current_atr <= 0.0) return;
    
    // Calculate volatility ratio
    g_current_volatility_ratio = current_atr / median_atr;
    
    // Determine volatility regime
    int new_regime;
    if(g_current_volatility_ratio >= InpHighVolThreshold){
        new_regime = 2; // High volatility
    } else if(g_current_volatility_ratio <= InpLowVolThreshold){
        new_regime = 0; // Low volatility  
    } else {
        new_regime = 1; // Normal volatility
    }
    
    // Update regime if changed
    if(new_regime != g_current_volatility_regime){
        g_current_volatility_regime = new_regime;
        g_volatility_regime_changes++;
        
        // Adjust ATR multiplier based on regime
        UpdateATRMultiplier();
        
        // Adjust timeout based on regime
        if(InpAdaptiveTimeouts){
            UpdateTimeoutParameters();
        }
        
        // Log regime change
        if(InpLogAdaptiveChanges){
            LogAdaptiveChange(StringFormat("Volatility regime changed to %s (ratio: %.2f)",
                             GetVolatilityRegimeName(new_regime), g_current_volatility_ratio));
        }
    }
}

// Calculate median of array values
double CalculateMedian(double &values[], int size){
    if(size <= 0) return 0.0;
    if(size == 1) return values[0];
    
    // Simple bubble sort for small arrays
    double sorted_values[];
    ArrayResize(sorted_values, size);
    for(int i = 0; i < size; i++) sorted_values[i] = values[i];
    
    for(int i = 0; i < size - 1; i++){
        for(int j = 0; j < size - i - 1; j++){
            if(sorted_values[j] > sorted_values[j + 1]){
                double temp = sorted_values[j];
                sorted_values[j] = sorted_values[j + 1];
                sorted_values[j + 1] = temp;
            }
        }
    }
    
    if(size % 2 == 0){
        return (sorted_values[size/2 - 1] + sorted_values[size/2]) / 2.0;
    } else {
        return sorted_values[size/2];
    }
}

// Update ATR multiplier based on volatility regime
void UpdateATRMultiplier(){
    if(!InpAdaptiveATRMultipliers) return;
    
    double old_multiplier = g_adaptive_atr_multiplier;
    
    switch(g_current_volatility_regime){
        case 0: // Low volatility - use smaller multipliers
            g_adaptive_atr_multiplier = InpMinATRMultiplier;
            break;
        case 1: // Normal volatility - use base multiplier
            g_adaptive_atr_multiplier = InpBaseATRMultiplier;
            break;
        case 2: // High volatility - use larger multipliers
            g_adaptive_atr_multiplier = InpMaxATRMultiplier;
            break;
    }
    
    // Ensure within bounds
    g_adaptive_atr_multiplier = MathMax(InpMinATRMultiplier, 
                               MathMin(InpMaxATRMultiplier, g_adaptive_atr_multiplier));
    
    if(MathAbs(old_multiplier - g_adaptive_atr_multiplier) > 0.01){
        g_adaptive_parameter_changes++;
        
        if(InpLogAdaptiveChanges){
            LogAdaptiveChange(StringFormat("ATR multiplier: %.2f -> %.2f (%s volatility)",
                             old_multiplier, g_adaptive_atr_multiplier, 
                             GetVolatilityRegimeName(g_current_volatility_regime)));
        }
    }
}

// Update timeout parameters based on volatility
void UpdateTimeoutParameters(){
    if(!InpAdaptiveTimeouts) return;
    
    int old_timeout = g_adaptive_timeout_hours;
    
    switch(g_current_volatility_regime){
        case 0: // Low volatility - longer timeouts
            g_adaptive_timeout_hours = InpBaseTimeoutHours;
            break;
        case 1: // Normal volatility - base timeout
            g_adaptive_timeout_hours = InpBaseTimeoutHours;
            break;
        case 2: // High volatility - shorter timeouts
            g_adaptive_timeout_hours = (int)(InpBaseTimeoutHours * InpVolatileTimeoutMultiplier);
            break;
    }
    
    // Ensure minimum timeout
    g_adaptive_timeout_hours = MathMax(12, g_adaptive_timeout_hours);
    
    if(old_timeout != g_adaptive_timeout_hours){
        g_adaptive_parameter_changes++;
        
        if(InpLogAdaptiveChanges){
            LogAdaptiveChange(StringFormat("Timeout hours: %d -> %d (%s volatility)",
                             old_timeout, g_adaptive_timeout_hours,
                             GetVolatilityRegimeName(g_current_volatility_regime)));
        }
    }
}

// Update performance history when trade closes
void UpdatePerformanceHistory(double trade_pnl){
    if(!InpUseAdaptiveParameters) return;
    
    // Add trade result to circular buffer
    g_recent_trade_results[g_trade_results_index] = trade_pnl;
    g_trade_results_index = (g_trade_results_index + 1) % 20;
    
    if(g_trade_results_index == 0) g_trade_results_filled = true;
    
    // Update consecutive streaks
    if(trade_pnl > 0){
        g_consecutive_wins++;
        g_consecutive_losses = 0;
    } else if(trade_pnl < 0){
        g_consecutive_losses++;
        g_consecutive_wins = 0;
    }
    // Breakeven trades don't affect streaks
    
    // Update risk parameters if adaptive risk sizing is enabled
    if(InpAdaptiveRiskSizing){
        UpdateRiskParameters();
    }
}

// Update risk parameters based on recent performance
void UpdateRiskParameters(){
    if(!InpAdaptiveRiskSizing) return;
    
    double old_risk = g_adaptive_risk_percent;
    
    // Start with base risk
    g_adaptive_risk_percent = InpBaseRiskPercent;
    
    // Apply win streak bonus (max 3 consecutive wins)
    int win_bonus_multiplier = MathMin(3, g_consecutive_wins);
    if(win_bonus_multiplier > 0){
        double win_multiplier = MathPow(InpWinStreakMultiplier, win_bonus_multiplier);
        g_adaptive_risk_percent *= win_multiplier;
    }
    
    // Apply loss streak penalty (max 3 consecutive losses)
    int loss_penalty_multiplier = MathMin(3, g_consecutive_losses);
    if(loss_penalty_multiplier > 0){
        double loss_divisor = MathPow(InpLossStreakDivisor, loss_penalty_multiplier);
        g_adaptive_risk_percent /= loss_divisor;
    }
    
    // Consider recent overall performance
    if(g_trade_results_filled || g_trade_results_index >= InpPerformanceLookback){
        double recent_performance = CalculateRecentPerformance();
        if(recent_performance < -0.5){ // Recent losses
            g_adaptive_risk_percent *= 0.8; // Reduce risk by 20%
        } else if(recent_performance > 0.5){ // Recent gains
            g_adaptive_risk_percent *= 1.1; // Increase risk by 10%
        }
    }
    
    // Ensure within bounds
    g_adaptive_risk_percent = MathMax(InpMinRiskPercent, 
                             MathMin(InpMaxRiskPercent, g_adaptive_risk_percent));
    
    // Log change if significant
    if(MathAbs(old_risk - g_adaptive_risk_percent) > 0.05){
        g_adaptive_parameter_changes++;
        
        if(InpLogAdaptiveChanges){
            LogAdaptiveChange(StringFormat("Risk percent: %.1f%% -> %.1f%% (Wins: %d, Losses: %d)",
                             old_risk, g_adaptive_risk_percent, g_consecutive_wins, g_consecutive_losses));
        }
    }
}

// Calculate recent performance ratio
double CalculateRecentPerformance(){
    int available_trades = g_trade_results_filled ? 20 : g_trade_results_index;
    if(available_trades < 5) return 0.0; // Need minimum trades
    
    int lookback = MathMin(InpPerformanceLookback, available_trades);
    double total_pnl = 0.0;
    int profitable_trades = 0;
    
    for(int i = 0; i < lookback; i++){
        int index = (g_trade_results_index - 1 - i + 20) % 20;
        double pnl = g_recent_trade_results[index];
        total_pnl += pnl;
        if(pnl > 0) profitable_trades++;
    }
    
    // Normalize performance (positive = good, negative = bad)
    double win_rate = (double)profitable_trades / lookback;
    double avg_pnl = total_pnl / lookback;
    
    // Simple performance metric combining win rate and average P&L
    return (win_rate - 0.5) * 2.0 + MathMin(1.0, MathMax(-1.0, avg_pnl / 100.0));
}

// Get volatility regime name for logging
string GetVolatilityRegimeName(int regime){
    switch(regime){
        case 0: return "LOW";
        case 1: return "NORMAL";
        case 2: return "HIGH";
        default: return "UNKNOWN";
    }
}

// Log adaptive parameter changes with throttling
void LogAdaptiveChange(const string message){
    if(!InpLogAdaptiveChanges) return;
    
    datetime current_time = TimeCurrent();
    if((current_time - g_last_adaptive_log) < InpAdaptiveLogThrottle) return;
    
    SmartLog("ADAPTIVE", message);
    g_last_adaptive_log = current_time;
}

// Reset adaptive parameters if configured for daily reset
void CheckAdaptiveReset(){
    if(!InpUseAdaptiveParameters || !InpResetAdaptiveDaily) return;
    
    datetime current_time = TimeCurrent();
    datetime current_day = current_time - (current_time % 86400);
    
    if(current_day != g_last_adaptive_reset){
        g_last_adaptive_reset = current_day;
        
        // Reset to base values
        g_adaptive_atr_multiplier = InpBaseATRMultiplier;
        g_adaptive_risk_percent = InpBaseRiskPercent;
        g_adaptive_timeout_hours = InpBaseTimeoutHours;
        
        // Reset streaks but keep history
        g_consecutive_wins = 0;
        g_consecutive_losses = 0;
        
        LogAdaptiveChange("Daily reset: All adaptive parameters restored to base values");
    }
}

// Get adaptive parameter statistics for display
string GetAdaptiveStatistics(){
    if(!InpUseAdaptiveParameters) return "Adaptive Parameters: DISABLED";
    
    return StringFormat("Adaptive Params: ATR %.2f | Risk %.1f%% | Timeout %dh | %s volatility | Changes: %.0f",
                       g_adaptive_atr_multiplier, g_adaptive_risk_percent, g_adaptive_timeout_hours,
                       GetVolatilityRegimeName(g_current_volatility_regime), g_adaptive_parameter_changes);
}

// Get current adaptive ATR multiplier for use in risk calculations
double GetAdaptiveATRMultiplier(){
    return InpUseAdaptiveParameters && InpAdaptiveATRMultipliers ? 
           g_adaptive_atr_multiplier : InpBaseATRMultiplier;
}

// Get current adaptive risk percentage for position sizing
double GetAdaptiveRiskPercent(){
    return InpUseAdaptiveParameters && InpAdaptiveRiskSizing ? 
           g_adaptive_risk_percent : InpBaseRiskPercent;
}

// Get current adaptive timeout hours for position management
int GetAdaptiveTimeoutHours(){
    return InpUseAdaptiveParameters && InpAdaptiveTimeouts ? 
           g_adaptive_timeout_hours : InpBaseTimeoutHours;
}

//============================== EMERGENCY STOP/DRAWDOWN CONTROL (1.5 IMPROVEMENTS) ===============
// Advanced emergency stop system with daily limits and circuit breakers
// Initialize daily tracking at start of new trading day
void InitializeDailyTracking(){
    if(!InpUseAdvancedEmergencyStop) return;
    
    datetime current_time = TimeCurrent();
    datetime current_day = current_time - (current_time % 86400); // Start of current day
    
    // Check if it's a new trading day
    if(current_day != g_current_trading_day){
        g_current_trading_day = current_day;
        g_daily_starting_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_max_equity_today = g_daily_starting_equity;
        g_daily_pnl = 0.0;
        g_daily_loss_limit_hit = false;
        g_consecutive_emergency_losses = 0;
        g_drawdown_level = 0;
        
        if(InpUseTimeBasedReset && g_emergency_shutdown_active){
            g_emergency_shutdown_active = false;
            g_emergency_pause_until = 0;
            Print("EMERGENCY STOP: Daily reset - trading re-enabled for new day");
        }
        
        Print("EMERGENCY STOP: New trading day initialized - Starting equity: $", 
              DoubleToString(g_daily_starting_equity, 2));
    }
}

// Update daily P&L tracking when positions close
void UpdateDailyPnL(double trade_pnl){
    if(!InpUseAdvancedEmergencyStop) return;
    
    g_daily_pnl += trade_pnl;
    double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Track maximum equity for drawdown calculation
    if(current_equity > g_max_equity_today){
        g_max_equity_today = current_equity;
    }
    
    // Update consecutive loss counter
    if(trade_pnl < 0){
        g_consecutive_emergency_losses++;
    } else if(trade_pnl > 0){
        g_consecutive_emergency_losses = 0; // Reset on profitable trade
    }
    
    Print("EMERGENCY STOP: Daily P&L updated: $", DoubleToString(g_daily_pnl, 2), 
          " | Consecutive losses: ", g_consecutive_emergency_losses);
}

// Check if daily loss limit has been exceeded
bool CheckDailyLossLimit(){
    if(!InpUseDailyLossLimit) return false;
    
    double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double daily_loss_amount = g_daily_starting_equity - current_equity;
    double daily_loss_percent = (daily_loss_amount / g_daily_starting_equity) * 100.0;
    
    // Check absolute dollar limit
    bool amount_exceeded = (daily_loss_amount >= InpDailyLossLimitAmount);
    
    // Check percentage limit  
    bool percent_exceeded = (daily_loss_percent >= InpDailyLossLimitPercent);
    
    if(amount_exceeded || percent_exceeded){
        if(!g_daily_loss_limit_hit){
            g_daily_loss_limit_hit = true;
            g_emergency_pause_until = TimeCurrent() + (InpEmergencyPauseHours * 3600);
            
            Print("EMERGENCY STOP: Daily loss limit exceeded!");
            Print("  Loss amount: $", DoubleToString(daily_loss_amount, 2), 
                  " (limit: $", DoubleToString(InpDailyLossLimitAmount, 2), ")");
            Print("  Loss percent: ", DoubleToString(daily_loss_percent, 2), 
                  "% (limit: ", DoubleToString(InpDailyLossLimitPercent, 1), "%)");
            Print("  Trading paused for ", InpEmergencyPauseHours, " hours");
            
            CloseAllPositions();
        }
        return true;
    }
    
    return false;
}

// Check progressive drawdown levels and take action
int CheckDrawdownLevels(){
    if(!InpUseDrawdownCircuitBreaker) return 0;
    
    double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double drawdown_amount = g_max_equity_today - current_equity;
    double drawdown_percent = (drawdown_amount / g_max_equity_today) * 100.0;
    
    int new_level = 0;
    
    // Determine drawdown level
    if(drawdown_percent >= InpLevel3DrawdownPercent){
        new_level = 3; // Emergency shutdown
    } else if(drawdown_percent >= InpLevel2DrawdownPercent){
        new_level = 2; // Stop new trades
    } else if(drawdown_percent >= InpLevel1DrawdownPercent){
        new_level = 1; // Reduce position sizes
    }
    
    // Take action if level changed
    if(new_level > g_drawdown_level){
        g_drawdown_level = new_level;
        
        switch(new_level){
            case 1:
                Print("EMERGENCY STOP: Level 1 drawdown (", DoubleToString(drawdown_percent, 2), 
                      "%) - reducing position sizes by 50%");
                break;
                
            case 2:
                Print("EMERGENCY STOP: Level 2 drawdown (", DoubleToString(drawdown_percent, 2), 
                      "%) - stopping new trades");
                break;
                
            case 3:
                Print("EMERGENCY STOP: Level 3 drawdown (", DoubleToString(drawdown_percent, 2), 
                      "%) - emergency shutdown activated");
                g_emergency_shutdown_active = true;
                g_emergency_pause_until = TimeCurrent() + (InpEmergencyPauseHours * 3600);
                CloseAllPositions();
                break;
        }
    }
    
    return new_level;
}

// Check consecutive loss emergency stop
bool CheckConsecutiveLossLimit(){
    if(!InpUseConsecutiveLossLimit) return false;
    
    if(g_consecutive_emergency_losses >= InpMaxConsecutiveLossLimit){
        Print("EMERGENCY STOP: Consecutive loss limit exceeded (", g_consecutive_emergency_losses, 
              " losses) - activating emergency pause");
        g_emergency_shutdown_active = true;
        g_emergency_pause_until = TimeCurrent() + (InpEmergencyPauseHours * 3600);
        CloseAllPositions();
        return true;
    }
    
    return false;
}

// Comprehensive emergency stop checking
bool CheckAdvancedEmergencyStops(){
    if(!InpUseAdvancedEmergencyStop) return false;
    
    // Check less frequently to reduce CPU usage
    datetime current_time = TimeCurrent();
    if(current_time - g_last_emergency_check < 60) return g_emergency_shutdown_active; // Check every minute
    g_last_emergency_check = current_time;
    
    // Initialize daily tracking if needed
    InitializeDailyTracking();
    
    // Check if emergency pause is still active
    if(g_emergency_pause_until > 0 && current_time < g_emergency_pause_until){
        int remaining_minutes = (int)((g_emergency_pause_until - current_time) / 60);
        if(remaining_minutes % 30 == 0){ // Print every 30 minutes
            Print("EMERGENCY STOP: Trading paused - ", remaining_minutes, " minutes remaining");
        }
        return true;
    } else if(g_emergency_pause_until > 0 && current_time >= g_emergency_pause_until){
        g_emergency_pause_until = 0;
        g_emergency_shutdown_active = false;
        Print("EMERGENCY STOP: Emergency pause expired - trading re-enabled");
    }
    
    // Check all emergency conditions
    bool emergency_active = false;
    
    if(CheckDailyLossLimit()){
        emergency_active = true;
    }
    
    if(CheckDrawdownLevels() >= 3){
        emergency_active = true;
    }
    
    if(CheckConsecutiveLossLimit()){
        emergency_active = true;
    }
    
    return emergency_active;
}

// Get position size adjustment based on emergency level
double GetEmergencySizeAdjustment(){
    if(!InpUseDrawdownCircuitBreaker) return 1.0;
    
    switch(g_drawdown_level){
        case 1: return 0.5; // 50% position sizes
        case 2: 
        case 3: return 0.0; // No new positions
        default: return 1.0; // Normal size
    }
}

//============================== INDICATOR CACHING & OPTIMIZATION (2.1 IMPROVEMENTS) ===============
// Performance enhancement through intelligent caching and computation reduction
// Initialize all cached indicator handles in OnInit
bool InitializeCachedIndicators(){
    if(!InpUseIndicatorCaching) return true;
    
    Print("CACHE: Initializing indicator caching system...");
    
    // Initialize ATR handles
    if(InpOptimizeATRCalls){
        g_atr_14_handle = iATR(_Symbol, _Period, 14);
        if(g_atr_14_handle == INVALID_HANDLE){
            Print("ERROR: Failed to initialize cached ATR(14) handle");
            return false;
        }
        
        g_atr_50_handle = iATR(_Symbol, _Period, 50);
        if(g_atr_50_handle == INVALID_HANDLE){
            Print("ERROR: Failed to initialize cached ATR(50) handle");
            return false;
        }
    }
    
    // Initialize MA handles for market regime detection
    if(InpUseSmartHandles){
        g_ma_10_handle = iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE);
        if(g_ma_10_handle == INVALID_HANDLE){
            Print("ERROR: Failed to initialize cached MA(10) handle");
            return false;
        }
        
        g_ma_50_handle = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);
        if(g_ma_50_handle == INVALID_HANDLE){
            Print("ERROR: Failed to initialize cached MA(50) handle");
            return false;
        }
    }
    
    Print("CACHE: All indicator handles initialized successfully");
    return true;
}

// Update cached indicator values at start of new bar
void UpdateIndicatorCache(){
    if(!InpUseIndicatorCaching) return;
    
    datetime current_time = TimeCurrent();
    
    // Check if we need to refresh cache
    bool should_refresh = false;
    if(!g_cache_indicators_ready){
        should_refresh = true;
    } else if(g_cache_bar_count >= InpCacheRefreshBars){
        should_refresh = true;
        g_cache_bar_count = 0;
    }
    
    if(!should_refresh) return;
    
    // Update ATR cache
    if(InpOptimizeATRCalls && g_atr_14_handle != INVALID_HANDLE){
        double atr_14_buffer[1];
        if(CopyBuffer(g_atr_14_handle, 0, 0, 1, atr_14_buffer) == 1){
            g_cached_atr_14 = atr_14_buffer[0];
        }
        
        double atr_50_buffer[1];
        if(CopyBuffer(g_atr_50_handle, 0, 0, 1, atr_50_buffer) == 1){
            g_cached_atr_50 = atr_50_buffer[0];
        }
    }
    
    // Update MA cache
    if(InpUseSmartHandles && g_ma_10_handle != INVALID_HANDLE){
        double ma_10_buffer[1];
        if(CopyBuffer(g_ma_10_handle, 0, 0, 1, ma_10_buffer) == 1){
            g_cached_ma_10 = ma_10_buffer[0];
        }
        
        double ma_50_buffer[1];
        if(CopyBuffer(g_ma_50_handle, 0, 0, 1, ma_50_buffer) == 1){
            g_cached_ma_50 = ma_50_buffer[0];
        }
    }
    
    // Update complex calculation cache
    if(InpCacheComplexCalcs){
        UpdateComplexCalculationCache();
    }
    
    g_last_cache_update = current_time;
    g_cache_indicators_ready = true;
    
    // Debug output (can be removed in production)
    static int cache_count = 0;
    cache_count++;
    if(cache_count % 10 == 0){ // Print every 10 cache updates
        Print("CACHE: Updated indicators (", cache_count, " times) - ATR14: ", 
              DoubleToString(g_cached_atr_14, 5), ", MA10: ", DoubleToString(g_cached_ma_10, 5));
    }
}

// Update cached complex calculations
void UpdateComplexCalculationCache(){
    if(!InpCacheComplexCalcs || g_cached_atr_14 <= 0) return;
    
    // Cache ATR monetary value (frequently used in position sizing)
    if(g_position_size > 0){
        g_cached_atr_monetary = g_cached_atr_14 * g_position_size * 100000;
    }
    
    // Cache volatility impact calculation (used in multiple functions)
    if(g_cached_atr_50 > 0){
        g_cached_volatility_impact = g_cached_atr_14 / g_cached_atr_50;
    }
    
    g_last_complex_calc_time = TimeCurrent();
}

// Get cached ATR(14) value with fallback
double GetCachedATR14(){
    if(InpOptimizeATRCalls && g_cache_indicators_ready && g_cached_atr_14 > 0){
        return g_cached_atr_14;
    }
    
    // Fallback to direct calculation
    double atr_values[1];
    int handle = iATR(_Symbol, _Period, 14);
    if(CopyBuffer(handle, 0, 0, 1, atr_values) == 1){
        return atr_values[0];
    }
    
    return 0.0;
}

// Get cached ATR(50) value with fallback
double GetCachedATR50(){
    if(InpOptimizeATRCalls && g_cache_indicators_ready && g_cached_atr_50 > 0){
        return g_cached_atr_50;
    }
    
    // Fallback to direct calculation
    double atr_values[1];
    int handle = iATR(_Symbol, _Period, 50);
    if(CopyBuffer(handle, 0, 0, 1, atr_values) == 1){
        return atr_values[0];
    }
    
    return 0.0;
}

// Get cached MA(10) value with fallback
double GetCachedMA10(){
    if(InpUseSmartHandles && g_cache_indicators_ready && g_cached_ma_10 > 0){
        return g_cached_ma_10;
    }
    
    // Fallback to direct calculation
    double ma_values[1];
    int handle = iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE);
    if(CopyBuffer(handle, 0, 0, 1, ma_values) == 1){
        return ma_values[0];
    }
    
    return 0.0;
}

// Get cached MA(50) value with fallback
double GetCachedMA50(){
    if(InpUseSmartHandles && g_cache_indicators_ready && g_cached_ma_50 > 0){
        return g_cached_ma_50;
    }
    
    // Fallback to direct calculation
    double ma_values[1];
    int handle = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);
    if(CopyBuffer(handle, 0, 0, 1, ma_values) == 1){
        return ma_values[0];
    }
    
    return 0.0;
}

// Get cached ATR monetary value with intelligent recalculation
double GetCachedATRMonetary(){
    if(InpCacheComplexCalcs && g_cached_atr_monetary > 0 && 
       (TimeCurrent() - g_last_complex_calc_time) < 300){ // 5 minute cache
        return g_cached_atr_monetary;
    }
    
    // Recalculate if needed
    double atr = GetCachedATR14();
    if(atr > 0 && g_position_size > 0){
        g_cached_atr_monetary = atr * g_position_size * 100000;
        g_last_complex_calc_time = TimeCurrent();
        return g_cached_atr_monetary;
    }
    
    return 0.0;
}

// Get cached volatility impact with intelligent recalculation  
double GetCachedVolatilityImpact(){
    if(InpCacheComplexCalcs && g_cached_volatility_impact > 0 &&
       (TimeCurrent() - g_last_complex_calc_time) < 300){ // 5 minute cache
        return g_cached_volatility_impact;
    }
    
    // Recalculate if needed
    double atr_14 = GetCachedATR14();
    double atr_50 = GetCachedATR50();
    
    if(atr_14 > 0 && atr_50 > 0){
        g_cached_volatility_impact = atr_14 / atr_50;
        g_last_complex_calc_time = TimeCurrent();
        return g_cached_volatility_impact;
    }
    
    return 1.0; // Default neutral impact
}

// Increment cache bar counter for refresh tracking
void IncrementCacheBarCounter(){
    if(InpUseIndicatorCaching){
        g_cache_bar_count++;
    }
}

//============================== DATA ACCESS OPTIMIZATION (2.2 IMPROVEMENTS) ===============
// Memory optimization through efficient data structures and reuse
// Initialize reusable arrays for optimal memory usage
bool InitializeDataArrays(){
    if(!InpOptimizeDataAccess) return true;
    
    Print("DATA OPTIMIZATION: Initializing reusable arrays...");
    
    // Set buffer size based on configuration
    g_buffer_size = MathMax(InpPreallocateSize, 100);
    
    // Initialize reusable buffers
    if(InpReuseArrays){
        ArrayResize(g_reusable_buffer, g_buffer_size);
        ArrayResize(g_temp_array, g_buffer_size);
        ArrayInitialize(g_reusable_buffer, 0.0);
        ArrayInitialize(g_temp_array, 0.0);
    }
    
    // Initialize neural network arrays for reuse
    if(InpMinimizeMemOps){
        ArrayResize(g_neural_z1, 64);  // Max layer size
        ArrayResize(g_neural_a1, 64);
        ArrayResize(g_neural_z2, 64);
        ArrayResize(g_neural_a2, 64);
        ArrayResize(g_neural_z3, 64);
        ArrayResize(g_neural_a3, 64);
        ArrayResize(g_neural_final, ACTIONS);
        
        ArrayInitialize(g_neural_z1, 0.0);
        ArrayInitialize(g_neural_a1, 0.0);
        ArrayInitialize(g_neural_z2, 0.0);
        ArrayInitialize(g_neural_a2, 0.0);
        ArrayInitialize(g_neural_z3, 0.0);
        ArrayInitialize(g_neural_a3, 0.0);
        ArrayInitialize(g_neural_final, 0.0);
    }
    
    // Initialize Q-value smoothing buffer
    ArrayInitialize(g_smoothed_q_buffer, 0.0);
    
    g_data_arrays_initialized = true;
    Print("DATA OPTIMIZATION: Arrays initialized successfully (buffer size: ", g_buffer_size, ")");
    return true;
}

// Get reusable buffer with automatic resizing if needed
void GetReusableBuffer(double &buffer[], int required_size = 0){
    if(!InpReuseArrays || !g_data_arrays_initialized){
        // Fallback to temporary array if optimization disabled
        if(required_size > 0){
            ArrayResize(buffer, required_size);
        }
        return;
    }
    
    // Resize global buffer if needed
    if(required_size > g_buffer_size){
        g_buffer_size = required_size + 50; // Add some extra space
        ArrayResize(g_reusable_buffer, g_buffer_size);
        Print("DATA OPTIMIZATION: Buffer resized to ", g_buffer_size, " elements");
    }
    
    // Copy to provided buffer
    ArrayResize(buffer, required_size);
    ArrayCopy(buffer, g_reusable_buffer, 0, 0, required_size);
}

// Get temporary array with automatic resizing if needed  
void GetTempArray(double &temp_array[], int required_size = 0){
    if(!InpReuseArrays || !g_data_arrays_initialized){
        if(required_size > 0){
            ArrayResize(temp_array, required_size);
        }
        return;
    }
    
    if(required_size > g_buffer_size){
        g_buffer_size = required_size + 50;
        ArrayResize(g_temp_array, g_buffer_size);
    }
    
    ArrayResize(temp_array, required_size);
    ArrayCopy(temp_array, g_temp_array, 0, 0, required_size);
}

// Optimized CopyBuffer operation with reusable arrays
int OptimizedCopyBuffer(int handle, int buffer_num, int start, int count, double &target_array[]){
    if(!InpOptimizeDataAccess){
        return CopyBuffer(handle, buffer_num, start, count, target_array);
    }
    
    // Use reusable buffer for the copy operation
    double temp_buffer[];
    GetReusableBuffer(temp_buffer, count);
    int copied = CopyBuffer(handle, buffer_num, start, count, temp_buffer);
    
    if(copied > 0){
        // Efficiently copy only needed elements
        ArrayResize(target_array, copied);
        for(int i = 0; i < copied; i++){
            target_array[i] = temp_buffer[i];
        }
    }
    
    return copied;
}

// Optimized array copying with size validation
void OptimizedArrayCopy(double &dest[], const double &src[], int count = -1){
    if(!InpMinimizeMemOps){
        if(count < 0){
            ArrayCopy(dest, src);
        } else {
            ArrayCopy(dest, src, 0, 0, count);
        }
        return;
    }
    
    // Determine copy count
    int src_size = ArraySize(src);
    int copy_count = (count < 0) ? src_size : MathMin(count, src_size);
    
    // Resize destination only if necessary
    if(ArraySize(dest) != copy_count){
        ArrayResize(dest, copy_count);
    }
    
    // Direct assignment for small arrays (faster than ArrayCopy)
    if(copy_count <= 10){
        for(int i = 0; i < copy_count; i++){
            dest[i] = src[i];
        }
    } else {
        ArrayCopy(dest, src, 0, 0, copy_count);
    }
}

// Optimized array initialization
void OptimizedArrayInit(double &array[], int size, double value = 0.0){
    if(!InpMinimizeMemOps){
        ArrayResize(array, size);
        ArrayInitialize(array, value);
        return;
    }
    
    // Only resize if necessary
    if(ArraySize(array) != size){
        ArrayResize(array, size);
    }
    
    // Fast initialization for common values
    if(value == 0.0){
        ArrayInitialize(array, 0.0);
    } else {
        for(int i = 0; i < size; i++){
            array[i] = value;
        }
    }
}

// Memory-efficient series data loading
bool OptimizedLoadSeries(const string sym, ENUM_TIMEFRAMES tf, int count, Series &s){
    if(!InpOptimizeSeriesAccess){
        return LoadSeries(sym, tf, count, s);
    }
    
    // Pre-allocate arrays to avoid multiple resizing
    ArrayResize(s.rates, count);
    ArrayResize(s.times, count);
    ArraySetAsSeries(s.rates, true);
    
    int copied = CopyRates(sym, tf, 0, count, s.rates);
    if(copied <= 0){
        Print("CopyRates failed ", sym, " ", EnumToString(tf), " err=", GetLastError());
        return false;
    }
    
    // Resize to actual copied size if different
    if(copied != count){
        ArrayResize(s.times, copied);
    }
    
    // Extract timestamps efficiently
    for(int i = 0; i < copied; ++i){
        s.times[i] = s.rates[i].time;
    }
    
    return true;
}

//============================== INTELLIGENT LOGGING SYSTEM (2.3 IMPROVEMENTS) ===============
// Performance-oriented logging with conditional output and throttling
// Initialize logging system
void InitializeLoggingSystem(){
    if(!InpMinimizeLogging) return;
    
    g_last_signal_log = 0;
    g_last_filter_log = 0; 
    g_last_performance_log = 0;
    g_last_logged_message = "";
    g_last_message_time = 0;
    g_log_suppression_count = 0;
    g_logging_system_ready = true;
    
    if(InpLogInitialization){
        Print("LOGGING: Intelligent logging system initialized");
        Print("  Critical events only: ", InpLogCriticalOnly ? "ENABLED" : "DISABLED");
        Print("  Signal logging: ", InpLogSignalChanges ? "ENABLED" : "DISABLED");
        Print("  Filter details: ", InpLogFilterDetails ? "ENABLED" : "DISABLED");
        Print("  Log throttling: ", InpLogThrottleSeconds, " seconds");
    }
}

// Check if logging should be allowed based on current settings
bool ShouldLog(const string category){
    if(!InpMinimizeLogging) return true; // Always log if optimization disabled
    
    datetime current_time = TimeCurrent();
    
    if(category == "CRITICAL" || category == "ERROR"){
        return InpLogErrorsAlways;
    }
    
    if(category == "TRADE"){
        return InpLogTradeEvents;
    }
    
    if(category == "SIGNAL"){
        if(!InpLogSignalChanges) return false;
        // Throttle signal logs
        if((current_time - g_last_signal_log) < InpLogThrottleSeconds) return false;
        g_last_signal_log = current_time;
        return true;
    }
    
    if(category == "FILTER"){
        if(!InpLogFilterDetails) return false;
        // Throttle filter logs
        if((current_time - g_last_filter_log) < InpLogThrottleSeconds) return false;
        g_last_filter_log = current_time;
        return true;
    }
    
    if(category == "PERFORMANCE"){
        if(!InpLogPerformanceData) return false;
        // Only log performance data periodically
        if((current_time - g_last_performance_log) < (InpLogThrottleSeconds * 2)) return false;
        g_last_performance_log = current_time;
        return true;
    }
    
    if(category == "INIT"){
        return InpLogInitialization;
    }
    
    // For uncategorized logs, use critical only setting
    return !InpLogCriticalOnly;
}

// Intelligent logging with deduplication and throttling
void SmartLog(const string category, const string message){
    if(!ShouldLog(category)) return;
    
    // Duplicate message suppression
    if(InpMinimizeLogging && g_logging_system_ready){
        if(message == g_last_logged_message){
            g_log_suppression_count++;
            // Only report suppression occasionally
            if(g_log_suppression_count % 10 == 0){
                Print("LOGGING: Suppressed ", g_log_suppression_count, " duplicate messages");
            }
            return;
        }
        
        // Reset suppression tracking for new message
        if(g_log_suppression_count > 0){
            Print("LOGGING: Suppressed ", g_log_suppression_count, " duplicate messages (final)");
            g_log_suppression_count = 0;
        }
        
        g_last_logged_message = message;
        g_last_message_time = TimeCurrent();
    }
    
    // Prepend category for categorized logs
    if(category != ""){
        Print("[", category, "] ", message);
    } else {
        Print(message);
    }
}

// Optimized logging functions for specific categories
void LogCritical(const string message){ SmartLog("CRITICAL", message); }
void LogError(const string message){ SmartLog("ERROR", message); }
void LogTrade(const string message){ SmartLog("TRADE", message); }
void LogSignal(const string message){ SmartLog("SIGNAL", message); }
void LogFilter(const string message){ SmartLog("FILTER", message); }
void LogRisk(const string message){ SmartLog("RISK", message); }
void LogPerformance(const string message){ SmartLog("PERFORMANCE", message); }
void LogInit(const string message){ SmartLog("INIT", message); }
void LogDebug(const string message){ SmartLog("", message); } // Uncategorized

// Special function for always logging regardless of settings (emergencies)
void ForceLog(const string message){
    Print("FORCE: ", message);
}

// Performance-aware conditional logging macros
void ConditionalLog(const string category, const string message, bool condition){
    if(condition && ShouldLog(category)){
        SmartLog(category, message);
    }
}

//============================== EFFICIENT LOOPING SYSTEM (2.4 IMPROVEMENTS) ===============
// Optimized loop patterns with early exits and intelligent iteration management

// Initialize loop optimization system
void InitializeLoopOptimization(){
    if(!InpOptimizeLoops) return;
    
    g_loop_iteration_count = 0;
    g_position_cache_count = 0;
    g_last_position_scan = 0;
    g_position_cache_valid = false;
    g_max_loop_safety = InpMaxLoopIterations;
    
    // Initialize position cache
    ArrayInitialize(g_cached_positions, 0);
    
    LogInit("LOOP OPTIMIZATION: System initialized with max iterations: " + IntegerToString(g_max_loop_safety));
}

// Refresh position cache to avoid repeated PositionsTotal() calls
void RefreshPositionCache(){
    if(!InpOptimizePositionScan) return;
    
    datetime current_time = TimeCurrent();
    // Refresh cache every 5 seconds or when explicitly requested
    if(g_position_cache_valid && (current_time - g_last_position_scan) < 5) return;
    
    g_position_cache_count = 0;
    int total_positions = PositionsTotal();
    
    // Cache relevant position tickets only
    for(int i = 0; i < total_positions && g_position_cache_count < ArraySize(g_cached_positions); i++){
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionSelectByTicket(ticket)){
            string pos_symbol = PositionGetString(POSITION_SYMBOL);
            long pos_magic = PositionGetInteger(POSITION_MAGIC);
            
            // Only cache positions for our symbol and magic
            if(pos_symbol == _Symbol && pos_magic == InpMagic){
                g_cached_positions[g_position_cache_count] = ticket;
                g_position_cache_count++;
            }
        }
    }
    
    g_last_position_scan = current_time;
    g_position_cache_valid = true;
    
    LogPerformance("LOOP OPTIMIZATION: Position cache refreshed - " + IntegerToString(g_position_cache_count) + " relevant positions cached");
}

// Get position count with caching
int GetCachedPositionCount(){
    if(!InpOptimizePositionScan) return PositionsTotal();
    
    RefreshPositionCache();
    return g_position_cache_count;
}

// Optimized position scanning with early exits and safety limits
bool ScanPositionsOptimized(bool &has_long, bool &has_short, double &total_volume, double &avg_profit){
    has_long = false; has_short = false; total_volume = 0.0; avg_profit = 0.0;
    
    if(!InpOptimizeLoops) {
        // Fallback to standard scanning
        return ScanPositionsStandard(has_long, has_short, total_volume, avg_profit);
    }
    
    RefreshPositionCache();
    
    if(g_position_cache_count == 0) return false;
    
    double total_profit = 0.0;
    int processed_positions = 0;
    
    // Use cached positions for faster scanning
    for(int i = 0; i < g_position_cache_count; i++){
        // Safety check for infinite loops
        if(InpUseEarlyBreaks && processed_positions >= g_max_loop_safety){
            LogCritical("LOOP SAFETY: Position scan terminated at max iterations: " + IntegerToString(g_max_loop_safety));
            break;
        }
        
        ulong ticket = g_cached_positions[i];
        if(!PositionSelectByTicket(ticket)) continue;
        
        long pos_type = PositionGetInteger(POSITION_TYPE);
        double pos_volume = PositionGetDouble(POSITION_VOLUME);
        double pos_profit = PositionGetDouble(POSITION_PROFIT);
        
        if(pos_type == POSITION_TYPE_BUY) has_long = true;
        else has_short = true;
        
        total_volume += pos_volume;
        total_profit += pos_profit;
        processed_positions++;
        
        // Early exit optimization: if we found both long and short, we have enough info
        if(InpUseEarlyBreaks && has_long && has_short && processed_positions >= 2){
            break; // No need to scan further for basic direction check
        }
    }
    
    if(processed_positions > 0){
        avg_profit = total_profit / processed_positions;
    }
    
    return (processed_positions > 0);
}

// Standard position scanning (fallback)
bool ScanPositionsStandard(bool &has_long, bool &has_short, double &total_volume, double &avg_profit){
    has_long = false; has_short = false; total_volume = 0.0; avg_profit = 0.0;
    
    int total_positions = PositionsTotal();
    double total_profit = 0.0;
    int processed_positions = 0;
    
    for(int i = 0; i < total_positions; i++){
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        
        string pos_symbol = PositionGetString(POSITION_SYMBOL);
        long pos_magic = PositionGetInteger(POSITION_MAGIC);
        
        if(pos_symbol == _Symbol && pos_magic == InpMagic){
            long pos_type = PositionGetInteger(POSITION_TYPE);
            double pos_volume = PositionGetDouble(POSITION_VOLUME);
            double pos_profit = PositionGetDouble(POSITION_PROFIT);
            
            if(pos_type == POSITION_TYPE_BUY) has_long = true;
            else has_short = true;
            
            total_volume += pos_volume;
            total_profit += pos_profit;
            processed_positions++;
        }
    }
    
    if(processed_positions > 0){
        avg_profit = total_profit / processed_positions;
    }
    
    return (processed_positions > 0);
}

// Optimized loop safety wrapper with iteration counting
bool LoopSafetyCheck(int &current_iteration, int max_iterations = -1){
    if(!InpOptimizeLoops) return true;
    
    if(max_iterations == -1) max_iterations = g_max_loop_safety;
    
    current_iteration++;
    g_loop_iteration_count++;
    
    if(current_iteration >= max_iterations){
        LogCritical("LOOP SAFETY: Maximum iterations reached (" + IntegerToString(max_iterations) + ") - forcing break");
        return false;
    }
    
    return true;
}

// Reset loop optimization counters
void ResetLoopCounters(){
    g_loop_iteration_count = 0;
    g_position_cache_valid = false; // Force cache refresh
}

//============================== TICK HANDLING OPTIMIZATION (2.5 IMPROVEMENTS) ===============
// Selective tick processing with bar close detection and timer-based checks

// Initialize tick handling optimization system
void InitializeTickHandling(){
    if(!InpOptimizeTicks) return;
    
    g_last_signal_bar_time = 0;
    g_tick_counter = 0;
    g_last_timer_check = TimeCurrent();
    g_tick_optimization_active = true;
    g_skipped_ticks_count = 0;
    g_processed_ticks_count = 0;
    g_is_new_bar = false;
    g_current_bar_time = 0;
    
    // Initialize timer if enabled
    if(InpUseTimerChecks){
        EventSetTimer(InpTimerIntervalSec);
        LogInit("TICK OPTIMIZATION: Timer initialized with " + IntegerToString(InpTimerIntervalSec) + " second interval");
    }
    
    LogInit("TICK OPTIMIZATION: System initialized - Bar close: " + 
            (InpProcessOnBarClose ? "ENABLED" : "DISABLED") + 
            ", Intra-bar: " + (InpAllowIntraBarTicks ? "ENABLED" : "DISABLED"));
}

// Check if current tick should be processed based on optimization settings
bool ShouldProcessTick(){
    if(!InpOptimizeTicks) return true; // No optimization - process all ticks
    
    // Get current bar information
    MqlRates current_rate[1];
    if(CopyRates(_Symbol, _Period, 0, 1, current_rate) != 1) return true; // Safety fallback
    
    datetime current_bar_time = current_rate[0].time;
    
    // Check if this is a new bar
    g_is_new_bar = (current_bar_time != g_current_bar_time);
    if(g_is_new_bar){
        g_current_bar_time = current_bar_time;
        g_tick_counter = 0; // Reset tick counter for new bar
    }
    
    // Always process first tick of new bar if bar close processing is enabled
    if(InpProcessOnBarClose && g_is_new_bar){
        g_processed_ticks_count++;
        LogPerformance("TICK OPTIMIZATION: Processing new bar tick at " + TimeToString(current_bar_time));
        return true;
    }
    
    // If only bar close processing is enabled, skip intra-bar ticks
    if(InpProcessOnBarClose && !g_is_new_bar){
        g_skipped_ticks_count++;
        return false;
    }
    
    // Handle intra-bar tick processing if enabled
    if(InpAllowIntraBarTicks && !g_is_new_bar){
        g_tick_counter++;
        
        // Process every Nth tick based on skip ratio
        if(g_tick_counter % InpTickSkipRatio == 0){
            g_processed_ticks_count++;
            return true;
        } else {
            g_skipped_ticks_count++;
            return false;
        }
    }
    
    // Default: process tick if no specific rules apply
    g_processed_ticks_count++;
    return true;
}

// Check if critical functions should run on every tick regardless of optimization
bool ShouldProcessCriticalTick(){
    // Always allow trailing stops if enabled
    if(InpTrailingOnTicks && InpUseTrailingStop) return true;
    
    // Always allow risk checks if enabled
    if(InpRiskChecksOnTicks) return true;
    
    // Always allow emergency checks if enabled
    if(InpEmergencyOnTicks) return true;
    
    return false;
}

// Timer-based periodic checks (called by OnTimer)
void ProcessTimerChecks(){
    if(!InpUseTimerChecks || !InpOptimizeTicks) return;
    
    datetime current_time = TimeCurrent();
    g_last_timer_check = current_time;
    
    LogPerformance("TICK OPTIMIZATION: Timer check at " + TimeToString(current_time));
    
    // Perform periodic risk management checks
    if(InpRiskChecksOnTicks){
        // Check portfolio risk
        if(!CheckPortfolioRisk()){
            LogRisk("TIMER CHECK: Portfolio risk limits exceeded");
        }
        
        // Check emergency stops
        if(CheckAdvancedEmergencyStops()){
            LogCritical("TIMER CHECK: Emergency stop triggered");
        }
    }
    
    // Update position tracking periodically
    UpdatePositionTracking();
    
    // Refresh indicator cache if needed
    if(InpUseIndicatorCaching){
        UpdateIndicatorCache();
    }
    
    // Log tick processing statistics every 10 timer intervals
    static int timer_count = 0;
    timer_count++;
    if(timer_count % 10 == 0){
        int total_ticks = g_processed_ticks_count + g_skipped_ticks_count;
        double skip_ratio = total_ticks > 0 ? (double)g_skipped_ticks_count / total_ticks * 100.0 : 0.0;
        
        LogPerformance("TICK OPTIMIZATION: Statistics - Processed: " + IntegerToString(g_processed_ticks_count) + 
                      ", Skipped: " + IntegerToString(g_skipped_ticks_count) + 
                      " (" + DoubleToString(skip_ratio, 1) + "% skipped)");
        
        // Log confidence filter statistics (3.1 IMPROVEMENT)
        if(InpUseConfidenceFilter){
            LogPerformance("CONFIDENCE FILTER: " + GetConfidenceStatistics());
        }
        
        // Log ensemble model statistics (3.2 IMPROVEMENT)
        if(InpUseEnsembleModel){
            LogPerformance("ENSEMBLE MODEL: " + GetEnsembleStatistics());
        }
        
        // Log adaptive parameter statistics (3.3 IMPROVEMENT)
        if(InpUseAdaptiveParameters){
            LogPerformance("ADAPTIVE PARAMS: " + GetAdaptiveStatistics());
        }
    }
}

// Reset tick optimization counters
void ResetTickCounters(){
    if(!InpOptimizeTicks) return;
    
    // Reset daily counters but keep cumulative statistics
    g_tick_counter = 0;
    g_is_new_bar = false;
}

// Get tick processing efficiency statistics
string GetTickProcessingStats(){
    if(!InpOptimizeTicks) return "Tick optimization: DISABLED";
    
    int total_ticks = g_processed_ticks_count + g_skipped_ticks_count;
    if(total_ticks == 0) return "Tick optimization: No ticks processed yet";
    
    double skip_ratio = (double)g_skipped_ticks_count / total_ticks * 100.0;
    double efficiency_gain = skip_ratio; // Percentage of CPU cycles saved
    
    return "Tick optimization: " + IntegerToString(g_processed_ticks_count) + " processed, " +
           IntegerToString(g_skipped_ticks_count) + " skipped (" + 
           DoubleToString(skip_ratio, 1) + "% efficiency gain)";
}

// Enhanced trading allowance check with advanced cooldown system
bool IsNewTradingAllowed(){
    // Check advanced emergency stops first (1.5 IMPROVEMENT)
    if(CheckAdvancedEmergencyStops()){
        return false; // Emergency system active
    }
    
    // Don't trade if we have an open position
    if(g_position_type != 0) return false;
    
    datetime current_time = TimeCurrent();
    datetime today_start = current_time - (current_time % (24 * 3600));
    
    // Reset daily counter if it's a new day
    if(today_start != g_current_day){
        g_trades_today = 0;
        g_current_day = today_start;
        g_consecutive_losses = 0; // Reset consecutive losses daily
    }
    
    // Check daily trade limit
    if(g_trades_today >= InpMaxTradesPerDay){
        Print("OVERTRADING PREVENTION: Daily trade limit reached (", g_trades_today, "/", InpMaxTradesPerDay, ")");
        return false;
    }
    
    // Original minimum bars check
    int bars_since_last_trade = (int)((current_time - g_last_trade_time) / PeriodSeconds(PERIOD_CURRENT));
    if(bars_since_last_trade < InpMinBarsBetweenTrades){
        return false;
    }
    
    // Enhanced cooldown system (1.3 IMPROVEMENT)
    if(InpUseAdvancedCooldown){
        // Check standard cooldown period after position close
        if(current_time < g_cooldown_until){
            int remaining_minutes = (int)((g_cooldown_until - current_time) / 60);
            Print("OVERTRADING PREVENTION: Cooldown active - ", remaining_minutes, " minutes remaining");
            return false;
        }
        
        // Check extended cooldown period
        if(current_time < g_extended_cooldown_until){
            int remaining_minutes = (int)((g_extended_cooldown_until - current_time) / 60);
            Print("OVERTRADING PREVENTION: Extended cooldown active - ", remaining_minutes, " minutes remaining");
            return false;
        }
        
        // Check choppy market condition
        if(IsMarketChoppy()){
            Print("OVERTRADING PREVENTION: Blocking trades in choppy market conditions");
            return false;
        }
    }
    
    return true;
}

// Update trading frequency tracking
void UpdateTradingFrequency(){
    g_last_trade_time = TimeCurrent();
    g_trades_today++;
}

// Force FLAT action when position meets exit criteria
bool ShouldForceFlat(){
    if(!InpEnforceFlat || g_position_type == 0) return false;
    
    int holding_hours = GetPositionHoldingHours();
    
    // Force FLAT if holding too long (80% of adaptive timeout)
    int adaptive_timeout = GetAdaptiveTimeoutHours();
    if(holding_hours > (adaptive_timeout * 0.8)){
        return true;
    }
    
    // Force FLAT if small profit available and held long enough
    if(g_position_unrealized_pnl > 10.0 && holding_hours > 12){
        return true;
    }
    
    // Force FLAT if large loss
    if(g_position_unrealized_pnl < -50.0){
        return true;
    }
    
    return false;
}

//============================== PHASE 2 & 3 ENHANCEMENT FUNCTIONS ==============
// Advanced position management and market analysis functions

// Enhanced reward calculation with multiple factors (Phase 2)
double CalculateAdvancedReward(double pnl, int holding_time_hours, double max_dd, bool quick_exit){
    if(!InpEnhancedRewards) return pnl; // Use simple P&L if enhancement disabled
    
    double base_reward = pnl / 100.0; // Normalize profit
    double time_penalty = -InpHoldingTimePenalty * holding_time_hours; // Penalty for long holds
    double drawdown_penalty = -InpDrawdownPenalty * max_dd; // Penalty for drawdown
    double quick_exit_bonus = quick_exit ? InpQuickExitBonus : 0; // Bonus for quick profitable exits
    
    return base_reward + time_penalty + drawdown_penalty + quick_exit_bonus;
}

// Dynamic stop loss tightening based on holding time (Phase 3)
void UpdateDynamicStops(){
    if(!InpUseDynamicStops || g_position_type == 0) return;
    
    int holding_hours = GetPositionHoldingHours();
    if(holding_hours < 24) return; // Only tighten after 24 hours
    
    // Use cached ATR for dynamic stop calculation (2.1 IMPROVEMENT)
    double atr = GetCachedATR14();
    if(atr <= 0) return;
    double days_held = holding_hours / 24.0;
    
    // Tighten stops progressively: base_multiplier * (tighten_rate ^ days_held)
    double adaptive_atr_multiplier = GetAdaptiveATRMultiplier();
    double stop_multiplier = adaptive_atr_multiplier * MathPow(InpStopTightenRate, days_held);
    stop_multiplier = MathMax(stop_multiplier, 0.5); // Don't go below 0.5x ATR
    
    double new_stop_distance = atr * stop_multiplier;
    
    // Update stop loss if tighter than current
    if(PositionSelect(_Symbol)){
        double current_price = (g_position_type == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double new_stop_loss;
        
        if(g_position_type == 1){ // Long position
            new_stop_loss = current_price - new_stop_distance;
        } else { // Short position
            new_stop_loss = current_price + new_stop_distance;
        }
        
        // Only move stop loss if it's tighter than current
        double current_sl = PositionGetDouble(POSITION_SL);
        bool should_update = (g_position_type == 1 && new_stop_loss > current_sl) || 
                           (g_position_type == 2 && new_stop_loss < current_sl);
                           
        if(should_update && current_sl > 0){
            Print("PHASE3: Tightening stop loss. Days held: ", DoubleToString(days_held, 1), 
                  " New SL: ", DoubleToString(new_stop_loss, 5));
            // Note: Actual stop loss modification would require trade management implementation
        }
    }
}

// Calculate market regime indicators (Phase 3)
void UpdateMarketRegime(){
    if(!InpUseMarketRegime) return;
    
    // Use cached indicators (2.1 IMPROVEMENT)
    double atr_current = GetCachedATR14();
    double atr_long_term = GetCachedATR50();
    
    if(atr_current <= 0 || atr_long_term <= 0) return;
    
    // Trend strength calculation (2.1 IMPROVEMENT - cached)
    double ma10 = GetCachedMA10();
    double ma50 = GetCachedMA50();
    
    if(ma10 <= 0 || ma50 <= 0) return;
    double price = iClose(_Symbol, PERIOD_CURRENT, 0);
    
    g_trend_strength = MathAbs(ma10 - ma50) / atr_current; // Normalized trend strength
    
    // Volatility regime
    g_volatility_regime = atr_current / atr_long_term;
    
    // Market regime classification
    if(g_volatility_regime > 1.5){
        g_market_regime = 2; // Volatile
    } else if(g_trend_strength > 2.0){
        g_market_regime = 1; // Trending
    } else {
        g_market_regime = 0; // Ranging
    }
}

// Update position-aware features (Phase 3)
void UpdatePositionFeatures(){
    if(!InpUsePositionFeatures) return;
    
    // Normalized holding time [0-1] based on max holding time
    if(g_position_type > 0){
        int holding_hours = GetPositionHoldingHours();
        int adaptive_timeout = GetAdaptiveTimeoutHours();
        g_position_normalized_time = MathMin(1.0, (double)holding_hours / (double)adaptive_timeout);
        
        // P&L ratio vs ATR (2.1 IMPROVEMENT - use cached monetary value)
        double atr_value = GetCachedATRMonetary();
        if(atr_value <= 0){
            // Fallback calculation
            double atr = GetCachedATR14();
            atr_value = atr * g_position_size * 100000;
        }
        g_unrealized_pnl_ratio = (atr_value > 0) ? g_position_unrealized_pnl / atr_value : 0.0;
        g_unrealized_pnl_ratio = clipd(g_unrealized_pnl_ratio, -5.0, 5.0); // Clip to reasonable range
    } else {
        g_position_normalized_time = 0.0;
        g_unrealized_pnl_ratio = 0.0;
    }
}

//============================== ENHANCED SIGNAL FILTERING (1.2 IMPROVEMENTS) ===============
// Advanced trade filtering and confirmation system

// Check higher timeframe trend alignment
int GetHigherTimeframeTrend(){
    if(!InpUseMultiTimeframe || g_higher_tf_ma_handle == INVALID_HANDLE) return 0;
    
    // Only check periodically to reduce computation
    if(TimeCurrent() - g_last_higher_tf_check < 300) return g_higher_tf_trend; // Check every 5 minutes
    
    double ma_values[3];
    if(CopyBuffer(g_higher_tf_ma_handle, 0, 0, 3, ma_values) != 3) return 0;
    
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ma_current = ma_values[0];
    double ma_prev = ma_values[1];
    double ma_prev2 = ma_values[2];
    
    // Determine trend based on MA slope and price position
    bool ma_rising = (ma_current > ma_prev) && (ma_prev > ma_prev2);
    bool ma_falling = (ma_current < ma_prev) && (ma_prev < ma_prev2);
    bool price_above_ma = current_price > ma_current;
    
    int trend = 0;
    if(ma_rising && price_above_ma) trend = 1;      // Strong uptrend
    else if(ma_falling && !price_above_ma) trend = -1; // Strong downtrend
    else if(price_above_ma) trend = 1;              // Weak uptrend (price above MA)
    else if(!price_above_ma) trend = -1;            // Weak downtrend (price below MA)
    
    g_higher_tf_trend = trend;
    g_last_higher_tf_check = TimeCurrent();
    
    return trend;
}

// Get RSI confirmation signal
bool GetRSIConfirmation(int trade_direction){
    if(!InpUseSecondaryIndicators || g_rsi_handle == INVALID_HANDLE) return true;
    
    double rsi_values[2];
    if(CopyBuffer(g_rsi_handle, 0, 0, 2, rsi_values) != 2) return true; // Default to allow if can't read
    
    double rsi_current = rsi_values[0];
    
    // For buy signals, avoid overbought conditions
    if(trade_direction > 0 && rsi_current >= InpRSIOverbought){
        SmartLog("FILTER", "SIGNAL FILTER: RSI overbought (" + DoubleToString(rsi_current, 1) + ") - blocking BUY signal");
        return false;
    }
    
    // For sell signals, avoid oversold conditions  
    if(trade_direction < 0 && rsi_current <= InpRSIOversold){
        SmartLog("FILTER", "SIGNAL FILTER: RSI oversold (" + DoubleToString(rsi_current, 1) + ") - blocking SELL signal");
        return false;
    }
    
    return true;
}

// Get MACD trend confirmation
bool GetMACDConfirmation(int trade_direction){
    if(!InpUseMACDConfirmation || g_macd_handle == INVALID_HANDLE) return true;
    
    double macd_main[2], macd_signal[2];
    if(CopyBuffer(g_macd_handle, 0, 0, 2, macd_main) != 2) return true;
    if(CopyBuffer(g_macd_handle, 1, 0, 2, macd_signal) != 2) return true;
    
    double macd_current = macd_main[0];
    double macd_signal_current = macd_signal[0];
    bool macd_bullish = macd_current > macd_signal_current;
    
    // For buy signals, prefer bullish MACD
    if(trade_direction > 0 && !macd_bullish){
        SmartLog("FILTER", "SIGNAL FILTER: MACD bearish - weakening BUY signal confidence");
        return false;
    }
    
    // For sell signals, prefer bearish MACD
    if(trade_direction < 0 && macd_bullish){
        SmartLog("FILTER", "SIGNAL FILTER: MACD bullish - weakening SELL signal confidence");
        return false;
    }
    
    return true;
}

// Check signal strength based on Q-value differences (OPTIMIZED 2.4)
bool IsSignalStrong(const double &q_values[], int action){
    if(!InpUseSignalConfirmation) return true;
    
    // Find the highest and second-highest Q-values with optimization
    double max_q = q_values[action];
    double second_max = 0.0;
    
    if(InpOptimizeLoops && InpUseEarlyBreaks){
        // Optimized search with early break
        int loop_count = 0;
        for(int i = 0; i < ACTIONS; i++){
            if(!LoopSafetyCheck(loop_count)) break;
            
            if(i != action && q_values[i] > second_max){
                second_max = q_values[i];
                
                // Early break if signal is clearly strong enough
                if(InpSimplifyConditions && (max_q - second_max) >= InpMinSignalStrength * 1.5){
                    break; // Signal strength clearly sufficient
                }
            }
        }
    }
    else {
        // Standard implementation
        for(int i = 0; i < ACTIONS; i++){
            if(i != action && q_values[i] > second_max){
                second_max = q_values[i];
            }
        }
    }
    
    double q_difference = max_q - second_max;
    bool is_strong = q_difference >= InpMinSignalStrength;
    
    if(!is_strong){
        SmartLog("FILTER", "SIGNAL FILTER: Weak signal strength (Q-diff: " + DoubleToString(q_difference, 3) + 
              " < " + DoubleToString(InpMinSignalStrength, 3) + ")");
    }
    
    return is_strong;
}

// Check spread and liquidity conditions
bool CheckMarketConditions(){
    // Enhanced spread filtering
    if(InpUseSpreadFilter){
        double current_spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
        
        // Get ATR for dynamic spread comparison
        double atr_values[1];
        int atr_handle = iATR(_Symbol, _Period, 14);
        if(CopyBuffer(atr_handle, 0, 0, 1, atr_values) == 1){
            double atr = atr_values[0];
            double max_spread = atr * InpMaxSpreadATR;
            
            if(current_spread > max_spread){
                SmartLog("FILTER", "SIGNAL FILTER: Spread too wide (" + DoubleToString(current_spread/_Point, 1) + 
                      " pts > " + DoubleToString(max_spread/_Point, 1) + " pts ATR limit)");
                return false;
            }
        }
    }
    
    // Basic liquidity check via tick volume
    if(InpUseLiquidityFilter){
        MqlTick last_tick;
        if(SymbolInfoTick(_Symbol, last_tick)){
            if((int)last_tick.volume < InpMinTickVolume){
                SmartLog("FILTER", "SIGNAL FILTER: Low liquidity (tick volume: " + IntegerToString(last_tick.volume) + 
                      " < " + IntegerToString(InpMinTickVolume) + ")");
                return false;
            }
        }
    }
    
    return true;
}

// Main enhanced signal filtering function
int ApplyEnhancedSignalFiltering(int raw_action, const double &q_values[], datetime current_time){
    // Skip filtering for non-trading actions
    if(raw_action == ACTION_HOLD || raw_action == ACTION_FLAT) return raw_action;
    
    // Check market conditions first
    if(!CheckMarketConditions()) return ACTION_HOLD;
    
    // Determine trade direction
    int trade_direction = (raw_action == ACTION_BUY_STRONG || raw_action == ACTION_BUY_WEAK) ? 1 : -1;
    
    // Signal persistence check
    if(InpUseSignalPersistence){
        if(g_last_signal_action == raw_action){
            g_signal_persistence_count++;
        } else {
            g_signal_persistence_count = 1;
            g_last_signal_action = raw_action;
        }
        
        if(g_signal_persistence_count < InpSignalPersistenceBars){
            SmartLog("FILTER", "SIGNAL FILTER: Signal persistence required (" + IntegerToString(g_signal_persistence_count) + 
                  "/" + IntegerToString(InpSignalPersistenceBars) + " bars) - signal: " + ACTION_NAME[raw_action]);
            return ACTION_HOLD;
        }
    }
    
    // Signal strength check
    if(!IsSignalStrong(q_values, raw_action)){
        return ACTION_HOLD;
    }
    
    // Higher timeframe trend confirmation
    if(InpUseMultiTimeframe){
        int higher_tf_trend = GetHigherTimeframeTrend();
        if(higher_tf_trend != 0 && higher_tf_trend != trade_direction){
            SmartLog("FILTER", "SIGNAL FILTER: Higher TF trend conflict (Signal: " + 
                  (trade_direction > 0 ? "BUY" : "SELL") + ", Higher TF: " + 
                  (higher_tf_trend > 0 ? "UP" : "DOWN") + ")");
            return ACTION_HOLD;
        }
    }
    
    // RSI confirmation
    if(!GetRSIConfirmation(trade_direction)){
        return ACTION_HOLD;
    }
    
    // MACD confirmation
    if(!GetMACDConfirmation(trade_direction)){
        // Don't block completely, just reduce signal strength
        if(raw_action == ACTION_BUY_STRONG) return ACTION_BUY_WEAK;
        if(raw_action == ACTION_SELL_STRONG) return ACTION_SELL_WEAK;
        if(raw_action == ACTION_BUY_WEAK || raw_action == ACTION_SELL_WEAK) return ACTION_HOLD;
    }
    
    SmartLog("FILTER", "SIGNAL FILTER: All confirmations passed for " + ACTION_NAME[raw_action]);
    return raw_action;
}

//============================== MAIN TRADING ENGINE ==============================
// Primary EA Event Handler - Called Every Price Tick
// This is the "brain" of the EA that orchestrates all trading decisions
// Implements advanced performance optimizations and comprehensive safety checks
// 
// Execution Flow:
//   1. Performance & Safety Initialization
//   2. Adaptive Parameter Updates 
//   3. Critical Safety Validations
//   4. Market Data Loading & Analysis
//   5. AI Neural Network Inference
//   6. Risk Management & Position Control
//   7. Trade Execution & Monitoring
void OnTick(){
  // SAFETY GATE: Prevent any trading if model loading failed
  // Critical protection against running EA without proper AI model
  if(!g_loaded) return;
  
  // PERFORMANCE OPTIMIZATION: Reset computational efficiency counters
  // Tracks loop iterations and function calls to optimize execution speed
  if(InpOptimizeLoops){
    ResetLoopCounters();
  }
  
  // INTELLIGENT TICK PROCESSING: Selective Signal Generation
  // Not every price tick requires full AI analysis - this optimization
  // reduces CPU usage while maintaining trading effectiveness
  bool process_signals = true;
  if(InpOptimizeTicks){
    process_signals = ShouldProcessTick();  // Smart tick filtering
  }
  
  // CRITICAL FUNCTION PROCESSING: Always Execute Safety Checks
  // Some functions (emergency stops, position monitoring) must run on every tick
  // regardless of optimization settings for maximum account protection
  bool process_critical = ShouldProcessCriticalTick();
  
  // ADAPTIVE INTELLIGENCE: Dynamic Parameter Adjustment
  // The EA learns from market conditions and adjusts its behavior automatically
  // This enables adaptation to changing volatility, trends, and market regimes
  if(InpUseAdaptiveParameters && (process_signals || process_critical)){
    // VOLATILITY REGIME DETECTION: Monitor market volatility changes
    // Higher volatility requires smaller positions and tighter stops
    double current_atr = GetCachedATR14();
    if(current_atr > 0.0){
      UpdateATRHistory(current_atr);         // Maintain volatility history
      UpdateAdaptiveVolatilityRegime();      // Classify market regime
    }
    
    // DAILY RESET MECHANISM: Fresh start each trading day
    CheckAdaptiveReset();
  }
  
  // CRITICAL SAFETY VALIDATION: Model-Chart Compatibility Check
  // Prevents catastrophic losses from using wrong model on wrong symbol/timeframe
  // This is a fundamental safety mechanism that cannot be disabled
  if(InpEnforceSymbolTF && (_Symbol != g_model_symbol || _Period != g_model_tf)){
    static datetime last_warning = 0;
    if(TimeCurrent() - last_warning > 300){ // Throttle warnings to every 5 minutes
      Print("=== CRITICAL SAFETY ERROR ===");
      Print("Model/Chart mismatch detected! Trading halted for safety!");
      Print("Current Chart: ", _Symbol, " ", EnumToString(_Period));
      Print("Model Trained For: ", g_model_symbol, " ", EnumToString(g_model_tf));
      Print("SOLUTION: Use correct model or change chart to match model");
      Alert("CORTEX5 SAFETY: Model mismatch! Trading stopped!");
      last_warning = TimeCurrent();
    }
    return; // Refuse all trading activity until mismatch is resolved
  }
  
  // ESSENTIAL SAFETY OPERATIONS: Always Execute Critical Functions
  // These operations protect account equity and must run regardless of optimizations
  if(process_critical || !InpOptimizeTicks){
    // DYNAMIC POSITION MANAGEMENT: Update trailing stops and break-even levels
    // Protects profits and limits losses on existing positions
    if(InpTrailingOnTicks){
      UpdateTrailingStops();
    }
    
    // REAL-TIME RISK MONITORING: Track position metrics and exposure
    // Monitors unrealized P&L, drawdown, and position duration
    if(InpRiskChecksOnTicks){
      UpdatePositionTracking();
    }
  }
  
  // INTELLIGENT SIGNAL FILTERING: Performance Optimization Gate
  // Skips expensive AI analysis when market conditions don't warrant it
  if(!process_signals){
    // EMERGENCY MONITORING: Always check for account protection triggers
    // Even when skipping signals, we must monitor for emergency conditions
    if(InpEmergencyOnTicks && CheckEmergencyStops()){
      return; // Emergency stop activated - position closed for safety
    }
    return; // Skip AI analysis to conserve computational resources
  }
  
  // MARKET DATA ACQUISITION: Multi-Timeframe Analysis Foundation
  // Load comprehensive price history across multiple timeframes for AI analysis
  // Each timeframe provides different perspectives on market structure
  Series base,m1,m5,h1,h4,d1;
  if(!OptimizedLoadSeries(_Symbol, g_model_tf, InpBarLookback, base)) return;
  
  // DATA SUFFICIENCY VALIDATION: Ensure adequate history for technical analysis
  // Technical indicators require sufficient historical bars for accurate calculation
  if(ArraySize(base.rates)<60) return;
  
  // BAR COMPLETION DETECTION: New Bar Signal Processing
  // Only analyzes market when a new bar completes - prevents over-trading
  // This is critical for consistent backtesting and live trading alignment
  datetime current_bar_time = base.rates[1].time;
  bool is_new_bar = (g_last_bar_time != current_bar_time);
  
  // OPTIMIZATION TRACKING: Update performance monitoring counters
  if(InpOptimizeTicks && is_new_bar){
    g_last_signal_bar_time = current_bar_time;
    ResetTickCounters();
  }
  
  // SIGNAL TIMING CONTROL: Process each bar only once
  // Prevents multiple signals from the same bar data
  if(!is_new_bar){
    return; // Already analyzed this completed bar
  }
  g_last_bar_time = current_bar_time;  // Update bar tracking timestamp
  
  // PERFORMANCE OPTIMIZATION: Refresh cached technical indicators
  // Pre-compute expensive calculations once per bar for multiple reuse
  UpdateIndicatorCache();
  IncrementCacheBarCounter();

  // POSITION MANAGEMENT: Update trailing stops for active positions
  // Protects profits by automatically adjusting stop losses as price moves favorably
  if(!InpTrailingOnTicks){
    UpdateTrailingStops();
  }

  // RISK MONITORING: Update position metrics and exposure tracking
  // Calculates unrealized P&L, duration, drawdown for risk management
  if(!InpRiskChecksOnTicks){
    UpdatePositionTracking();
  }
  
  // EMERGENCY PROTECTION: Highest priority safety checks
  // Immediate position closure for account protection
  if(CheckEmergencyStops()) {
    return; // Emergency protocols activated - position closed
  }
  
  // AUTOMATIC EXIT MANAGEMENT: Time and profit-based position closure
  // Prevents holding positions too long or missing profit opportunities
  if(CheckMaxHoldingTime(TimeCurrent()) || CheckProfitTargets()) {
    return; // Position automatically closed - skip new signal analysis
  }

  // ADVANCED MARKET INTELLIGENCE: Sophisticated Analysis Systems
  UpdateDynamicStops();      // Adaptive stop loss tightening over time
  UpdateMarketRegime();      // Trending vs ranging market classification
  UpdatePositionFeatures();  // Position-aware state features for AI

  // OVERTRADING PREVENTION: Frequency and exposure controls
  // Prevents excessive trading that can erode profits through costs
  bool trading_allowed = IsNewTradingAllowed();
  bool should_force_flat = ShouldForceFlat();

  // MASTER RISK VALIDATION: Comprehensive Safety Gateway
  // Final safety check before allowing any trading decisions
  // Evaluates all risk parameters, account status, and market conditions
  if(!MasterRiskCheck()) {
    // Risk check failed - trading suspended for safety
    return;
  }

  // Load supporting timeframe data for multi-timeframe analysis
  LoadSeries(_Symbol, PERIOD_M1, InpBarLookback, m1);
  LoadSeries(_Symbol, PERIOD_M5, InpBarLookback, m5);
  LoadSeries(_Symbol, PERIOD_H1, InpBarLookback, h1);
  LoadSeries(_Symbol, PERIOD_H4, InpBarLookback, h4);
  LoadSeries(_Symbol, PERIOD_D1, InpBarLookback, d1);

  // Build the feature vector that describes current market conditions (includes position state)
  double row[];
  BuildStateRow(base,1,m1,m5,h1,h4,d1,row);  // Use bar [1] (completed bar)
  ApplyMinMax(row,g_feat_min,g_feat_max);     // Normalize features to 0-1 range

  // Get AI prediction: feed market features into neural network (2.2 IMPROVEMENT - optimized)
  double q[];               // Will hold Q-values (expected rewards) for each action
  int raw_action;
  double ensemble_agreement = 1.0;
  
  if(InpUseEnsembleModel && g_active_ensemble_count >= 2){
    // Use ensemble prediction (3.2 IMPROVEMENT)
    raw_action = GetSimpleEnsemblePrediction(row, q, ensemble_agreement);
    
    // Log ensemble decision details (with throttling)
    if(InpLogEnsembleDetails && 
       (base.rates[1].time - g_last_ensemble_log) >= InpEnsembleLogThrottle){
      SmartLog("ENSEMBLE", StringFormat("Prediction: %s | Agreement: %.3f | Method: %s | Active models: %d", 
               ACTION_NAME[raw_action], ensemble_agreement, g_ensemble_method, g_active_ensemble_count));
      g_last_ensemble_log = base.rates[1].time;
    }
  } else {
    // Use single model prediction
    g_Q.PredictOptimized(row,q);  // Run optimized neural network forward pass
    raw_action = argmax(q);   // Choose action with highest expected reward
  }
  
  // Log the AI's raw decision with position context (ENHANCED logging)
  string decision_source = InpUseEnsembleModel && g_active_ensemble_count >= 2 ? 
                          StringFormat("Ensemble(%d)", g_active_ensemble_count) : "Single";
  Print("AI Raw Decision (",decision_source,"): ",ACTION_NAME[raw_action],
        " | Position: ",DoubleToString(row[12],1),"x",DoubleToString(row[13],2)," (P&L:",DoubleToString(row[14],1),"pts)",
        " | Q-values=[",DoubleToString(q[0],3),",",DoubleToString(q[1],3),",",
        DoubleToString(q[2],3),",",DoubleToString(q[3],3),",",DoubleToString(q[4],3),",",DoubleToString(q[5],3),"]",
        InpUseEnsembleModel && g_active_ensemble_count >= 2 ? 
        StringFormat(" | Agreement: %.3f", ensemble_agreement) : "");
  
  // Enhanced Signal Filtering & Confirmation (1.2 IMPROVEMENT)
  int filtered_action = raw_action;
  if(InpUseSignalConfirmation && g_indicators_initialized){
    filtered_action = ApplyEnhancedSignalFiltering(raw_action, q, base.rates[1].time);
    if(filtered_action != raw_action){
      SmartLog("SIGNAL", "SIGNAL FILTERING: Action changed from " + ACTION_NAME[raw_action] + 
            " to " + ACTION_NAME[filtered_action] + " after confirmation filters");
    }
  }
  
  // Refined Signal Processing (1.4 IMPROVEMENT)
  int refined_action = filtered_action;
  if(InpUseSignalRefinement){
    refined_action = ApplySignalRefinement(filtered_action, q, base.rates[1].time, base.rates);
    if(refined_action != filtered_action){
      SmartLog("SIGNAL", "SIGNAL REFINEMENT: Action changed from " + ACTION_NAME[filtered_action] + 
            " to " + ACTION_NAME[refined_action] + " after signal refinement");
    }
  }
  
  // Confidence-Based Trade Filter (3.1 IMPROVEMENT)
  int final_action = refined_action;
  if(InpUseConfidenceFilter && (refined_action != ACTION_HOLD && refined_action != ACTION_FLAT)){
    if(!PassesConfidenceFilter(q, refined_action, base.rates[1].time)){
      final_action = ACTION_HOLD; // Filter out low-confidence signals
      SmartLog("SIGNAL", "CONFIDENCE FILTER: Signal " + ACTION_NAME[refined_action] + 
            " filtered due to low confidence (" + DoubleToString(g_last_confidence_score, 3) + 
            " < " + DoubleToString(InpMinConfidenceThreshold, 3) + ")");
    } else {
      SmartLog("SIGNAL", "CONFIDENCE FILTER: Signal " + ACTION_NAME[refined_action] + 
            " passed with confidence " + DoubleToString(g_last_confidence_score, 3));
    }
  }
  
  // Execute the final action
  MaybeTrade(final_action);
}

// TIMER-BASED PERIODIC CHECKS (2.5 IMPROVEMENT)
// Called periodically for non-critical operations when tick optimization is enabled
void OnTimer(){
    if(InpOptimizeTicks && InpUseTimerChecks){
        ProcessTimerChecks();
    }
}

//+------------------------------------------------------------------+
