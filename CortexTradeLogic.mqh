//+------------------------------------------------------------------+
//|                                            CortexTradeLogic.mqh |
//|                          Unified Trade Logic Module for Cortex5 |
//|                    Shared functions for EA, Backtester, & Tools |
//+------------------------------------------------------------------+
//| This include file contains common trade logic functions used    |
//| across all Cortex5 components to ensure consistency and prevent |
//| discrepancies between live trading, backtesting, and analysis.  |
//|                                                                  |
//| IMPROVEMENT 3.4: Unified Module for Trade Logic                 |
//| - Centralized entry/exit conditions                             |
//| - Shared risk management calculations                            |
//| - Consistent position sizing logic                              |
//| - Common ATR and technical analysis functions                   |
//+------------------------------------------------------------------+

#ifndef CORTEX_TRADE_LOGIC_MQH
#define CORTEX_TRADE_LOGIC_MQH

#property copyright "Cortex Trading System"
#property version   "1.00"

//============================== CONSTANTS ==============================
// These constants must match across all components to ensure model compatibility

// STATE_SIZE defines the number of input features fed to the neural network
// This includes price data, technical indicators, volume metrics, and market regime features
// CRITICAL: Changing this requires retraining all models - must remain consistent
#define STATE_SIZE 45  // IMPROVEMENT 4.3: Updated to match enhanced 45-feature state vector

// ACTIONS defines the total number of trading decisions the AI can make
// The system uses 6 discrete actions to provide granular control over position sizing
#define ACTIONS 6

// Trading Action Definitions - Primary Constants
// These represent the core decision space for the reinforcement learning agent
#define BUY_STRONG  0  // High-confidence long position (larger size)
#define BUY_WEAK    1  // Low-confidence long position (smaller size)
#define SELL_STRONG 2  // High-confidence short position (larger size)
#define SELL_WEAK   3  // Low-confidence short position (smaller size)
#define HOLD        4  // Maintain current position (no new trades)
#define FLAT        5  // Close all positions and stay neutral

// Backward Compatibility Constants - ACTION_ Prefixed
// Maintained for legacy code compatibility across different system versions
#define ACTION_BUY_STRONG  0
#define ACTION_BUY_WEAK    1
#define ACTION_SELL_STRONG 2
#define ACTION_SELL_WEAK   3
#define ACTION_HOLD        4
#define ACTION_FLAT        5

// Position Type Definitions
// Used internally to track current market exposure and calculate P&L
#define POS_NONE  0  // No open position
#define POS_LONG  1  // Long position (profit from price increases)
#define POS_SHORT 2  // Short position (profit from price decreases)

//============================== STRUCTURES ==============================

// Comprehensive Trade Record Structure
// This structure captures all essential trade data for performance analysis,
// risk management, and system optimization. Used consistently across all components
// to ensure unified trade tracking and reporting.
struct TradeRecord {
    // Core Trade Timing Information
    datetime open_time;        // When the position was opened (entry timestamp)
    datetime close_time;       // When the position was closed (exit timestamp)
    
    // Trade Decision and Direction
    int      action;           // Original AI action that triggered this trade (0-5)
    int      position_type;    // Market direction: POS_LONG, POS_SHORT, or POS_NONE
    
    // Price and Size Information
    double   entry_price;      // Actual fill price when entering the position
    double   exit_price;       // Actual fill price when closing the position
    double   lots;             // Position size in standard lots
    
    // Financial Results
    double   profit_loss;      // Net profit/loss for this trade (including costs)
    double   balance_after;    // Account balance immediately after trade completion
    double   drawdown_pct;     // Current drawdown as percentage of peak equity
    
    // IMPROVEMENT 7.2: Advanced Trade Analytics
    double   mae;              // Maximum Adverse Excursion (worst unrealized loss)
    double   mfe;              // Maximum Favorable Excursion (best unrealized profit)
    int      holding_time_hours; // Total trade duration in hours
    string   exit_reason;      // Descriptive reason for trade closure
    double   commission;       // Total trading costs (spread + commission)
    double   confidence_score; // AI model's confidence level for this trade (0.0-1.0)
    
    // Copy constructor to avoid deprecation warnings
    TradeRecord(const TradeRecord &other) {
        open_time = other.open_time;
        close_time = other.close_time;
        action = other.action;
        position_type = other.position_type;
        entry_price = other.entry_price;
        exit_price = other.exit_price;
        lots = other.lots;
        profit_loss = other.profit_loss;
        balance_after = other.balance_after;
        drawdown_pct = other.drawdown_pct;
        mae = other.mae;
        mfe = other.mfe;
        holding_time_hours = other.holding_time_hours;
        exit_reason = other.exit_reason;
        commission = other.commission;
        confidence_score = other.confidence_score;
    }
    
    // Default constructor
    TradeRecord() {
        open_time = 0;
        close_time = 0;
        action = HOLD;
        position_type = POS_NONE;
        entry_price = 0.0;
        exit_price = 0.0;
        lots = 0.0;
        profit_loss = 0.0;
        balance_after = 0.0;
        drawdown_pct = 0.0;
        mae = 0.0;
        mfe = 0.0;
        holding_time_hours = 0;
        exit_reason = "";
        commission = 0.0;
        confidence_score = 0.5;
    }
};

// Comprehensive Risk Metrics Structure
// Aggregates key performance indicators for risk assessment and system evaluation
// Used by diagnostic tools and live monitoring to ensure trading safety
struct RiskMetrics {
    double max_drawdown;       // Maximum peak-to-trough equity decline (worst case)
    double current_drawdown;   // Current equity decline from recent peak
    double win_rate;           // Percentage of profitable trades (0-100)
    double profit_factor;      // Ratio of gross profit to gross loss
    double sharpe_ratio;       // Risk-adjusted return metric (higher is better)
    double risk_reward_ratio;  // Average win size divided by average loss size
    int    total_trades;       // Total number of completed trades
    int    winning_trades;     // Number of profitable trades
    int    losing_trades;      // Number of unprofitable trades
};

//============================== ATR AND TECHNICAL ANALYSIS ==============================

// Average True Range (ATR) Calculation
// ATR measures market volatility and is crucial for:
// - Position sizing (larger positions in lower volatility)
// - Stop loss placement (stops farther out in high volatility)
// - Risk management (adjusting exposure based on market conditions)
// This function ensures consistent volatility measurement across all system components
double CalculateATR(string symbol, ENUM_TIMEFRAMES timeframe, int period, int shift = 0) {
    double atr_values[];
    // Attempt to retrieve ATR indicator values from MetaTrader 5
    // Uses built-in iATR indicator for accurate volatility measurement
    if(CopyBuffer(iATR(symbol, timeframe, period), 0, shift, 1, atr_values) <= 0) {
        Print("ERROR: Failed to get ATR values for ", symbol);
        return 0.0;  // Return 0 on failure to signal error condition
    }
    return atr_values[0];  // Return most recent ATR value
}

// Proxy ATR Calculation for Historical Data
// Used during backtesting when live indicator data isn't available
// Manually calculates True Range from OHLC price data to maintain accuracy
// True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
double ATR_Proxy(const MqlRates &rates[], int index, int period) {
    // Boundary check to ensure we have sufficient historical data
    if(index + period >= ArraySize(rates)) return 0.0;
    
    double sum = 0.0;  // Accumulator for True Range values
    int count = 0;     // Counter for valid calculations
    
    // Calculate True Range for each period and accumulate
    for(int i = 0; i < period && (index + i + 1) < ArraySize(rates); i++) {
        int idx = index + i;
        
        // True Range Calculation - captures intraday and gap volatility
        // TR = max of: (High-Low), |High-PrevClose|, |Low-PrevClose|
        // This accounts for overnight gaps and intraday price movement
        double tr = MathMax(rates[idx].high - rates[idx].low, 
                           MathMax(MathAbs(rates[idx].high - rates[idx + 1].close), 
                                  MathAbs(rates[idx].low - rates[idx + 1].close)));
        sum += tr;
        count++;
    }
    
    // Return average True Range over the specified period
    return (count > 0) ? sum / count : 0.0;
}

// Exponential Moving Average (EMA) Slope Calculation
// Determines the directional momentum of price trends by comparing current EMA to previous EMA
// Positive slope indicates upward trend, negative slope indicates downward trend
// Used as a feature in the AI model's state vector for trend analysis
double EMA_Slope(const MqlRates &rates[], int index, int period) {
    // SAFETY: Comprehensive bounds checking to prevent array access violations
    // Critical for backtesting where array size is limited by historical data
    int array_size = ArraySize(rates);
    if(index >= array_size || index < 0 || period <= 0) return 0.0;
    if(index + period + 1 >= array_size) return 0.0;
    
    // EMA smoothing factor - standard exponential weighting formula
    double alpha = 2.0 / (period + 1);
    double ema = 0.0;  // Current EMA value
    int n = 0;         // Sample counter
    
    // Calculate Current EMA Value
    // Uses exponential weighting: EMA = α * Current_Price + (1-α) * Previous_EMA
    for(int k = period; k >= 0; k--) {
        int idx = index + k;
        if(idx >= array_size || idx < 0) continue;  // Skip invalid indices
        
        if(n == 0) ema = rates[idx].close;  // Initialize with first price
        else ema = alpha * rates[idx].close + (1 - alpha) * ema;  // Apply EMA formula
        n++;
    }
    
    // Calculate Previous EMA Value (one period earlier)
    // This allows us to determine the rate of change (slope) of the trend
    double ema_prev = 0.0;  // Previous EMA value
    n = 0;                  // Reset counter
    for(int k = period + 1; k >= 1; k--) {
        int idx = index + k;
        if(idx >= array_size || idx < 0) continue;  // Skip invalid indices
        
        if(n == 0) ema_prev = rates[idx].close;  // Initialize with first price
        else ema_prev = alpha * rates[idx].close + (1 - alpha) * ema_prev;  // Apply EMA formula
        n++;
    }
    
    // Return EMA slope (rate of change)
    // Positive = uptrend (bullish), Negative = downtrend (bearish)
    return ema - ema_prev;
}

//============================== POSITION SIZING ==============================

// Unified Position Size Calculation with Risk Management
// Implements Kelly Criterion-inspired position sizing that adapts to:
// - Signal strength (strong vs weak signals get different sizes)
// - Market volatility (ATR-based adjustments)
// - Account risk tolerance (percentage-based risk limits)
// - Broker constraints (minimum/maximum lot sizes)
// This ensures consistent risk management across all system components
double CalculateUnifiedPositionSize(bool is_strong_signal,     // True for STRONG signals, false for WEAK
                                   double risk_percent,        // Percentage of equity to risk per trade
                                   double atr_multiplier,      // ATR multiplier for stop loss distance
                                   double lots_strong = 0.1,   // Default lot size for strong signals
                                   double lots_weak = 0.05,    // Default lot size for weak signals
                                   bool use_risk_sizing = true // Enable dynamic risk-based sizing
                                   ) {
    
    // Simple fixed sizing if dynamic risk management is disabled
    if(!use_risk_sizing) {
        return is_strong_signal ? lots_strong : lots_weak;
    }
    
    // Get current market volatility for position sizing adjustments
    // Higher volatility = smaller positions to maintain constant risk
    double atr = CalculateATR(_Symbol, PERIOD_CURRENT, 14);
    if(atr <= 0) return is_strong_signal ? lots_strong : lots_weak; // Fallback to fixed sizing
    
    // Calculate stop loss distance in points
    // Wider stops in volatile markets, tighter stops in calm markets
    double stop_loss_points = atr * atr_multiplier / _Point;
    
    // Calculate maximum risk amount based on account equity
    // This prevents risking more than the specified percentage on any single trade
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = equity * risk_percent / 100.0;
    
    // Adjust risk based on signal strength
    // Strong signals get full risk allocation, weak signals get reduced allocation
    double signal_multiplier = is_strong_signal ? 1.0 : 0.5;
    risk_amount *= signal_multiplier;
    
    // Get symbol-specific trading parameters for accurate calculations
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);  // Value per tick in account currency
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);    // Minimum price increment
    
    // Validate trading parameters before calculation
    if(tick_value <= 0 || tick_size <= 0 || stop_loss_points <= 0) {
        return is_strong_signal ? lots_strong : lots_weak; // Fallback to fixed sizing
    }
    
    // Calculate optimal lot size using position sizing formula:
    // Lot Size = Risk Amount / (Stop Loss Distance × Tick Value per Point)
    double lots = risk_amount / (stop_loss_points * tick_value / tick_size);
    
    // Apply broker-specific constraints to ensure valid order placement
    double min_lots = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);   // Minimum allowed position size
    double max_lots = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);   // Maximum allowed position size
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);  // Lot size increment (e.g., 0.01)
    
    // Enforce minimum lot size requirement
    lots = MathMax(lots, min_lots);
    // Enforce maximum lot size limit
    lots = MathMin(lots, max_lots);
    // Round to valid lot step increment
    lots = MathRound(lots / lot_step) * lot_step;
    
    return lots;
}

//============================== ENTRY CONDITIONS ==============================

// Comprehensive Trade Entry Validation System
// Acts as a gatekeeper to prevent trades during unfavorable market conditions
// Checks multiple filters including:
// - Action validity (no HOLD/FLAT trades)
// - Time-based filters (avoid weekends, news hours)
// - Spread filters (avoid high-cost trading periods)
// - Market state validation
bool ValidateTradeEntry(int action,                          // AI-generated trading action
                       double confidence_threshold = 0.0,   // Minimum confidence level (if available)
                       bool use_time_filters = true,        // Enable time-based trade filtering
                       bool use_spread_filter = true,       // Enable spread-based trade filtering
                       double max_spread_points = 20.0      // Maximum allowed spread in points
                       ) {
    
    // Validate action is within expected range
    if(action < 0 || action >= ACTIONS) {
        return false;  // Invalid action code
    }
    
    // Filter out non-trading actions
    // HOLD and FLAT are portfolio management actions, not entry signals
    if(action == HOLD || action == FLAT) {
        return false;  // These actions don't create new positions
    }
    
    // Apply time-based trading restrictions
    if(use_time_filters) {
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        
        // Weekend Filter: Avoid trading during market closure
        // Day 0 = Sunday, Day 6 = Saturday (market closed)
        if(dt.day_of_week == 0 || dt.day_of_week == 6) {
            return false;  // Market closed on weekends
        }
        
        // Trading Hours Filter: Avoid low liquidity periods
        // Hours 2-22 GMT typically have good liquidity for major pairs
        // Avoids Asian session gaps and late NY session thin liquidity
        if(dt.hour < 2 || dt.hour > 22) {
            return false;  // Outside optimal trading hours
        }
    }
    
    // Apply spread-based cost filtering
    if(use_spread_filter) {
        // Get current bid-ask spread in points
        double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
        
        // Reject trades when spreads are too wide (high transaction costs)
        // Wide spreads often occur during news events or low liquidity
        if(spread > max_spread_points * _Point) {
            return false;  // Spread too wide - high transaction costs
        }
    }
    
    return true;  // All validation checks passed - trade entry allowed
}

//============================== EXIT CONDITIONS ==============================

// Comprehensive Position Exit Logic
// Implements multiple exit criteria to protect capital and lock in profits:
// - Profit targets based on ATR multiples
// - Time-based exits to avoid holding positions too long
// - Emergency stop losses for damage control
// - Unrealized P&L monitoring
// This function ensures consistent exit logic across all trading components
bool CheckUnifiedExitConditions(int position_type,               // Current position direction
                                double entry_price,             // Original entry price
                                datetime entry_time,            // When position was opened
                                double current_price,           // Current market price
                                double atr_value,               // Current ATR for targets
                                double profit_target_atr = 1.8, // ATR multiple for profit target
                                int max_holding_hours = 72,     // Maximum position hold time
                                double emergency_stop_loss = 150.0, // Emergency loss limit ($)
                                bool use_profit_targets = true, // Enable profit taking
                                bool use_time_limits = true,    // Enable time-based exits
                                bool use_emergency_stops = true // Enable emergency stops
                                ) {
    
    // Skip exit check if no position is open
    if(position_type == POS_NONE) return false;
    
    // Calculate how long the position has been held
    datetime current_time = TimeCurrent();
    int holding_hours = (int)((current_time - entry_time) / 3600);
    
    // Calculate current unrealized profit/loss
    double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    double lots = 0.1; // Default lot size - TODO: should be passed as parameter
    double unrealized_pnl = 0;
    
    // Calculate P&L based on position direction
    if(position_type == POS_LONG) {
        // Long position: profit when current price > entry price
        unrealized_pnl = (current_price - entry_price) * lots * contract_size;
    } else {
        // Short position: profit when current price < entry price
        unrealized_pnl = (entry_price - current_price) * lots * contract_size;
    }
    
    // EMERGENCY STOP LOSS: Prevent catastrophic losses
    // Exits immediately if loss exceeds emergency threshold
    if(use_emergency_stops && unrealized_pnl < -emergency_stop_loss) {
        return true; // Force exit - emergency stop triggered
    }
    
    // TIME LIMIT CHECK: Prevent holding positions indefinitely
    // Long-held positions often become stale and less profitable
    if(use_time_limits && holding_hours > max_holding_hours) {
        return true; // Force exit - maximum hold time exceeded
    }
    
    // PROFIT TARGET CHECK: Lock in gains when target is reached
    // Uses ATR-based targets that adapt to market volatility
    if(use_profit_targets && unrealized_pnl > 0) {
        double profit_target = atr_value * profit_target_atr * lots * contract_size;
        if(unrealized_pnl >= profit_target) {
            return true; // Take profit - target reached
        }
    }
    
    // EXTENSIBLE EXIT FRAMEWORK
    // Additional exit conditions can be added here:
    // - Trailing stops (dynamic stop loss adjustment)
    // - Technical indicator reversals (RSI overbought/oversold)
    // - Volatility breakouts (unusual market conditions)
    // - News event filters (close before major announcements)
    
    return false;  // No exit conditions met - continue holding position
}

//============================== RISK MANAGEMENT ==============================

// ATR-Based Stop Loss and Take Profit Calculation
// Dynamically sets risk management levels based on market volatility:
// - Stop losses are placed at ATR multiples to avoid noise-induced exits
// - Take profits are set using risk-reward ratios for consistent profitability
// - All levels are normalized to broker tick sizes for valid order placement
void CalculateStopAndTarget(double entry_price,      // Trade entry price
                           int position_type,        // POS_LONG or POS_SHORT
                           double atr_value,         // Current ATR value
                           double atr_multiplier,    // ATR multiple for stop distance
                           double rr_ratio,          // Risk-reward ratio (e.g., 1:2)
                           double &stop_loss,        // Output: calculated stop loss price
                           double &take_profit       // Output: calculated take profit price
                           ) {
    
    // Calculate stop loss distance based on market volatility
    // Wider stops in volatile markets, tighter stops in calm markets
    double atr_distance = atr_value * atr_multiplier;
    
    // Set stop loss and take profit based on position direction
    if(position_type == POS_LONG) {
        // Long position: stop below entry, target above entry
        stop_loss = entry_price - atr_distance;                    // Stop loss below entry
        take_profit = entry_price + (atr_distance * rr_ratio);     // Target using R:R ratio
    } else if(position_type == POS_SHORT) {
        // Short position: stop above entry, target below entry
        stop_loss = entry_price + atr_distance;                    // Stop loss above entry
        take_profit = entry_price - (atr_distance * rr_ratio);     // Target using R:R ratio
    } else {
        // No position - set levels to zero
        stop_loss = 0;
        take_profit = 0;
    }
    
    // Normalize prices to valid tick increments
    // This ensures orders are placed at broker-acceptable price levels
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tick_size > 0) {
        stop_loss = MathRound(stop_loss / tick_size) * tick_size;      // Round stop to valid tick
        take_profit = MathRound(take_profit / tick_size) * tick_size;  // Round target to valid tick
    }
}

// Comprehensive Risk Metrics Calculator
// Analyzes historical trade performance to generate key risk indicators
// Used for system evaluation, optimization, and live monitoring
// Calculates industry-standard metrics for professional risk assessment
RiskMetrics CalculateRiskMetrics(const TradeRecord &trades[], int trade_count) {
    RiskMetrics metrics = {0};  // Initialize all metrics to zero
    
    // Return empty metrics if no trades to analyze
    if(trade_count <= 0) return metrics;
    
    // Initialize calculation variables
    double total_profit = 0;    // Sum of all profitable trades
    double total_loss = 0;      // Sum of all losing trades
    double running_balance = 0; // Cumulative account balance
    double peak_balance = 0;    // Highest balance reached
    double max_dd = 0;          // Maximum drawdown experienced
    
    metrics.total_trades = trade_count;
    
    // Process each trade to calculate cumulative metrics
    for(int i = 0; i < trade_count; i++) {
        // Update running balance with this trade's P&L
        running_balance += trades[i].profit_loss;
        
        // Track new equity peaks
        if(running_balance > peak_balance) {
            peak_balance = running_balance;
        }
        
        // Calculate current drawdown from peak
        double current_dd = peak_balance - running_balance;
        if(current_dd > max_dd) {
            max_dd = current_dd;  // Update maximum drawdown
        }
        
        // Categorize trade as winning or losing
        if(trades[i].profit_loss > 0) {
            metrics.winning_trades++;
            total_profit += trades[i].profit_loss;
        } else if(trades[i].profit_loss < 0) {
            metrics.losing_trades++;
            total_loss += MathAbs(trades[i].profit_loss);  // Convert to positive value
        }
        // Note: Break-even trades (profit_loss == 0) are not counted in either category
    }
    
    // Finalize calculated metrics
    metrics.max_drawdown = max_dd;                                    // Worst drawdown experienced
    metrics.current_drawdown = peak_balance - running_balance;        // Current drawdown from peak
    
    // Win Rate: Percentage of profitable trades
    metrics.win_rate = (metrics.total_trades > 0) ? 
                      (double)metrics.winning_trades / metrics.total_trades * 100.0 : 0.0;
    
    // Profit Factor: Ratio of gross profit to gross loss
    // Values > 1.0 indicate profitable system, > 2.0 is excellent
    metrics.profit_factor = (total_loss > 0) ? total_profit / total_loss : 0.0;
    
    // Risk-Reward Ratio: Average win size divided by average loss size
    // Higher values indicate better trade quality
    double avg_win = (metrics.winning_trades > 0) ? total_profit / metrics.winning_trades : 0.0;
    double avg_loss = (metrics.losing_trades > 0) ? total_loss / metrics.losing_trades : 0.0;
    metrics.risk_reward_ratio = (avg_loss > 0) ? avg_win / avg_loss : 0.0;
    
    return metrics;
}

//============================== UTILITY FUNCTIONS ==============================

// Action Code to String Converter
// Provides human-readable labels for AI-generated trading actions
// Used in logging, debugging, and trade reports for clarity
string ActionToString(int action) {
    switch(action) {
        case BUY_STRONG:  return "BUY_STRONG";
        case BUY_WEAK:    return "BUY_WEAK";
        case SELL_STRONG: return "SELL_STRONG";
        case SELL_WEAK:   return "SELL_WEAK";
        case HOLD:        return "HOLD";
        case FLAT:        return "FLAT";
        default:          return "UNKNOWN";
    }
}

// Position Type to String Converter
// Converts internal position codes to readable format
// Useful for trade reporting and system diagnostics
string PositionTypeToString(int pos_type) {
    switch(pos_type) {
        case POS_NONE:  return "NONE";
        case POS_LONG:  return "LONG";
        case POS_SHORT: return "SHORT";
        default:        return "UNKNOWN";
    }
}

// Duration Formatter for Human-Readable Output
// Converts raw hours into user-friendly time format
// Automatically handles day/hour conversion for better readability
string FormatDuration(int hours) {
    if(hours < 24) {
        return StringFormat("%dh", hours);  // Show hours only if less than a day
    } else {
        int days = hours / 24;                    // Calculate full days
        int remaining_hours = hours % 24;         // Calculate remaining hours
        return StringFormat("%dd %dh", days, remaining_hours);  // Show both days and hours
    }
}

// Standardized Trade Execution Logger
// Provides consistent log formatting across all system components
// Includes component identification, action details, and optional reasoning
// Critical for trade audit trails and system debugging
void LogTradeExecution(string component,     // Which component executed the trade (EA, Backtester, etc.)
                      int action,           // Trading action taken
                      double price,         // Execution price
                      double lots,          // Position size
                      string reason = ""    // Optional explanation for the trade
                      ) {
    string action_str = ActionToString(action);
    // Format: [Component] ACTION: X.XX lots @ X.XXXXX
    string log_msg = StringFormat("[%s] %s: %.2f lots @ %.5f", 
                                 component, action_str, lots, price);
    // Append reason if provided
    if(reason != "") {
        log_msg += " (" + reason + ")";
    }
    Print(log_msg);  // Output to MT5 terminal and log files
}

//+------------------------------------------------------------------+
//| End of CortexTradeLogic.mqh                                     |
//+------------------------------------------------------------------+

#endif // CORTEX_TRADE_LOGIC_MQH