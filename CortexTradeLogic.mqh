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
// These constants must match across all components
#define STATE_SIZE 45  // IMPROVEMENT 4.3: Updated to match enhanced 45-feature state vector
#define ACTIONS 6

// Action definitions
#define BUY_STRONG  0
#define BUY_WEAK    1
#define SELL_STRONG 2
#define SELL_WEAK   3
#define HOLD        4
#define FLAT        5

// Backward compatibility - ACTION_ prefixed constants
#define ACTION_BUY_STRONG  0
#define ACTION_BUY_WEAK    1
#define ACTION_SELL_STRONG 2
#define ACTION_SELL_WEAK   3
#define ACTION_HOLD        4
#define ACTION_FLAT        5

// Position types
#define POS_NONE  0
#define POS_LONG  1
#define POS_SHORT 2

//============================== STRUCTURES ==============================

// Trade record structure for consistent logging
struct TradeRecord {
    datetime open_time;
    datetime close_time;
    int      action;         // Trading action taken
    int      position_type;  // POS_LONG or POS_SHORT  
    double   entry_price;
    double   exit_price;
    double   lots;
    double   profit_loss;
    double   balance_after;  // Account balance after trade
    double   drawdown_pct;   // Drawdown percentage
    // IMPROVEMENT 7.2: Enhanced trade tracking
    double   mae;              // Maximum Adverse Excursion
    double   mfe;              // Maximum Favorable Excursion
    int      holding_time_hours; // Trade duration in hours
    string   exit_reason;      // Why the trade was closed
    double   commission;       // Trading costs
    double   confidence_score; // Model confidence for this trade
    
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

// Risk metrics structure
struct RiskMetrics {
    double max_drawdown;
    double current_drawdown;
    double win_rate;
    double profit_factor;
    double sharpe_ratio;
    double risk_reward_ratio;
    int    total_trades;
    int    winning_trades;
    int    losing_trades;
};

//============================== ATR AND TECHNICAL ANALYSIS ==============================

// Calculate Average True Range for volatility measurement
// This function provides consistent ATR calculation across all components
double CalculateATR(string symbol, ENUM_TIMEFRAMES timeframe, int period, int shift = 0) {
    double atr_values[];
    if(CopyBuffer(iATR(symbol, timeframe, period), 0, shift, 1, atr_values) <= 0) {
        Print("ERROR: Failed to get ATR values for ", symbol);
        return 0.0;
    }
    return atr_values[0];
}

// Proxy ATR calculation from price data (for backtesting)
double ATR_Proxy(const MqlRates &rates[], int index, int period) {
    if(index + period >= ArraySize(rates)) return 0.0;
    
    double sum = 0.0;
    int count = 0;
    
    for(int i = 0; i < period && (index + i + 1) < ArraySize(rates); i++) {
        int idx = index + i;
        
        // True Range = max of: (H-L), |H-prev_close|, |L-prev_close|
        double tr = MathMax(rates[idx].high - rates[idx].low, 
                           MathMax(MathAbs(rates[idx].high - rates[idx + 1].close), 
                                  MathAbs(rates[idx].low - rates[idx + 1].close)));
        sum += tr;
        count++;
    }
    
    return (count > 0) ? sum / count : 0.0;
}

// Calculate EMA slope for trend detection
double EMA_Slope(const MqlRates &rates[], int index, int period) {
    // BUGFIX: Improved bounds checking for array access safety
    int array_size = ArraySize(rates);
    if(index >= array_size || index < 0 || period <= 0) return 0.0;
    if(index + period + 1 >= array_size) return 0.0;
    
    double alpha = 2.0 / (period + 1);
    double ema = 0.0;
    int n = 0;
    
    // Calculate current EMA - BUGFIX: Added additional bounds checking
    for(int k = period; k >= 0; k--) {
        int idx = index + k;
        if(idx >= array_size || idx < 0) continue;
        
        if(n == 0) ema = rates[idx].close;
        else ema = alpha * rates[idx].close + (1 - alpha) * ema;
        n++;
    }
    
    // Calculate previous EMA - BUGFIX: Added additional bounds checking
    double ema_prev = 0.0;
    n = 0;
    for(int k = period + 1; k >= 1; k--) {
        int idx = index + k;
        if(idx >= array_size || idx < 0) continue;
        
        if(n == 0) ema_prev = rates[idx].close;
        else ema_prev = alpha * rates[idx].close + (1 - alpha) * ema_prev;
        n++;
    }
    
    return ema - ema_prev;  // Positive = uptrend, Negative = downtrend
}

//============================== POSITION SIZING ==============================

// Unified position size calculation with adaptive parameters
// This ensures consistent sizing logic across EA and backtester
double CalculateUnifiedPositionSize(bool is_strong_signal, 
                                   double risk_percent, 
                                   double atr_multiplier,
                                   double lots_strong = 0.1, 
                                   double lots_weak = 0.05,
                                   bool use_risk_sizing = true) {
    
    if(!use_risk_sizing) {
        return is_strong_signal ? lots_strong : lots_weak;
    }
    
    // Get current ATR for volatility-based sizing
    double atr = CalculateATR(_Symbol, PERIOD_CURRENT, 14);
    if(atr <= 0) return is_strong_signal ? lots_strong : lots_weak; // Fallback
    
    double stop_loss_points = atr * atr_multiplier / _Point;
    
    // Calculate position size based on risk percentage
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = equity * risk_percent / 100.0;
    
    // Account for signal strength
    double signal_multiplier = is_strong_signal ? 1.0 : 0.5;
    risk_amount *= signal_multiplier;
    
    // Calculate lot size
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tick_value <= 0 || tick_size <= 0 || stop_loss_points <= 0) {
        return is_strong_signal ? lots_strong : lots_weak; // Fallback
    }
    
    double lots = risk_amount / (stop_loss_points * tick_value / tick_size);
    
    // Apply broker limits
    double min_lots = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lots = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lots = MathMax(lots, min_lots);
    lots = MathMin(lots, max_lots);
    lots = MathRound(lots / lot_step) * lot_step;  // Round to step
    
    return lots;
}

//============================== ENTRY CONDITIONS ==============================

// Unified entry validation - checks all conditions before allowing trade
bool ValidateTradeEntry(int action, 
                       double confidence_threshold = 0.0,
                       bool use_time_filters = true,
                       bool use_spread_filter = true,
                       double max_spread_points = 20.0) {
    
    // Check if action is valid
    if(action < 0 || action >= ACTIONS) {
        return false;
    }
    
    // Skip HOLD and FLAT actions
    if(action == HOLD || action == FLAT) {
        return false;
    }
    
    // Check trading hours (if enabled)
    if(use_time_filters) {
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        
        // Avoid weekend gaps and low liquidity periods
        if(dt.day_of_week == 0 || dt.day_of_week == 6) {
            return false;
        }
        
        // Avoid major news hours (simplified - could be enhanced)
        if(dt.hour < 2 || dt.hour > 22) {
            return false;
        }
    }
    
    // Check spread conditions
    if(use_spread_filter) {
        double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
        if(spread > max_spread_points * _Point) {
            return false;
        }
    }
    
    return true;
}

//============================== EXIT CONDITIONS ==============================

// Unified exit condition checking
bool CheckUnifiedExitConditions(int position_type,
                                double entry_price,
                                datetime entry_time,
                                double current_price,
                                double atr_value,
                                double profit_target_atr = 1.8,
                                int max_holding_hours = 72,
                                double emergency_stop_loss = 150.0,
                                bool use_profit_targets = true,
                                bool use_time_limits = true,
                                bool use_emergency_stops = true) {
    
    if(position_type == POS_NONE) return false;
    
    datetime current_time = TimeCurrent();
    int holding_hours = (int)((current_time - entry_time) / 3600);
    
    // Calculate unrealized P&L
    double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    double lots = 0.1; // Default lot size - should be passed as parameter
    double unrealized_pnl = 0;
    
    if(position_type == POS_LONG) {
        unrealized_pnl = (current_price - entry_price) * lots * contract_size;
    } else {
        unrealized_pnl = (entry_price - current_price) * lots * contract_size;
    }
    
    // Emergency stop loss check
    if(use_emergency_stops && unrealized_pnl < -emergency_stop_loss) {
        return true; // Force exit
    }
    
    // Maximum holding time check
    if(use_time_limits && holding_hours > max_holding_hours) {
        return true; // Force exit
    }
    
    // Profit target check
    if(use_profit_targets && unrealized_pnl > 0) {
        double profit_target = atr_value * profit_target_atr * lots * contract_size;
        if(unrealized_pnl >= profit_target) {
            return true; // Take profit
        }
    }
    
    // Additional exit conditions can be added here:
    // - Trailing stops
    // - Technical indicator reversals
    // - Time-based exits
    
    return false;
}

//============================== RISK MANAGEMENT ==============================

// Calculate stop loss and take profit levels
void CalculateStopAndTarget(double entry_price, 
                           int position_type,
                           double atr_value,
                           double atr_multiplier,
                           double rr_ratio,
                           double &stop_loss,
                           double &take_profit) {
    
    double atr_distance = atr_value * atr_multiplier;
    
    if(position_type == POS_LONG) {
        stop_loss = entry_price - atr_distance;
        take_profit = entry_price + (atr_distance * rr_ratio);
    } else if(position_type == POS_SHORT) {
        stop_loss = entry_price + atr_distance;
        take_profit = entry_price - (atr_distance * rr_ratio);
    } else {
        stop_loss = 0;
        take_profit = 0;
    }
    
    // Normalize to tick size
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tick_size > 0) {
        stop_loss = MathRound(stop_loss / tick_size) * tick_size;
        take_profit = MathRound(take_profit / tick_size) * tick_size;
    }
}

// Calculate risk metrics from trade history
RiskMetrics CalculateRiskMetrics(const TradeRecord &trades[], int trade_count) {
    RiskMetrics metrics = {0};
    
    if(trade_count <= 0) return metrics;
    
    double total_profit = 0;
    double total_loss = 0;
    double running_balance = 0;
    double peak_balance = 0;
    double max_dd = 0;
    
    metrics.total_trades = trade_count;
    
    for(int i = 0; i < trade_count; i++) {
        running_balance += trades[i].profit_loss;
        
        if(running_balance > peak_balance) {
            peak_balance = running_balance;
        }
        
        double current_dd = peak_balance - running_balance;
        if(current_dd > max_dd) {
            max_dd = current_dd;
        }
        
        if(trades[i].profit_loss > 0) {
            metrics.winning_trades++;
            total_profit += trades[i].profit_loss;
        } else if(trades[i].profit_loss < 0) {
            metrics.losing_trades++;
            total_loss += MathAbs(trades[i].profit_loss);
        }
    }
    
    metrics.max_drawdown = max_dd;
    metrics.current_drawdown = peak_balance - running_balance;
    metrics.win_rate = (metrics.total_trades > 0) ? 
                      (double)metrics.winning_trades / metrics.total_trades * 100.0 : 0.0;
    metrics.profit_factor = (total_loss > 0) ? total_profit / total_loss : 0.0;
    
    // Calculate risk-reward ratio
    double avg_win = (metrics.winning_trades > 0) ? total_profit / metrics.winning_trades : 0.0;
    double avg_loss = (metrics.losing_trades > 0) ? total_loss / metrics.losing_trades : 0.0;
    metrics.risk_reward_ratio = (avg_loss > 0) ? avg_win / avg_loss : 0.0;
    
    return metrics;
}

//============================== UTILITY FUNCTIONS ==============================

// Convert action integer to string for logging
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

// Convert position type to string
string PositionTypeToString(int pos_type) {
    switch(pos_type) {
        case POS_NONE:  return "NONE";
        case POS_LONG:  return "LONG";
        case POS_SHORT: return "SHORT";
        default:        return "UNKNOWN";
    }
}

// Format duration in hours to readable string
string FormatDuration(int hours) {
    if(hours < 24) {
        return StringFormat("%dh", hours);
    } else {
        int days = hours / 24;
        int remaining_hours = hours % 24;
        return StringFormat("%dd %dh", days, remaining_hours);
    }
}

// Log trade execution with consistent format
void LogTradeExecution(string component, int action, double price, double lots, string reason = "") {
    string action_str = ActionToString(action);
    string log_msg = StringFormat("[%s] %s: %.2f lots @ %.5f", 
                                 component, action_str, lots, price);
    if(reason != "") {
        log_msg += " (" + reason + ")";
    }
    Print(log_msg);
}

//+------------------------------------------------------------------+
//| End of CortexTradeLogic.mqh                                     |
//+------------------------------------------------------------------+

#endif // CORTEX_TRADE_LOGIC_MQH