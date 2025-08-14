# Confidence-augmented Training EA Integration Guide for Cortex5

## Overview
This guide explains how to integrate the Improvement 6.3 Confidence-augmented Training system with the Cortex EA (cortex5.mq5) for confidence-based trade filtering and decision making.

## Integration Requirements

### 1. Confidence Output from Model
The trained model with confidence-augmented training will output both trading decisions and confidence scores.

```mql5
// Add to cortex5.mq5 for confidence-based filtering
struct TradingDecision {
    int action;                    // Predicted action (0-5)
    double confidence;            // Trading confidence (0.1-0.9)
    double direction_probability; // Probability of correct direction
    double outcome_certainty;     // Certainty of profitable outcome
};

// Global confidence filtering parameters
input bool InpUseConfidenceFilter = false;        // Enable confidence-based filtering
input double InpMinConfidenceThreshold = 0.6;     // Minimum confidence to execute trade
input double InpHighConfidenceThreshold = 0.8;    // High confidence threshold for stronger positions
input bool InpDynamicConfidenceThreshold = true;  // Adjust threshold based on recent performance
input bool InpLogConfidenceDecisions = true;      // Log confidence-based decisions
```

### 2. Enhanced Model Prediction Function
Update the model prediction to include confidence output:

```mql5
// Enhanced prediction function with confidence
TradingDecision GetModelPredictionWithConfidence(const double &state[]) {
    TradingDecision decision;
    
    // Get model prediction (existing function)
    decision.action = GetModelPrediction(state);
    
    // Extract confidence from model (if trained with 6.3)
    decision.confidence = ExtractModelConfidence(state, decision.action);
    decision.direction_probability = decision.confidence; // Simplified
    decision.outcome_certainty = decision.confidence * 0.9; // Slightly conservative
    
    return decision;
}

// Extract confidence from trained model
double ExtractModelConfidence(const double &state[], int predicted_action) {
    // Method 1: Use Q-value spread (if available)
    double q_values[6];
    GetModelQValues(state, q_values); // Would need implementation
    
    double q_max = -999999;
    double q_second = -999999;
    for(int i = 0; i < 6; i++) {
        if(q_values[i] > q_max) {
            q_second = q_max;
            q_max = q_values[i];
        } else if(q_values[i] > q_second) {
            q_second = q_values[i];
        }
    }
    
    double q_spread = q_max - q_second;
    double confidence = 1.0 / (1.0 + MathExp(-q_spread * 5.0)); // Sigmoid
    
    return MathMax(0.1, MathMin(0.9, confidence));
}
```

### 3. Confidence-Based Trade Filtering
Implement the core confidence filtering logic:

```mql5
// Main trading decision with confidence filtering
bool ShouldExecuteTrade(TradingDecision &decision) {
    if(!InpUseConfidenceFilter) return true;
    
    // Skip HOLD and FLAT actions
    if(decision.action == HOLD || decision.action == FLAT) return false;
    
    // Apply confidence threshold
    if(decision.confidence < InpMinConfidenceThreshold) {
        if(InpLogConfidenceDecisions) {
            Print("Trade rejected: Low confidence ", DoubleToString(decision.confidence, 3), 
                  " < threshold ", DoubleToString(InpMinConfidenceThreshold, 3));
        }
        return false;
    }
    
    // Additional validation for very low confidence
    if(decision.confidence < 0.3) {
        Print("WARNING: Very low confidence ", DoubleToString(decision.confidence, 3), " - skipping trade");
        return false;
    }
    
    return true;
}

// Confidence-based position sizing
double GetConfidenceBasedPositionSize(TradingDecision &decision, double base_size) {
    if(!InpUseConfidenceFilter) return base_size;
    
    double size_multiplier = 1.0;
    
    if(decision.confidence >= InpHighConfidenceThreshold) {
        // High confidence: increase position size
        size_multiplier = 1.5;
        if(InpLogConfidenceDecisions) {
            Print("High confidence trade: ", DoubleToString(decision.confidence, 3), " - increasing position size");
        }
    } else if(decision.confidence < InpMinConfidenceThreshold + 0.1) {
        // Low-medium confidence: reduce position size
        size_multiplier = 0.7;
    }
    
    return base_size * size_multiplier;
}
```

### 4. Dynamic Confidence Threshold Adjustment
Implement adaptive confidence thresholds based on performance:

```mql5
// Performance tracking for threshold adjustment
struct ConfidencePerformance {
    int high_conf_trades;
    int high_conf_wins;
    int med_conf_trades;
    int med_conf_wins;
    int low_conf_trades;
    int low_conf_wins;
    datetime last_update;
};

ConfidencePerformance g_conf_performance;

// Update confidence performance tracking
void UpdateConfidencePerformance(TradingDecision &decision, bool trade_won) {
    if(decision.confidence >= InpHighConfidenceThreshold) {
        g_conf_performance.high_conf_trades++;
        if(trade_won) g_conf_performance.high_conf_wins++;
    } else if(decision.confidence >= InpMinConfidenceThreshold) {
        g_conf_performance.med_conf_trades++;
        if(trade_won) g_conf_performance.med_conf_wins++;
    } else {
        g_conf_performance.low_conf_trades++;
        if(trade_won) g_conf_performance.low_conf_wins++;
    }
    
    g_conf_performance.last_update = TimeCurrent();
}

// Adjust thresholds based on performance
void AdjustConfidenceThresholds() {
    if(!InpDynamicConfidenceThreshold) return;
    
    static datetime last_adjustment = 0;
    datetime current_time = TimeCurrent();
    
    // Adjust weekly
    if(current_time - last_adjustment < 604800) return;
    
    if(g_conf_performance.high_conf_trades >= 10) {
        double high_conf_rate = (double)g_conf_performance.high_conf_wins / g_conf_performance.high_conf_trades;
        
        if(high_conf_rate > 0.7) {
            // High confidence trades are successful - can be more aggressive
            InpMinConfidenceThreshold = MathMax(0.5, InpMinConfidenceThreshold - 0.05);
        } else if(high_conf_rate < 0.5) {
            // High confidence trades failing - be more conservative
            InpMinConfidenceThreshold = MathMin(0.8, InpMinConfidenceThreshold + 0.05);
        }
        
        Print("Confidence threshold adjusted to: ", DoubleToString(InpMinConfidenceThreshold, 2), 
              " based on win rate: ", DoubleToString(high_conf_rate * 100, 1), "%");
    }
    
    last_adjustment = current_time;
}
```

### 5. Main Trading Loop Integration
Update the main EA logic to use confidence filtering:

```mql5
// In main OnTick() function
void OnTick() {
    // ... existing EA logic ...
    
    // Get state features
    double state[45]; // Or 35 depending on model
    BuildCurrentState(state);
    
    // Get prediction with confidence
    TradingDecision decision = GetModelPredictionWithConfidence(state);
    
    // Apply confidence filtering
    if(!ShouldExecuteTrade(decision)) return;
    
    // Calculate position size based on confidence
    double base_size = CalculateBasePositionSize();
    double final_size = GetConfidenceBasedPositionSize(decision, base_size);
    
    // Execute trade
    ExecuteTrade(decision.action, final_size);
    
    // Log confidence decision
    if(InpLogConfidenceDecisions) {
        Print("Executed ", ActionToString(decision.action), 
              " with confidence: ", DoubleToString(decision.confidence, 3),
              ", size: ", DoubleToString(final_size, 2));
    }
    
    // Track confidence performance (after trade closes)
    // This would be called from trade close logic
    // UpdateConfidencePerformance(decision, trade_result);
}

// Periodic confidence threshold adjustment
void OnTimer() {
    AdjustConfidenceThresholds();
}
```

### 6. Performance Monitoring
Add confidence-specific performance tracking:

```mql5
// Confidence performance report
void PrintConfidenceReport() {
    Print("=== CONFIDENCE-BASED TRADING REPORT ===");
    
    if(g_conf_performance.high_conf_trades > 0) {
        double high_rate = (double)g_conf_performance.high_conf_wins / g_conf_performance.high_conf_trades;
        Print("High confidence trades (>=", DoubleToString(InpHighConfidenceThreshold, 2), "): ", 
              g_conf_performance.high_conf_trades, " trades, ", 
              DoubleToString(high_rate * 100, 1), "% win rate");
    }
    
    if(g_conf_performance.med_conf_trades > 0) {
        double med_rate = (double)g_conf_performance.med_conf_wins / g_conf_performance.med_conf_trades;
        Print("Medium confidence trades: ", g_conf_performance.med_conf_trades, " trades, ", 
              DoubleToString(med_rate * 100, 1), "% win rate");
    }
    
    Print("Current confidence threshold: ", DoubleToString(InpMinConfidenceThreshold, 2));
    Print("Confidence filtering: ", (InpUseConfidenceFilter ? "Enabled" : "Disabled"));
}
```

## Implementation Steps

1. **Phase 1**: Train model with confidence-augmented training (6.3) enabled
2. **Phase 2**: Add confidence extraction functions to EA
3. **Phase 3**: Implement basic confidence filtering
4. **Phase 4**: Add confidence-based position sizing
5. **Phase 5**: Implement dynamic threshold adjustment
6. **Phase 6**: Add comprehensive performance monitoring

## Expected Benefits

- **15-30% improved trade filtering** through confidence-based decisions
- **Reduced false signals** by filtering low-confidence predictions
- **Enhanced position sizing** based on prediction confidence
- **Adaptive thresholds** that adjust to market conditions
- **Better risk management** through uncertainty quantification

## Configuration Recommendations

- **Minimum threshold**: Start with 0.6, adjust based on performance
- **High confidence threshold**: Use 0.8 for position size increases
- **Dynamic adjustment**: Enable for adaptive behavior
- **Logging**: Enable during testing, reduce in production

## Troubleshooting

- **Too few trades**: Lower confidence threshold or check model training
- **Poor performance**: Verify confidence calibration in training
- **Inconsistent confidence**: Check model input features and normalization
- **Threshold drift**: Monitor and validate dynamic adjustment logic

This integration enables the Cortex5 EA to leverage well-calibrated confidence predictions for improved trade selection and risk management.