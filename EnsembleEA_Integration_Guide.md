# Ensemble Model Integration Guide for Cortex EA

## Overview
This guide explains how to modify the Cortex EA (cortex5.mq5) to support ensemble models trained with improvement 6.1.

## Required EA Modifications

### 1. Multiple Model Loading
```mql5
// Add ensemble support globals
input bool InpUseEnsemble = false;
input string InpEnsemblePrefix = "Ensemble_Model";
input int InpEnsembleSize = 3;
input string InpCombinationMethod = "VOTE"; // VOTE, AVERAGE, WEIGHTED

// Model instance arrays
CDoubleDuelingDRQN g_ensemble_models[];
bool g_ensemble_loaded = false;
```

### 2. Model Loading Function
```mql5
bool LoadEnsembleModels() {
    if(!InpUseEnsemble) {
        return LoadSingleModel(InpModelFileName);
    }
    
    ArrayResize(g_ensemble_models, InpEnsembleSize);
    
    for(int i = 0; i < InpEnsembleSize; i++) {
        string filename = StringFormat("%s_%d.dat", InpEnsemblePrefix, i + 1);
        if(!g_ensemble_models[i].LoadModel(filename)) {
            Print("Failed to load ensemble model: ", filename);
            return false;
        }
    }
    
    g_ensemble_loaded = true;
    Print("Loaded ", InpEnsembleSize, " ensemble models successfully");
    return true;
}
```

### 3. Ensemble Prediction Function
```mql5
int GetEnsemblePrediction(const double &state[]) {
    if(!g_ensemble_loaded || !InpUseEnsemble) {
        return GetSingleModelPrediction(state);
    }
    
    // Get predictions from all models
    double predictions[][];
    ArrayResize(predictions, InpEnsembleSize);
    
    for(int i = 0; i < InpEnsembleSize; i++) {
        ArrayResize(predictions[i], 6);
        g_ensemble_models[i].Predict(state, predictions[i]);
    }
    
    // Combine predictions based on method
    return CombineEnsemblePredictions(predictions, InpEnsembleSize, InpCombinationMethod);
}
```

### 4. Required Support Functions
```mql5
int CombineEnsemblePredictions(double predictions[][], int num_models, string method) {
    // Implementation matching the training script's combination logic
    // VOTE: Majority voting
    // AVERAGE: Average Q-values
    // WEIGHTED: Performance-weighted average
}
```

## Integration Steps

1. **Backup Current EA**: Save a copy of cortex5.mq5 before modifications
2. **Add Ensemble Globals**: Insert ensemble-related global variables
3. **Modify OnInit()**: Update model loading to support ensemble
4. **Update Signal Logic**: Replace single model predictions with ensemble predictions
5. **Add Error Handling**: Ensure graceful fallback if ensemble models fail to load
6. **Update Logging**: Log ensemble decisions and model agreement statistics

## Coordination Notes

- Ensemble models must use consistent STATE_SIZE and ACTIONS
- All models should be trained on the same data for fair comparison
- Consider model versioning for incremental updates
- Monitor ensemble diversity in live trading
- Implement model performance tracking for weighted combinations

## Performance Considerations

- Ensemble prediction takes ~3-5x longer than single model
- Memory usage increases with ensemble size
- Consider using ensemble only during high-importance decisions
- Implement caching for frequently accessed predictions

## Testing Protocol

1. Test individual model loading first
2. Verify ensemble combination logic with known test cases
3. Compare ensemble vs single model performance in backtest
4. Monitor live performance and model agreement rates
5. Implement A/B testing to validate ensemble benefits

## Maintenance

- Retrain ensemble periodically with fresh data
- Monitor individual model performance drift
- Remove underperforming models from ensemble
- Update combination weights based on recent performance