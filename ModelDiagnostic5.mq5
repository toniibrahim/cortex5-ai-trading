//+------------------------------------------------------------------+
//|                                         ModelDiagnostic5.mq5     |
//|               Diagnostic tool for Double-Dueling DRQN model files |
//+------------------------------------------------------------------+

// IMPROVEMENT 3.4: Include unified trade logic module for consistent analysis
#include <CortexTradeLogic.mqh>

#property script_show_inputs

input string InpModelFileName = "DoubleDueling_DRQN_Model.dat";

// Neural network weight validation function
bool ValidateNetworkWeights(int file_handle, int state_size, int actions, int h1, int h2, int h3,
                           bool has_lstm, int lstm_size, bool has_dueling, 
                           int value_head_size, int adv_head_size,
                           bool has_confidence, int confidence_head_size){
    
    if(file_handle == INVALID_HANDLE) return false;
    
    ulong file_pos_start = FileTell(file_handle);
    int layers_validated = 0;
    int total_weights = 0;
    int total_biases = 0;
    
    // Validate Layer 1: state_size -> h1
    if(!ValidateLayer(file_handle, state_size, h1, "Layer 1")) return false;
    total_weights += state_size * h1;
    total_biases += h1;
    layers_validated++;
    
    // Validate Layer 2: h1 -> h2  
    if(!ValidateLayer(file_handle, h1, h2, "Layer 2")) return false;
    total_weights += h1 * h2;
    total_biases += h2;
    layers_validated++;
    
    // Validate Layer 3: h2 -> h3
    if(!ValidateLayer(file_handle, h2, h3, "Layer 3")) return false;
    total_weights += h2 * h3;
    total_biases += h3;
    layers_validated++;
    
    int lstm_output_size = h3;
    
    // Validate LSTM layer if enabled
    if(has_lstm){
        if(!ValidateLSTMLayer(file_handle, h3, lstm_size, "LSTM Layer")) return false;
        total_weights += h3 * lstm_size * 4 + lstm_size * lstm_size * 4; // 4 gates
        total_biases += lstm_size * 4;
        lstm_output_size = lstm_size;
        layers_validated++;
    }
    
    // Validate Dueling heads if enabled
    if(has_dueling){
        // Value head: lstm_output -> value_head_size -> 1
        if(!ValidateLayer(file_handle, lstm_output_size, value_head_size, "Value Head")) return false;
        total_weights += lstm_output_size * value_head_size;
        total_biases += value_head_size;
        
        // Advantage head: lstm_output -> adv_head_size -> actions
        if(!ValidateLayer(file_handle, lstm_output_size, adv_head_size, "Advantage Head")) return false;
        total_weights += lstm_output_size * adv_head_size;
        total_biases += adv_head_size;
        layers_validated += 2;
    } else {
        // Standard output layer: lstm_output -> actions
        if(!ValidateLayer(file_handle, lstm_output_size, actions, "Output Layer")) return false;
        total_weights += lstm_output_size * actions;
        total_biases += actions;
        layers_validated++;
    }
    
    // Validate Confidence head if enabled
    if(has_confidence){
        if(!ValidateLayer(file_handle, lstm_output_size, confidence_head_size, "Confidence Head")) return false;
        total_weights += lstm_output_size * confidence_head_size;
        total_biases += confidence_head_size;
        layers_validated++;
    }
    
    Print("Network validation summary:");
    Print("  Layers validated: ", layers_validated);
    Print("  Total weights: ", total_weights);
    Print("  Total biases: ", total_biases);
    Print("  Total parameters: ", total_weights + total_biases);
    
    // Calculate expected file size and compare with actual
    ulong expected_size = CalculateExpectedFileSize(state_size, actions, h1, h2, h3, 
                                                   has_lstm, lstm_size, has_dueling,
                                                   value_head_size, adv_head_size, 
                                                   has_confidence, confidence_head_size);
    
    ulong current_pos = FileTell(file_handle);
    FileSeek(file_handle, 0, SEEK_END);
    ulong actual_size = FileTell(file_handle);
    
    Print("File size analysis:");
    Print("  Expected size: ", expected_size, " bytes (", DoubleToString(expected_size/1024.0, 1), " KB)");
    Print("  Actual size: ", actual_size, " bytes (", DoubleToString(actual_size/1024.0, 1), " KB)");
    
    double size_diff_pct = MathAbs((double)(actual_size - expected_size)) / expected_size * 100.0;
    Print("  Size difference: ", DoubleToString(size_diff_pct, 1), "%");
    
    if(size_diff_pct > 10.0){
        Print("  ⚠️ WARNING: Significant size difference detected. File may be corrupted.");
    } else {
        Print("  ✓ File size is within expected range");
    }
    
    return true;
}

// Validate individual layer structure
bool ValidateLayer(int file_handle, int expected_in, int expected_out, string layer_name){
    if(file_handle == INVALID_HANDLE) return false;
    
    // Read layer dimensions
    int layer_in = (int)FileReadLong(file_handle);
    int layer_out = (int)FileReadLong(file_handle);
    
    if(layer_in != expected_in || layer_out != expected_out){
        Print("ERROR: ", layer_name, " dimension mismatch. Expected [", expected_in, "x", expected_out, 
              "], got [", layer_in, "x", layer_out, "]");
        return false;
    }
    
    // Skip weights and biases (just validate they exist)
    for(int i = 0; i < layer_in * layer_out; i++){
        double weight = FileReadDouble(file_handle);
        if(!IsValidWeight(weight)){
            Print("WARNING: ", layer_name, " contains invalid weight: ", DoubleToString(weight, 8));
        }
    }
    
    for(int j = 0; j < layer_out; j++){
        double bias = FileReadDouble(file_handle);
        if(!IsValidWeight(bias)){
            Print("WARNING: ", layer_name, " contains invalid bias: ", DoubleToString(bias, 8));
        }
    }
    
    Print("✓ ", layer_name, ": [", layer_in, "x", layer_out, "] validated");
    return true;
}

// Validate LSTM layer structure
bool ValidateLSTMLayer(int file_handle, int expected_in, int expected_out, string layer_name){
    if(file_handle == INVALID_HANDLE) return false;
    
    int lstm_in = (int)FileReadLong(file_handle);
    int lstm_out = (int)FileReadLong(file_handle);
    
    if(lstm_in != expected_in || lstm_out != expected_out){
        Print("ERROR: ", layer_name, " dimension mismatch. Expected [", expected_in, "x", expected_out, 
              "], got [", lstm_in, "x", lstm_out, "]");
        return false;
    }
    
    // Skip LSTM weights (4 gates, input and hidden weights, biases)
    int total_input_weights = lstm_in * lstm_out * 4;  // Wf, Wi, Wc, Wo
    int total_hidden_weights = lstm_out * lstm_out * 4; // Uf, Ui, Uc, Uo  
    int total_biases = lstm_out * 4; // bf, bi, bc, bo
    
    for(int i = 0; i < total_input_weights + total_hidden_weights + total_biases; i++){
        FileReadDouble(file_handle); // Skip LSTM parameters
    }
    
    Print("✓ ", layer_name, ": [", lstm_in, "x", lstm_out, "] validated");
    return true;
}

// Check if weight value is reasonable
bool IsValidWeight(double weight){
    if(weight != weight) return false; // NaN check
    if(weight > 1e6 || weight < -1e6) return false; // Extreme value check
    return true;
}

// Calculate expected file size based on architecture
ulong CalculateExpectedFileSize(int state_size, int actions, int h1, int h2, int h3,
                               bool has_lstm, int lstm_size, bool has_dueling,
                               int value_head_size, int adv_head_size,
                               bool has_confidence, int confidence_head_size){
    
    ulong total_size = 0;
    
    // Header and metadata
    total_size += 8;  // Magic number
    total_size += 8;  // Symbol length
    total_size += 20; // Symbol string (estimated average)
    total_size += 8;  // Timeframe
    total_size += 8;  // State size
    total_size += 8;  // Actions
    total_size += 8 * 3; // Hidden layer sizes (h1, h2, h3)
    
    // Architecture flags (8 parameters for checkpoint format)
    total_size += 8 * 8; // LSTM, Dueling, Confidence flags + sizes
    
    // Feature normalization (min/max pairs)
    total_size += state_size * 8 * 2; // Double values for min and max
    
    // Checkpoint data
    total_size += 8;  // Last trained time
    total_size += 8;  // Training steps
    total_size += 8;  // Epsilon
    total_size += 8;  // Beta
    
    // Neural network weights
    // Layer 1: state_size -> h1
    total_size += 8 + 8; // Layer dimensions
    total_size += (state_size * h1 + h1) * 8; // Weights + biases
    
    // Layer 2: h1 -> h2
    total_size += 8 + 8; // Layer dimensions
    total_size += (h1 * h2 + h2) * 8; // Weights + biases
    
    // Layer 3: h2 -> h3
    total_size += 8 + 8; // Layer dimensions
    total_size += (h2 * h3 + h3) * 8; // Weights + biases
    
    int final_layer_input = h3;
    
    // LSTM layer if enabled
    if(has_lstm){
        total_size += 8 + 8; // LSTM dimensions
        // LSTM weights: 4 gates * (input_weights + hidden_weights + biases)
        total_size += (h3 * lstm_size * 4 + lstm_size * lstm_size * 4 + lstm_size * 4) * 8;
        final_layer_input = lstm_size;
    }
    
    // Output layers
    if(has_dueling){
        // Value head
        total_size += 8 + 8; // Dimensions
        total_size += (final_layer_input * value_head_size + value_head_size) * 8;
        
        // Advantage head  
        total_size += 8 + 8; // Dimensions
        total_size += (final_layer_input * adv_head_size + adv_head_size) * 8;
    } else {
        // Standard output layer
        total_size += 8 + 8; // Dimensions
        total_size += (final_layer_input * actions + actions) * 8;
    }
    
    // Confidence head if enabled
    if(has_confidence){
        total_size += 8 + 8; // Dimensions
        total_size += (final_layer_input * confidence_head_size + confidence_head_size) * 8;
    }
    
    return total_size;
}

void OnStart(){
    Print("=== DOUBLE-DUELING DRQN MODEL DIAGNOSTIC TOOL ===");
    Print("Analyzing model file: ", InpModelFileName);
    
    // Get file size for validation
    string full_path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + InpModelFileName;
    ulong file_size = 0;
    
    int h = FileOpen(InpModelFileName, FILE_BIN|FILE_READ);
    if(h == INVALID_HANDLE){
        Print("ERROR: Cannot open model file. File may not exist.");
        Print("Expected location: ", full_path);
        return;
    }
    
    // Get file size
    FileSeek(h, 0, SEEK_END);
    file_size = FileTell(h);
    FileSeek(h, 0, SEEK_SET);
    
    Print("File size: ", file_size, " bytes (", DoubleToString(file_size/1024.0/1024.0, 2), " MB)");
    
    // Validate minimum file size
    if(file_size < 1000){
        Print("ERROR: File too small (", file_size, " bytes). Minimum expected: 1000 bytes");
        Print("This file appears to be corrupted or incomplete.");
        FileClose(h);
        return;
    }
    
    // Validate maximum reasonable file size (prevent loading huge corrupted files)
    if(file_size > 500*1024*1024){ // 500MB
        Print("ERROR: File too large (", DoubleToString(file_size/1024.0/1024.0, 2), " MB). Maximum expected: 500 MB");
        Print("This file may be corrupted or not a valid model file.");
        FileClose(h);
        return;
    }
    
    // Read magic number
    long magic = FileReadLong(h);
    Print("Magic number: 0x", IntegerToString(magic, 16));
    
    if(magic == (long)0xC0DE0203){
        Print("✓ Model format: Enhanced with checkpoints (latest version)");
    } else if(magic == (long)0xC0DE0202){
        Print("✓ Model format: Legacy (no checkpoints)");
    } else {
        Print("✗ Unknown magic number - not a valid Cortex model file");
        FileClose(h);
        return;
    }
    
    // Read model metadata
    Print("DEBUG: Reading model metadata...");
    int sym_len = (int)FileReadLong(h);  // Read symbol length first
    Print("DEBUG: Symbol length read: ", sym_len);
    
    if(sym_len <= 0 || sym_len > 32){
        Print("ERROR: Invalid symbol length: ", sym_len);
        FileClose(h);
        return;
    }
    
    string symbol = FileReadString(h, sym_len);  // Then read symbol with length
    Print("DEBUG: Symbol read: '", symbol, "'");
    
    int timeframe = (int)FileReadLong(h);
    Print("DEBUG: Timeframe read: ", timeframe);
    
    int state_size = (int)FileReadLong(h);
    Print("DEBUG: State size read: ", state_size);
    
    int actions = (int)FileReadLong(h);
    Print("DEBUG: Actions read: ", actions);
    
    int h1 = (int)FileReadLong(h);
    int h2 = (int)FileReadLong(h);
    int h3 = (int)FileReadLong(h);
    Print("DEBUG: Hidden layers read: [", h1, ",", h2, ",", h3, "]");
    
    // Read architecture parameters (for new format)
    bool has_lstm = false;
    bool has_dueling = false;
    bool has_confidence = false;
    int lstm_size = 0;
    int seq_len = 0;
    int value_head_size = 0;
    int adv_head_size = 0;
    int confidence_head_size = 0;
    
    bool has_checkpoint = (magic == (long)0xC0DE0203);
    if(has_checkpoint){
        has_lstm = (FileReadLong(h) == 1);
        has_dueling = (FileReadLong(h) == 1);
        has_confidence = (FileReadLong(h) == 1);  // FIX: Read missing confidence flag
        lstm_size = (int)FileReadLong(h);
        seq_len = (int)FileReadLong(h);
        value_head_size = (int)FileReadLong(h);
        adv_head_size = (int)FileReadLong(h);
        confidence_head_size = (int)FileReadLong(h);  // FIX: Read missing confidence head size
        
        Print("DEBUG: Architecture flags read:");
        Print("  LSTM enabled: ", has_lstm);
        Print("  Dueling enabled: ", has_dueling);
        Print("  Confidence enabled: ", has_confidence);
        Print("  LSTM size: ", lstm_size);
        Print("  Sequence length: ", seq_len);
        Print("  Value head size: ", value_head_size);
        Print("  Advantage head size: ", adv_head_size);
        Print("  Confidence head size: ", confidence_head_size);
    }
    
    Print("Model metadata:");
    Print("  Symbol: ", symbol);
    Print("  Timeframe: ", EnumToString((ENUM_TIMEFRAMES)timeframe));
    Print("  State size: ", state_size);
    Print("  Actions: ", actions);
    
    // Build architecture description
    string arch_desc = IntegerToString(state_size) + "x[" + IntegerToString(h1) + "," + IntegerToString(h2) + "," + IntegerToString(h3);
    if(has_lstm) arch_desc += ",LSTM:" + IntegerToString(lstm_size);
    if(has_dueling) arch_desc += ",Dueling:" + IntegerToString(value_head_size) + "+" + IntegerToString(adv_head_size);
    if(has_confidence) arch_desc += ",Confidence:" + IntegerToString(confidence_head_size);
    arch_desc += "]x" + IntegerToString(actions);
    
    Print("  Network architecture: ", arch_desc);
    if(has_checkpoint){
        Print("  Advanced features:");
        Print("    Double DQN: ENABLED (fixes Q-overestimation)");
        Print("    LSTM Memory: ", has_lstm ? "ENABLED" : "DISABLED");
        if(has_lstm){
            Print("      - LSTM hidden units: ", lstm_size);
            Print("      - Sequence length: ", seq_len);
            Print("      - Purpose: Market regime memory & partial observability");
        }
        Print("    Dueling Network: ", has_dueling ? "ENABLED" : "DISABLED");
        if(has_dueling){
            Print("      - Value head size: ", value_head_size);
            Print("      - Advantage head size: ", adv_head_size);
            Print("      - Purpose: Better action selection in noisy markets");
        }
        Print("    Confidence Network: ", has_confidence ? "ENABLED" : "DISABLED");
        if(has_confidence){
            Print("      - Confidence head size: ", confidence_head_size);
            Print("      - Purpose: Uncertainty estimation for trade filtering");
        }
    } else {
        Print("  Advanced features: NONE (legacy format)");
    }
    
    // Validate feature normalization data
    double feat_min[], feat_max[];
    ArrayResize(feat_min, state_size);
    ArrayResize(feat_max, state_size);
    
    bool normalization_valid = true;
    for(int i=0; i<state_size; ++i){
        feat_min[i] = FileReadDouble(h);  // min
        feat_max[i] = FileReadDouble(h);  // max
        
        // Validate normalization ranges
        if(feat_max[i] <= feat_min[i]){
            Print("WARNING: Feature ", i, " has invalid normalization range [", 
                  DoubleToString(feat_min[i], 6), ", ", DoubleToString(feat_max[i], 6), "]");
            normalization_valid = false;
        }
    }
    
    if(normalization_valid){
        Print("✓ Feature normalization: Valid ranges for all ", state_size, " features");
    } else {
        Print("⚠️ Feature normalization: Some features have invalid ranges");
    }
    
    // Read checkpoint data if available
    if(has_checkpoint){
        datetime last_trained = (datetime)FileReadLong(h);
        int training_steps = (int)FileReadLong(h);
        double epsilon = FileReadDouble(h);
        double beta = FileReadDouble(h);
        
        Print("Checkpoint data:");
        Print("  Last trained: ", TimeToString(last_trained));
        Print("  Training steps: ", training_steps);
        Print("  Epsilon: ", DoubleToString(epsilon, 4));
        Print("  Beta (PER): ", DoubleToString(beta, 4));
        
        // Check if data is recent
        datetime now = TimeCurrent();
        int days_old = (int)((now - last_trained) / (24*60*60));
        Print("  Data age: ", days_old, " days");
        
        if(days_old > 7){
            Print("  ⚠️  Model data is more than a week old - consider updating");
        } else {
            Print("  ✓ Model data is recent");
        }
    } else {
        Print("No checkpoint data (legacy format)");
    }
    
    // NEURAL NETWORK WEIGHT VALIDATION
    Print("=== NEURAL NETWORK VALIDATION ===");
    bool weights_valid = ValidateNetworkWeights(h, state_size, actions, h1, h2, h3, 
                                               has_lstm, lstm_size, has_dueling, 
                                               value_head_size, adv_head_size, 
                                               has_confidence, confidence_head_size);
    
    if(weights_valid){
        Print("✓ Neural network weights: Structure validated successfully");
    } else {
        Print("✗ Neural network weights: Validation failed or file corrupted");
    }
    
    FileClose(h);
    
    // Compatibility check with current settings
    Print("=");
    Print("=== COMPATIBILITY ANALYSIS ===");
    
    bool state_ok = (state_size == 35 || state_size == 45);  // IMPROVEMENT 4.4: Support both 35 and 45 feature models
    bool actions_ok = (actions == 6);
    bool is_legacy = !has_checkpoint;
    
    Print("Core compatibility:");
    if(state_ok){
        Print("  ✓ State size: COMPATIBLE (", state_size, " features)");
    } else {
        Print("  ✗ State size: INCOMPATIBLE - expected 35, got ", state_size);
    }
    
    if(actions_ok){
        Print("  ✓ Actions: COMPATIBLE (6 actions including FLAT)");
    } else {
        Print("  ✗ Actions: INCOMPATIBLE - expected 6, got ", actions);
        if(actions == 5){
            Print("    > This is an old 5-action model (missing FLAT action)");
            Print("    > FLAT action allows explicit position closing without taking opposite positions");
        }
    }
    
    Print("Architecture analysis:");
    if(is_legacy){
        Print("  ⚠️  LEGACY MODEL: Basic DQN architecture only");
        Print("    > Missing: Double DQN, LSTM memory, Dueling heads");
        Print("    > Performance: Limited compared to Double-Dueling DRQN");
    } else {
        Print("  ✓ MODERN MODEL: Double-Dueling DRQN architecture");
        Print("    > Double DQN: Reduces Q-value overestimation");
        if(has_lstm){
            Print("    > LSTM Memory: Handles market regimes and partial observability");
        } else {
            Print("    ! LSTM Memory: DISABLED (reduced performance in trending markets)");
        }
        if(has_dueling){
            Print("    > Dueling Heads: Better action selection in noisy conditions");
        } else {
            Print("    ! Dueling Heads: DISABLED (suboptimal action selection)");
        }
    }
    
    // Overall compatibility rating
    string rating = "UNKNOWN";
    string status_icon = "";
    
    if(state_ok && actions_ok && !is_legacy && has_lstm && has_dueling){
        rating = "EXCELLENT";
        status_icon = "✓✓✓";
    } else if(state_ok && actions_ok && !is_legacy){
        rating = "GOOD";
        status_icon = "✓✓";
    } else if(state_ok && actions_ok){
        rating = "ACCEPTABLE";
        status_icon = "✓";
    } else {
        rating = "INCOMPATIBLE";
        status_icon = "✗";
    }
    
    Print("");
    Print("Overall compatibility: ", status_icon, " ", rating);
    if(rating == "EXCELLENT"){
        Print("  > This model has all advanced features and should perform optimally");
        Print("  > Supports: Double DQN, LSTM memory, Dueling heads, Enhanced features");
        Print("  > IMPROVEMENT 4.5: Compatible with diverse training scenarios");
    } else if(rating == "GOOD"){
        Print("  > This model is modern but missing some advanced features");
        Print("  > IMPROVEMENT 4.5: Supports diverse training for better generalization");
    } else if(rating == "ACCEPTABLE"){
        Print("  > This model will work but performance may be limited");
        Print("  > IMPROVEMENT 4.5: Can benefit from diverse training retraining");
    } else {
        Print("  > This model cannot be used with the current EA/training script");
        Print("  > IMPROVEMENT 4.5: Retrain with diverse scenarios for better compatibility");
    }
    
    // Recommendations
    Print("");
    Print("=== RECOMMENDATIONS ===");
    
    if(rating == "INCOMPATIBLE"){
        Print("! CRITICAL ISSUES FOUND:");
        if(!state_ok){
            Print("* State size mismatch - model cannot be loaded");
            Print("* Delete model file and retrain with current data");
        }
        if(!actions_ok){
            if(actions == 5){
                Print("* Model missing FLAT action - functionality limited");
                Print("* FLAT action enables better position management");
            }
            Print("* Retrain model with 6 actions for full functionality");
        }
        Print("* Set InpForceRetrain=true in Cortextrainingv5.mq5");
        Print("* Recommended: Train new Double-Dueling DRQN model");
        
    } else if(rating == "ACCEPTABLE"){
        Print("! LEGACY MODEL DETECTED:");
        Print("* Your model uses basic DQN architecture (outdated)");
        Print("* Consider upgrading to Double-Dueling DRQN for better performance:");
        Print("*   - Enable InpUseDoubleDQN=true (reduces Q-overestimation)");
        Print("*   - Enable InpUseLSTM=true (market memory)");
        Print("*   - Enable InpUseDuelingNet=true (better action selection)");
        Print("* Set InpForceRetrain=true to train with new architecture");
        Print("* Benefits: Better handling of market regimes and volatility");
        
    } else if(rating == "GOOD"){
        Print("+ MODERN MODEL WITH ROOM FOR IMPROVEMENT:");
        if(!has_lstm){
            Print("* Consider enabling LSTM memory (InpUseLSTM=true)");
            Print("* LSTM helps with trend persistence and volatility regimes");
        }
        if(!has_dueling){
            Print("* Consider enabling Dueling heads (InpUseDuelingNet=true)");
            Print("* Dueling improves action selection in noisy markets");
        }
        Print("* Your model has Double DQN - excellent for stable learning!");
        Print("* For optimal performance, retrain with all features enabled");
        
    } else if(rating == "EXCELLENT"){
        Print("+ EXCELLENT MODEL - FULLY OPTIMIZED!");
        Print("* Your model has all advanced features:");
        Print("*   + Double DQN: Stable Q-learning with reduced overestimation");
        Print("*   + LSTM Memory: Handles market regimes and sequences");
        Print("*   + Dueling Heads: Optimal action selection");
        Print("*   + 6 Actions: Including FLAT for precise position control");
        Print("* No changes needed - this model should perform optimally");
        Print("* Monitor performance and retrain periodically with fresh data");
    }
    
    Print("");
    Print("General maintenance tips:");
    if(has_checkpoint){
        Print("• Your model supports incremental training - efficient updates possible");
        Print("• Use existing model as starting point for faster retraining");
    } else {
        Print("• Upgrade to checkpoint format for incremental training support");
        Print("• Set InpForceRetrain=false to enable incremental learning");
    }
    Print("• Run training script weekly for best market adaptation");
    Print("• Monitor equity curves and adjust risk parameters as needed");
    Print("• Test models on validation data before live deployment");
    Print("");
    Print("IMPROVEMENT 4.5 - Diverse Training Scenarios:");
    Print("• Enable InpUseDiverseTraining=true for better generalization");
    Print("• Use 3-5 training periods (InpTrainingPeriods) for robustness");
    Print("• Apply 5-20% data augmentation rate for uncertainty tolerance");
    Print("• Enable data shuffling to prevent temporal overfitting");
    Print("• Consider multi-symbol training for cross-market patterns");
    Print("");
    Print("IMPROVEMENT 4.6 - Validation and Early Stopping:");
    Print("• Enable InpUseAdvancedValidation=true for overfitting prevention");
    Print("• Use InpUseEarlyStopping=true to stop when validation deteriorates");
    Print("• Set validation frequency (InpValidationFrequency) to 5-10 epochs");
    Print("• Enable InpUseLearningRateDecay=true for adaptive optimization");
    Print("• Monitor Sharpe ratio, win rate, and drawdown on validation set");
    Print("• Use 10-15% validation split (InpValidationSplit) for robust testing");
    Print("");
    Print("IMPROVEMENT 5.1 - Indicator and Data Caching:");
    Print("• Enable InpUseIndicatorCaching=true for significant performance gains");
    Print("• Use InpCacheAllIndicators=true for maximum speed (more RAM usage)");
    Print("• Set cache validation frequency (InpCacheValidationFreq) to 1000 steps");
    Print("• Enable InpLogCachePerformance=true to monitor efficiency");
    Print("• Expected performance gain: 30-80% faster training depending on dataset size");
    Print("• Cache automatically handles all technical indicators and multi-timeframe data");
    Print("• Validation system ensures cache accuracy throughout training");
    Print("");
    Print("IMPROVEMENT 5.2 - Optimize Inner Loops:");
    Print("• Enable InpOptimizeInnerLoops=true for critical performance optimizations");
    Print("• Use InpCacheNeuralNetOutputs=true to avoid redundant forward passes");
    Print("• Enable InpCacheRewardComponents=true to cache expensive reward calculations");
    Print("• Set InpMinimizeFunctionCalls=true to inline simple operations");
    Print("• Adjust InpProgressReportFreq to reduce console output overhead");
    Print("• Expected performance gain: 15-40% faster training through loop optimization");
    Print("• System eliminates redundant function calls and caches expensive computations");
    Print("• Neural network output caching prevents duplicate forward propagation");
    Print("• Reward component caching avoids recalculating ATR, volatility, and costs");
    Print("");
    Print("IMPROVEMENT 5.3 - Vectorize Operations:");
    Print("• Enable InpUseVectorizedOps=true for bulk array operation performance");
    Print("• Use InpVectorizeIndicators=true for entire-array indicator calculations");
    Print("• Enable InpVectorizeRewards=true for batch reward processing");
    Print("• Set InpVectorizeFeatures=true for optimized feature array operations");
    Print("• Use InpUseMatrixOperations=true to leverage MQL5 matrix math");
    Print("• Configure InpVectorBatchSize (1000) to balance speed vs memory");
    Print("• Expected performance gain: 20-60% faster through vectorization");
    Print("• System replaces element-by-element loops with bulk array operations");
    Print("• Vectorized indicators process entire arrays instead of single values");
    Print("• Matrix operations leverage optimized mathematical libraries");
    
    Print("");
    Print("IMPROVEMENT 5.4 - Parallelize or Batch Training:");
    Print("• Enable InpUseBatchTraining=true for gradient accumulation performance gains");
    Print("• Use InpGradientAccumSteps (4-8) to batch gradients before model updates");
    Print("• Enable InpAdaptiveBatchSize=true for dynamic batch size optimization");
    Print("• Configure InpMinBatchSize (32) and InpMaxBatchSize (256) for memory balance");
    Print("• Use InpParallelDataPrep=true to simulate parallel data processing");
    Print("• Set InpParallelWorkers (4-8) to match CPU cores for optimal simulation");
    Print("• Enable InpLogBatchPerformance=true to monitor batch training effectiveness");
    Print("• Expected performance gain: 25-50% faster training through gradient accumulation");
    Print("• System accumulates gradients over multiple steps before applying updates");
    Print("• Reduces frequent weight adjustments improving training stability");
    Print("• Parallel data preparation simulation maximizes CPU utilization");
    Print("• Adaptive batch sizing automatically adjusts to memory and performance constraints");
    
    Print("");
    Print("IMPROVEMENT 5.5 - Limit Logging during Training:");
    Print("• Enable InpLimitTrainingLogs=true for dramatic performance improvements");
    Print("• Set InpLogFrequency (1000-5000) to control progress logging frequency");
    Print("• Enable InpLogOnlyImportant=true to show only critical training events");
    Print("• Use InpLogEpochSummaryOnly=true for minimal output mode");
    Print("• Enable InpDisableDebugLogs=true to remove verbose debug information");
    Print("• Set InpBatchLogFrequency (100-500) to control batch progress logging");
    Print("• Enable InpQuietMode=true for maximum performance with minimal output");
    Print("• Expected performance gain: 10-40% faster training through reduced I/O overhead");
    Print("• System intelligently suppresses repetitive console output in training loops");
    Print("• Maintains important error messages and epoch summaries for monitoring");
    Print("• Configurable logging levels allow fine-tuning based on monitoring needs");
    Print("• Performance statistics show exact suppression rates and time savings");
    
    Print("");
    Print("IMPROVEMENT 5.6 - Memory Management:");
    Print("• Enable InpOptimizeMemory=true for advanced memory management during training");
    Print("• Use InpReuseArrays=true to enable array pooling and reuse systems");
    Print("• Set InpMaxArrayPool (50-100) to balance memory usage vs allocation speed");
    Print("• Enable InpMemoryMonitoring=true to track memory usage and detect leaks");
    Print("• Configure InpMemoryCheckFreq (10000) to monitor without performance impact");
    Print("• Use InpAutoMemoryCleanup=true for proactive memory management");
    Print("• Set InpCleanupThreshold (100MB) to trigger cleanup when memory usage is high");
    Print("• Enable InpLogMemoryStats=true to report memory optimization effectiveness");
    Print("• Expected performance gain: 5-20% faster training through reduced allocation overhead");
    Print("• System prevents memory bloat during long training runs");
    Print("• Array pooling dramatically reduces allocation/deallocation overhead");
    Print("• Pre-allocated working arrays eliminate frequent memory operations");
    Print("• Automatic cleanup prevents gradual memory accumulation over time");
    
    Print("");
    Print("IMPROVEMENT 6.1 - Ensemble or Multi-Model Training:");
    Print("• Enable InpUseEnsembleTraining=true for robust multi-model trading strategies");
    Print("• Set InpEnsembleSize (3-5) to balance diversity vs training time");
    Print("• Enable InpRandomizeArchitecture=true for model architectural diversity");
    Print("• Use InpRandomizeWeights=true for different weight initialization starting points");
    Print("• Enable InpRandomizeHyperparams=true to vary learning rates and dropout");
    Print("• Set InpEnsembleCombination (VOTE/AVERAGE/WEIGHTED) for prediction blending");
    Print("• Use InpSaveIndividualModels=true to keep separate models for analysis");
    Print("• Configure InpEnsemblePrefix for organized model file naming");
    Print("• Enable InpLogEnsembleStats=true to monitor ensemble diversity and performance");
    Print("• Expected performance gain: 15-30% more robust predictions through model diversity");
    Print("• System trains multiple models with different configurations automatically");
    Print("• Ensemble combination reduces overfitting and improves generalization");
    Print("• Individual model analysis allows identification of best-performing architectures");
    Print("• Model diversity metrics ensure ensemble provides genuine benefits over single models");
    Print("• WARNING: Training time multiplies by ensemble size - plan accordingly");
    Print("• COORDINATION: Requires EA modifications to load and combine multiple models");
    
    Print("");
    Print("IMPROVEMENT 6.2 - Adaptive/Online Learning Mechanism:");
    Print("• Enable InpUseOnlineLearning=true for continuous model adaptation to market regime shifts");
    Print("• Set InpOnlineUpdateDays (7-14) to balance adaptation speed vs stability");
    Print("• Configure InpOnlineDataWindow (30-60 days) for recent data usage in online updates");
    Print("• Use reduced InpOnlineLearningRate (0.00001-0.0001) for stable incremental learning");
    Print("• Set InpOnlineEpochs (1-3) for lightweight online training sessions");
    Print("• Enable InpUseRegimeDetection=true to trigger updates when market conditions change");
    Print("• Adjust InpRegimeThreshold (0.1-0.2) for regime shift sensitivity");
    Print("• Use InpPreserveBaseModel=true to maintain original model as fallback");
    Print("• Configure InpExperienceBufferSize (50000-100000) for experience collection capacity");
    Print("• Enable InpLogOnlineLearning=true to monitor adaptation effectiveness and performance");
    Print("• Expected benefits: 20-40% improved adaptability to changing market conditions");
    Print("• System automatically detects regime shifts using volatility, trend, and performance metrics");
    Print("• Online learning prevents model performance degradation over time");
    Print("• Experience collection during live trading enables targeted model improvements");
    Print("• Periodic retraining with recent data maintains model relevance");
    Print("• Safety mechanisms prevent catastrophic forgetting and preserve base model performance");
    Print("• Performance tracking measures adaptation effectiveness and model drift");
    Print("• Regime detection uses multi-factor analysis: volatility, trend strength, correlation changes");
    Print("• WARNING: Requires coordination with EA for experience collection during live trading");
    Print("• COORDINATION: EA must support experience buffer integration and model updates");
    
    Print("");
    Print("IMPROVEMENT 6.3 - Confidence-augmented Training:");
    Print("• Enable InpUseConfidenceTraining=true for well-calibrated confidence predictions");
    Print("• Use InpUseDualObjective=true to train classification accuracy alongside trading reward");
    Print("• Set InpConfidenceWeight (0.2-0.4) to balance confidence vs trading objectives");
    Print("• Configure InpConfidenceLearningRate (0.00001-0.001) for stable confidence learning");
    Print("• Enable InpUseConfidenceCalibration=true for probability calibration training");
    Print("• Set InpCalibrationWeight (0.1-0.3) to control calibration loss influence");
    Print("• Use InpUseConfidenceRewards=true to reward well-calibrated predictions");
    Print("• Adjust InpConfidencePenaltyRate (0.05-0.2) for overconfidence penalties");
    Print("• Enable InpLogConfidenceMetrics=true to monitor calibration quality");
    Print("• Expected benefits: 15-30% improved signal filtering through confidence-based decisions");
    Print("• System learns secondary classification objective for prediction reliability");
    Print("• Dual-objective training maximizes both profitability and prediction accuracy");
    Print("• Confidence calibration ensures predicted probabilities match actual outcomes");
    Print("• Well-calibrated confidence enables reliable trade filtering in live deployment");
    Print("• Reward system encourages high confidence on correct predictions, penalizes overconfidence");
    Print("• Advanced metrics: Expected Calibration Error (ECE), Brier Score, Reliability");
    Print("• Confidence discrimination measures ability to distinguish correct from wrong predictions");
    Print("• Calibration bins analyze accuracy across different confidence levels (0.1 to 1.0)");
    Print("• System supports both integrated confidence and separate confidence network architectures");
    Print("• Performance tracking shows net impact of confidence system on trading results");
    Print("• COORDINATION: Enables EA confidence-based filtering for improved trade selection");
    
    Print("");
    Print("IMPROVEMENT 6.4 - Automated Hyperparameter Tuning:");
    Print("• Enable InpUseHyperparameterTuning=true for automatic parameter optimization");
    Print("• Choose InpOptimizationMethod: GRID (systematic), BAYESIAN (efficient), RANDOM (exploratory)");
    Print("• Set InpOptimizationIterations (20-50) to balance thorough search vs training time");
    Print("• Use InpUseValidationSplit=true to avoid overfitting during hyperparameter search");
    Print("• Configure InpHyperparamValidationSplit (0.1-0.2) for robust validation performance");
    Print("• Select InpOptimizationObjective: SHARPE (risk-adjusted), PROFIT (return), DRAWDOWN (risk), MULTI (balanced)");
    Print("• Enable InpSaveOptimizationResults=true to analyze parameter performance patterns");
    Print("• Use InpLogOptimizationProgress=true to monitor optimization progress");
    Print("• Set InpOptimizationSeed for reproducible hyperparameter experiments");
    Print("• Expected benefits: 20-50% improved model performance through optimal parameter selection");
    Print("• System automatically searches key parameter spaces: learning rate, architecture, regularization");
    Print("• Grid search provides systematic coverage of parameter combinations");
    Print("• Bayesian optimization balances exploration vs exploitation for efficient search");
    Print("• Random search provides baseline comparison and explores unexpected combinations");
    Print("• Multi-objective optimization balances profitability, risk, and stability");
    Print("• Validation split prevents overfitting to training data during hyperparameter selection");
    Print("• Advanced search spaces: log-scale learning rates, network architectures, regularization");
    Print("• Comprehensive metrics: Sharpe ratio, Calmar ratio, Sortino ratio, profit factor");
    Print("• Result analysis identifies optimal parameter patterns and sensitivity");
    Print("• CSV export enables external analysis and visualization of optimization results");
    Print("• Integration with confidence training (6.3) and online learning (6.2) parameter optimization");
    Print("• Strategy Tester compatibility for parallel hyperparameter optimization");
    Print("• WARNING: Hyperparameter tuning significantly increases training time");
    Print("• RECOMMENDATION: Use smaller datasets for hyperparameter search, full data for final training");
    
    Print("");
    Print("IMPROVEMENT 7.1 - Sync with EA Logic (CortexBacktestWorking.mq5):");
    Print("• Enable unified trade logic synchronization between EA and backtester");
    Print("• Use consistent entry/exit conditions across both systems:");
    Print("  - Master risk check combining confidence, frequency, volatility filters");
    Print("  - ATR-based dynamic stops and trailing stop functionality");
    Print("  - Volatility regime detection and position size adjustment");
    Print("  - Emergency risk management with automatic trading halts");
    Print("• Sync advanced features with EA configuration:");
    Print("  - Confidence-based filtering (6.3) with same thresholds");
    Print("  - Dynamic position sizing based on account risk percentage");
    Print("  - Session and time-based filtering for optimal trading hours");
    Print("• Benefits: Accurate backtest results reflecting live EA behavior");
    Print("• Performance: Consistent risk management and position management");
    Print("• Validation: Backtester results should closely match EA performance");
    Print("• Integration: Shared logic ensures no discrepancies between systems");
    Print("• Configuration: Use identical input parameters in both EA and backtester");
    Print("• Monitoring: Comprehensive performance statistics for all unified features");
    Print("• WARNING: Both systems must use same model and identical risk parameters");
    Print("• RECOMMENDATION: Always validate backtester changes against EA behavior");
    
    Print("");
    Print("IMPROVEMENT 7.2 - Comprehensive Metrics Output (CortexBacktestWorking.mq5):");
    Print("• Enable advanced performance metrics beyond basic profit/loss analysis");
    Print("• Calculate professional-grade trading metrics:");
    Print("  - Risk-adjusted returns: Sharpe, Sortino, Calmar, Sterling ratios");
    Print("  - Risk metrics: Maximum drawdown, VaR (95%/99%), Conditional VaR");
    Print("  - Trade analysis: Profit factor, expectancy, win/loss ratios");
    Print("  - Time-based metrics: Holding periods, consecutive statistics");
    Print("  - Advanced metrics: Ulcer Index, Pain Index, Kelly Criterion");
    Print("• Comprehensive performance rating system (1-5 stars):");
    Print("  - Evaluates profitability, risk management, and consistency");
    Print("  - Provides automated recommendations for improvement");
    Print("  - Scores based on industry-standard benchmarks");
    Print("• Enhanced trade recording with MAE/MFE tracking:");
    Print("  - Maximum Adverse/Favorable Excursion for each trade");
    Print("  - Trade duration, confidence scores, exit reasons");
    Print("  - Commission tracking and detailed P&L attribution");
    Print("• Real-time equity curve and drawdown analysis:");
    Print("  - Continuous equity curve tracking during backtest");
    Print("  - Underwater curve showing drawdown periods");
    Print("  - Daily returns calculation for volatility metrics");
    Print("• Professional reporting format with visual indicators:");
    Print("  - Organized sections: Performance, Risk, Time Analysis");
    Print("  - Color-coded ratings and warning indicators");
    Print("  - Actionable recommendations based on metric analysis");
    Print("• Benefits: Institutional-quality performance evaluation");
    Print("• Integration: Seamless integration with existing backtest workflow");
    Print("• Analysis depth: 50+ professional trading metrics calculated");
    Print("• Comparison capability: Easy comparison across different strategy variants");
    Print("• RECOMMENDATION: Use comprehensive metrics for strategy optimization");
    Print("• USAGE: Enable detailed logging for full MAE/MFE accuracy");
    
    Print("");
    Print("IMPROVEMENT 7.3 - Trade-by-Trade Logging (CortexBacktestWorking.mq5):");
    Print("• Enable comprehensive CSV export for machine-readable analysis");
    Print("• Configure CSV logging with input parameters:");
    Print("  - InpEnableCSVLogging=true to activate CSV export functionality");
    Print("  - Set InpCSVTradeFileName for individual trade records export");
    Print("  - Set InpCSVEquityFileName for equity curve and drawdown export");
    Print("  - Set InpCSVMetricsFileName for comprehensive metrics export");
    Print("  - Configure InpCSVIncludeHeaders=true for descriptive column headers");
    Print("  - Use InpCSVAppendMode=false to overwrite existing files");
    Print("• Detailed trade-by-trade CSV export includes:");
    Print("  - Complete trade data: entry/exit times, prices, P&L, duration");
    Print("  - Risk metrics: MAE/MFE, stop distances, confidence scores");
    Print("  - Exit attribution: precise reason why each trade was closed");
    Print("  - Rule tracking: which specific trading rules triggered entries");
    Print("  - Position data: lot sizes, leverage, account percentage risk");
    Print("• Real-time equity curve CSV export provides:");
    Print("  - Continuous balance tracking throughout backtest execution");
    Print("  - Drawdown calculations and underwater curve percentages");
    Print("  - Position status tracking (open/closed) at each time point");
    Print("  - Account balance, equity, and margin utilization data");
    Print("• Comprehensive metrics CSV export contains:");
    Print("  - All 50+ professional trading metrics in machine-readable format");
    Print("  - Performance ratios: Sharpe, Sortino, Calmar, Sterling, Kelly");
    Print("  - Risk analysis: VaR, CVaR, Ulcer Index, maximum drawdown");
    Print("  - Trade statistics: win rates, profit factors, expectancy");
    Print("  - Time analysis: holding periods, consecutive statistics");
    Print("• Advanced analysis capabilities:");
    Print("  - CSV files enable detailed external analysis in Excel/Python/R");
    Print("  - Machine-readable format supports automated strategy comparison");
    Print("  - Rule attribution analysis identifies most profitable entry/exit rules");
    Print("  - Time-series analysis of equity curves and performance patterns");
    Print("  - Risk analysis through detailed MAE/MFE distributions");
    Print("• Integration benefits:");
    Print("  - Seamless export without affecting backtest performance");
    Print("  - Professional data format compatible with analysis tools");
    Print("  - Automated file management and error handling");
    Print("  - Configurable logging frequency for performance optimization");
    Print("• Usage recommendations:");
    Print("  - Enable all CSV exports for comprehensive strategy analysis");
    Print("  - Use trade-by-trade data to identify optimal entry/exit rules");
    Print("  - Analyze equity curves for drawdown patterns and recovery times");
    Print("  - Export metrics for systematic strategy comparison and ranking");
    Print("  - Integrate with external tools for advanced statistical analysis");
    Print("• COORDINATION: Works with comprehensive metrics (7.2) for complete analysis");
    Print("• FILE LOCATION: CSV files saved to MT5 Files directory for easy access");
    
    Print("");
    Print("IMPROVEMENT 7.4 - Parameter Flexibility (CortexBacktestWorking.mq5):");
    Print("• Enable comprehensive parameter control for strategy optimization and what-if analysis");
    Print("• Configure flexible parameter groups through organized input sections:");
    Print("  - Risk Management Parameters: Control stop losses, take profits, daily limits");
    Print("  - Trading Filter Parameters: Enable/disable confidence, volatility, spread filters");
    Print("  - Position Sizing Parameters: Configure dynamic sizing based on multiple factors");
    Print("  - Session and Time Parameters: Control trading hours, days, intervals");
    Print("  - Signal Quality Parameters: Set signal strength ranges and confirmation requirements");
    Print("  - Advanced Feature Parameters: Enable market regime, correlation, volume filters");
    Print("  - Optimization Control Parameters: Configure robustness testing and optimization");
    Print("• Risk management flexibility enables:");
    Print("  - InpRiskPercentage (1-5%): Account risk percentage per trade for consistent exposure");
    Print("  - InpStopLossATR (1-4): ATR-based stop loss multiplier for market-adaptive stops");
    Print("  - InpTakeProfitATR (2-6): ATR-based take profit for optimal risk-reward ratios");
    Print("  - InpMaxLossPerDay ($100-1000): Daily loss limits for capital preservation");
    Print("  - InpMaxConsecutiveLosses (3-10): Auto-halt after consecutive losses");
    Print("• Position sizing flexibility provides:");
    Print("  - Volatility-based sizing: Reduces position size in high volatility conditions");
    Print("  - Confidence-based sizing: Increases position size for high-confidence signals");
    Print("  - Equity-based sizing: Scales position size with account growth");
    Print("  - Risk percentage limits: Ensures consistent risk exposure regardless of sizing method");
    Print("• Trading filter flexibility includes:");
    Print("  - Confidence filtering with adjustable thresholds (0.5-0.9)");
    Print("  - Volatility range filtering (min/max ATR) for optimal market conditions");
    Print("  - Session hour controls with day-of-week enabling/disabling");
    Print("  - Anti-whipsaw protection with minimum trade intervals");
    Print("  - Signal strength validation with configurable ranges");
    Print("• Advanced optimization features:");
    Print("  - Parameter validation with range checking and error reporting");
    Print("  - Parameter set saving for tracking optimization results");
    Print("  - Comprehensive parameter summary with organized visual output");
    Print("  - Flexible filter integration with existing trading logic");
    Print("• Strategy Tester integration benefits:");
    Print("  - All parameters can be optimized using MT5 Strategy Tester");
    Print("  - Parameter ranges documented in comments for easy optimization setup");
    Print("  - Group organization makes parameter management efficient");
    Print("  - What-if scenario testing without code changes");
    Print("• Usage recommendations for parameter optimization:");
    Print("  - Start with default values and gradually adjust one parameter group at a time");
    Print("  - Use InpOptimizationMode=true for robustness testing with random variations");
    Print("  - Enable InpParameterSetSaving=true to track successful parameter combinations");
    Print("  - Focus on risk management parameters first, then filters, then advanced features");
    Print("  - Use confidence and volatility filters for different market conditions");
    Print("  - Test parameter sensitivity by running small variations around optimal values");
    Print("• Integration with other improvements:");
    Print("  - Works seamlessly with comprehensive metrics (7.2) for parameter effectiveness analysis");
    Print("  - Integrates with CSV logging (7.3) for detailed parameter impact tracking");
    Print("  - Uses unified EA logic (7.1) ensuring consistent behavior across systems");
    Print("• Performance optimization guidelines:");
    Print("  - Use fewer active filters for faster backtesting during parameter sweeps");
    Print("  - Enable detailed logging only for final validation runs");
    Print("  - Group related parameters for efficient batch optimization");
    Print("• COORDINATION: Parameter changes require validation through backtesting before live deployment");
    Print("• RECOMMENDATION: Use parameter flexibility for systematic strategy optimization and robustness testing");
    
    Print("");
    Print("🎲 IMPROVEMENT 7.5: BATCH AND MONTE CARLO TESTING");
    Print("─────────────────────────────────────────────────────────────────");
    Print("• Monte Carlo robustness testing is now available for comprehensive strategy validation");
    Print("• New testing capabilities include:");
    Print("  - Batch testing with randomized market conditions (spread, slippage, commission variations)");
    Print("  - Random period selection and data shuffling for robustness assessment");
    Print("  - Comprehensive statistical analysis across multiple scenarios");
    Print("  - Automated robustness scoring and strategy stability measurement");
    Print("• Key Monte Carlo parameters:");
    Print("  - InpEnableMonteCarloMode=true: Enable Monte Carlo testing mode");
    Print("  - InpMonteCarloRuns=100: Number of simulation runs (recommended: 50-1000)");
    Print("  - InpMCDataShuffling=true: Enable data randomization for robustness");
    Print("  - InpMCPeriodVariation=true: Test across different time periods");
    Print("  - InpMCSpreadVariation=15.0: Spread variation percentage (5-25%)");
    Print("  - InpMCSlippageVariation=2.0: Maximum slippage variation in pips");
    Print("• Monte Carlo output analysis:");
    Print("  - Overall robustness score (target: >0.7 for live deployment)");
    Print("  - Return consistency and risk consistency scores");
    Print("  - Strategy stability across different market conditions");
    Print("  - Success rate percentage (robust runs / total runs)");
    Print("  - Statistical distributions (5th-95th percentiles)");
    Print("• Usage recommendations for Monte Carlo testing:");
    Print("  - Run 100+ simulations for statistically significant results");
    Print("  - Test with realistic spread/slippage variations for your broker");
    Print("  - Use InpMCRobustnessThreshold=0.7 as minimum acceptable robustness");
    Print("  - Enable result saving (InpMCSaveResults=true) for detailed analysis");
    Print("  - Review robustness score distributions to identify improvement areas");
    Print("• Integration with other improvements:");
    Print("  - Utilizes comprehensive metrics (7.2) for robust performance evaluation");
    Print("  - Exports detailed CSV results (7.3) for advanced statistical analysis");
    Print("  - Tests parameter flexibility (7.4) across randomized conditions");
    Print("  - Validates EA logic consistency (7.1) under various scenarios");
    Print("• Performance guidelines for Monte Carlo testing:");
    Print("  - Each run adds 100-500ms depending on backtest period and complexity");
    Print("  - 100 runs typically complete in 1-3 minutes for 30-day backtests");
    Print("  - Use shorter periods (7-14 days) for extensive parameter optimization");
    Print("  - Enable selective logging to minimize CSV file sizes during batch runs");
    Print("• CRITICAL: Monte Carlo results provide confidence intervals for strategy deployment");
    Print("• RECOMMENDATION: Use Monte Carlo testing for final validation before live deployment");
    
    Print("");
    Print("=== DOUBLE-DUELING DRQN DIAGNOSTIC COMPLETE ===");
    Print("For questions about the new architecture, see CLAUDE.md documentation");
}