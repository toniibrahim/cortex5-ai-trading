//+------------------------------------------------------------------+
//|                                         ModelDiagnostic5.mq5     |
//|               Diagnostic tool for Double-Dueling DRQN model files |
//+------------------------------------------------------------------+
#property script_show_inputs

input string InpModelFileName = "DoubleDueling_DRQN_Model.dat";

void OnStart(){
    Print("=== DOUBLE-DUELING DRQN MODEL DIAGNOSTIC TOOL ===");
    Print("Analyzing model file: ", InpModelFileName);
    
    int h = FileOpen(InpModelFileName, FILE_BIN|FILE_READ);
    if(h == INVALID_HANDLE){
        Print("ERROR: Cannot open model file. File may not exist.");
        Print("Expected location: ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\", InpModelFileName);
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
    int lstm_size = 0;
    int seq_len = 0;
    int value_head_size = 0;
    int adv_head_size = 0;
    
    bool has_checkpoint = (magic == (long)0xC0DE0203);
    if(has_checkpoint){
        has_lstm = (FileReadLong(h) == 1);
        has_dueling = (FileReadLong(h) == 1);
        lstm_size = (int)FileReadLong(h);
        seq_len = (int)FileReadLong(h);
        value_head_size = (int)FileReadLong(h);
        adv_head_size = (int)FileReadLong(h);
        
        Print("DEBUG: Architecture flags read:");
        Print("  LSTM enabled: ", has_lstm);
        Print("  Dueling enabled: ", has_dueling);
        Print("  LSTM size: ", lstm_size);
        Print("  Sequence length: ", seq_len);
        Print("  Value head size: ", value_head_size);
        Print("  Advantage head size: ", adv_head_size);
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
    } else {
        Print("  Advanced features: NONE (legacy format)");
    }
    
    // Skip feature normalization data
    for(int i=0; i<state_size; ++i){
        FileReadDouble(h);  // min
        FileReadDouble(h);  // max
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
    
    FileClose(h);
    
    // Compatibility check with current settings
    Print("=");
    Print("=== COMPATIBILITY ANALYSIS ===");
    
    bool state_ok = (state_size == 35);
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
    } else if(rating == "GOOD"){
        Print("  > This model is modern but missing some advanced features");
    } else if(rating == "ACCEPTABLE"){
        Print("  > This model will work but performance may be limited");
    } else {
        Print("  > This model cannot be used with the current EA/training script");
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
    Print("=== DOUBLE-DUELING DRQN DIAGNOSTIC COMPLETE ===");
    Print("For questions about the new architecture, see CLAUDE.md documentation");
}