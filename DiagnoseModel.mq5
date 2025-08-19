//+------------------------------------------------------------------+
//|                                              DiagnoseModel.mq5   |
//|                    Simple Model File Diagnostic Script           |
//+------------------------------------------------------------------+
#include <CortexTradeLogic.mqh>
#property script_show_inputs

input string InpModelFileName = "DoubleDueling_DRQN_Model.dat";

void OnStart(){
    Print("=== MODEL FILE DIAGNOSTIC ===");
    Print("Analyzing: ", InpModelFileName);
    Print("Expected STATE_SIZE: ", STATE_SIZE);
    Print("Expected ACTIONS: ", ACTIONS);
    Print("");
    
    int h = FileOpen(InpModelFileName, FILE_BIN|FILE_READ);
    if(h == INVALID_HANDLE){
        Print("ERROR: Cannot open model file: ", InpModelFileName);
        Print("Error code: ", GetLastError());
        return;
    }
    
    // Get file size
    FileSeek(h, 0, SEEK_END);
    ulong file_size = FileTell(h);
    FileSeek(h, 0, SEEK_SET);
    Print("File size: ", file_size, " bytes");
    
    // Read and verify magic number
    if(FileTell(h) + 8 > file_size) {
        Print("ERROR: File too small for magic number");
        FileClose(h);
        return;
    }
    
    long magic = FileReadLong(h);
    Print("Magic number: 0x", IntegerToString(magic, 16));
    
    bool has_checkpoint = false;
    if(magic == (long)0xC0DE0203) {
        Print("Format: New format with training checkpoints");
        has_checkpoint = true;
    } else if(magic == (long)0xC0DE0202) {
        Print("Format: Old format without checkpoints");
        has_checkpoint = false;
    } else {
        Print("ERROR: Unknown magic number - file may be corrupted");
        FileClose(h);
        return;
    }
    
    // Read symbol and timeframe
    if(FileTell(h) + 8 > file_size) {
        Print("ERROR: File truncated at symbol section");
        FileClose(h);
        return;
    }
    
    int sym_len = (int)FileReadLong(h);
    Print("Symbol length: ", sym_len);
    
    if(sym_len < 0 || sym_len > 20) {
        Print("ERROR: Invalid symbol length");
        FileClose(h);
        return;
    }
    
    if(FileTell(h) + sym_len + 4 > file_size) {
        Print("ERROR: File truncated at symbol data");
        FileClose(h);
        return;
    }
    
    string model_symbol = "";
    for(int i = 0; i < sym_len; i++) {
        model_symbol += CharToString((uchar)FileReadInteger(h, 1));
    }
    
    int tf_int = (int)FileReadLong(h);
    ENUM_TIMEFRAMES model_tf = (ENUM_TIMEFRAMES)tf_int;
    
    Print("Model symbol: '", model_symbol, "'");
    Print("Model timeframe: ", EnumToString(model_tf), " (", tf_int, ")");
    
    // Read architecture parameters
    if(FileTell(h) + 20 > file_size) {
        Print("ERROR: File truncated at architecture section");
        FileClose(h);
        return;
    }
    
    int stsz = (int)FileReadLong(h);
    int acts = (int)FileReadLong(h);
    int h1 = (int)FileReadLong(h);
    int h2 = (int)FileReadLong(h);
    int h3 = (int)FileReadLong(h);
    
    Print("");
    Print("=== ARCHITECTURE ANALYSIS ===");
    Print("File STATE_SIZE: ", stsz);
    Print("File ACTIONS: ", acts);
    Print("Hidden layers: [", h1, ", ", h2, ", ", h3, "]");
    
    // Advanced features (if checkpoint format)
    bool file_has_lstm = false;
    bool file_has_dueling = false;
    int file_lstm_size = 0;
    int file_value_head = 0;
    int file_adv_head = 0;
    
    if(has_checkpoint) {
        if(FileTell(h) + 24 > file_size) {
            Print("ERROR: File truncated at advanced features section");
            FileClose(h);
            return;
        }
        
        file_has_lstm = ((int)FileReadLong(h) != 0);
        file_has_dueling = ((int)FileReadLong(h) != 0);
        file_lstm_size = (int)FileReadLong(h);
        file_value_head = (int)FileReadLong(h);
        file_adv_head = (int)FileReadLong(h);
        
        Print("LSTM enabled: ", file_has_lstm ? "YES" : "NO");
        if(file_has_lstm) Print("LSTM size: ", file_lstm_size);
        Print("Dueling enabled: ", file_has_dueling ? "YES" : "NO");
        if(file_has_dueling) {
            Print("Value head size: ", file_value_head);
            Print("Advantage head size: ", file_adv_head);
        }
    }
    
    // Feature normalization parameters
    if(FileTell(h) + (stsz * 16) > file_size) {
        Print("ERROR: File truncated at feature normalization section");
        FileClose(h);
        return;
    }
    
    Print("");
    Print("=== FEATURE NORMALIZATION ===");
    double feat_min_sample = FileReadDouble(h);
    double feat_max_sample = FileReadDouble(h);
    Print("Feature 0 range: [", DoubleToString(feat_min_sample, 6), ", ", DoubleToString(feat_max_sample, 6), "]");
    
    // Skip remaining feature ranges
    for(int i = 1; i < stsz; i++) {
        FileReadDouble(h); // min
        FileReadDouble(h); // max
    }
    
    if(has_checkpoint) {
        Print("");
        Print("=== CHECKPOINT DATA ===");
        if(FileTell(h) + 32 > file_size) {
            Print("ERROR: File truncated at checkpoint section");
            FileClose(h);
            return;
        }
        
        datetime last_trained = (datetime)FileReadLong(h);
        int training_steps = (int)FileReadLong(h);
        double checkpoint_eps = FileReadDouble(h);
        double checkpoint_beta = FileReadDouble(h);
        
        Print("Last trained: ", TimeToString(last_trained));
        Print("Training steps: ", training_steps);
        Print("Epsilon: ", DoubleToString(checkpoint_eps, 4));
        Print("Beta: ", DoubleToString(checkpoint_beta, 4));
    }
    
    ulong remaining_bytes = file_size - FileTell(h);
    Print("");
    Print("=== COMPATIBILITY CHECK ===");
    Print("Expected STATE_SIZE: ", STATE_SIZE, " | File has: ", stsz, " | Match: ", (stsz == STATE_SIZE ? "✓" : "✗"));
    Print("Expected ACTIONS: ", ACTIONS, " | File has: ", acts, " | Match: ", (acts == ACTIONS ? "✓" : "✗"));
    
    if(stsz == STATE_SIZE && acts == ACTIONS) {
        Print("✓ MODEL IS COMPATIBLE");
    } else {
        Print("✗ MODEL IS INCOMPATIBLE");
        Print("  The EA expects ", STATE_SIZE, " features and ", ACTIONS, " actions");
        Print("  This model has ", stsz, " features and ", acts, " actions");
    }
    
    Print("");
    Print("Remaining file data: ", remaining_bytes, " bytes (network weights and biases)");
    Print("Diagnostic complete.");
    
    FileClose(h);
}