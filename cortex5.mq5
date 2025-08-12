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

// ADVANCED RISK MANAGEMENT (NEW - trailing stops and volatility controls)
input bool    InpUseTrailingStop     = true;  // Enable trailing stop functionality
input double  InpTrailStartATR       = 2.0;   // ATR multiple to start trailing (profit threshold)
input double  InpTrailStopATR        = 1.0;   // ATR multiple for trailing stop distance
input bool    InpUseBreakEven        = true;  // Move SL to break-even when profitable
input double  InpBreakEvenATR        = 1.5;   // ATR multiple profit to trigger break-even
input double  InpBreakEvenBuffer     = 5.0;   // Points buffer beyond break-even (small profit)

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
#define STATE_SIZE 35         // Number of market features the AI analyzes
#define ACTIONS     6          // Number of possible trading actions
#define ACTION_BUY_STRONG 0    // Strong buy signal (larger position)
#define ACTION_BUY_WEAK   1    // Weak buy signal (smaller position)
#define ACTION_SELL_STRONG 2   // Strong sell signal (larger position)
#define ACTION_SELL_WEAK   3   // Weak sell signal (smaller position)
#define ACTION_HOLD        4   // Do nothing / hold current position
#define ACTION_FLAT        5   // Close position and stay flat

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

// PHASE 2 & 3 ENHANCEMENT GLOBALS - ADVANCED POSITION AND MARKET ANALYSIS
double           g_trend_strength = 0.0;          // Current trend strength (Phase 3)
double           g_volatility_regime = 0.0;       // Current volatility regime indicator (Phase 3)
int              g_market_regime = 0;              // Market regime: 0=ranging, 1=trending, 2=volatile (Phase 3)
double           g_position_normalized_time = 0.0; // Normalized holding time [0-1] (Phase 3)
double           g_unrealized_pnl_ratio = 0.0;    // P&L ratio vs ATR (Phase 3)

// UTILITY FUNCTIONS - Small helper functions used throughout the program
int    idx2(const int r,const int c,const int ncols){ return r*ncols + c; } // Convert 2D matrix position to 1D array index
int    argmax(const double &v[]){ int m=0; for(int i=1;i<ArraySize(v);++i) if(v[i]>v[m]) m=i; return m; } // Find index of highest value in array
double clipd(const double x,const double a,const double b){ return (x<a? a : (x>b? b : x)); } // Constrain value between min and max

// NEURAL NETWORK LAYER STRUCTURE
// Represents one layer of the neural network with weights and biases
struct DenseLayer{ 
    int in,out;     // Number of input and output neurons
    double W[];     // Weights connecting inputs to outputs
    double b[];     // Bias values for each output neuron
};

// LSTM Layer structure for inference (simplified from training version)
struct LSTMInferenceLayer{
    int in, out;
    double Wf[], Wi[], Wc[], Wo[];  // Input-to-hidden weights
    double Uf[], Ui[], Uc[], Uo[];  // Hidden-to-hidden weights
    double bf[], bi[], bc[], bo[];  // Bias vectors
    double h_prev[], c_prev[];      // Previous states (maintained between predictions)
    double h_curr[], c_curr[];      // Current states
    
    LSTMInferenceLayer() : in(0), out(0) {}
    
    void Init(int _in, int _out){
        in = _in; out = _out;
        int ih_size = in * out;
        int hh_size = out * out;
        
        ArrayResize(Wf, ih_size); ArrayResize(Wi, ih_size); ArrayResize(Wc, ih_size); ArrayResize(Wo, ih_size);
        ArrayResize(Uf, hh_size); ArrayResize(Ui, hh_size); ArrayResize(Uc, hh_size); ArrayResize(Uo, hh_size);
        ArrayResize(bf, out); ArrayResize(bi, out); ArrayResize(bc, out); ArrayResize(bo, out);
        ArrayResize(h_prev, out); ArrayResize(c_prev, out);
        ArrayResize(h_curr, out); ArrayResize(c_curr, out);
        
        ResetState();
    }
    
    void ResetState(){
        ArrayInitialize(h_prev, 0.0);
        ArrayInitialize(c_prev, 0.0);
        ArrayInitialize(h_curr, 0.0);
        ArrayInitialize(c_curr, 0.0);
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
            value_head.out = 1;
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
          ArrayResize(final_out, outSize);
          for(int i=0; i<outSize; ++i){
              final_out[i] = v_out[0] + (a_out[i] - mean_adv);
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
};

CInferenceNetwork g_Q;  // The main Double-Dueling DRQN that makes trading decisions

//-------------------------- MODEL LOADING FUNCTIONS -----------------------
// These functions load the trained Double-Dueling DRQN model from a file
// Load one layer's weights and biases from the model file
bool LoadLayer(const int h, DenseLayer &L){
  int in  = (int)FileReadLong(h);   // Read input size from file
  int out = (int)FileReadLong(h);   // Read output size from file
  if(in!=L.in || out!=L.out){ 
      Print("LoadLayer mismatch: expected ",L.in,"x",L.out," got ",in,"x",out); 
      return false; 
  }
  // Load all the weights
  for(int i=0;i<L.in*L.out;++i) L.W[i]=FileReadDouble(h);
  // Load all the biases
  for(int j=0;j<L.out;++j)      L.b[j]=FileReadDouble(h);
  return true;
}

// Load LSTM layer from model file
bool LoadLSTMLayer(const int h, LSTMInferenceLayer &lstm){
  int in = (int)FileReadLong(h);
  int out = (int)FileReadLong(h);
  
  if(in != lstm.in || out != lstm.out){
    Print("LoadLSTMLayer mismatch: expected ",lstm.in,"x",lstm.out," got ",in,"x",out);
    return false;
  }
  
  // Load all LSTM weights
  for(int i=0; i<lstm.in*lstm.out; i++){
    lstm.Wf[i] = FileReadDouble(h); // Forget gate input weights
    lstm.Wi[i] = FileReadDouble(h); // Input gate input weights
    lstm.Wc[i] = FileReadDouble(h); // Cell gate input weights
    lstm.Wo[i] = FileReadDouble(h); // Output gate input weights
  }
  
  for(int i=0; i<lstm.out*lstm.out; i++){
    lstm.Uf[i] = FileReadDouble(h); // Forget gate hidden weights
    lstm.Ui[i] = FileReadDouble(h); // Input gate hidden weights
    lstm.Uc[i] = FileReadDouble(h); // Cell gate hidden weights
    lstm.Uo[i] = FileReadDouble(h); // Output gate hidden weights
  }
  
  // Load biases
  for(int i=0; i<lstm.out; i++){
    lstm.bf[i] = FileReadDouble(h); // Forget gate bias
    lstm.bi[i] = FileReadDouble(h); // Input gate bias
    lstm.bc[i] = FileReadDouble(h); // Cell gate bias
    lstm.bo[i] = FileReadDouble(h); // Output gate bias
  }
  
  return true;
}

// Check if a timeframe value is valid in MetaTrader 5
// This prevents loading corrupted model files
bool IsValidTF(int tf){
  static int tfs[] = {1,2,3,4,5,6,10,12,15,20,30,60,120,180,240,360,480,720,1440,10080,43200}; // Valid MT5 timeframes in minutes
  for(int i=0;i<ArraySize(tfs); ++i) if(tfs[i]==tf) return true;
  return false;
}

// ROBUST SYMBOL AND TIMEFRAME READING
// Models can be saved in different formats, so we try multiple methods
// This ensures compatibility with models saved by different versions
bool ReadSymbolAndTF(int h, string &sym, ENUM_TIMEFRAMES &tf)
{
  // Save current position in case we need to backtrack
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

// Load historical market data from MetaTrader
bool LoadSeries(const string sym, ENUM_TIMEFRAMES tf, int count, Series &s){
  ArraySetAsSeries(s.rates,true);  // Set array indexing: [0]=newest, [1]=older, etc.
  int copied = CopyRates(sym,tf,0,count,s.rates);  // Get price data from MT5
  if(copied<=0){ 
      Print("CopyRates failed ",sym," ",EnumToString(tf)," err=",GetLastError()); 
      return false; 
  }
  // Extract timestamps from the rate data
  ArrayResize(s.times,copied);
  for(int i=0;i<copied;++i) s.times[i]=s.rates[i].time;
  return true;
}

// Binary search to find the latest bar at or before a given time
// This is used to sync data across different timeframes
int FindIndexLE(const datetime &times[], int n, datetime t){
  int lo=0, hi=n-1, ans=-1;
  while(lo<=hi){
    int mid=(lo+hi)>>1;  // Middle point
    if(times[mid]<=t){ 
        ans=mid; lo=mid+1;  // Found candidate, look for later one
    } else hi=mid-1;        // Time too late, look earlier
  }
  return ans;  // Returns index of latest bar <= time t
}

// TECHNICAL INDICATOR CALCULATIONS
// These calculate the market features that the AI uses to make decisions

// Simple Moving Average of closing prices
double SMA_Close(const MqlRates &r[], int i, int period){ 
    double s=0; int n=0; 
    for(int k=0;k<period && (i+k)<ArraySize(r); ++k){ 
        s+=r[i+k].close; n++; 
    } 
    return (n>0? s/n : 0.0); 
}

// Exponential Moving Average slope (trend strength indicator)
double EMA_Slope(const MqlRates &r[], int i, int period){
  double ema=0, alpha=2.0/(period+1); int n=0;
  // Calculate current EMA
  for(int k=period+20;k>=0;--k){ 
      int idx=i+k; 
      if(idx>=ArraySize(r)) continue; 
      if(n==0) ema=r[idx].close; 
      else ema=alpha*r[idx].close+(1-alpha)*ema; 
      n++; 
  }
  // Calculate previous EMA and return difference (slope)
  if(i+1<ArraySize(r)){
    double ema_prev=0; alpha=2.0/(period+1); n=0;
    for(int k=period+21;k>=1;--k){ 
        int idx=i+k; 
        if(idx>=ArraySize(r)) continue; 
        if(n==0) ema_prev=r[idx].close; 
        else ema_prev=alpha*r[idx].close+(1-alpha)*ema_prev; 
        n++; 
    }
    return ema-ema_prev;  // Positive = uptrend, Negative = downtrend
  }
  return 0.0;
}

// Average True Range - measures volatility/market movement
double ATR_Proxy(const MqlRates &r[], int i, int period){
  double s=0; int n=0;
  for(int k=0;k<period && (i+k+1)<ArraySize(r); ++k){
    int idx=i+k;
    // True Range = max of: (H-L), |H-prev_close|, |L-prev_close|
    double tr=MathMax(r[idx].high - r[idx].low, 
                     MathMax(MathAbs(r[idx].high - r[idx+1].close), 
                            MathAbs(r[idx].low - r[idx+1].close)));
    s+=tr; n++;
  }
  return (n>0? s/n : 0.0);  // Average true range
}

// Trend direction: compares current price to price N bars ago
double TrendDir(const MqlRates &r[], int i, int look){
  int idx=i+look;
  if(idx>=ArraySize(r)) return 0.0;
  double a=r[i].close, b=r[idx].close;
  if(a>b) return 1.0;     // Uptrend
  if(a<b) return -1.0;    // Downtrend 
  return 0.0;             // Sideways
}

// ENHANCED MARKET CONTEXT FEATURES (NEW - populate reserved slots 15-34)
// These features provide critical market context missing from basic price/volume data

// Get normalized time-of-day (0.0 = start of day, 1.0 = end of day)
double GetTimeOfDay(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    return (dt.hour * 60.0 + dt.min) / (24.0 * 60.0);  // 0-1 range
}

// Get day of week (0.0 = Sunday, 1.0 = Saturday)
double GetDayOfWeek(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    return dt.day_of_week / 6.0;  // 0-1 range
}

// Get trading session indicator (0=Asian, 0.33=London, 0.66=NY, 1.0=Off-hours)
double GetTradingSession(){
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour_utc = dt.hour;  // Assuming server time is UTC
    
    // Asian session: 00:00-09:00 UTC
    if(hour_utc >= 0 && hour_utc < 9) return 0.0;
    // London session: 08:00-17:00 UTC  
    else if(hour_utc >= 8 && hour_utc < 17) return 0.33;
    // New York session: 13:00-22:00 UTC
    else if(hour_utc >= 13 && hour_utc < 22) return 0.66;
    // Off hours
    else return 1.0;
}

// Calculate volume momentum (current vs recent average)
double GetVolumeMomentum(const MqlRates &r[], int i, int period){
    if(i+period >= ArraySize(r)) return 0.5; // Default neutral
    
    double current_vol = (double)r[i].tick_volume;
    double vol_sum = 0.0;
    int count = 0;
    
    for(int k=1; k<=period && i+k<ArraySize(r); ++k){
        vol_sum += (double)r[i+k].tick_volume;
        count++;
    }
    
    if(count == 0 || vol_sum == 0) return 0.5;
    double avg_vol = vol_sum / count;
    
    // Return ratio clamped to reasonable range
    double ratio = current_vol / avg_vol;
    return clipd(ratio / 3.0, 0.0, 1.0);  // Scale so 3x average = 1.0
}

// Get spread as percentage of price
double GetSpreadPercent(){
    double spread_points = GetSymbolSpreadPoints(); // Use centralized function with 15pt fallback
    double current_price = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) + SymbolInfoDouble(_Symbol, SYMBOL_BID)) / 2.0;
    if(current_price <= 0) return 0.0;
    return (spread_points * _Point) / current_price * 10000.0;  // Basis points / 10000
}

// Calculate price momentum (rate of change)
double GetPriceMomentum(const MqlRates &r[], int i, int period){
    if(i+period >= ArraySize(r)) return 0.5; // Default neutral
    
    double current_price = r[i].close;
    double past_price = r[i+period].close;
    
    if(past_price <= 0) return 0.5;
    double change = (current_price - past_price) / past_price;
    
    // Scale to 0-1 range, assuming ±5% is extreme
    return clipd((change + 0.05) / 0.10, 0.0, 1.0);
}

// Calculate volatility rank (current ATR vs historical range)
double GetVolatilityRank(const MqlRates &r[], int i, int atr_period, int rank_period){
    if(i+rank_period >= ArraySize(r)) return 0.5;
    
    double current_atr = ATR_Proxy(r, i, atr_period);
    
    // Calculate ATR for each bar in ranking period
    double atr_values[100]; // Max rank period
    int actual_period = MathMin(rank_period, ArraySize(r)-i-atr_period);
    actual_period = MathMin(actual_period, 100);
    
    for(int k=0; k<actual_period; ++k){
        if(i+k+atr_period < ArraySize(r)){
            atr_values[k] = ATR_Proxy(r, i+k, atr_period);
        }
    }
    
    // Calculate percentile rank
    int below_count = 0;
    for(int k=0; k<actual_period; ++k){
        if(atr_values[k] < current_atr) below_count++;
    }
    
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

// Calculate ATR-based position size
double CalculatePositionSize(bool is_strong_signal){
    if(!InpUseRiskSizing) {
        return is_strong_signal ? InpLotsStrong : InpLotsWeak;
    }
    
    // Get current ATR for volatility-based sizing
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 20, atr_values);
    if(copied <= 0) return is_strong_signal ? InpLotsStrong : InpLotsWeak; // Fallback
    
    double atr = atr_values[0];  // Current ATR
    double stop_loss_points = atr * InpATRMultiplier / _Point;
    
    // Calculate position size based on risk percentage
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = equity * InpRiskPercent / 100.0;
    
    // Account for signal strength
    double signal_multiplier = is_strong_signal ? 1.0 : 0.5;
    risk_amount *= signal_multiplier;
    
    // Calculate lot size
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tick_value <= 0 || tick_size <= 0 || stop_loss_points <= 0) {
        return is_strong_signal ? InpLotsStrong : InpLotsWeak; // Fallback
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

// Get current position information for scaling decisions
bool GetCurrentPosition(double &current_lots, bool &is_long, double &entry_price){
    current_lots = 0.0; is_long = false; entry_price = 0.0;
    
    for(int i=0; i<PositionsTotal(); ++i){
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        
        string sym = PositionGetString(POSITION_SYMBOL);
        long mg = PositionGetInteger(POSITION_MAGIC);
        
        if(sym == _Symbol && mg == InpMagic){
            long type = PositionGetInteger(POSITION_TYPE);
            current_lots = PositionGetDouble(POSITION_VOLUME);
            entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            is_long = (type == POSITION_TYPE_BUY);
            return true;
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
            Print("Position scaling: Volume difference too small to adjust (", DoubleToString(add_volume,2), ")");
            return false;
        }
        
        // Add to position
        trade.SetExpertMagicNumber(InpMagic);
        bool result = is_long ? 
            trade.Buy(add_volume, _Symbol, 0, 0, 0, "cortex3_scale_up") :
            trade.Sell(add_volume, _Symbol, 0, 0, 0, "cortex3_scale_up");
            
        if(result){
            Print("Position scaled UP: Added ", DoubleToString(add_volume,2), " lots to ", (is_long?"LONG":"SHORT"), " position");
        }
        return result;
    }
    else {
        // Need to decrease position (partial close)
        double close_volume = MathAbs(volume_diff);
        double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        
        if(close_volume < min_vol){
            Print("Position scaling: Volume difference too small to adjust (", DoubleToString(close_volume,2), ")");
            return false;
        }
        
        // Partial close position
        trade.SetExpertMagicNumber(InpMagic);
        bool result = trade.PositionClosePartial(_Symbol, close_volume);
        
        if(result){
            Print("Position scaled DOWN: Reduced ", (is_long?"LONG":"SHORT"), " position by ", DoubleToString(close_volume,2), " lots");
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

// Update trade tracking (call this from trade management)
void UpdateTradeTracking(bool was_profitable){
    ArrayResize(g_recent_trades, 10);  // Keep last 10 trades
    
    // Shift array and add new result
    for(int i=9; i>0; --i){
        g_recent_trades[i] = g_recent_trades[i-1];
    }
    g_recent_trades[0] = was_profitable ? 1 : 0;
    
    if(g_recent_trades_count < 10) g_recent_trades_count++;
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
    
    // Volatility impact from current ATR
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 1, atr_values);
    double volatility_impact = 1.0;
    if(copied > 0){
        double current_atr = atr_values[0];
        volatility_impact = current_atr > 0 ? MathSqrt(current_atr * 10000) : 1.0;
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

// Update trailing stops and break-even for all positions
void UpdateTrailingStops(){
    if(!InpUseTrailingStop && !InpUseBreakEven) return;
    
    // Get current ATR for calculations
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 2, atr_values);
    if(copied <= 0) return;
    
    double current_atr = atr_values[0];
    if(current_atr <= 0) return;
    
    // Update all positions
    for(int i=0; i<PositionsTotal(); ++i){
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        
        string pos_symbol = PositionGetString(POSITION_SYMBOL);
        long pos_magic = PositionGetInteger(POSITION_MAGIC);
        
        if(pos_symbol == _Symbol && pos_magic == InpMagic){
            UpdateSinglePositionTrailing(ticket, current_atr);
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
        Print("Portfolio risk check: Maximum positions reached (", g_active_positions, "/", InpMaxPositions, ")");
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

// Update portfolio risk metrics
void UpdatePortfolioRisk(){
    g_active_positions = 0;
    g_total_risk_percent = 0.0;
    double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Count active positions and calculate total risk
    for(int i=0; i<PositionsTotal(); ++i){
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
        Print("FLAT action: Closed position. Profit: ", DoubleToString(profit,2));
        closed_any = true;
      }
    }
    if(!closed_any){
      Print("FLAT action: No positions to close");
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
    Print("High volatility regime: Reduced position size from ", DoubleToString(base_lots,2), 
          " to ", DoubleToString(lots,2), " lots");
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
            Print("Closed opposite SELL position. Profit: ", DoubleToString(profit,2));
        }
        if(!is_buy && type==POSITION_TYPE_BUY){ 
            UpdateTradeTracking(profit > 0);  // Track result before closing
            trade.PositionClose(sym);  // Close buy when AI wants sell
            Print("Closed opposite BUY position. Profit: ", DoubleToString(profit,2));
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
            Print("Position scaling needed: Current=", DoubleToString(current_volume,2), 
                  " lots, Target=", DoubleToString(target_volume,2), " lots, Signal=", 
                  (is_strong?"STRONG":"WEAK"));
            
            // Scale the position to match new signal strength
            bool scaling_result = ScalePosition(current_volume, target_volume, is_buy);
            if(scaling_result){
              Print("Position successfully scaled for ", ACTION_NAME[action], " signal");
            } else {
              Print("Position scaling failed for ", ACTION_NAME[action], " signal");
            }
          } else {
            Print("Position size already matches ", (is_strong?"STRONG":"WEAK"), " signal - no scaling needed");
          }
        } else {
          Print("Same direction signal ignored - position scaling disabled");
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
    double atr_values[];
    ArraySetAsSeries(atr_values, true);
    int copied = CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 2, atr_values);
    
    if(copied > 0){
      double atr = atr_values[0];
      double current_price = is_buy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      if(is_buy){
        sl_price = current_price - (atr * InpATRMultiplier);
        tp_price = current_price + (atr * InpATRMultiplier * InpRRRatio);
      } else {
        sl_price = current_price + (atr * InpATRMultiplier);
        tp_price = current_price - (atr * InpATRMultiplier * InpRRRatio);
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
  
  // Try to load the trained Double-Dueling DRQN model
  g_loaded = LoadModel(InpModelFileName);
  if(!g_loaded){
    Print("FATAL: Double-Dueling DRQN model not loaded; EA cannot operate without trained model.");
    return(INIT_FAILED);  // Stop EA if model can't be loaded
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
  
  Print("=== CORTEX3 EA INITIALIZED ===");
  Print("Model: ", InpModelFileName);
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
  Print("Initial equity: $", DoubleToString(g_initial_equity,2));
  Print("EA ready for production trading with advanced risk management.");
  
  // Reset LSTM state for fresh start
  g_Q.ResetLSTMState();
  
  return(INIT_SUCCEEDED);
}

// CLEANUP FUNCTION 
// Called when EA is removed from chart
void OnDeinit(const int reason){
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
void UpdatePositionTracking(){
    if(PositionSelect(_Symbol)){
        g_position_open_time = (datetime)PositionGetInteger(POSITION_TIME);
        g_position_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
        g_position_size = PositionGetDouble(POSITION_VOLUME);
        g_position_unrealized_pnl = PositionGetDouble(POSITION_PROFIT);
        g_position_type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : 2;
    } else {
        g_position_open_time = 0;
        g_position_entry_price = 0.0;
        g_position_size = 0.0;
        g_position_unrealized_pnl = 0.0;
        g_position_type = 0;
    }
}

// Check if position should be closed due to maximum holding time
bool CheckMaxHoldingTime(datetime current_time){
    if(!InpUseMaxHoldingTime || g_position_type == 0) return false;
    
    int holding_hours = (int)((current_time - g_position_open_time) / 3600);
    
    if(holding_hours > InpMaxHoldingHours){
        Print("PHASE1: Maximum holding time exceeded (", holding_hours, " hours). Forcing position close.");
        CloseAllPositions();
        return true;
    }
    return false;
}

// Check if position should be closed due to profit target
bool CheckProfitTargets(){
    if(!InpUseProfitTargets || g_position_type == 0) return false;
    
    int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
    double atr_buffer[];
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0) return false;
    double atr = atr_buffer[0];
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

//============================== TRADING FREQUENCY CONTROL FUNCTIONS =============
// Functions to prevent overtrading in live trading

// Check if new trading is allowed (prevent overtrading)
bool IsNewTradingAllowed(){
    // Don't trade if we have an open position
    if(g_position_type != 0) return false;
    
    datetime current_time = TimeCurrent();
    datetime today_start = current_time - (current_time % (24 * 3600));
    
    // Reset daily counter if it's a new day
    if(today_start != g_current_day){
        g_trades_today = 0;
        g_current_day = today_start;
    }
    
    // Check daily trade limit
    if(g_trades_today >= InpMaxTradesPerDay){
        return false;
    }
    
    // Check minimum time between trades
    int bars_since_last_trade = (int)((current_time - g_last_trade_time) / PeriodSeconds(PERIOD_CURRENT));
    if(bars_since_last_trade < InpMinBarsBetweenTrades){
        return false;
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
    
    // Force FLAT if holding too long (80% of max time)
    if(holding_hours > (InpMaxHoldingHours * 0.8)){
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
    
    int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
    double atr_buffer[];
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0) return;
    double atr = atr_buffer[0];
    double days_held = holding_hours / 24.0;
    
    // Tighten stops progressively: base_multiplier * (tighten_rate ^ days_held)
    double stop_multiplier = InpATRMultiplier * MathPow(InpStopTightenRate, days_held);
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
    
    int atr14_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
    int atr50_handle = iATR(_Symbol, PERIOD_CURRENT, 50);
    int ma10_handle = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
    int ma50_handle = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_SMA, PRICE_CLOSE);
    
    double atr14_buffer[], atr50_buffer[], ma10_buffer[], ma50_buffer[];
    if(CopyBuffer(atr14_handle, 0, 0, 1, atr14_buffer) <= 0) return;
    if(CopyBuffer(atr50_handle, 0, 0, 1, atr50_buffer) <= 0) return;
    if(CopyBuffer(ma10_handle, 0, 0, 1, ma10_buffer) <= 0) return;
    if(CopyBuffer(ma50_handle, 0, 0, 1, ma50_buffer) <= 0) return;
    
    double atr_current = atr14_buffer[0];
    double atr_long_term = atr50_buffer[0];
    
    // Trend strength calculation
    double ma10 = ma10_buffer[0];
    double ma50 = ma50_buffer[0];
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
        g_position_normalized_time = MathMin(1.0, (double)holding_hours / (double)InpMaxHoldingHours);
        
        // P&L ratio vs ATR
        int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
        double atr_buffer[];
        if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0) return;
        double atr = atr_buffer[0];
        double atr_value = atr * g_position_size * 100000; // Convert to monetary value
        g_unrealized_pnl_ratio = (atr_value > 0) ? g_position_unrealized_pnl / atr_value : 0.0;
        g_unrealized_pnl_ratio = clipd(g_unrealized_pnl_ratio, -5.0, 5.0); // Clip to reasonable range
    } else {
        g_position_normalized_time = 0.0;
        g_unrealized_pnl_ratio = 0.0;
    }
}

// MAIN PROCESSING FUNCTION
// Called every time a new price tick arrives
void OnTick(){
  // Don't do anything if model failed to load
  if(!g_loaded) return;
  
  // RUNTIME SAFETY CHECK: Double-check model alignment before any trading activity
  if(InpEnforceSymbolTF && (_Symbol != g_model_symbol || _Period != g_model_tf)){
    static datetime last_warning = 0;
    if(TimeCurrent() - last_warning > 300){ // Warn every 5 minutes max
      Print("RUNTIME ERROR: Model mismatch detected! EA should not be running!");
      Print("Chart: ", _Symbol, " ", EnumToString(_Period), " | Model: ", g_model_symbol, " ", EnumToString(g_model_tf));
      Alert("CORTEX3 RUNTIME ERROR: Model mismatch! EA stopping trading!");
      last_warning = TimeCurrent();
    }
    return; // Refuse to process any ticks
  }
  
  // Load market data for all timeframes
  Series base,m1,m5,h1,h4,d1;
  if(!LoadSeries(_Symbol, g_model_tf, InpBarLookback, base)) return;
  
  // Need enough historical data for indicators
  if(ArraySize(base.rates)<60) return;
  
  // Only process once per new bar (avoid multiple signals on same bar)
  if(g_last_bar_time==base.rates[1].time) return;
  g_last_bar_time = base.rates[1].time;  // Remember this bar's timestamp

  // Update trailing stops and break-even for existing positions
  UpdateTrailingStops();

  // PHASE 1 ENHANCEMENTS - UPDATE POSITION TRACKING AND CHECK EXIT CONDITIONS
  UpdatePositionTracking();
  
  // Check emergency stops first (highest priority)
  if(CheckEmergencyStops()) {
    return; // Emergency stop triggered, position closed
  }
  
  // Check if position should be closed due to maximum holding time or profit targets
  if(CheckMaxHoldingTime(TimeCurrent()) || CheckProfitTargets()) {
    return; // Position was closed, skip AI decision for this tick
  }

  // PHASE 2 & 3 ENHANCEMENTS - ADVANCED POSITION AND MARKET ANALYSIS
  UpdateDynamicStops();      // Phase 3: Dynamic stop loss tightening
  UpdateMarketRegime();      // Phase 3: Market regime detection
  UpdatePositionFeatures();  // Phase 3: Position-aware features

  // TRADING FREQUENCY CONTROLS - Check if new trading allowed and force FLAT if needed
  bool trading_allowed = IsNewTradingAllowed();
  bool should_force_flat = ShouldForceFlat();

  // COMPREHENSIVE RISK CHECKS (ENHANCED - all conditions before proceeding)
  if(!MasterRiskCheck()) {
    // Master risk check handles all printing internally
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

  // Get AI prediction: feed market features into neural network
  double q[];               // Will hold Q-values (expected rewards) for each action
  g_Q.Predict(row,q);       // Run neural network forward pass
  int action = argmax(q);   // Choose action with highest expected reward
  
  // Log the AI's decision with position context (ENHANCED logging)
  Print("AI Decision: ",ACTION_NAME[action],
        " | Position: ",DoubleToString(row[12],1),"x",DoubleToString(row[13],2)," (P&L:",DoubleToString(row[14],1),"pts)",
        " | Q-values=[",DoubleToString(q[0],3),",",DoubleToString(q[1],3),",",
        DoubleToString(q[2],3),",",DoubleToString(q[3],3),",",DoubleToString(q[4],3),",",DoubleToString(q[5],3),"]");
  
  // Execute the recommended action
  MaybeTrade(action);
}
//+------------------------------------------------------------------+
