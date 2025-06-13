import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import json
import logging
import requests
import sqlite3
from datetime import datetime, timedelta
import ccxt
import talib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import warnings
import importlib.util
import sys
import threading
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from scipy import fft

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trading.log'),
        logging.StreamHandler()
    ]
)

# Constants - Adjusted for more sensitivity
SEQ_LENGTH = 40
HIDDEN_SIZE = 256
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICTION_HORIZON = 3  # Reduced from 5 for shorter-term trades

# Default webhook URL (will be overridden by config)
DEFAULT_WEBHOOK_URL = 'http://localhost:5000/webhook/default'

# Default webhook mappings in case config file doesn't exist
DEFAULT_WEBHOOK_URLS = {
    'BTC/USDT': 'http://localhost:5000/webhook/btc',
    'SOL/USDT': 'http://localhost:5000/webhook/sol',
    'default': 'http://localhost:5000/webhook/default'
}

# Default ticker map
DEFAULT_TICKER_MAP = {
    'BTC/USDT': 'BTCUSDT',
    'SOL/USDT': 'SOL',
}

# Configuration for Telegram notifications (optional)
TELEGRAM_CONFIG = {
    'enabled': False,
    'token': '',  # Your bot token
    'chat_id': '',  # Your chat ID
}

# ADJUSTED: More aggressive model configuration
DEFAULT_MODEL_CONFIG = {
    'use_ensemble': True,
    'num_ensemble_models': 3,
    'use_mc_dropout': True,
    'confidence_threshold': 0.35  # Lowered from 0.7 for more signals
}

# ADJUSTED: More aggressive risk management
DEFAULT_RISK_CONFIG = {
    'use_trailing_stop': True,
    'use_partial_profits': True,
    'max_position_size': 1.0,  # Increased from 0.8
    'consecutive_loss_limit': 5   # Increased from 3
}

# Default backtesting configuration
DEFAULT_BACKTEST_CONFIG = {
    'commission_rate': 0.001,  # 0.1% commission
    'slippage_rate': 0.0005,   # 0.05% slippage
    'initial_capital': 10000   # $10,000 starting capital
}

# Load configuration
CONFIG_FILE = 'crypto_config.py'

try:
    if os.path.exists(CONFIG_FILE):
        spec = importlib.util.spec_from_file_location("crypto_config", CONFIG_FILE)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get webhook URLs and ticker mappings
        WEBHOOK_URLS = getattr(config_module, 'WEBHOOK_URLS', DEFAULT_WEBHOOK_URLS)
        TICKER_MAP = getattr(config_module, 'TICKER_MAP', DEFAULT_TICKER_MAP)
        SUCCESSFUL_FORMATS = getattr(config_module, 'SUCCESSFUL_FORMATS', {})
        TELEGRAM_CONFIG = getattr(config_module, 'TELEGRAM_CONFIG', TELEGRAM_CONFIG)
        
        # Get enhanced configuration options
        MODEL_CONFIG = getattr(config_module, 'MODEL_CONFIG', DEFAULT_MODEL_CONFIG)
        RISK_CONFIG = getattr(config_module, 'RISK_CONFIG', DEFAULT_RISK_CONFIG)
        BACKTEST_CONFIG = getattr(config_module, 'BACKTEST_CONFIG', DEFAULT_BACKTEST_CONFIG)
        
        logging.info(f"Loaded configuration from {CONFIG_FILE}")
    else:
        WEBHOOK_URLS = DEFAULT_WEBHOOK_URLS
        TICKER_MAP = DEFAULT_TICKER_MAP
        SUCCESSFUL_FORMATS = {}
        MODEL_CONFIG = DEFAULT_MODEL_CONFIG
        RISK_CONFIG = DEFAULT_RISK_CONFIG
        BACKTEST_CONFIG = DEFAULT_BACKTEST_CONFIG
        
        # Create a default config file
        with open(CONFIG_FILE, 'w') as f:
            f.write('"""' + '\n')
            f.write('Configuration file for CryptoPrime Signal Generator.' + '\n')
            f.write('Edit this file to configure webhooks and ticker formats for different cryptocurrencies.' + '\n')
            f.write('"""' + '\n\n')
            f.write('# Webhook URLs for different cryptocurrencies' + '\n')
            f.write('WEBHOOK_URLS = ' + json.dumps(WEBHOOK_URLS, indent=4) + '\n\n')
            f.write('# Map from trading pairs (exchange format) to API tickers' + '\n')
            f.write('TICKER_MAP = ' + json.dumps(TICKER_MAP, indent=4) + '\n\n')
            f.write('# Dictionary to store successful formats during runtime' + '\n')
            f.write('SUCCESSFUL_FORMATS = {}' + '\n\n')
            f.write('# Telegram notification configuration' + '\n')
            f.write('TELEGRAM_CONFIG = ' + json.dumps(TELEGRAM_CONFIG, indent=4) + '\n\n')
            f.write('# Model configuration' + '\n')
            f.write('MODEL_CONFIG = ' + json.dumps(MODEL_CONFIG, indent=4) + '\n\n')
            f.write('# Risk management configuration' + '\n')
            f.write('RISK_CONFIG = ' + json.dumps(RISK_CONFIG, indent=4) + '\n\n')
            f.write('# Backtesting configuration' + '\n')
            f.write('BACKTEST_CONFIG = ' + json.dumps(BACKTEST_CONFIG, indent=4) + '\n')
        
        logging.info(f"Created default configuration file: {CONFIG_FILE}")
except Exception as e:
    logging.error(f"Error loading configuration: {str(e)}")
    WEBHOOK_URLS = DEFAULT_WEBHOOK_URLS
    TICKER_MAP = DEFAULT_TICKER_MAP
    SUCCESSFUL_FORMATS = {}
    MODEL_CONFIG = DEFAULT_MODEL_CONFIG
    RISK_CONFIG = DEFAULT_RISK_CONFIG
    BACKTEST_CONFIG = DEFAULT_BACKTEST_CONFIG

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

#################################
# 1. MODEL ARCHITECTURE
#################################

class FeatureAttention(nn.Module):
    """
    Dynamic feature importance weighting module
    """
    def __init__(self, num_features):
        super().__init__()
        self.importance_weights = nn.Parameter(torch.ones(num_features))
        self.selector = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Calculate feature importance scores
        feature_scores = self.selector(x.mean(dim=1))
        # Apply importance weighting
        return x * feature_scores.unsqueeze(1) * self.importance_weights

class MultiScaleProcessor(nn.Module):
    """
    Multi-scale temporal feature extractor
    """
    def __init__(self, input_size):
        super().__init__()
        self.short_term = nn.Conv1d(input_size, 96, kernel_size=3, padding=1)
        self.medium_term = nn.Conv1d(input_size, 96, kernel_size=7, padding=3)
        self.long_term = nn.Conv1d(input_size, 96, kernel_size=15, padding=7)
        self.very_long_term = nn.Conv1d(input_size, 96, kernel_size=30, padding=15)
        
        # Add batch normalization
        self.bn_short = nn.BatchNorm1d(96)
        self.bn_medium = nn.BatchNorm1d(96)
        self.bn_long = nn.BatchNorm1d(96)
        self.bn_very_long = nn.BatchNorm1d(96)
        
    def forward(self, x):
        # Convert to shape [batch, features, sequence]
        x = x.permute(0, 2, 1)
        
        # Apply different scale convolutions with batch norm
        short = F.leaky_relu(self.bn_short(self.short_term(x)))
        medium = F.leaky_relu(self.bn_medium(self.medium_term(x)))
        long = F.leaky_relu(self.bn_long(self.long_term(x)))
        very_long = F.leaky_relu(self.bn_very_long(self.very_long_term(x)))
        
        # Apply pooling for different time horizons
        short = F.max_pool1d(short, 2)
        medium = F.avg_pool1d(medium, 2)
        long = F.max_pool1d(long, 2)
        very_long = F.adaptive_avg_pool1d(very_long, short.size(2))
        
        # Combine multi-scale features
        multi_scale = torch.cat([short, medium, long, very_long], dim=1)
        
        # Convert back to [batch, sequence, features]
        return multi_scale.permute(0, 2, 1)

class TemporalFusionModel(nn.Module):
    """
    Improved model architecture inspired by Temporal Fusion Transformer
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature attention module
        self.feature_attention = FeatureAttention(input_size)
        
        # Multi-scale temporal processing
        self.multi_scale = MultiScaleProcessor(input_size)
        
        # Combined feature dimension after multi-scale processing
        multi_scale_size = 96 * 4  # short, medium, long, very long term features
        
        # Feature processing layer
        self.feature_layer = nn.Sequential(
            nn.Linear(multi_scale_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Add gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # LSTM for sequence encoding
        self.lstm_encoder = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=2,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=True
        )
        
        # Multi-head attention layers
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size*2,
            num_heads=8,
            dropout=DROPOUT
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size*4, hidden_size*2)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size*2)
        self.norm2 = nn.LayerNorm(hidden_size*2)
        
        # Output heads
        # 1. Direction prediction (buy, hold, sell)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, 3)
        )
        
        # 2. Stop loss prediction
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 3. Take profit prediction
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 4. Trade duration prediction
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 5. Confidence score (new)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mc_dropout=False):
        # Enable MC dropout during inference if requested
        if mc_dropout:
            self.train()  # Enable dropout layers
            
            # Only set dropout layers to train mode
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
                elif not isinstance(module, nn.Dropout) and hasattr(module, 'train'):
                    module.eval()
        
        # Apply feature attention
        x_attended = self.feature_attention(x)
        
        # Apply multi-scale processing
        x_multi = self.multi_scale(x_attended)
        
        # Process input features
        features = self.feature_layer(x_multi)
        
        # Apply gating mechanism
        gate_values = self.feature_gate(features)
        gated_features = features * gate_values
        
        # Extract static features from the first timestep
        static_features = self.static_encoder(gated_features[:, 0, :])
        
        # Expand static features to the sequence length
        static_features = static_features.unsqueeze(1).expand(-1, gated_features.size(1), -1)
        
        # Combine with gated features
        combined_features = gated_features + static_features
        
        # LSTM processing
        lstm_out, _ = self.lstm_encoder(combined_features)
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # First residual connection
        attn_out = self.norm1(lstm_out + attn_out.transpose(0, 1))
        
        # FFN with residual connection
        ffn_out = self.ffn(attn_out)
        output = self.norm2(attn_out + ffn_out)
        
        # Get final representation (last timestep)
        final_repr = output[:, -1, :]
        
        # Output heads
        direction_logits = self.direction_head(final_repr)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Stop loss prediction (0-20% range for more flexibility)
        stop_loss = self.sl_head(final_repr) * 0.2
        
        # Take profit prediction (0-30% range for more flexibility)
        take_profit = self.tp_head(final_repr) * 0.3
        
        # Trade duration prediction (1-73 hours)
        duration = self.duration_head(final_repr) * 72 + 1
        
        # Prediction confidence (0-1)
        confidence = self.confidence_head(final_repr)
        
        return direction_probs, stop_loss, take_profit, duration, confidence, final_repr

# Ensemble model that combines predictions from multiple models
class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.num_models = len(models)
        
        # If weights not provided, use equal weighting
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def __call__(self, x, mc_dropout=False):
        return self.predict(x, mc_dropout)
    
    def predict(self, x, mc_dropout=False):
        """
        Make a prediction using weighted ensemble
        """
        # Initialize aggregation variables
        all_direction_probs = []
        all_stop_losses = []
        all_take_profits = []
        all_durations = []
        all_confidences = []
        all_features = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                direction_probs, stop_loss, take_profit, duration, confidence, features = model(x, mc_dropout=mc_dropout)
                
                # Weight the predictions
                weight = self.weights[i]
                all_direction_probs.append(direction_probs.cpu().numpy() * weight)
                all_stop_losses.append(float(stop_loss.cpu().numpy()) * weight)
                all_take_profits.append(float(take_profit.cpu().numpy()) * weight)
                all_durations.append(float(duration.cpu().numpy()) * weight)
                all_confidences.append(float(confidence.cpu().numpy()) * weight)
                all_features.append(features)
        
        # Sum the weighted predictions
        ensemble_direction = np.sum(all_direction_probs, axis=0)
        ensemble_sl = np.sum(all_stop_losses)
        ensemble_tp = np.sum(all_take_profits)
        ensemble_duration = np.sum(all_durations)
        ensemble_confidence = np.sum(all_confidences)
        
        # Calculate uncertainty between models
        if self.num_models > 1:
            direction_std = np.std([probs[0] for probs in all_direction_probs], axis=0)
            uncertainty = np.mean(direction_std)
            
            # Reduce confidence when models disagree
            ensemble_confidence = max(0.1, ensemble_confidence * (1 - uncertainty))
        
        # Convert back to tensors
        ensemble_direction_tensor = torch.tensor(ensemble_direction, device=x.device)
        ensemble_sl_tensor = torch.tensor([ensemble_sl], device=x.device)
        ensemble_tp_tensor = torch.tensor([ensemble_tp], device=x.device)
        ensemble_duration_tensor = torch.tensor([ensemble_duration], device=x.device)
        ensemble_confidence_tensor = torch.tensor([ensemble_confidence], device=x.device)
        
        # Use features from the first model
        return ensemble_direction_tensor, ensemble_sl_tensor, ensemble_tp_tensor, ensemble_duration_tensor, ensemble_confidence_tensor, all_features[0]

#################################
# 2. TRADE FEEDBACK SYSTEM
#################################

class TradeFeedbackSystem:
    """
    System to track trade outcomes and provide feedback for model improvement
    """
    def __init__(self, db_path='trades.db'):
        self.db_path = db_path
        self.conn = self._init_db()
        
    def _init_db(self):
        """Initialize database connection and create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            action TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_pct REAL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            duration_hours REAL,
            exit_reason TEXT,
            success INTEGER,
            predicted_sl REAL,
            predicted_tp REAL,
            predicted_duration REAL,
            position_size REAL
        )
        ''')
        
        # Create feedback data table for model improvements
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_feedback (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            timestamp TIMESTAMP,
            feature_data BLOB,
            actual_outcome INTEGER,
            confidence REAL
        )
        ''')
        
        # Create market regime table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regimes (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            timestamp TIMESTAMP,
            regime TEXT,
            volatility REAL,
            trend_strength REAL
        )
        ''')
        
        conn.commit()
        return conn
    
    def record_trade_entry(self, symbol, action, entry_price, predicted_sl, predicted_tp, predicted_duration, position_size=1.0):
        """Record a new trade entry"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO trades 
            (symbol, action, entry_price, entry_time, predicted_sl, predicted_tp, predicted_duration, position_size) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, action, entry_price, datetime.now(), predicted_sl, predicted_tp, predicted_duration, position_size)
        )
        
        self.conn.commit()
        return cursor.lastrowid
    
    def record_trade_exit(self, trade_id, exit_price, exit_reason="manual"):
        """Record a trade exit and calculate profit/loss"""
        cursor = self.conn.cursor()
        
        # Get trade information
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = cursor.fetchone()
        
        if not trade:
            logging.error(f"Trade ID {trade_id} not found")
            return False
        
        # Extract trade data
        symbol = trade[1]
        action = trade[2]
        entry_price = trade[3]
        position_size = trade[15] if len(trade) > 15 else 1.0
        entry_time = datetime.fromisoformat(trade[7])
        exit_time = datetime.now()
        
        # Calculate duration in hours
        duration_hours = (exit_time - entry_time).total_seconds() / 3600
        
        # Calculate profit/loss
        if action == 'buy':
            profit_loss = (exit_price - entry_price) * position_size
            profit_pct = (exit_price / entry_price - 1) * 100
        elif action == 'sell':
            profit_loss = (entry_price - exit_price) * position_size
            profit_pct = (entry_price / exit_price - 1) * 100
        else:
            profit_loss = 0
            profit_pct = 0
        
        # Determine success (1 = profitable, 0 = loss)
        success = 1 if profit_pct > 0 else 0
        
        # Update trade record
        cursor.execute(
            """
            UPDATE trades 
            SET exit_price = ?, exit_time = ?, profit_loss = ?, profit_pct = ?, 
                duration_hours = ?, exit_reason = ?, success = ?
            WHERE id = ?
            """,
            (exit_price, exit_time, profit_loss, profit_pct, duration_hours, 
             exit_reason, success, trade_id)
        )
        
        self.conn.commit()
        
        logging.info(f"Trade exit recorded - Symbol: {symbol}, Action: {action}, "
                     f"P/L: {profit_pct:.2f}%, Duration: {duration_hours:.1f}h, Reason: {exit_reason}")
        
        return True
    
    def save_market_regime(self, symbol, regime, volatility, trend_strength):
        """Save market regime information"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO market_regimes
            (symbol, timestamp, regime, volatility, trend_strength)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, datetime.now(), regime, volatility, trend_strength)
        )
        
        self.conn.commit()
    
    def save_model_feedback(self, symbol, feature_data, actual_outcome, confidence=1.0):
        """Save feature data with actual outcome for model retraining"""
        cursor = self.conn.cursor()
        
        # Serialize feature data
        feature_data_bytes = json.dumps(feature_data.tolist()).encode()
        
        cursor.execute(
            """
            INSERT INTO model_feedback
            (symbol, timestamp, feature_data, actual_outcome, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, datetime.now(), feature_data_bytes, actual_outcome, confidence)
        )
        
        self.conn.commit()
    
    def get_feedback_data(self, symbol, limit=1000):
        """Retrieve feedback data for model retraining"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            SELECT feature_data, actual_outcome, confidence 
            FROM model_feedback
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, limit)
        )
        
        feedback_data = []
        for row in cursor.fetchall():
            feature_data = np.array(json.loads(row[0]))
            actual_outcome = row[1]
            confidence = row[2]
            feedback_data.append((feature_data, actual_outcome, confidence))
        
        return feedback_data
    
    def get_performance_metrics(self, symbol, days=30):
        """Get performance metrics for a symbol over a given time period"""
        cursor = self.conn.cursor()
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as winning_trades,
                AVG(profit_pct) as avg_profit,
                AVG(CASE WHEN success = 1 THEN profit_pct ELSE NULL END) as avg_win,
                AVG(CASE WHEN success = 0 THEN profit_pct ELSE NULL END) as avg_loss,
                AVG(duration_hours) as avg_duration
            FROM trades
            WHERE symbol = ? AND entry_time >= ?
            """,
            (symbol, since_date)
        )
        
        return cursor.fetchone()
    
    def get_model_adjustment_factors(self, symbol):
        """Calculate model adjustment factors based on trade history"""
        cursor = self.conn.cursor()
        
        # Get win rate for different actions
        cursor.execute(
            """
            SELECT action, 
                   COUNT(*) as total,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE symbol = ? AND exit_price IS NOT NULL
            GROUP BY action
            """,
            (symbol,)
        )
        
        action_stats = {}
        for row in cursor.fetchall():
            action, total, wins = row
            win_rate = wins / total if total > 0 else 0.5
            # Calculate adjustment factor based on win rate
            # If win rate is high, we should encourage this action
            # If win rate is low, we should discourage it
            adjustment = (win_rate - 0.5) * 2  # Scale to [-1, 1]
            action_stats[action] = {
                'win_rate': win_rate,
                'adjustment': adjustment
            }
        
        # Get SL/TP effectiveness
        cursor.execute(
            """
            SELECT 
                AVG(CASE WHEN exit_reason = 'tp_hit' THEN 1 ELSE 0 END) as tp_rate,
                AVG(CASE WHEN exit_reason = 'sl_hit' THEN 1 ELSE 0 END) as sl_rate,
                AVG(predicted_tp) as avg_predicted_tp,
                AVG(predicted_sl) as avg_predicted_sl,
                AVG(CASE WHEN success = 1 THEN profit_pct ELSE NULL END) / 100 as avg_actual_win
            FROM trades
            WHERE symbol = ? AND exit_price IS NOT NULL
            """,
            (symbol,)
        )
        
        sl_tp_stats = cursor.fetchone()
        if sl_tp_stats and sl_tp_stats[0] is not None:
            tp_rate, sl_rate, avg_predicted_tp, avg_predicted_sl, avg_actual_win = sl_tp_stats
            
            # Calculate TP adjustment factor
            if avg_actual_win and avg_predicted_tp:
                tp_adjustment = avg_actual_win / avg_predicted_tp if avg_predicted_tp > 0 else 1.0
            else:
                tp_adjustment = 1.0
                
            # Calculate SL adjustment factor
            if sl_rate > 0.3:  # If stop losses are hit too often
                sl_adjustment = 1.2  # Suggest wider stop losses
            elif sl_rate < 0.1:  # If stop losses are rarely hit
                sl_adjustment = 0.9  # Suggest tighter stop losses
            else:
                sl_adjustment = 1.0
        else:
            tp_adjustment = 1.0
            sl_adjustment = 1.0
        
        return {
            'action_adjustments': action_stats,
            'tp_adjustment': tp_adjustment,
            'sl_adjustment': sl_adjustment
        }
    
    def get_recent_market_regime(self, symbol, days=7):
        """Get the recent market regime for a symbol"""
        cursor = self.conn.cursor()
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute(
            """
            SELECT regime, COUNT(*) as count
            FROM market_regimes
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY regime
            ORDER BY count DESC
            LIMIT 1
            """,
            (symbol, since_date)
        )
        
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "unknown"
    
    def get_consecutive_losses(self, symbol, limit=5):
        """Get the number of consecutive losses for a symbol"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            SELECT success
            FROM trades
            WHERE symbol = ? AND exit_price IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            (symbol, limit)
        )
        
        results = cursor.fetchall()
        
        # Count consecutive losses from most recent
        consecutive_losses = 0
        for row in results:
            success = row[0]
            if success == 0:
                consecutive_losses += 1
            else:
                break
        
        return consecutive_losses
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

#################################
# 3. SIGNAL DISPATCHER
#################################

class ImprovedSignalDispatcher:
    """
    Enhanced system for reliable signal delivery with improved performance
    """
    def __init__(self):
        self.webhook_urls = WEBHOOK_URLS
        self.ticker_map = TICKER_MAP
        self.telegram_config = TELEGRAM_CONFIG
        self.max_retries = 3
        self.backoff_factor = 2
        self.active_trades = {}  # To track active trades for each symbol
        self.rate_limits = {}    # To track API rate limits
        self.executor = ThreadPoolExecutor(max_workers=5)  # For parallel processing
        
        # Create asyncio event loop for webhook sending
        self.loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._start_loop, args=(self.loop,), daemon=True)
        thread.start()
    
    def _start_loop(self, loop):
        """Start the event loop in a background thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    def get_webhook_url(self, symbol):
        """Get the correct webhook URL for the given symbol"""
        if symbol in self.webhook_urls:
            return self.webhook_urls[symbol]
        return self.webhook_urls.get('default', DEFAULT_WEBHOOK_URL)
    
    def get_api_ticker(self, symbol):
        """Convert the trading symbol to the API-compatible ticker format"""
        # First check if we have a mapping
        if symbol in self.ticker_map:
            return self.ticker_map[symbol]
        
        # Check if we've successfully used a format before
        if symbol in SUCCESSFUL_FORMATS:
            return SUCCESSFUL_FORMATS[symbol]
        
        # Default to just the base currency
        return symbol.split('/')[0]
    
    async def send_webhook_async(self, webhook_url, payload, attempt=1):
        """Send webhook asynchronously with retry logic"""
        try:
            # Check rate limits
            if webhook_url in self.rate_limits:
                last_request, count = self.rate_limits[webhook_url]
                # If we've made too many requests in a short time, add a delay
                if count > 5 and time.time() - last_request < 60:
                    delay = 60 - (time.time() - last_request)
                    logging.info(f"Rate limiting for {webhook_url}, waiting {delay:.1f} seconds")
                    await asyncio.sleep(delay)
            
            # Update rate limit tracking
            self.rate_limits[webhook_url] = (time.time(), self.rate_limits.get(webhook_url, (0, 0))[1] + 1)
            
            # Make the request
            logging.info(f"Sending webhook to {webhook_url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=10) as response:
                    # Get the response text
                    response_text = await response.text()
                    
                    if response.status == 200:
                        logging.info(f"Webhook sent successfully: {response_text}")
                        return True, response_text
                    else:
                        logging.error(f"Failed to send webhook. Status code: {response.status}, Response: {response_text}")
                        
                        # If we get a rate limit error, back off more aggressively
                        if response.status == 429:
                            await asyncio.sleep(30)  # Wait 30 seconds for rate limit reset
                        
                        # If this is a "no matching pair" error, return specific code
                        if response.status == 404 and "No matching bot pair found" in response_text:
                            return False, "no_matching_pair"
                        
                        # For other errors, retry if attempts remain
                        if attempt < self.max_retries:
                            wait_time = self.backoff_factor ** attempt
                            logging.info(f"Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                            return await self.send_webhook_async(webhook_url, payload, attempt + 1)
                        
                        return False, response_text
        
        except asyncio.TimeoutError:
            logging.error(f"Timeout sending webhook (attempt {attempt})")
            
            # Retry if attempts remain
            if attempt < self.max_retries:
                wait_time = self.backoff_factor ** attempt
                logging.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                return await self.send_webhook_async(webhook_url, payload, attempt + 1)
            
            return False, "timeout"
        
        except Exception as e:
            logging.error(f"Error sending webhook (attempt {attempt}): {str(e)}")
            
            # Retry if attempts remain
            if attempt < self.max_retries:
                wait_time = self.backoff_factor ** attempt
                logging.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                return await self.send_webhook_async(webhook_url, payload, attempt + 1)
            
            return False, str(e)
    
    def send_webhook(self, symbol, action, price, **kwargs):
        """Send a trading signal to the appropriate webhook with enhanced reliability"""
        # Get the API-compatible ticker format
        api_ticker = self.get_api_ticker(symbol)
        
        # Get the correct webhook URL for this symbol
        webhook_url = self.get_webhook_url(symbol)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the payload based on action type
        if action == "buy":
            payload = {
                "ticker": api_ticker,
                "action": "buy",
                "price": str(price),
                "time": current_time
            }
            message = "BUY"
        elif action == "sell":
            payload = {
                "ticker": api_ticker,
                "action": "sell",
                "price": str(price),
                "time": current_time
            }
            message = "SELL"
        elif action == "exit_buy":
            payload = {
                "ticker": api_ticker,
                "action": "exit_buy",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "")),
                "per": str(kwargs.get("per", "")),
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            message = "EXIT BUY"
        elif action == "exit_sell":
            payload = {
                "ticker": api_ticker,
                "action": "exit_sell",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "")),
                "per": str(kwargs.get("per", "")),
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            message = "EXIT SELL"
        
        # Add additional contextual data if provided
        for key, value in kwargs.items():
            if key not in payload and key not in ["size", "per", "sl", "tp"]:
                payload[key] = str(value)
        
        # Log the signal
        logging.info(f"Sending {message} signal for {symbol} at {price}")
        
        success = False
        attempted_formats = [api_ticker]
        
        # Use asyncio to send the webhook
        try:
            # Create future for async task
            future = asyncio.run_coroutine_threadsafe(
                self.send_webhook_async(webhook_url, payload), 
                self.loop
            )
            
            # Wait for result with timeout
            success, response = future.result(timeout=30)
            
            # If successful, save the format
            if success:
                if symbol not in SUCCESSFUL_FORMATS:
                    SUCCESSFUL_FORMATS[symbol] = api_ticker
                    self._update_config_file()
            
            # If format not matched, try alternatives
            elif response == "no_matching_pair":
                logging.info(f"No matching pair found for {api_ticker}, trying alternatives")
                alt_success = self._try_alternative_formats(symbol, action, price, webhook_url, payload, attempted_formats)
                success = success or alt_success
        
        except Exception as e:
            logging.error(f"Error in webhook sending: {str(e)}")
            success = False
            
            # Try synchronous fallback
            try:
                logging.info("Trying synchronous webhook as fallback")
                response = requests.post(webhook_url, json=payload, timeout=10)
                success = response.status_code == 200
            except Exception as e2:
                logging.error(f"Synchronous fallback also failed: {str(e2)}")
        
        # If webhook failed, try sending via Telegram as backup
        if not success and self.telegram_config['enabled']:
            success = self.send_telegram(symbol, action, price, **kwargs)
        
        return success
    
    def _try_alternative_formats(self, symbol, action, price, webhook_url, payload, attempted_formats):
        """Try alternative ticker formats"""
        # Define fallback formats based on the symbol
        fallback_formats = []
        
        if symbol == 'BTC/USDT':
            fallback_formats = ['BTCUSDT', 'BTC-USD', 'XBTUSD', 'BTC']
        elif symbol == 'SOL/USDT':
            fallback_formats = ['SOLUSDT', 'SOL-USD', 'SOL']
        else:
            base, quote = symbol.split('/')
            fallback_formats = [f"{base}{quote}", f"{base}-{quote}", base]
        
        # Try each fallback format that hasn't been attempted
        for format_to_try in fallback_formats:
            if format_to_try in attempted_formats:
                continue
            
            attempted_formats.append(format_to_try)
            
            # Update the payload with the new ticker format
            payload["ticker"] = format_to_try
            
            try:
                logging.info(f"Trying fallback ticker format: {format_to_try}")
                
                # Use asyncio to send the webhook
                future = asyncio.run_coroutine_threadsafe(
                    self.send_webhook_async(webhook_url, payload), 
                    self.loop
                )
                fallback_success, _ = future.result(timeout=30)
                
                if fallback_success:
                    logging.info(f"Webhook sent successfully with fallback format: {format_to_try}")
                    
                    # Update the ticker map for future use
                    self.ticker_map[symbol] = format_to_try
                    SUCCESSFUL_FORMATS[symbol] = format_to_try
                    
                    # Update the config file
                    self._update_config_file()
                    
                    return True
            
            except Exception as e:
                logging.error(f"Error trying fallback format {format_to_try}: {str(e)}")
                
                # Try synchronous as fallback for this format
                try:
                    response = requests.post(webhook_url, json=payload, timeout=10)
                    if response.status_code == 200:
                        logging.info(f"Synchronous webhook succeeded with format {format_to_try}")
                        
                        # Update the ticker map for future use
                        self.ticker_map[symbol] = format_to_try
                        SUCCESSFUL_FORMATS[symbol] = format_to_try
                        self._update_config_file()
                        
                        return True
                except Exception:
                    pass
        
        return False
    
    async def send_telegram_async(self, message):
        """Send a Telegram message asynchronously"""
        if not self.telegram_config['enabled']:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['token']}/sendMessage"
            payload = {
                "chat_id": self.telegram_config['chat_id'],
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        response_text = await response.text()
                        logging.error(f"Failed to send Telegram notification. Status code: {response.status}, Response: {response_text}")
                        return False
        
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {str(e)}")
            return False
    
    def send_telegram(self, symbol, action, price, **kwargs):
        """Send a signal via Telegram as a backup channel"""
        if not self.telegram_config['enabled']:
            return False
        
        try:
            # Format the message
            message = f"🚨 {symbol} SIGNAL 🚨\n\n"
            message += f"Action: {action.upper()}\n"
            message += f"Price: {price}\n"
            
            # Add additional data
            if 'sl' in kwargs and kwargs['sl']:
                message += f"Stop Loss: {kwargs['sl']}\n"
            if 'tp' in kwargs and kwargs['tp']:
                message += f"Take Profit: {kwargs['tp']}\n"
            if 'per' in kwargs and kwargs['per']:
                message += f"Profit/Loss: {kwargs['per']}\n"
            if 'duration' in kwargs and kwargs['duration']:
                message += f"Expected Duration: {kwargs['duration']:.1f} hours\n"
            if 'confidence' in kwargs and kwargs['confidence']:
                message += f"Signal Confidence: {float(kwargs['confidence'])*100:.1f}%\n"
            if 'size' in kwargs and kwargs['size']:
                message += f"Position Size: {kwargs['size']}\n"
            
            message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send to Telegram asynchronously
            future = asyncio.run_coroutine_threadsafe(
                self.send_telegram_async(message), 
                self.loop
            )
            success = future.result(timeout=10)
            
            if success:
                logging.info(f"Telegram notification sent successfully for {symbol} {action}")
                return True
            else:
                # Try synchronous as fallback
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_config['token']}/sendMessage"
                    payload = {
                        "chat_id": self.telegram_config['chat_id'],
                        "text": message,
                        "parse_mode": "HTML"
                    }
                    response = requests.post(url, json=payload, timeout=10)
                    return response.status_code == 200
                except Exception:
                    return False
        
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {str(e)}")
            return False
    
    def _update_config_file(self):
        """Update the configuration file with successful ticker formats"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_content = f.read()
            
            # Update the TICKER_MAP in the config
            import re
            ticker_pattern = r"TICKER_MAP = \{.*?\}"
            ticker_replacement = f"TICKER_MAP = {json.dumps(self.ticker_map, indent=4)}"
            config_content = re.sub(ticker_pattern, ticker_replacement, config_content, flags=re.DOTALL)
            
            # Update SUCCESSFUL_FORMATS
            formats_pattern = r"SUCCESSFUL_FORMATS = \{.*?\}"
            formats_replacement = f"SUCCESSFUL_FORMATS = {json.dumps(SUCCESSFUL_FORMATS, indent=4)}"
            config_content = re.sub(formats_pattern, formats_replacement, config_content, flags=re.DOTALL)
            
            with open(CONFIG_FILE, 'w') as f:
                f.write(config_content)
            
            logging.info(f"Updated config file with new ticker mappings")
        except Exception as e:
            logging.error(f"Error updating config file: {str(e)}")
    
    def register_trade(self, symbol, trade_id, action, entry_price, sl, tp, duration, position_size=1.0):
        """Register an active trade"""
        self.active_trades[symbol] = {
            'trade_id': trade_id,
            'action': action,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_time': datetime.now(),
            'expected_duration': duration,
            'position_size': position_size,
            'partial_exit': False
        }
    
    def get_active_trade(self, symbol):
        """Get information about an active trade"""
        return self.active_trades.get(symbol)
    
    def update_trade_info(self, symbol, **kwargs):
        """Update information for an active trade"""
        if symbol in self.active_trades:
            for key, value in kwargs.items():
                self.active_trades[symbol][key] = value
            return True
        return False
    
    def remove_trade(self, symbol):
        """Remove a trade after it's closed"""
        if symbol in self.active_trades:
            del self.active_trades[symbol]
    
    def shutdown(self):
        """Clean up resources"""
        self.executor.shutdown()
        self.loop.stop()

#################################
# 4. DATA HANDLING
#################################

def fetch_crypto_data(symbol, timeframe='1h', lookback_days=90):
    """
    Fetch historical crypto data from ccxt compatible exchange
    """
    try:
        # Use Binance by default (can be changed to any other exchange)
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        # Convert to milliseconds timestamp
        since = int(start.timestamp() * 1000)
        
        logging.info(f"Fetching data for {symbol} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        logging.info(f"Successfully fetched {len(df)} data points for {symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        # Try fallback exchange if the first one fails
        try:
            exchange = ccxt.kucoin({'enableRateLimit': True})
            since = int(start.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            logging.info(f"Successfully fetched {len(df)} data points for {symbol} using fallback exchange")
            return df
        except Exception as fallback_error:
            logging.error(f"Fallback exchange also failed: {str(fallback_error)}")
            return None

def detect_market_regime(df, lookback=20):
    """
    Detect the current market regime (trending, ranging, volatile)
    """
    if len(df) < lookback:
        return "unknown", 0, 0
    
    # Get relevant data
    prices = df['close'].iloc[-lookback:].values
    volumes = df['volume'].iloc[-lookback:].values
    
    # Volatility (ATR-based)
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else np.mean(
        np.abs(df['high'].iloc[-lookback:] - df['low'].iloc[-lookback:])
    )
    price_volatility = atr / np.mean(prices)
    
    # Trend strength (ADX-based)
    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
    
    # Volume profile
    volume_trend = np.corrcoef(np.arange(lookback), volumes)[0, 1]
    
    # Price directionality
    price_return = (prices[-1] / prices[0]) - 1
    price_direction = np.abs(price_return)
    
    # Moving average relationship
    ma_short = np.mean(prices[-5:])
    ma_long = np.mean(prices)
    ma_relationship = ma_short / ma_long - 1
    
    # Determine regime
    if adx > 25 and np.abs(ma_relationship) > 0.01:
        # Strong trend
        regime = "trending"
    elif price_volatility > 0.04:  # High volatility threshold
        regime = "volatile"
    else:
        # Ranging/sideways
        regime = "ranging"
    
    return regime, price_volatility, adx

def calculate_features(df):
    """
    Calculate advanced technical indicators with enhanced features
    """
    # Make a copy to avoid fragmentation warnings
    df = df.copy()
    
    # Convert all data to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values that might have appeared
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    features = pd.DataFrame(index=df.index)
    
    # Price data
    features['open'] = df['open']
    features['high'] = df['high']
    features['low'] = df['low']
    features['close'] = df['close']
    features['volume'] = df['volume']
    
    # Basic price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['range'] = df['high'] - df['low']
    features['range_pct'] = features['range'] / df['close']
    
    # Volatility features
    for window in [5, 7, 14]:  # Added shorter window for faster response
        features[f'volatility_{window}d'] = features['returns'].rolling(window=window).std() * np.sqrt(window)
    
    # Relative strength of recent moves
    features['bull_power'] = df['high'] - talib.EMA(df['close'], timeperiod=13)
    features['bear_power'] = df['low'] - talib.EMA(df['close'], timeperiod=13)
    
    # Moving Averages - Added shorter timeframes for quicker signals
    for period in [3, 5, 10, 20, 50, 100, 200]:
        features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Distance from price to MA
        features[f'dist_to_sma_{period}'] = (df['close'] - features[f'sma_{period}']) / df['close']
        features[f'dist_to_ema_{period}'] = (df['close'] - features[f'ema_{period}']) / df['close']
    
    # Moving average crossovers
    features['sma_3_10_cross'] = np.where(features['sma_3'] > features['sma_10'], 1, -1)  # Faster crossover
    features['sma_5_20_cross'] = np.where(features['sma_5'] > features['sma_20'], 1, -1)
    features['sma_20_50_cross'] = np.where(features['sma_20'] > features['sma_50'], 1, -1)
    features['ema_3_10_cross'] = np.where(features['ema_3'] > features['ema_10'], 1, -1)  # Faster crossover
    features['ema_5_20_cross'] = np.where(features['ema_5'] > features['ema_20'], 1, -1)
    
    # Bollinger Bands
    for period in [10, 20]:  # Added shorter timeframe for BB
        upper, middle, lower = talib.BBANDS(
            df['close'], timeperiod=period, nbdevup=2, nbdevdn=2
        )
        features[f'bb_upper_{period}'] = upper
        features[f'bb_lower_{period}'] = lower
        features[f'bb_width_{period}'] = (upper - lower) / middle
        features[f'bb_pos_{period}'] = (df['close'] - lower) / (upper - lower)
    
    # RSI and momentum - Added shorter timeframes for quicker response
    for period in [3, 7, 14]:
        features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
    
    # RSI divergence detection
    features['price_higher'] = df['close'] > df['close'].shift(3)
    features['rsi_lower'] = features['rsi_14'] < features['rsi_14'].shift(3)
    features['bearish_div'] = (features['price_higher'] & features['rsi_lower']).astype(float)
    
    features['price_lower'] = df['close'] < df['close'].shift(3)
    features['rsi_higher'] = features['rsi_14'] > features['rsi_14'].shift(3)
    features['bullish_div'] = (features['price_lower'] & features['rsi_higher']).astype(float)
    
    # MACD - Adjusted for faster response
    macd, signal, hist = talib.MACD(
        df['close'], fastperiod=8, slowperiod=17, signalperiod=9  # Modified for faster signals
    )
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = hist
    features['macd_cross'] = np.where(macd > signal, 1, -1)
    
    # MACD histogram analysis
    features['macd_hist_change'] = features['macd_hist'].diff(1)
    features['macd_hist_slope'] = features['macd_hist'].diff(3)
    
    # MACD divergence detection (enhanced)
    features['macd_div_bullish'] = ((df['close'] < df['close'].shift(5)) & 
                                   (features['macd'] > features['macd'].shift(5))).astype(float)
    features['macd_div_bearish'] = ((df['close'] > df['close'].shift(5)) & 
                                   (features['macd'] < features['macd'].shift(5))).astype(float)
    
    # Trend indicators
    features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    features['adx_trend'] = np.where(features['adx'] > 25, 1, 0)
    
    # ATR (Average True Range)
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['atr_pct'] = features['atr'] / df['close']
    
    # Heikin Ashi indicators
    features['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    features['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    features['ha_high'] = df[['high', 'open', 'close']].max(axis=1)
    features['ha_low'] = df[['low', 'open', 'close']].min(axis=1)
    features['ha_trend'] = np.where(features['ha_close'] > features['ha_open'], 1, -1)
    
    # Momentum indicators
    features['mom'] = talib.MOM(df['close'], timeperiod=10)
    features['mom_pct'] = features['mom'] / df['close'].shift(10)
    
    # Stochastic oscillator - Adjusted for faster response
    features['slowk'], features['slowd'] = talib.STOCH(
        df['high'], df['low'], df['close'], 
        fastk_period=10, slowk_period=3, slowk_matype=0,  # Faster stochastic
        slowd_period=3, slowd_matype=0
    )
    features['stoch_cross'] = np.where(features['slowk'] > features['slowd'], 1, -1)
    
    # Directional Movement Index
    features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    features['di_cross'] = np.where(features['plus_di'] > features['minus_di'], 1, -1)
    
    # Volume indicators
    features['volume_sma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma']
    features['up_volume'] = df['volume'] * (df['close'] > df['close'].shift(1)).astype(float)
    features['down_volume'] = df['volume'] * (df['close'] < df['close'].shift(1)).astype(float)
    
    # Price-volume relationship (new)
    features['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
    
    # Price patterns
    features['higher_high_3d'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    features['lower_low_3d'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    features['higher_high_3d'] = features['higher_high_3d'].astype(float)
    features['lower_low_3d'] = features['lower_low_3d'].astype(float)
    
    # Channel breakouts - Adjusted for shorter timeframe
    features['upper_channel'] = df['high'].rolling(15).max()  # Reduced from 20
    features['lower_channel'] = df['low'].rolling(15).min()   # Reduced from 20
    features['channel_pos'] = (df['close'] - features['lower_channel']) / (features['upper_channel'] - features['lower_channel'])
    
    # Short-term price dynamics - New for faster trading
    features['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    features['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    features['price_accel'] = df['close'].diff(1).diff(1)
    
    # Fourier transform features (new)
    try:
        # Get the most recent 64 prices (power of 2 for FFT efficiency)
        close_prices = df['close'].values[-64:]
        # Compute FFT
        fft_result = fft.fft(close_prices)
        # Get amplitudes
        amplitudes = np.abs(fft_result)[:32]  # Use only first half (rest are complex conjugates)
        # Get phases
        phases = np.angle(fft_result)[:32]
        
        # Add dominant frequency components (top 3)
        dominant_indices = np.argsort(amplitudes)[-3:]
        for i, idx in enumerate(dominant_indices):
            if idx > 0:  # Skip DC component
                features[f'fft_amp_{i}'] = amplitudes[idx]
                features[f'fft_phase_{i}'] = phases[idx]
                
        # Add total spectral power
        features['fft_total_power'] = np.sum(amplitudes[1:] ** 2)  # Skip DC component
    except Exception as e:
        logging.warning(f"Error calculating FFT features: {str(e)}")
        # Add placeholder columns
        features['fft_amp_0'] = 0
        features['fft_amp_1'] = 0
        features['fft_amp_2'] = 0
        features['fft_phase_0'] = 0
        features['fft_phase_1'] = 0
        features['fft_phase_2'] = 0
        features['fft_total_power'] = 0
    
    # Market regime features (new)
    try:
        # Detect regime for each window
        regimes = []
        volatilities = []
        trend_strengths = []
        
        for i in range(len(df)):
            if i < 20:
                regime, vol, trend = "unknown", 0, 0
            else:
                regime, vol, trend = detect_market_regime(df.iloc[:i+1])
            
            # Convert regime to numeric
            if regime == "trending":
                regime_numeric = 1
            elif regime == "volatile":
                regime_numeric = 2
            elif regime == "ranging":
                regime_numeric = 3
            else:  # unknown
                regime_numeric = 0
                
            regimes.append(regime_numeric)
            volatilities.append(vol)
            trend_strengths.append(trend)
        
        features['market_regime'] = regimes
        features['market_volatility'] = volatilities
        features['market_trend_strength'] = trend_strengths
    except Exception as e:
        logging.warning(f"Error calculating market regime features: {str(e)}")
        # Add placeholder columns
        features['market_regime'] = 0
        features['market_volatility'] = 0
        features['market_trend_strength'] = 0
    
    # Ensure all columns are float type
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype == 'bool':
            features[col] = pd.to_numeric(features[col], errors='coerce').astype(float)
    
    # Fill any NaN values from calculations
    features = features.fillna(0).astype('float32')
    
    return features

def prepare_sequences(df, seq_length=SEQ_LENGTH, prediction_horizon=PREDICTION_HORIZON):
    """
    Prepare sequences for training with ADJUSTED thresholds for more signals
    """
    # Extract price data
    close_prices = df['close'].values
    
    # Scale features with RobustScaler
    scaler = RobustScaler()
    # Scale all columns except timestamps
    columns_to_scale = [col for col in df.columns if col != 'timestamp']
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[columns_to_scale]),
        columns=columns_to_scale,
        index=df.index
    )
    
    # Convert all to float32 for consistency
    df_scaled = df_scaled.astype('float32')
    
    # Create sequences
    X, y = [], []
    stop_losses, take_profits, durations = [], [], []
    sample_weights = []
    
    # For each possible sequence
    for i in range(len(df) - seq_length - prediction_horizon):
        # Input sequence
        seq = df_scaled.iloc[i:i+seq_length].values
        
        # Target: Future price movement
        current_price = close_prices[i+seq_length]
        future_price = close_prices[i+seq_length+prediction_horizon]
        price_change = (future_price / current_price) - 1
        
        # Get recent volatility for adaptive thresholds
        recent_atr_pct = df['atr_pct'].iloc[i+seq_length-10:i+seq_length].mean()
        
        # ADJUSTED: Much more sensitive thresholds
        # Base thresholds are now 0.2% instead of 0.5%
        buy_threshold = max(0.002, recent_atr_pct * 0.5)   # Reduced from 1.0
        sell_threshold = min(-0.002, -recent_atr_pct * 0.5)  # Reduced from 1.0
        
        # ADJUSTED: More aggressive labeling
        if price_change > buy_threshold:
            target = 0  # Buy
            # ADJUSTED: More conservative targets for shorter trades
            take_profit = min(max(price_change * 1.5, 0.005), 0.1)  # Min 0.5%, max 10%
            stop_loss = max(min(recent_atr_pct * 1.0, 0.05), 0.003)  # Min 0.3%, max 5%
            duration = min(12, max(2, prediction_horizon * 2))  # Shorter durations
            weight = 3.0  # Higher weight for buy/sell signals
        elif price_change < sell_threshold:
            target = 2  # Sell
            take_profit = min(max(abs(price_change) * 1.5, 0.005), 0.1)
            stop_loss = max(min(recent_atr_pct * 1.0, 0.05), 0.003)
            duration = min(12, max(2, prediction_horizon * 2))
            weight = 3.0  # Higher weight for buy/sell signals
        else:
            target = 1  # Hold
            take_profit = max(recent_atr_pct * 1.5, 0.01)
            stop_loss = max(recent_atr_pct * 0.8, 0.003)
            duration = 6  # Default duration for hold
            weight = 0.5  # Lower weight for hold signals to encourage trading
        
        X.append(seq)
        y.append(target)
        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        durations.append(duration)
        sample_weights.append(weight)
    
    return np.array(X), np.array(y), np.array(stop_losses), np.array(take_profits), np.array(durations), np.array(sample_weights)

def prepare_sequences_with_regimes(df, seq_length=SEQ_LENGTH, prediction_horizon=PREDICTION_HORIZON):
    """
    Prepare sequences with market regime labels
    """
    # Get the original X, y, etc.
    X, y, stop_losses, take_profits, durations, sample_weights = prepare_sequences(df, seq_length, prediction_horizon)
    
    # Extract regime data for each sequence
    regimes = []
    price_changes = []
    
    # Extract price data
    close_prices = df['close'].values
    
    # For each possible sequence
    for i in range(len(df) - seq_length - prediction_horizon):
        # Regime at prediction time
        regime = df['market_regime'].iloc[i+seq_length] if 'market_regime' in df.columns else 0
        regimes.append(regime)
        
        # Price change for Sharpe ratio calculation
        current_price = close_prices[i+seq_length]
        future_price = close_prices[i+seq_length+prediction_horizon]
        price_change = (future_price / current_price) - 1
        price_changes.append(price_change)
    
    return X, y, stop_losses, take_profits, durations, sample_weights, np.array(regimes), np.array(price_changes)

def prepare_sequence_for_prediction(df, seq_length=SEQ_LENGTH):
    """
    Prepare a single sequence for prediction with enhanced robustness
    """
    # Make a copy to avoid fragmentation issues
    df = df.copy()
    
    # Ensure all data is numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace any remaining NaN values
    df = df.fillna(0)
    
    # Scale all features with RobustScaler
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Convert to float32
    df_scaled = df_scaled.astype('float32')
    
    # Get the last sequence
    sequence = df_scaled.iloc[-seq_length:].values
    
    # Ensure the sequence has the correct shape
    if len(sequence) < seq_length:
        padding = np.zeros((seq_length - len(sequence), sequence.shape[1]), dtype='float32')
        sequence = np.vstack((padding, sequence))
    
    return sequence

#################################
# 5. MODEL TRAINING WITH FEEDBACK
#################################

class SharpeRatioLoss(nn.Module):
    """
    Custom loss function optimizing for Sharpe ratio
    """
    def __init__(self, alpha=0.5, risk_free_rate=0.0):
        super().__init__()
        self.alpha = alpha  # Weight between accuracy and Sharpe
        self.risk_free_rate = risk_free_rate
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, direction_probs, direction_targets, price_changes):
        # Standard cross-entropy loss for classification
        ce_loss = self.ce_loss(direction_probs, direction_targets)
        
        # Calculate returns for each sample based on predicted probabilities
        # We want buy (0) when price goes up, sell (2) when price goes down
        buy_probs = direction_probs[:, 0]
        sell_probs = direction_probs[:, 2]
        
        # Simulate returns (positive when buy & price up or sell & price down)
        simulated_returns = buy_probs * price_changes + sell_probs * (-price_changes)
        
        # Calculate Sharpe ratio
        mean_return = torch.mean(simulated_returns)
        std_return = torch.std(simulated_returns) + 1e-6  # Avoid division by zero
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        
        # Negative sharpe because we want to minimize loss
        sharpe_loss = -sharpe_ratio
        
        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * sharpe_loss
        
        return total_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()

class CustomDataset(Dataset):
    """
    Dataset for trading data with sample weights
    """
    def __init__(self, sequences, targets, stop_losses, take_profits, durations, weights=None, regimes=None, price_changes=None):
        self.sequences = sequences
        self.targets = targets
        self.stop_losses = stop_losses
        self.take_profits = take_profits
        self.durations = durations
        self.weights = weights
        self.regimes = regimes
        self.price_changes = price_changes
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = [
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([int(self.targets[idx])]),  # Convert to int
            torch.FloatTensor([self.stop_losses[idx]]),
            torch.FloatTensor([self.take_profits[idx]]),
            torch.FloatTensor([self.durations[idx]]),
            torch.FloatTensor([self.weights[idx] if self.weights is not None else 1.0])
        ]
        
        if self.regimes is not None:
            item.append(torch.LongTensor([int(self.regimes[idx])]))  # Convert to int
        
        if self.price_changes is not None:
            item.append(torch.FloatTensor([self.price_changes[idx]]))
        
        return tuple(item)

def train_model(symbol, timeframe='1h', lookback_days=90, feedback_system=None, train_data=None):
    """
    Train a model for a specific symbol with enhanced training
    """
    logging.info(f"Training model for {symbol}...")
    
    # Fetch data if not provided
    if train_data is None:
        df = fetch_crypto_data(symbol, timeframe, lookback_days)
        if df is None or len(df) < SEQ_LENGTH + 10:
            logging.error(f"Insufficient data for {symbol}")
            return None
        
        # Calculate features
        feature_df = calculate_features(df)
    else:
        feature_df = train_data
    
    # Prepare sequences with regimes and price changes
    X, y, stop_losses, take_profits, durations, sample_weights, regimes, price_changes = prepare_sequences_with_regimes(feature_df)
    
    if len(X) == 0:
        logging.error(f"No sequences could be prepared for {symbol}")
        return None
    
    # Check class balance
    class_counts = np.bincount(y)
    logging.info(f"Class distribution - Buy: {class_counts[0]}, Hold: {class_counts[1]}, Sell: {class_counts[2]}")
    
    # Calculate class weights for focal loss
    if len(class_counts) == 3:  # Ensure all 3 classes exist
        # ADJUSTED: Give more weight to buy/sell classes
        class_weights = np.array([5.0, 1.0, 5.0])  # Buy and Sell get 5x weight vs Hold
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    else:
        class_weights = None
    
    # Incorporate feedback data if available
    if feedback_system:
        feedback_data = feedback_system.get_feedback_data(symbol)
        if feedback_data and len(feedback_data) > 0:
            logging.info(f"Incorporating {len(feedback_data)} feedback samples into training")
            
            # Extract feedback sequences and outcomes
            feedback_X = []
            feedback_y = []
            feedback_weights = []
            
            for feature_data, actual_outcome, confidence in feedback_data:
                feedback_X.append(feature_data)
                feedback_y.append(actual_outcome)
                feedback_weights.append(confidence * 5.0)  # Give even higher weight to actual outcomes
            
            # Add feedback data to training set
            # Note: We only add target labels from feedback, not SL/TP values
            if len(feedback_X) > 0:
                # Generate dummy values for SL, TP, and duration for feedback data
                dummy_sl = np.mean(stop_losses) * np.ones(len(feedback_X))
                dummy_tp = np.mean(take_profits) * np.ones(len(feedback_X))
                dummy_duration = np.mean(durations) * np.ones(len(feedback_X))
                dummy_regime = np.zeros(len(feedback_X))
                dummy_price_change = np.zeros(len(feedback_X))
                
                # Combine with regular training data
                X = np.vstack((X, np.array(feedback_X)))
                y = np.append(y, np.array(feedback_y))
                sample_weights = np.append(sample_weights, np.array(feedback_weights))
                stop_losses = np.append(stop_losses, dummy_sl)
                take_profits = np.append(take_profits, dummy_tp)
                durations = np.append(durations, dummy_duration)
                regimes = np.append(regimes, dummy_regime)
                price_changes = np.append(price_changes, dummy_price_change)
                
                logging.info(f"Updated dataset size after feedback incorporation: {len(X)}")
                
                # Recalculate class distribution
                new_class_counts = np.bincount(y.astype(int))
                logging.info(f"Updated class distribution - Buy: {new_class_counts[0]}, "
                             f"Hold: {new_class_counts[1]}, Sell: {new_class_counts[2]}")
    
    # Split data
    X_train, X_val, y_train, y_val, sl_train, sl_val, tp_train, tp_val, dur_train, dur_val, w_train, w_val, r_train, r_val, pc_train, pc_val = train_test_split(
        X, y, stop_losses, take_profits, durations, sample_weights, regimes, price_changes, test_size=0.2, shuffle=False
    )
    
    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, sl_train, tp_train, dur_train, w_train, r_train, pc_train)
    val_dataset = CustomDataset(X_val, y_val, sl_val, tp_val, dur_val, w_val, r_val, pc_val)
    
    # Create weighted sampler for training to handle class imbalance
    train_weights = torch.FloatTensor(w_train)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_size = X_train.shape[2]
    logging.info(f"Input feature dimension: {input_size}")
    
    model = TemporalFusionModel(input_size=input_size).to(DEVICE)
    
    # Adjust model parameters based on historical performance if feedback is available
    if feedback_system:
        adjustment_factors = feedback_system.get_model_adjustment_factors(symbol)
        if adjustment_factors and 'tp_adjustment' in adjustment_factors:
            logging.info(f"Applying model adjustments based on historical performance")
            
            # Apply take profit adjustment
            tp_adjustment = adjustment_factors['tp_adjustment']
            if hasattr(model, 'tp_head') and 0.5 <= tp_adjustment <= 2.0:
                logging.info(f"Adjusting TP prediction by factor: {tp_adjustment}")
                # This is a simplified approach - in practice you would want to modify
                # the final layer weights or adjust outputs during prediction
                for param in model.tp_head[-2].parameters():
                    if isinstance(param, nn.Parameter):
                        param.data = param.data * tp_adjustment
            
            # Apply SL adjustment
            sl_adjustment = adjustment_factors['sl_adjustment']
            if hasattr(model, 'sl_head') and 0.5 <= sl_adjustment <= 2.0:
                logging.info(f"Adjusting SL prediction by factor: {sl_adjustment}")
                for param in model.sl_head[-2].parameters():
                    if isinstance(param, nn.Parameter):
                        param.data = param.data * sl_adjustment
    
    # Use SharpeRatioLoss along with FocalLoss
    direction_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    sharpe_criterion = SharpeRatioLoss(alpha=0.7)  # 70% CE loss, 30% Sharpe
    sl_criterion = nn.MSELoss()
    tp_criterion = nn.MSELoss()
    duration_criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dir_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Get data
            sequences = batch[0].to(DEVICE)
            dir_targets = batch[1].squeeze().to(DEVICE)
            sl_targets = batch[2].to(DEVICE)
            tp_targets = batch[3].to(DEVICE)
            dur_targets = batch[4].to(DEVICE)
            price_changes = batch[7].to(DEVICE) if len(batch) > 7 else None
            
            # Forward pass
            dir_probs, sl_preds, tp_preds, dur_preds, _, _ = model(sequences)
            
            # Calculate losses
            if price_changes is not None:
                # Use Sharpe ratio loss if price changes available
                dir_loss = sharpe_criterion(dir_probs, dir_targets, price_changes)
            else:
                # Otherwise use standard focal loss
                dir_loss = direction_criterion(dir_probs, dir_targets)
                
            sl_loss = sl_criterion(sl_preds, sl_targets)
            tp_loss = tp_criterion(tp_preds, tp_targets)
            dur_loss = duration_criterion(dur_preds, dur_targets)
            
            # Combined loss (weighted)
            loss = dir_loss * 0.7 + sl_loss * 0.1 + tp_loss * 0.1 + dur_loss * 0.1
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(dir_probs, 1)
            train_total += dir_targets.size(0)
            train_dir_correct += (predicted == dir_targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dir_correct = 0
        val_total = 0
        val_class_correct = [0, 0, 0]  # Correct predictions for each class
        val_class_total = [0, 0, 0]    # Total predictions for each class
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                sequences = batch[0].to(DEVICE)
                dir_targets = batch[1].squeeze().to(DEVICE)
                sl_targets = batch[2].to(DEVICE)
                tp_targets = batch[3].to(DEVICE)
                dur_targets = batch[4].to(DEVICE)
                price_changes = batch[7].to(DEVICE) if len(batch) > 7 else None
                
                # Forward pass
                dir_probs, sl_preds, tp_preds, dur_preds, _, _ = model(sequences)
                
                # Calculate losses
                if price_changes is not None:
                    # Use Sharpe ratio loss if price changes available
                    dir_loss = sharpe_criterion(dir_probs, dir_targets, price_changes)
                else:
                    # Otherwise use standard focal loss
                    dir_loss = direction_criterion(dir_probs, dir_targets)
                    
                sl_loss = sl_criterion(sl_preds, sl_targets)
                tp_loss = tp_criterion(tp_preds, tp_targets)
                dur_loss = duration_criterion(dur_preds, dur_targets)
                
                # Combined loss
                loss = dir_loss * 0.7 + sl_loss * 0.1 + tp_loss * 0.1 + dur_loss * 0.1
                
                val_loss += loss.item()
                
                # Track accuracy for each class
                _, predicted = torch.max(dir_probs, 1)
                val_total += dir_targets.size(0)
                val_dir_correct += (predicted == dir_targets).sum().item()
                
                # Track per-class accuracy
                for i in range(3):  # 3 classes: Buy, Hold, Sell
                    class_mask = (dir_targets == i)
                    val_class_total[i] += class_mask.sum().item()
                    if val_class_total[i] > 0:
                        val_class_correct[i] += (predicted[class_mask] == i).sum().item()
        
        # Calculate average losses and accuracy metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_dir_correct / train_total if train_total > 0 else 0
        val_accuracy = 100 * val_dir_correct / val_total if val_total > 0 else 0
        
        # Calculate per-class accuracy
        class_accuracy = []
        for i in range(3):
            if val_class_total[i] > 0:
                class_accuracy.append(100 * val_class_correct[i] / val_class_total[i])
            else:
                class_accuracy.append(0)
        
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"Train Acc: {train_accuracy:.2f}% | "
                   f"Val Acc: {val_accuracy:.2f}% | "
                   f"Buy Acc: {class_accuracy[0]:.2f}% | "
                   f"Hold Acc: {class_accuracy[1]:.2f}% | "
                   f"Sell Acc: {class_accuracy[2]:.2f}%")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save model
            safe_symbol = symbol.replace('/', '_')
            model_path = f'models/{safe_symbol}_model.pth'
            torch.save(model.state_dict(), model_path)
            
            # Save model info
            model_info = {
                'symbol': symbol,
                'input_size': input_size,
                'accuracy': val_accuracy,
                'per_class_accuracy': class_accuracy,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_size': len(X),
                'epochs_trained': epoch + 1
            }
            
            with open(f'models/{safe_symbol}_model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logging.info(f"Model saved to {model_path}")
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model

def train_diverse_models(symbol, num_models=3):
    """
    Train multiple diverse models for ensemble
    """
    models = []
    model_info = []
    
    # Fetch data
    df = fetch_crypto_data(symbol, timeframe='1h', lookback_days=120)
    if df is None or len(df) < SEQ_LENGTH + 10:
        logging.error(f"Insufficient data for {symbol}")
        return None, None
    
    # Calculate features
    feature_df = calculate_features(df)
    
    # Variations for diversity
    sequence_lengths = [30, 40, 50]
    hidden_sizes = [128, 196, 256]
    
    for i in range(num_models):
        logging.info(f"Training model {i+1}/{num_models} for ensemble")
        
        # Vary params for diversity
        seq_len = sequence_lengths[i % len(sequence_lengths)]
        h_size = hidden_sizes[i % len(hidden_sizes)]
        
        # Use different data samples (time periods or feature subsets)
        if i == 0:
            # Full dataset, most recent
            train_data = feature_df
        elif i == 1:
            # Older data
            train_data = feature_df.iloc[:-len(feature_df)//4]
        else:
            # Random sample of features (80%) but ensure essential columns are included
            essential_columns = ['open', 'high', 'low', 'close', 'volume']
            other_columns = [col for col in feature_df.columns if col not in essential_columns]
            
            # Select 80% of non-essential columns
            num_to_select = int(len(other_columns) * 0.8)
            selected_other = np.random.choice(other_columns, size=num_to_select, replace=False)
            
            # Combine essential columns with selected features
            feature_subset = essential_columns + list(selected_other)
            train_data = feature_df[feature_subset]
        
        # Initialize model with varying parameters
        input_size = train_data.shape[1]
        model = TemporalFusionModel(input_size=input_size, hidden_size=h_size).to(DEVICE)
        
        # Prepare sequences with the specific seq_length
        X, y, stop_losses, take_profits, durations, sample_weights = prepare_sequences(
            train_data, seq_length=seq_len
        )
        
        # Train the model (simplified training loop)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        epochs = 30
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass
                direction_probs, _, _, _, _, _ = model(inputs)
                
                # Calculate loss
                loss = criterion(direction_probs, targets)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    
                    # Forward pass
                    direction_probs, _, _, _, _, _ = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(direction_probs, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(direction_probs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            logging.info(f"Model {i+1}, Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {avg_train_loss:.4f} | "
                       f"Val Loss: {avg_val_loss:.4f} | "
                       f"Accuracy: {accuracy:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Add to ensemble
        models.append(model)
        
        # Store model info
        model_info.append({
            'seq_length': seq_len,
            'hidden_size': h_size,
            'accuracy': accuracy,
            'val_loss': avg_val_loss
        })
        
        # Save model
        safe_symbol = symbol.replace('/', '_')
        torch.save(model.state_dict(), f'models/{safe_symbol}_ensemble_{i}.pth')
    
    # Calculate ensemble weights based on validation performance
    weights = [1.0 / (info['val_loss'] + 1e-5) for info in model_info]
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights=weights)
    
    # Save ensemble info
    ensemble_info = {
        'models': model_info,
        'weights': weights,
        'symbol': symbol,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    safe_symbol = symbol.replace('/', '_')
    with open(f'models/{safe_symbol}_ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    return ensemble, ensemble_info

def load_ensemble_model(symbol):
    """
    Load an ensemble model for a specific symbol
    """
    try:
        safe_symbol = symbol.replace('/', '_')
        ensemble_info_path = f'models/{safe_symbol}_ensemble_info.json'
        
        # Check if ensemble info exists
        if not os.path.exists(ensemble_info_path):
            logging.warning(f"No ensemble found for {symbol}.")
            return None
        
        # Load ensemble info
        with open(ensemble_info_path, 'r') as f:
            ensemble_info = json.load(f)
        
        # Load individual models
        models = []
        weights = ensemble_info.get('weights', [])
        
        # First fetch the data to get input_size
        df = fetch_crypto_data(symbol, timeframe='1h', lookback_days=30)
        if df is None:
            logging.error(f"Could not fetch data for {symbol}")
            return None
            
        feature_df = calculate_features(df)
        input_size = feature_df.shape[1]
        
        for i, model_info in enumerate(ensemble_info['models']):
            model_path = f'models/{safe_symbol}_ensemble_{i}.pth'
            
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_path} not found.")
                continue
            
            # Initialize model with the same architecture
            hidden_size = model_info.get('hidden_size', HIDDEN_SIZE)
            
            model = TemporalFusionModel(input_size=input_size, hidden_size=hidden_size).to(DEVICE)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            models.append(model)
        
        # Create ensemble
        if models:
            ensemble = EnsembleModel(models, weights=weights)
            logging.info(f"Ensemble model for {symbol} loaded successfully with {len(models)} models.")
            return ensemble
        else:
            logging.error(f"Failed to load any models for ensemble.")
            return None
        
    except Exception as e:
        logging.error(f"Failed to load ensemble model for {symbol}: {str(e)}")
        return None

#################################
# 6. BACKTESTING FRAMEWORK
#################################

class BacktestEngine:
    """
    Comprehensive backtesting engine for evaluating model performance
    """
    def __init__(self, symbol, model, lookback_days=90, commission=0.001):
        self.symbol = symbol
        self.model = model
        self.lookback_days = lookback_days
        self.commission = commission  # 0.1% commission per trade
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0
        self.partial_exits = {}
    
    def run_backtest(self, start_date=None, end_date=None):
        """
        Run backtest from start_date to end_date
        """
        # Fetch data
        df = fetch_crypto_data(self.symbol, timeframe='1h', lookback_days=self.lookback_days)
        if df is None or len(df) < SEQ_LENGTH + 10:
            logging.error(f"Insufficient data for {self.symbol}")
            return None
        
        # Calculate features
        feature_df = calculate_features(df)
        
        # Filter by date if provided
        if start_date:
            feature_df = feature_df[feature_df.index >= start_date]
        if end_date:
            feature_df = feature_df[feature_df.index <= end_date]
        
        # Initialize tracking variables
        initial_balance = 10000  # $10,000 starting capital
        balance = initial_balance
        position = None
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0
        equity_history = []
        
        # Start with SEQ_LENGTH to have enough data
        for i in range(SEQ_LENGTH, len(feature_df)):
            current_time = feature_df.index[i]
            
            # Get price data
            current_price = float(feature_df['close'].iloc[i])
            high_price = float(feature_df['high'].iloc[i])
            low_price = float(feature_df['low'].iloc[i])
            
            # Update equity history
            if position == "long":
                equity = balance + (position_size * (current_price / entry_price - 1) * initial_balance)
            elif position == "short":
                equity = balance + (position_size * (entry_price / current_price - 1) * initial_balance)
            else:
                equity = balance
                
            equity_history.append((current_time, equity))
            
            # Check for stop loss or take profit if in a position
            if position == "long":
                # Check if low price hit stop loss
                if low_price <= stop_loss:
                    # Calculate profit/loss
                    pnl = ((stop_loss / entry_price) - 1) * position_size * initial_balance
                    # Apply commission
                    pnl -= (position_size * initial_balance * self.commission * 2)  # Entry and exit
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': feature_df.index[i-1],
                        'exit_time': current_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl': pnl,
                        'pnl_pct': pnl / (position_size * initial_balance) * 100,
                        'exit_reason': 'sl_hit',
                        'position_size': position_size
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0
                
                # Check if high price hit take profit
                elif high_price >= take_profit:
                    # Calculate profit/loss
                    pnl = ((take_profit / entry_price) - 1) * position_size * initial_balance
                    # Apply commission
                    pnl -= (position_size * initial_balance * self.commission * 2)  # Entry and exit
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': feature_df.index[i-1],
                        'exit_time': current_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl': pnl,
                        'pnl_pct': pnl / (position_size * initial_balance) * 100,
                        'exit_reason': 'tp_hit',
                        'position_size': position_size
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0
                
                # Check for partial profit taking (simulation)
                elif position_size > 0 and i not in self.partial_exits:
                    # If price has moved 40% toward take profit
                    tp_distance = take_profit - entry_price
                    partial_profit_threshold = entry_price + (tp_distance * 0.4)
                    
                    if high_price >= partial_profit_threshold and partial_profit_threshold > entry_price * 1.02:
                        # Take profit on half the position
                        partial_size = position_size / 2
                        pnl = ((partial_profit_threshold / entry_price) - 1) * partial_size * initial_balance
                        # Apply commission
                        pnl -= (partial_size * initial_balance * self.commission)  # Just exit commission
                        
                        # Update balance and position size
                        balance += pnl
                        position_size -= partial_size
                        
                        # Record partial exit
                        self.partial_exits[i] = True
                        
                        # Record trade
                        self.trades.append({
                            'entry_time': feature_df.index[i-1],
                            'exit_time': current_time,
                            'position': f"{position}_partial",
                            'entry_price': entry_price,
                            'exit_price': partial_profit_threshold,
                            'pnl': pnl,
                            'pnl_pct': pnl / (partial_size * initial_balance) * 100,
                            'exit_reason': 'partial_profit',
                            'position_size': partial_size
                        })
            
            elif position == "short":
                # Check if high price hit stop loss
                if high_price >= stop_loss:
                    # Calculate profit/loss
                    pnl = ((entry_price / stop_loss) - 1) * position_size * initial_balance
                    # Apply commission
                    pnl -= (position_size * initial_balance * self.commission * 2)  # Entry and exit
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': feature_df.index[i-1],
                        'exit_time': current_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl': pnl,
                        'pnl_pct': pnl / (position_size * initial_balance) * 100,
                        'exit_reason': 'sl_hit',
                        'position_size': position_size
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0
                
                # Check if low price hit take profit
                elif low_price <= take_profit:
                    # Calculate profit/loss
                    pnl = ((entry_price / take_profit) - 1) * position_size * initial_balance
                    # Apply commission
                    pnl -= (position_size * initial_balance * self.commission * 2)  # Entry and exit
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': feature_df.index[i-1],
                        'exit_time': current_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl': pnl,
                        'pnl_pct': pnl / (position_size * initial_balance) * 100,
                        'exit_reason': 'tp_hit',
                        'position_size': position_size
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0
                
                # Check for partial profit taking (simulation)
                elif position_size > 0 and i not in self.partial_exits:
                    # If price has moved 40% toward take profit
                    tp_distance = entry_price - take_profit
                    partial_profit_threshold = entry_price - (tp_distance * 0.4)
                    
                    if low_price <= partial_profit_threshold and partial_profit_threshold < entry_price * 0.98:
                        # Take profit on half the position
                        partial_size = position_size / 2
                        pnl = ((entry_price / partial_profit_threshold) - 1) * partial_size * initial_balance
                        # Apply commission
                        pnl -= (partial_size * initial_balance * self.commission)  # Just exit commission
                        
                        # Update balance and position size
                        balance += pnl
                        position_size -= partial_size
                        
                        # Record partial exit
                        self.partial_exits[i] = True
                        
                        # Record trade
                        self.trades.append({
                            'entry_time': feature_df.index[i-1],
                            'exit_time': current_time,
                            'position': f"{position}_partial",
                            'entry_price': entry_price,
                            'exit_price': partial_profit_threshold,
                            'pnl': pnl,
                            'pnl_pct': pnl / (partial_size * initial_balance) * 100,
                            'exit_reason': 'partial_profit',
                            'position_size': partial_size
                        })
            
            # Get model prediction
            if position is None:
                # Prepare sequence for prediction
                sequence = prepare_sequence_for_prediction(feature_df.iloc[i-SEQ_LENGTH:i])
                
                # Convert to tensor
                x = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
                
                # Get prediction with Monte Carlo Dropout for uncertainty estimation
                with torch.no_grad():
                    # Run multiple predictions with dropout
                    n_samples = 10
                    all_directions = []
                    all_sl = []
                    all_tp = []
                    
                    for _ in range(n_samples):
                        direction_probs, stop_loss_pred, take_profit_pred, duration_pred, confidence_pred, _ = self.model(x, mc_dropout=True)
                        all_directions.append(direction_probs[0].cpu().numpy())
                        all_sl.append(float(stop_loss_pred[0].cpu().numpy()))
                        all_tp.append(float(take_profit_pred[0].cpu().numpy()))
                    
                    # Average predictions
                    avg_direction = np.mean(np.array(all_directions), axis=0)
                    avg_sl = np.mean(all_sl)
                    avg_tp = np.mean(all_tp)
                    
                    # Calculate prediction uncertainty
                    direction_std = np.std(np.array(all_directions), axis=0)
                    uncertainty = np.mean(direction_std)
                    
                    # Get the predicted action
                    action_idx = np.argmax(avg_direction)
                    
                    # ADJUSTED: Lower thresholds for more trades
                    buy_threshold = 0.35  # Lowered from 0.4
                    sell_threshold = 0.35  # Lowered from 0.4
                    uncertainty_threshold = 0.25  # Increased from 0.2
                    
                    # Convert to action string (0: Buy, 1: Hold, 2: Sell)
                    if action_idx == 0 and avg_direction[0] > buy_threshold and uncertainty < uncertainty_threshold:
                        action = "buy"
                        # Calculate dynamic position size based on confidence and uncertainty
                        confidence = avg_direction[0]
                        pos_size = min(0.5 + (confidence * 0.5), 1.0) * (1.0 - uncertainty)
                        pos_size = max(0.2, min(1.0, pos_size))  # Keep between 20-100%
                        
                        # Enter long position
                        position = "long"
                        entry_price = current_price
                        stop_loss = current_price * (1 - avg_sl)
                        take_profit = current_price * (1 + avg_tp)
                        position_size = pos_size
                        
                    elif action_idx == 2 and avg_direction[2] > sell_threshold and uncertainty < uncertainty_threshold:
                        action = "sell"
                        # Calculate dynamic position size based on confidence and uncertainty
                        confidence = avg_direction[2]
                        pos_size = min(0.5 + (confidence * 0.5), 1.0) * (1.0 - uncertainty)
                        pos_size = max(0.2, min(1.0, pos_size))  # Keep between 20-100%
                        
                        # Enter short position
                        position = "short"
                        entry_price = current_price
                        stop_loss = current_price * (1 + avg_sl)
                        take_profit = current_price * (1 - avg_tp)
                        position_size = pos_size
                    
                    # Deduct commission for entry
                    if position:
                        balance -= (position_size * initial_balance * self.commission)
        
        # Close any open position at the end of the backtest
        if position:
            if position == "long":
                pnl = ((current_price / entry_price) - 1) * position_size * initial_balance
            else:  # short
                pnl = ((entry_price / current_price) - 1) * position_size * initial_balance
                
            # Apply commission
            pnl -= (position_size * initial_balance * self.commission)  # Just exit commission
            
            # Update balance
            balance += pnl
            
            # Record trade
            self.trades.append({
                'entry_time': feature_df.index[i-1],
                'exit_time': current_time,
                'position': position,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl / (position_size * initial_balance) * 100,
                'exit_reason': 'backtest_end',
                'position_size': position_size
            })
        
        # Calculate performance metrics
        self.equity_curve = equity_history
        self.calculate_performance_metrics(initial_balance)
        
        return self.performance_metrics
    
    def calculate_performance_metrics(self, initial_balance):
        """
        Calculate performance metrics from backtest results
        """
        if not self.trades:
            self.performance_metrics = {
                'total_trades': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': initial_balance,
                'return_pct': 0
            }
            return
        
        # Calculate metrics from trades
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor (gross profit / gross loss)
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average profit
        avg_profit = sum(t['pnl'] for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_profit_pct = sum(t['pnl_pct'] for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Calculate average trade duration
        avg_duration = np.mean([
            (t['exit_time'] - t['entry_time']).total_seconds() / 3600  # hours
            for t in self.trades
        ]) if self.trades else 0
        
        # Calculate max drawdown
        equity = [e[1] for e in self.equity_curve]
        peak = initial_balance
        drawdown = 0
        max_drawdown = 0
        
        for eq in equity:
            if eq > peak:
                peak = eq
            
            drawdown = (peak - eq) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate final balance
        final_balance = equity[-1] if equity else initial_balance
        return_pct = (final_balance / initial_balance - 1) * 100
        
        # Calculate Sharpe ratio
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365*24) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Store metrics
        self.performance_metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown * 100,  # as percentage
            'sharpe_ratio': sharpe_ratio,
            'final_balance': final_balance,
            'return_pct': return_pct,
            'avg_duration': avg_duration
        }
    
    def generate_report(self):
        """
        Generate a detailed backtest report
        """
        if not hasattr(self, 'performance_metrics'):
            return "No backtest results available. Run backtest first."
        
        # Format metrics
        pm = self.performance_metrics
        
        report = f"========= Backtest Report for {self.symbol} =========\n\n"
        report += f"Total Trades: {pm['total_trades']}\n"
        report += f"Win Rate: {pm['win_rate']*100:.2f}%\n"
        report += f"Profit Factor: {pm['profit_factor']:.2f}\n"
        report += f"Average Profit: ${pm['avg_profit']:.2f} ({pm['avg_profit_pct']:.2f}%)\n"
        report += f"Average Duration: {pm['avg_duration']:.1f} hours\n"
        report += f"Maximum Drawdown: {pm['max_drawdown']:.2f}%\n"
        report += f"Sharpe Ratio: {pm['sharpe_ratio']:.2f}\n"
        report += f"Final Balance: ${pm['final_balance']:.2f}\n"
        report += f"Total Return: {pm['return_pct']:.2f}%\n\n"
        
        # Add trade breakdown
        win_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        loss_trades = sum(1 for t in self.trades if t['pnl'] <= 0)
        
        report += f"Winning Trades: {win_trades}\n"
        report += f"Losing Trades: {loss_trades}\n\n"
        
        # Trade reasons
        sl_hits = sum(1 for t in self.trades if t['exit_reason'] == 'sl_hit')
        tp_hits = sum(1 for t in self.trades if t['exit_reason'] == 'tp_hit')
        partial_profits = sum(1 for t in self.trades if t['exit_reason'] == 'partial_profit')
        
        report += f"Stop Loss Hits: {sl_hits} ({sl_hits/pm['total_trades']*100:.1f}% of trades)\n"
        report += f"Take Profit Hits: {tp_hits} ({tp_hits/pm['total_trades']*100:.1f}% of trades)\n"
        report += f"Partial Profits Taken: {partial_profits}\n\n"
        
        # Position types
        longs = sum(1 for t in self.trades if t['position'] == 'long')
        shorts = sum(1 for t in self.trades if t['position'] == 'short')
        
        report += f"Long Positions: {longs} ({longs/pm['total_trades']*100:.1f}%)\n"
        report += f"Short Positions: {shorts} ({shorts/pm['total_trades']*100:.1f}%)\n\n"
        
        # Best and worst trades
        if self.trades:
            best_trade = max(self.trades, key=lambda x: x['pnl_pct'])
            worst_trade = min(self.trades, key=lambda x: x['pnl_pct'])
            
            report += f"Best Trade: {best_trade['position'].upper()} on {best_trade['entry_time']} for {best_trade['pnl_pct']:.2f}%\n"
            report += f"Worst Trade: {worst_trade['position'].upper()} on {worst_trade['entry_time']} for {worst_trade['pnl_pct']:.2f}%\n"
        
        return report
    
    def plot_equity_curve(self):
        """
        Create an equity curve plot
        """
        if not self.equity_curve:
            return "No equity data available. Run backtest first."
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Extract data
            dates = [e[0] for e in self.equity_curve]
            equity = [e[1] for e in self.equity_curve]
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates, equity, label='Equity Curve')
            
            # Add trade markers
            for trade in self.trades:
                if trade['pnl'] > 0:
                    marker = '^'
                    color = 'green'
                else:
                    marker = 'v'
                    color = 'red'
                
                plt.scatter(trade['exit_time'], trade['exit_price'], marker=marker, color=color, s=50)
            
            # Format
            plt.title(f'Backtest Equity Curve - {self.symbol}')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            
            # Format date axis
            date_format = DateFormatter('%Y-%m-%d')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'backtest_{self.symbol.replace("/", "_")}.png')
            plt.close()
            
            return f'Equity curve saved as backtest_{self.symbol.replace("/", "_")}.png'
        
        except Exception as e:
            return f"Error creating equity curve: {str(e)}"

#################################
# 7. ENHANCED SIGNAL GENERATION
#################################

class EnhancedSignalGenerator:
    """
    Generates trading signals based on model predictions with improved risk management
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.models = {}
        self.positions = {symbol: None for symbol in symbols}  # None: no position, 'long': long, 'short': short
        self.entry_prices = {symbol: 0 for symbol in symbols}
        self.stop_losses = {symbol: 0 for symbol in symbols}
        self.take_profits = {symbol: 0 for symbol in symbols}
        self.position_times = {symbol: None for symbol in symbols}
        self.max_position_duration = {symbol: 24 for symbol in symbols}  # Default max duration in hours
        self.trade_ids = {symbol: None for symbol in symbols}
        self.last_signals = {symbol: {'action': None, 'time': None} for symbol in symbols}
        self.position_sizes = {symbol: 0 for symbol in symbols}
        self.partial_profits_taken = {symbol: False for symbol in symbols}
        self.consecutive_losses = {symbol: 0 for symbol in symbols}
        self.cooldown_until = {symbol: None for symbol in symbols}
        
        # Create signal dispatcher
        self.signal_dispatcher = ImprovedSignalDispatcher()
        
        # Initialize trade feedback system
        self.feedback_system = TradeFeedbackSystem()
        
        # Load models for each symbol
        for symbol in symbols:
            self.load_model(symbol)
    
    def load_model(self, symbol):
        """
        Load a model for a specific symbol with ensemble support
        """
        try:
            # First try to load ensemble model if enabled
            if MODEL_CONFIG['use_ensemble']:
                ensemble = load_ensemble_model(symbol)
                if ensemble:
                    self.models[symbol] = ensemble
                    logging.info(f"Ensemble model for {symbol} loaded successfully.")
                    return True
            
            # Fall back to single model if ensemble not available
            safe_symbol = symbol.replace('/', '_')
            model_path = f'models/{safe_symbol}_model.pth'
            model_info_path = f'models/{safe_symbol}_model_info.json'
            
            # Check if model exists
            if not os.path.exists(model_path):
                logging.warning(f"No model found for {symbol}. The model needs to be trained first.")
                return False
            
            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            input_size = model_info.get('input_size', 100)
            
            # Initialize model
            model = TemporalFusionModel(input_size=input_size).to(DEVICE)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            # Store model
            self.models[symbol] = model
            
            logging.info(f"Model for {symbol} loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to load model for {symbol}: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol, current_price, confidence, volatility):
        """
        Dynamically size positions based on volatility and model confidence
        """
        # Get the base position size from risk config
        base_size = RISK_CONFIG['max_position_size']
        
        # Check if in cooldown period due to consecutive losses
        if self.cooldown_until.get(symbol) and datetime.now() < self.cooldown_until[symbol]:
            logging.info(f"{symbol} is in cooldown period until {self.cooldown_until[symbol]}")
            return 0.0  # No trading during cooldown
        
        # Scale by confidence (0.5-1.0)
        confidence_factor = 0.5 + (confidence * 0.5)
        
        # Inverse volatility scaling: trade smaller in high volatility
        volatility_factor = 1.0 / (1.0 + (volatility * 5))
        
        # Scale by market regime
        regime_factor = 1.0
        recent_regime = self.feedback_system.get_recent_market_regime(symbol)
        if recent_regime == 'volatile':
            regime_factor = 0.7
        elif recent_regime == 'trending':
            regime_factor = 1.2
        
        # Adjust for consecutive losses
        loss_factor = max(0.5, 1.0 - (self.consecutive_losses.get(symbol, 0) * 0.2))
        
        # Calculate final position size
        position_size = base_size * confidence_factor * volatility_factor * regime_factor * loss_factor
        
        # Cap position size
        position_size = min(position_size, RISK_CONFIG['max_position_size'])
        
        return position_size
    
    def update_trailing_stop(self, symbol, current_price):
        """
        Update trailing stop-loss based on price movement
        """
        if symbol not in self.positions or self.positions[symbol] is None:
            return
        
        position = self.positions[symbol]
        entry_price = self.entry_prices[symbol]
        current_stop = self.stop_losses[symbol]
        
        # Calculate price movement percentage
        if position == "long":
            price_movement_pct = (current_price - entry_price) / entry_price
            # Update stop loss if price has moved favorably
            if price_movement_pct > 0.03:  # 3% profit
                # Move stop loss to breakeven
                new_stop = max(current_stop, entry_price)
                
                # If price has moved more than 5%, trail with half the movement above 5%
                if price_movement_pct > 0.05:
                    new_stop = max(new_stop, entry_price * (1 + (price_movement_pct - 0.05) / 2))
                
                # Update if better than current stop
                if new_stop > current_stop:
                    self.stop_losses[symbol] = new_stop
                    logging.info(f"{symbol} trailing stop updated to {new_stop:.2f}")
                    
                    # Update in signal dispatcher
                    self.signal_dispatcher.update_trade_info(symbol, sl=new_stop)
        
        elif position == "short":
            price_movement_pct = (entry_price - current_price) / entry_price
            # Update stop loss if price has moved favorably
            if price_movement_pct > 0.03:  # 3% profit
                # Move stop loss to breakeven
                new_stop = min(current_stop, entry_price)
                
                # If price has moved more than 5%, trail with half the movement above 5%
                if price_movement_pct > 0.05:
                    new_stop = min(new_stop, entry_price * (1 - (price_movement_pct - 0.05) / 2))
                
                # Update if better than current stop
                if new_stop < current_stop:
                    self.stop_losses[symbol] = new_stop
                    logging.info(f"{symbol} trailing stop updated to {new_stop:.2f}")
                    
                    # Update in signal dispatcher
                    self.signal_dispatcher.update_trade_info(symbol, sl=new_stop)
    
    def implement_partial_profits(self, symbol, current_price):
        """
        Take partial profits as price moves favorably
        """
        if symbol not in self.positions or self.positions[symbol] is None:
            return False
        
        position = self.positions[symbol]
        entry_price = self.entry_prices[symbol]
        take_profit = self.take_profits[symbol]
        
        # If we've already taken partial profits, skip
        if self.partial_profits_taken[symbol]:
            return False
        
        # Calculate price movement percentage
        if position == "long":
            price_movement_pct = (current_price - entry_price) / entry_price
            
            # Take 50% profit at 40% of the way to target
            tp_distance = (take_profit - entry_price) / entry_price
            partial_profit_threshold = tp_distance * 0.4
            
            if price_movement_pct >= partial_profit_threshold and price_movement_pct > 0.02:
                # Calculate profit
                profit_percent = price_movement_pct * 100
                position_size = self.position_sizes[symbol]
                
                # Only take partial profit if position size is significant
                if position_size > 0.2:
                    # Calculate size to exit (half the position)
                    exit_size = position_size / 2
                    
                    # Send partial exit signal
                    self.signal_dispatcher.send_webhook(
                        symbol,
                        "exit_buy",
                        current_price,
                        size=exit_size,
                        per=f"{profit_percent:.2f}%",
                        reason="partial_profit"
                    )
                    
                    # Mark partial profits as taken
                    self.partial_profits_taken[symbol] = True
                    
                    # Update position size
                    self.position_sizes[symbol] = position_size - exit_size
                    
                    # Update take profit for remaining position
                    new_tp = take_profit * 1.1  # Move TP 10% higher
                    self.take_profits[symbol] = new_tp
                    
                    # Update in signal dispatcher
                    self.signal_dispatcher.update_trade_info(
                        symbol, 
                        position_size=self.position_sizes[symbol],
                        tp=new_tp,
                        partial_exit=True
                    )
                    
                    logging.info(f"{symbol} PARTIAL PROFIT TAKEN at {current_price} ({profit_percent:.2f}%)")
                    return True
        
        elif position == "short":
            price_movement_pct = (entry_price - current_price) / entry_price
            
            # Take 50% profit at 40% of the way to target
            tp_distance = (entry_price - take_profit) / entry_price
            partial_profit_threshold = tp_distance * 0.4
            
            if price_movement_pct >= partial_profit_threshold and price_movement_pct > 0.02:
                # Calculate profit
                profit_percent = price_movement_pct * 100
                position_size = self.position_sizes[symbol]
                
                # Only take partial profit if position size is significant
                if position_size > 0.2:
                    # Calculate size to exit (half the position)
                    exit_size = position_size / 2
                    
                    # Send partial exit signal
                    self.signal_dispatcher.send_webhook(
                        symbol,
                        "exit_sell",
                        current_price,
                        size=exit_size,
                        per=f"{profit_percent:.2f}%",
                        reason="partial_profit"
                    )
                    
                    # Mark partial profits as taken
                    self.partial_profits_taken[symbol] = True
                    
                    # Update position size
                    self.position_sizes[symbol] = position_size - exit_size
                    
                    # Update take profit for remaining position
                    new_tp = take_profit * 0.9  # Move TP 10% lower
                    self.take_profits[symbol] = new_tp
                    
                    # Update in signal dispatcher
                    self.signal_dispatcher.update_trade_info(
                        symbol, 
                        position_size=self.position_sizes[symbol],
                        tp=new_tp,
                        partial_exit=True
                    )
                    
                    logging.info(f"{symbol} PARTIAL PROFIT TAKEN at {current_price} ({profit_percent:.2f}%)")
                    return True
        
        return False
    
    def should_signal(self, symbol, action, current_time):
        """
        Check if we should send a signal (avoid duplicate signals)
        """
        last_signal = self.last_signals[symbol]
        
        # If no previous signal, or different action, we should signal
        if last_signal['action'] is None or last_signal['action'] != action:
            return True
        
        # If same action, check time difference (minimum 1 hour between same signals)
        if last_signal['time'] is not None:
            time_diff = (current_time - last_signal['time']).total_seconds() / 3600
            return time_diff >= 1  # Reduced from 2h to 1h for more responsive trading
        
        return True
    
    def check_position_age(self, symbol, current_time):
        """
        Check if a position should be closed based on time duration
        """
        if self.positions[symbol] is not None and self.position_times[symbol] is not None:
            time_diff = (current_time - self.position_times[symbol]).total_seconds() / 3600
            max_duration = self.max_position_duration[symbol]
            
            if time_diff > max_duration:
                logging.info(f"{symbol} position duration {time_diff:.1f}h exceeded maximum {max_duration:.1f}h")
                return True
        
        return False
    
    def get_current_price(self, symbol):
        """
        Get the current price for a symbol
        """
        try:
            # Use ccxt to get current price
            exchange = ccxt.binance({'enableRateLimit': True})
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def close_position(self, symbol, current_price, reason="manual"):
        """
        Close an existing position and record the outcome
        """
        position = self.positions[symbol]
        if position is None:
            logging.warning(f"No active position to close for {symbol}")
            return False
        
        try:
            # Calculate profit/loss
            entry_price = self.entry_prices[symbol]
            position_size = self.position_sizes[symbol]
            
            if position == "long":
                profit_loss = current_price - entry_price
                profit_percent = (profit_loss / entry_price) * 100
                action = "exit_buy"
            else:  # short
                profit_loss = entry_price - current_price
                profit_percent = (profit_loss / entry_price) * 100
                action = "exit_sell"
            
            # Send exit signal
            self.signal_dispatcher.send_webhook(
                symbol,
                action,
                current_price,
                size=position_size,
                per=f"{profit_percent:.2f}%",
                sl=self.stop_losses[symbol],
                tp=self.take_profits[symbol],
                reason=reason
            )
            
            # Record trade outcome
            if self.trade_ids[symbol] is not None:
                self.feedback_system.record_trade_exit(
                    self.trade_ids[symbol], 
                    current_price,
                    exit_reason=reason
                )
            
            # Update consecutive losses tracking
            if profit_percent <= 0:
                self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
                
                # Check if we need to enter cooldown
                if self.consecutive_losses[symbol] >= RISK_CONFIG['consecutive_loss_limit']:
                    logging.warning(f"{symbol} has {self.consecutive_losses[symbol]} consecutive losses. Entering cooldown.")
                    # Set cooldown for 24 hours
                    self.cooldown_until[symbol] = datetime.now() + timedelta(hours=24)
            else:
                # Reset consecutive losses on profitable trade
                self.consecutive_losses[symbol] = 0
                
                # Clear any cooldown
                self.cooldown_until[symbol] = None
            
            # Reset position tracking
            self.positions[symbol] = None
            self.position_times[symbol] = None
            self.trade_ids[symbol] = None
            self.position_sizes[symbol] = 0
            self.partial_profits_taken[symbol] = False
            self.signal_dispatcher.remove_trade(symbol)
            
            logging.info(f"{symbol} {position.upper()} position closed at {current_price} ({reason})")
            logging.info(f"Profit/Loss: {profit_percent:.2f}%")
            
            return True
            
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal for the given symbol and data with enhanced risk management
        """
        try:
            # Check if model exists
            if symbol not in self.models:
                logging.error(f"No model found for {symbol}")
                return {'symbol': symbol, 'action': 'error', 'error': 'No model loaded'}
            
            model = self.models[symbol]
            
            # Prepare the sequence for prediction
            sequence = prepare_sequence_for_prediction(data)
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            
            # Get prediction with Monte Carlo dropout if enabled
            with torch.no_grad():
                use_mc_dropout = MODEL_CONFIG['use_mc_dropout']
                
                if isinstance(model, EnsembleModel):
                    # Use ensemble prediction
                    direction_probs, stop_loss, take_profit, duration, confidence, last_features = model(x, mc_dropout=use_mc_dropout)
                    
                    # For ensemble, convert tensor outputs to scalars
                    confidence_value = float(confidence.cpu().numpy()[0])
                    sl_value = float(stop_loss.cpu().numpy()[0])
                    tp_value = float(take_profit.cpu().numpy()[0])
                    duration_value = float(duration.cpu().numpy()[0])
                    direction_probs_np = direction_probs.cpu().numpy()[0]
                    
                else:
                    # Single model prediction
                    direction_probs, stop_loss, take_profit, duration, confidence, last_features = model(x, mc_dropout=use_mc_dropout)
                    
                    # Get scalar values
                    confidence_value = float(confidence[0].cpu().numpy())
                    sl_value = float(stop_loss[0].cpu().numpy())
                    tp_value = float(take_profit[0].cpu().numpy())
                    duration_value = float(duration[0].cpu().numpy())
                    direction_probs_np = direction_probs[0].cpu().numpy()
                
                # Get the predicted action
                action_idx = torch.argmax(direction_probs[0]).item()
                
                # Convert to action string (0: Buy, 1: Hold, 2: Sell)
                if action_idx == 0:
                    action = "buy"
                elif action_idx == 2:
                    action = "sell"
                else:
                    action = "hold"
                
                # Get current price
                current_price = float(data['close'].iloc[-1])
                
                # Get current time
                current_time = datetime.now()
                
                # Get recent market regime
                market_regime, volatility, _ = detect_market_regime(data)
                self.feedback_system.save_market_regime(symbol, market_regime, volatility, direction_probs_np[action_idx])
                
                # Get current position status from our tracker
                current_position = self.positions[symbol]
                
                # Check if in cooldown
                if self.cooldown_until.get(symbol) and current_time < self.cooldown_until[symbol]:
                    logging.info(f"{symbol} is in cooldown until {self.cooldown_until[symbol]}")
                    return {
                        'symbol': symbol,
                        'action': 'hold',
                        'direction_probs': direction_probs_np.tolist(),
                        'current_price': current_price,
                        'position': current_position,
                        'cooldown': True,
                        'cooldown_until': self.cooldown_until[symbol].strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                # Apply trailing stop logic if enabled and in a position
                if RISK_CONFIG['use_trailing_stop'] and current_position is not None:
                    self.update_trailing_stop(symbol, current_price)
                
                # Apply partial profit taking if enabled and in a position
                if RISK_CONFIG['use_partial_profits'] and current_position is not None:
                    self.implement_partial_profits(symbol, current_price)
                
                # Check if existing position should be closed due to time
                if current_position is not None:
                    if self.check_position_age(symbol, current_time):
                        self.close_position(symbol, current_price, "time_exit")
                        current_position = None
                
                # Check if we should send a signal
                confidence_threshold = MODEL_CONFIG['confidence_threshold']
                if action != "hold" and direction_probs_np[action_idx] >= confidence_threshold and self.should_signal(symbol, action, current_time):
                    # Update last signal time
                    self.last_signals[symbol] = {'action': action, 'time': current_time}
                    
                    # Calculate position size based on confidence and volatility
                    position_size = self.calculate_position_size(
                        symbol, 
                        current_price, 
                        confidence_value, 
                        volatility
                    )
                    
                    # Skip if position size is too small
                    if position_size < 0.1:
                        logging.info(f"Skipping {symbol} {action} signal due to small position size ({position_size:.2f})")
                        return {
                            'symbol': symbol,
                            'action': 'hold',
                            'direction_probs': direction_probs_np.tolist(),
                            'confidence': confidence_value,
                            'position_size': position_size,
                            'current_price': current_price,
                            'position': current_position,
                            'market_regime': market_regime
                        }
                    
                    # Handle position entry
                    if action == "buy" and current_position != "long":
                        # If in a short position, exit first
                        if current_position == "short":
                            self.close_position(symbol, current_price, "reversal")
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = current_price * (1 - sl_value)
                        take_profit_price = current_price * (1 + tp_value)
                        
                        # Send buy signal
                        self.signal_dispatcher.send_webhook(
                            symbol, 
                            "buy", 
                            current_price, 
                            sl=f"{stop_loss_price:.2f}",
                            tp=f"{take_profit_price:.2f}",
                            duration=f"{duration_value:.1f}h",
                            confidence=f"{confidence_value:.2f}",
                            size=f"{position_size:.2f}"
                        )
                        
                        # Record trade entry
                        trade_id = self.feedback_system.record_trade_entry(
                            symbol, 
                            "buy", 
                            current_price, 
                            sl_value, 
                            tp_value, 
                            duration_value,
                            position_size
                        )
                        
                        # Update position tracking
                        self.positions[symbol] = "long"
                        self.entry_prices[symbol] = current_price
                        self.stop_losses[symbol] = stop_loss_price
                        self.take_profits[symbol] = take_profit_price
                        self.position_times[symbol] = current_time
                        self.max_position_duration[symbol] = duration_value
                        self.trade_ids[symbol] = trade_id
                        self.position_sizes[symbol] = position_size
                        self.partial_profits_taken[symbol] = False
                        
                        # Register with signal dispatcher
                        self.signal_dispatcher.register_trade(
                            symbol, 
                            trade_id, 
                            "buy", 
                            current_price, 
                            stop_loss_price, 
                            take_profit_price, 
                            duration_value,
                            position_size
                        )
                        
                        logging.info(f"{symbol} BUY signal at {current_price} | "
                                   f"SL: {stop_loss_price:.2f} | "
                                   f"TP: {take_profit_price:.2f} | "
                                   f"Duration: {duration_value:.1f}h | "
                                   f"Size: {position_size:.2f}")
                    
                    elif action == "sell" and current_position != "short":
                        # If in a long position, exit first
                        if current_position == "long":
                            self.close_position(symbol, current_price, "reversal")
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = current_price * (1 + sl_value)
                        take_profit_price = current_price * (1 - tp_value)
                        
                        # Send sell signal
                        self.signal_dispatcher.send_webhook(
                            symbol, 
                            "sell", 
                            current_price, 
                            sl=f"{stop_loss_price:.2f}",
                            tp=f"{take_profit_price:.2f}",
                            duration=f"{duration_value:.1f}h",
                            confidence=f"{confidence_value:.2f}",
                            size=f"{position_size:.2f}"
                        )
                        
                        # Record trade entry
                        trade_id = self.feedback_system.record_trade_entry(
                            symbol, 
                            "sell", 
                            current_price, 
                            sl_value, 
                            tp_value, 
                            duration_value,
                            position_size
                        )
                        
                        # Update position tracking
                        self.positions[symbol] = "short"
                        self.entry_prices[symbol] = current_price
                        self.stop_losses[symbol] = stop_loss_price
                        self.take_profits[symbol] = take_profit_price
                        self.position_times[symbol] = current_time
                        self.max_position_duration[symbol] = duration_value
                        self.trade_ids[symbol] = trade_id
                        self.position_sizes[symbol] = position_size
                        self.partial_profits_taken[symbol] = False
                        
                        # Register with signal dispatcher
                        self.signal_dispatcher.register_trade(
                            symbol, 
                            trade_id, 
                            "sell", 
                            current_price, 
                            stop_loss_price, 
                            take_profit_price, 
                            duration_value,
                            position_size
                        )
                        
                        logging.info(f"{symbol} SELL signal at {current_price} | "
                                   f"SL: {stop_loss_price:.2f} | "
                                   f"TP: {take_profit_price:.2f} | "
                                   f"Duration: {duration_value:.1f}h | "
                                   f"Size: {position_size:.2f}")
                
                # Check for stop loss or take profit if in a position
                if current_position == "long":
                    # Check stop loss
                    if current_price <= self.stop_losses[symbol]:
                        self.close_position(symbol, current_price, "sl_hit")
                    
                    # Check take profit
                    elif current_price >= self.take_profits[symbol]:
                        self.close_position(symbol, current_price, "tp_hit")
                        
                        # Record successful feature data for model improvement
                        self.feedback_system.save_model_feedback(
                            symbol, 
                            sequence, 
                            0,  # 0 = buy was successful
                            confidence=1.5  # Higher confidence for successful trades
                        )
                
                elif current_position == "short":
                    # Check stop loss
                    if current_price >= self.stop_losses[symbol]:
                        self.close_position(symbol, current_price, "sl_hit")
                    
                    # Check take profit
                    elif current_price <= self.take_profits[symbol]:
                        self.close_position(symbol, current_price, "tp_hit")
                        
                        # Record successful feature data for model improvement
                        self.feedback_system.save_model_feedback(
                            symbol, 
                            sequence, 
                            2,  # 2 = sell was successful
                            confidence=1.5  # Higher confidence for successful trades
                        )
                
                return {
                    'symbol': symbol,
                    'action': action,
                    'direction_probs': direction_probs_np.tolist(),
                    'stop_loss': sl_value,
                    'take_profit': tp_value,
                    'duration': duration_value,
                    'confidence': confidence_value,
                    'current_price': current_price,
                    'position': self.positions[symbol],
                    'market_regime': market_regime,
                    'volatility': volatility
                }
                
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'action': 'error',
                'error': str(e)
            }
    
    def get_performance_report(self, symbol, days=30):
        """
        Get a performance report for a symbol
        """
        metrics = self.feedback_system.get_performance_metrics(symbol, days)
        if not metrics:
            return f"No trade data available for {symbol}"
        
        total_trades, winning_trades, avg_profit, avg_win, avg_loss, avg_duration = metrics
        
        if total_trades == 0:
            return f"No trades recorded for {symbol} in the last {days} days"
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        report = f"Performance Report for {symbol} (Last {days} days):\n"
        report += f"Total Trades: {total_trades}\n"
        report += f"Win Rate: {win_rate:.2f}%\n"
        report += f"Average Profit: {avg_profit:.2f}%\n"
        report += f"Average Win: {avg_win:.2f}%\n"
        report += f"Average Loss: {avg_loss:.2f}%\n"
        report += f"Average Trade Duration: {avg_duration:.1f} hours\n"
        
        # Add market regime info
        regime = self.feedback_system.get_recent_market_regime(symbol)
        report += f"Recent Market Regime: {regime}\n"
        
        # Add consecutive losses
        report += f"Consecutive Losses: {self.consecutive_losses.get(symbol, 0)}\n"
        
        return report
    
    def shutdown(self):
        """Clean up resources"""
        self.signal_dispatcher.shutdown()
        self.feedback_system.close()

#################################
# 8. MAIN EXECUTION
#################################

def run_walk_forward_optimization(symbol, hyperparams, lookback_days=180):
    """
    Run walk-forward optimization to find optimal hyperparameters
    """
    # Declare global variables at the beginning
    global SEQ_LENGTH, HIDDEN_SIZE, DROPOUT, BATCH_SIZE, LEARNING_RATE
    
    # Fetch data
    df = fetch_crypto_data(symbol, timeframe='1h', lookback_days=lookback_days)
    if df is None or len(df) < SEQ_LENGTH + 50:
        logging.error(f"Insufficient data for {symbol}")
        return None
    
    # Calculate features
    feature_df = calculate_features(df)
    
    # Determine test windows (walk-forward)
    window_size = 14 * 24  # 14 days of hourly data
    test_windows = []
    
    for i in range(0, len(feature_df) - window_size, window_size // 2):  # 50% overlap
        train_end = i + int(window_size * 0.8)
        test_start = train_end
        test_end = min(test_start + window_size // 2, len(feature_df))
        
        # Ensure minimum size
        if test_end - test_start < 24:  # At least 24 hours of testing
            continue
        
        test_windows.append((i, train_end, test_end))
    
    # Set up hyperparameter grid
    param_combinations = []
    
    # Generate all combinations
    for seq_len in hyperparams.get('seq_length', [SEQ_LENGTH]):
        for hidden_size in hyperparams.get('hidden_size', [HIDDEN_SIZE]):
            for dropout in hyperparams.get('dropout', [DROPOUT]):
                for learning_rate in hyperparams.get('learning_rate', [LEARNING_RATE]):
                    for batch_size in hyperparams.get('batch_size', [BATCH_SIZE]):
                        param_combinations.append({
                            'seq_length': seq_len,
                            'hidden_size': hidden_size,
                            'dropout': dropout,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        })
    
    # Store results
    results = []
    
    # Run optimization
    for params in param_combinations:
        logging.info(f"Testing hyperparameters: {params}")
        
        window_results = []
        
        for window_idx, (train_start, train_end, test_end) in enumerate(test_windows):
            try:
                # Get training and test data
                train_data = feature_df.iloc[train_start:train_end]
                test_data = feature_df.iloc[train_end:test_end]
                
                # Set global params (global declaration moved to function start)
                SEQ_LENGTH = params['seq_length']
                HIDDEN_SIZE = params['hidden_size']
                DROPOUT = params['dropout']
                BATCH_SIZE = params['batch_size']
                LEARNING_RATE = params['learning_rate']
                
                
                # Train model
                model = train_model(symbol, train_data=train_data)
                
                if model is None:
                    continue
                
                # Backtest on test data
                backtest = BacktestEngine(symbol, model, commission=0.001)
                backtest_results = backtest.run_backtest(
                    start_date=test_data.index[0],
                    end_date=test_data.index[-1]
                )
                
                if backtest_results:
                    window_results.append(backtest_results)
                    
                    logging.info(f"Window {window_idx+1}/{len(test_windows)}: "
                               f"Return: {backtest_results['return_pct']:.2f}%, "
                               f"Sharpe: {backtest_results['sharpe_ratio']:.2f}")
            
            except Exception as e:
                logging.error(f"Error in walk-forward window {window_idx}: {str(e)}")
        
        # Calculate average metrics across windows
        if window_results:
            avg_return = np.mean([r['return_pct'] for r in window_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in window_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in window_results])
            avg_win_rate = np.mean([r['win_rate'] for r in window_results])
            
            results.append({
                'params': params,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'avg_win_rate': avg_win_rate,
                'window_results': window_results
            })
            
            logging.info(f"Results for params {params}: "
                       f"Avg Return: {avg_return:.2f}%, "
                       f"Avg Sharpe: {avg_sharpe:.2f}")
    
    # Sort by Sharpe Ratio
    results.sort(key=lambda x: x['avg_sharpe'], reverse=True)
    
    # Reset global parameters to optimal values
    if results:
        optimal_params = results[0]['params']
        logging.info(f"Optimal parameters found: {optimal_params}")
        
        # global SEQ_LENGTH, HIDDEN_SIZE, DROPOUT, BATCH_SIZE, LEARNING_RATE  # <-- REMOVE THIS LINE TOO
        SEQ_LENGTH = optimal_params['seq_length']
        HIDDEN_SIZE = optimal_params['hidden_size']
        DROPOUT = optimal_params['dropout']
        BATCH_SIZE = optimal_params['batch_size']
        LEARNING_RATE = optimal_params['learning_rate']
        
        # Save optimal parameters
        with open(f'optimal_params_{symbol.replace("/", "_")}.json', 'w') as f:
            json.dump(optimal_params, f, indent=2)
    
    return results

def train_models(symbols=['BTC/USDT', 'SOL/USDT'], feedback_system=None, use_ensemble=False):
    """
    Train models for each symbol
    """
    for symbol in symbols:
        logging.info(f"==== Training model for {symbol} ====")
        
        if use_ensemble:
            logging.info(f"Training ensemble model with {MODEL_CONFIG['num_ensemble_models']} sub-models")
            train_diverse_models(symbol, num_models=MODEL_CONFIG['num_ensemble_models'])
        else:
            train_model(symbol, feedback_system=feedback_system)

def run_backtests(symbols=['BTC/USDT', 'SOL/USDT'], lookback_days=90):
    """
    Run backtests for all symbols
    """
    results = {}
    
    for symbol in symbols:
        logging.info(f"==== Backtesting {symbol} ====")
        
        # Try to load ensemble model first
        model = load_ensemble_model(symbol)
        
        # Fall back to single model if ensemble not available
        if model is None:
            safe_symbol = symbol.replace('/', '_')
            model_path = f'models/{safe_symbol}_model.pth'
            model_info_path = f'models/{safe_symbol}_model_info.json'
            
            if not os.path.exists(model_path):
                logging.warning(f"No model found for {symbol}. Skipping backtest.")
                continue
            
            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            input_size = model_info.get('input_size', 100)
            
            # Initialize model
            model = TemporalFusionModel(input_size=input_size).to(DEVICE)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        
        # Run backtest
        backtest = BacktestEngine(symbol, model, lookback_days=lookback_days)
        metrics = backtest.run_backtest()
        
        if metrics:
            # Generate and save report
            report = backtest.generate_report()
            with open(f'backtest_report_{symbol.replace("/", "_")}.txt', 'w') as f:
                f.write(report)
            
            # Generate and save equity curve
            curve_result = backtest.plot_equity_curve()
            
            # Store results
            results[symbol] = metrics
            
            logging.info(f"Backtest for {symbol} completed:")
            logging.info(f"  Return: {metrics['return_pct']:.2f}%")
            logging.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            logging.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logging.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            logging.info(f"  Report saved to backtest_report_{symbol.replace('/', '_')}.txt")
        else:
            logging.error(f"Backtest for {symbol} failed")
    
    return results

def run_signal_generation(symbols=['BTC/USDT', 'SOL/USDT'], interval=300):
    """
    Run the signal generator continuously
    """
    logging.info(f"Starting continuous signal generation for symbols: {symbols}")
    
    # Initialize signal generator
    signal_generator = EnhancedSignalGenerator(symbols)
    
    # Count how many models were loaded successfully
    loaded_models = sum(1 for symbol in symbols if symbol in signal_generator.models)
    if loaded_models == 0:
        logging.error("No models were loaded. Please train models first.")
        return
    
    logging.info(f"Signal generator initialized with {loaded_models}/{len(symbols)} models.")
    
    try:
        while True:
            for symbol in symbols:
                try:
                    # Skip symbols with no model
                    if symbol not in signal_generator.models:
                        logging.warning(f"Skipping {symbol} - no model available")
                        continue
                    
                    # Fetch latest data
                    df = fetch_crypto_data(symbol, timeframe='1h', lookback_days=30)
                    
                    if df is not None and len(df) > SEQ_LENGTH:
                        # Calculate features
                        feature_df = calculate_features(df)
                        
                        # Generate signal
                        result = signal_generator.generate_signal(symbol, feature_df)
                        
                        # Log result
                        if result.get('action') != 'error':
                            probs = result['direction_probs']
                            logging.info(f"Signal for {symbol}: {result['action'].upper()} | "
                                       f"Buy: {probs[0]:.4f}, Hold: {probs[1]:.4f}, Sell: {probs[2]:.4f} | "
                                       f"Price: {result['current_price']} | "
                                       f"Regime: {result.get('market_regime', 'unknown')}")
                        else:
                            logging.error(f"Error generating signal for {symbol}: {result.get('error')}")
                    else:
                        logging.warning(f"Insufficient data for {symbol}")
                        
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
            
            # Wait for the next interval
            logging.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("Signal generation stopped by user.")
    except Exception as e:
        logging.error(f"Error in signal generation loop: {str(e)}")
    finally:
        # Clean up resources
        signal_generator.shutdown()

def retrain_with_feedback(symbols=['BTC/USDT', 'SOL/USDT']):
    """
    Retrain models using the feedback data
    """
    logging.info("Retraining models with feedback data...")
    
    # Initialize feedback system
    feedback_system = TradeFeedbackSystem()
    
    # Train models with feedback
    train_models(symbols, feedback_system)
    
    # Close feedback system
    feedback_system.close()

if __name__ == "__main__":
    # Define trading symbols
    symbols = ['BTC/USDT', 'SOL/USDT']
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--mode", choices=["train", "backtest", "run", "optimize", "train_ensemble", "retrain"], 
                        default="run", help="Operation mode")
    parser.add_argument("--symbols", nargs="+", default=symbols, 
                        help="Trading symbols to use")
    parser.add_argument("--interval", type=int, default=300, 
                        help="Signal generation interval in seconds")
    parser.add_argument("--lookback", type=int, default=90, 
                        help="Lookback days for data")
    args = parser.parse_args()
    
    symbols = args.symbols
    
    # Execute based on mode
    if args.mode == "train":
        # Train individual models
        train_models(symbols, use_ensemble=False)
    
    elif args.mode == "train_ensemble":
        # Train ensemble models
        train_models(symbols, use_ensemble=True)
    
    elif args.mode == "backtest":
        # Run backtests
        run_backtests(symbols, args.lookback)
    
    elif args.mode == "optimize":
        # Define hyperparameters to search
        hyperparams = {
            'seq_length': [30, 40, 50],
            'hidden_size': [128, 196, 256],
            'dropout': [0.3, 0.4, 0.5],
            'learning_rate': [1e-4, 2e-4, 5e-4],
            'batch_size': [16, 32, 64]
        }
        
        # Run optimization for each symbol
        for symbol in symbols:
            run_walk_forward_optimization(symbol, hyperparams, lookback_days=args.lookback)
    
    elif args.mode == "retrain":
        # Retrain models with feedback
        retrain_with_feedback(symbols)
    
    else:  # "run" mode
        # Run signal generation
        run_signal_generation(symbols, interval=args.interval)
        