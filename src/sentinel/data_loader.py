# This file focus on fetching data through API calls and other data sources
#Here is exactly how the data flow works in this architecture:
#Fetching: When view.py calls loader.get_macro_data(), the script fetches the data from the internet (FRED/Yahoo).
#Caching (The "Save" Mechanism): The line @st.cache_data tells Streamlit to store the result in a hidden internal cache (in memory).
#Next time you click a button: It doesn't re-download the data; it reads it instantly from this memory cache.
#If you close the browser tab: The data is eventually cleared from memory.
#Returning: The data is passed directly to the variable in view.py (e.g., macro_data = ...) for immediate use.


import streamlit as st
from fredapi import Fred
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- CONFIGURATION & SECURITY ---
FRED_KEY = None
try:
    FRED_KEY = st.secrets["api_keys"]["fred_api_key"]
except (FileNotFoundError, KeyError):
    pass

class FinancialDataLoader:
    def __init__(self):
        if FRED_KEY:
            self.fred = Fred(api_key=FRED_KEY)
        else:
            self.fred = None

    def get_my_portfolio_definition(self):
        """Returns user's specific portfolio mix."""
        # [Paste your huge dictionary here - omitted for brevity in this snippet]
        # Just ensure 'BIL' is mapped correctly as in your previous prompt
        return {
            "Money Market": {"ticker": "BIL", "weight": 0.0575},
            "VTI": {"ticker": "VTI", "weight": 0.60}, # Example truncated
            "VXUS": {"ticker": "VXUS", "weight": 0.3425} 
        }

    @st.cache_data(ttl=86400)
    def get_macro_data(_self, years=10):
        """
        Fetches Macro Data (Mortgage Rates, Treasury).
        
        Args:
            years: Number of years of historical data to fetch (default: 10)
            
        Returns:
            DataFrame with columns: '30y Mortgage', '15y Mortgage', '10y Treasury'
        """
        if not _self.fred: 
            return pd.DataFrame()
        try:
            start_date = (datetime.now() - timedelta(days=365*(years+1))).strftime('%Y-%m-%d')
            # Fetch mortgage rates (30-year and 15-year)
            mortgage_30 = _self.fred.get_series('MORTGAGE30US', observation_start=start_date)
            mortgage_15 = _self.fred.get_series('MORTGAGE15US', observation_start=start_date)
            treasury_10 = _self.fred.get_series('DGS10', observation_start=start_date)
            
            # Combine into DataFrame
            df = pd.DataFrame({
                '30y Mortgage': mortgage_30,
                '15y Mortgage': mortgage_15,
                '10y Treasury': treasury_10
            })
            
            # Forward fill to handle different frequencies (weekly vs daily)
            df = df.ffill().dropna()
            
            # Return last N years of data
            end_date = datetime.now()
            start_date_filter = end_date - timedelta(days=365*years)
            df = df[df.index >= start_date_filter]
            
            return df
        except Exception as e:
            st.error(f"FRED Error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_portfolio_stats(_self, portfolio_dict=None, period="10y"):
        """
        Calculates Custom Portfolio Risk/Return.
        
        Args:
            portfolio_dict: Either a dict like {"VTI": 0.6, "VXUS": 0.4} 
                           OR a dict like {"Name": {"ticker": "VTI", "weight": 0.6}, ...}
                           If None, uses get_my_portfolio_definition()
        """
        if portfolio_dict is None: 
            portfolio_dict = _self.get_my_portfolio_definition()
        
        # Handle two possible portfolio_dict formats:
        # Format 1: {"VTI": 0.6, "VXUS": 0.4} - ticker: weight mapping
        # Format 2: {"Name": {"ticker": "VTI", "weight": 0.6}, ...} - nested structure
        ticker_to_weight = {}
        
        # Check if values are dictionaries (Format 2) or numbers (Format 1)
        first_value = list(portfolio_dict.values())[0] if portfolio_dict else None
        
        if isinstance(first_value, dict) and "ticker" in first_value and "weight" in first_value:
            # Format 2: Convert nested structure to flat ticker:weight mapping
            # Consolidate weights by ticker (in case same ticker appears multiple times)
            for name, data in portfolio_dict.items():
                ticker = data.get("ticker")
                weight = data.get("weight", 0.0)
                if ticker and weight > 0:
                    ticker_to_weight[ticker] = ticker_to_weight.get(ticker, 0.0) + weight
        else:
            # Format 1: Already in ticker:weight format
            ticker_to_weight = portfolio_dict
        
        # Extract tickers and weights
        tickers = list(ticker_to_weight.keys())
        weights = list(ticker_to_weight.values())
        
        # Normalize weights to sum to 1.0 (in case of rounding errors)
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            raise ValueError("Portfolio weights sum to zero")
        
        try:
            data = yf.download(tickers, period=period, progress=False)['Close']
            if data.empty: 
                raise ValueError("No data returned from Yahoo Finance")
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Handle single ticker vs multiple tickers
            if len(tickers) == 1:
                # Single ticker: portfolio return = ticker return
                port_ret = returns.iloc[:, 0] if len(returns.columns) > 0 else returns
            else:
                # Multiple tickers: align weights with columns (yfinance returns columns alphabetically)
                # Create a weight list aligned with the data columns
                aligned_weights = []
                for col in returns.columns:
                    # Find weight for this ticker
                    ticker_weight = ticker_to_weight.get(col, 0.0)
                    aligned_weights.append(ticker_weight)
                
                # Renormalize (in case some tickers failed to download)
                aligned_total = sum(aligned_weights)
                if aligned_total > 0:
                    aligned_weights = [w / aligned_total for w in aligned_weights]
                else:
                    raise ValueError("All tickers failed to download")
                
                # Calculate portfolio returns: weighted average of individual returns
                port_ret = returns.dot(aligned_weights)

            # Annualize statistics (252 trading days per year)
            return {
                "name": "Personal Portfolio",
                "annual_return": port_ret.mean() * 252,
                "annual_volatility": port_ret.std() * (252**0.5)
            }
        except Exception as e:
            st.error(f"Error calculating portfolio stats: {e}")
            return {"annual_return": 0.07, "annual_volatility": 0.15}

    @st.cache_data(ttl=3600)
    def get_benchmark_stats(_self, period="10y"):
        """Fetches S&P 500 stats for comparison."""
        try:
            spy = yf.Ticker("^GSPC") # S&P 500 Index
            hist = spy.history(period=period)
            returns = hist['Close'].pct_change().dropna()
            return {
                "name": "S&P 500 Benchmark",
                "annual_return": returns.mean() * 252,
                "annual_volatility": returns.std() * (252**0.5)
            }
        except:
             return {"annual_return": 0.10, "annual_volatility": 0.16}

    @st.cache_data(ttl=3600)
    def get_sp500_returns_series(_self, years=10):
        """
        Fetches S&P 500 cumulative returns time series.
        
        Args:
            years: Number of years of historical data (default: 10)
            
        Returns:
            Series with date index and cumulative return values (normalized to start at 1.0)
        """
        try:
            period = f"{years}y"
            spy = yf.Ticker("^GSPC")  # S&P 500 Index
            hist = spy.history(period=period)['Close']
            
            # Calculate cumulative returns (normalized to start at 1.0)
            cumulative_returns = (1 + hist.pct_change().fillna(0)).cumprod()
            
            return cumulative_returns
        except Exception as e:
            st.error(f"Error fetching S&P 500 data: {e}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600)
    def get_portfolio_returns_series(_self, portfolio_dict=None, years=10):
        """
        Fetches portfolio cumulative returns time series.
        
        Args:
            portfolio_dict: Portfolio definition (if None, uses get_my_portfolio_definition())
            years: Number of years of historical data (default: 10)
            
        Returns:
            Series with date index and cumulative return values (normalized to start at 1.0)
        """
        if portfolio_dict is None:
            portfolio_dict = _self.get_my_portfolio_definition()
        
        # Handle portfolio format (same logic as get_portfolio_stats)
        ticker_to_weight = {}
        first_value = list(portfolio_dict.values())[0] if portfolio_dict else None
        
        if isinstance(first_value, dict) and "ticker" in first_value and "weight" in first_value:
            for name, data in portfolio_dict.items():
                ticker = data.get("ticker")
                weight = data.get("weight", 0.0)
                if ticker and weight > 0:
                    ticker_to_weight[ticker] = ticker_to_weight.get(ticker, 0.0) + weight
        else:
            ticker_to_weight = portfolio_dict
        
        try:
            period = f"{years}y"
            tickers = list(ticker_to_weight.keys())
            weights = list(ticker_to_weight.values())
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                return pd.Series(dtype=float)
            
            # Download historical data
            data = yf.download(tickers, period=period, progress=False)['Close']
            if data.empty:
                return pd.Series(dtype=float)
            
            # Calculate portfolio returns
            returns = data.pct_change().fillna(0)
            
            if len(tickers) == 1:
                port_returns = returns.iloc[:, 0]
            else:
                # Align weights with columns
                aligned_weights = [ticker_to_weight.get(col, 0.0) for col in returns.columns]
                aligned_total = sum(aligned_weights)
                if aligned_total > 0:
                    aligned_weights = [w / aligned_total for w in aligned_weights]
                    port_returns = returns.dot(aligned_weights)
                else:
                    return pd.Series(dtype=float)
            
            # Calculate cumulative returns (normalized to start at 1.0)
            cumulative_returns = (1 + port_returns).cumprod()
            
            return cumulative_returns
        except Exception as e:
            st.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)