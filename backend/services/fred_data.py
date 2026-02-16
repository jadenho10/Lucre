"""
FRED API data service for fetching historical market data.

Uses Federal Reserve Economic Data (FRED) to get real historical data
for stocks, bonds, and inflation to parameterize Monte Carlo simulations.
"""

import os
from typing import Optional
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FRED API key from environment
FRED_API_KEY = os.getenv("FREDAPI")

# FRED series IDs
FRED_SERIES = {
    "sp500": "SP500",           # S&P 500 Index
    "treasury_10yr": "DGS10",   # 10-Year Treasury Constant Maturity Rate
    "treasury_3mo": "DTB3",     # 3-Month Treasury Bill Rate
    "cpi": "CPIAUCSL",          # Consumer Price Index for All Urban Consumers
    "inflation_expectation": "T10YIE",  # 10-Year Breakeven Inflation Rate
}

# Fallback parameters if FRED API fails
FALLBACK_PARAMETERS = {
    "stocks": {"mean": 0.07, "std": 0.18},
    "bonds": {"mean": 0.02, "std": 0.06},
    "inflation": {"mean": 0.025, "std": 0.012},
    "correlation": 0.1,
}


class FredDataService:
    """
    Service for fetching and processing financial data from FRED.
    
    Calculates historical return statistics to parameterize
    Monte Carlo simulations with real market data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with FRED API key."""
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError("FRED API key not found. Set FREDAPI environment variable.")
        self.fred = Fred(api_key=self.api_key)
        self._cache = {}
    
    def _fetch_series(self, series_id: str, start_date: str = "1970-01-01") -> Optional[np.ndarray]:
        """Fetch a FRED series and return as numpy array."""
        try:
            data = self.fred.get_series(series_id, observation_start=start_date)
            # Drop NaN values
            data = data.dropna()
            return data
        except Exception as e:
            print(f"Warning: Failed to fetch {series_id}: {e}")
            return None
    
    def get_stock_returns(self, years: int = 30) -> dict:
        """
        Calculate annualized stock return statistics from S&P 500 data.
        
        Returns:
            Dict with 'mean' and 'std' of annual returns
        """
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        sp500 = self._fetch_series(FRED_SERIES["sp500"], start_date)
        
        if sp500 is None or len(sp500) < 252:  # Need at least 1 year of daily data
            print("Using fallback stock parameters")
            return FALLBACK_PARAMETERS["stocks"]
        
        # Calculate daily returns
        daily_returns = sp500.pct_change().dropna()
        
        # Annualize statistics (252 trading days)
        annual_mean = daily_returns.mean() * 252
        annual_std = daily_returns.std() * np.sqrt(252)
        
        # Adjust for inflation to get real returns (approximate)
        inflation = self.get_inflation_stats(years)
        real_mean = annual_mean - inflation["mean"]
        
        return {
            "mean": float(real_mean),
            "std": float(annual_std),
            "nominal_mean": float(annual_mean),
        }
    
    def get_bond_returns(self, years: int = 30) -> dict:
        """
        Calculate bond return statistics from Treasury yield data.
        
        Uses 10-year Treasury yields as proxy for bond returns.
        
        Returns:
            Dict with 'mean' and 'std' of annual returns
        """
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        treasury = self._fetch_series(FRED_SERIES["treasury_10yr"], start_date)
        
        if treasury is None or len(treasury) < 252:
            print("Using fallback bond parameters")
            return FALLBACK_PARAMETERS["bonds"]
        
        # Treasury yields are in percent, convert to decimal
        yields = treasury / 100
        
        # For simplicity, use yield level as expected return proxy
        # In reality, bond returns depend on yield changes and duration
        annual_mean = yields.mean()
        
        # Calculate return volatility from yield changes
        # Duration approximation: 1% yield change ≈ 7% price change for 10yr
        yield_changes = yields.diff().dropna()
        duration = 7.0
        price_std = yield_changes.std() * duration * np.sqrt(252)
        
        # Adjust for inflation to get real returns
        inflation = self.get_inflation_stats(years)
        real_mean = annual_mean - inflation["mean"]
        
        return {
            "mean": float(real_mean),
            "std": float(price_std),
            "nominal_mean": float(annual_mean),
        }
    
    def get_inflation_stats(self, years: int = 30) -> dict:
        """
        Calculate inflation statistics from CPI data.
        
        Returns:
            Dict with 'mean' and 'std' of annual inflation
        """
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        cpi = self._fetch_series(FRED_SERIES["cpi"], start_date)
        
        if cpi is None or len(cpi) < 12:  # Need at least 1 year of monthly data
            print("Using fallback inflation parameters")
            return FALLBACK_PARAMETERS["inflation"]
        
        # Calculate year-over-year inflation rate
        annual_inflation = cpi.pct_change(periods=12).dropna()
        
        return {
            "mean": float(annual_inflation.mean()),
            "std": float(annual_inflation.std()),
        }
    
    def get_stock_bond_correlation(self, years: int = 30) -> float:
        """
        Calculate correlation between stock and bond returns.
        
        Returns:
            Correlation coefficient
        """
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        sp500 = self._fetch_series(FRED_SERIES["sp500"], start_date)
        treasury = self._fetch_series(FRED_SERIES["treasury_10yr"], start_date)
        
        if sp500 is None or treasury is None:
            return FALLBACK_PARAMETERS["correlation"]
        
        # Align dates
        import pandas as pd
        df = pd.DataFrame({"stocks": sp500, "bonds": treasury}).dropna()
        
        if len(df) < 252:
            return FALLBACK_PARAMETERS["correlation"]
        
        # Calculate returns
        stock_returns = df["stocks"].pct_change().dropna()
        # Bond returns inversely related to yield changes (simplified)
        bond_returns = -df["bonds"].diff().dropna() * 7 / 100  # Duration effect
        
        # Align and calculate correlation
        aligned = pd.DataFrame({
            "stocks": stock_returns,
            "bonds": bond_returns
        }).dropna()
        
        correlation = aligned["stocks"].corr(aligned["bonds"])
        
        return float(correlation) if not np.isnan(correlation) else FALLBACK_PARAMETERS["correlation"]
    
    @lru_cache(maxsize=1)
    def get_all_parameters(self, lookback_years: int = 30) -> dict:
        """
        Get all market parameters for Monte Carlo simulation.
        
        Returns:
            Dict with stocks, bonds, inflation parameters and correlation
        """
        print(f"Fetching market data from FRED (last {lookback_years} years)...")
        
        stocks = self.get_stock_returns(lookback_years)
        bonds = self.get_bond_returns(lookback_years)
        inflation = self.get_inflation_stats(lookback_years)
        correlation = self.get_stock_bond_correlation(lookback_years)
        
        parameters = {
            "stocks": stocks,
            "bonds": bonds,
            "inflation": inflation,
            "correlation": correlation,
        }
        
        print(f"Stock returns: {stocks['mean']:.2%} ± {stocks['std']:.2%}")
        print(f"Bond returns: {bonds['mean']:.2%} ± {bonds['std']:.2%}")
        print(f"Inflation: {inflation['mean']:.2%} ± {inflation['std']:.2%}")
        print(f"Stock-Bond correlation: {correlation:.2f}")
        
        return parameters


# Singleton instance
_fred_service: Optional[FredDataService] = None


def get_fred_service() -> FredDataService:
    """Get or create the FRED data service singleton."""
    global _fred_service
    if _fred_service is None:
        _fred_service = FredDataService()
    return _fred_service


def get_market_parameters(lookback_years: int = 30) -> dict:
    """
    Convenience function to get market parameters.
    
    Falls back to hardcoded parameters if FRED API fails.
    """
    try:
        service = get_fred_service()
        return service.get_all_parameters(lookback_years)
    except Exception as e:
        print(f"Warning: FRED API failed ({e}), using fallback parameters")
        return FALLBACK_PARAMETERS
