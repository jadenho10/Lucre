import datetime
import numpy as np
from typing import Optional
import time

from schemas.simulation import (
    SimulationInput,
    SimulationResult,
    PercentileOutcome,
    YearlyProjection,
    RiskProfile,
)
from services.fred_data import get_market_parameters


# Default fallback parameters (real returns, inflation-adjusted)
# Used if FRED API is unavailable
DEFAULT_ASSET_PARAMETERS = {
    "stocks": {
        "mean": 0.07,      # 7% real return
        "std": 0.18,       # 18% standard deviation
    },
    "bonds": {
        "mean": 0.02,      # 2% real return
        "std": 0.06,       # 6% standard deviation
    },
    "inflation": {
        "mean": 0.025,     # 2.5% average inflation
        "std": 0.012,      # 1.2% standard deviation
    },
    "correlation": 0.1,    # Stock-bond correlation
}

# Risk profile to allocation mapping
RISK_ALLOCATIONS = {
    RiskProfile.CONSERVATIVE: 0.30,
    RiskProfile.MODERATE_CONSERVATIVE: 0.40,
    RiskProfile.MODERATE: 0.60,
    RiskProfile.MODERATE_AGGRESSIVE: 0.70,
    RiskProfile.AGGRESSIVE: 0.80,
    RiskProfile.VERY_AGGRESSIVE: 0.90,
}


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for retirement planning.
    
    Simulates thousands of possible market futures using correlated
    stock/bond returns and inflation to compute probability of
    retirement success.
    
    Uses real market data from FRED API when available.
    """
    
    def __init__(self, seed: Optional[int] = None, use_fred_data: bool = True, lookback_years: int = 30):
        """
        Initialize the Monte Carlo engine.
        
        Args:
            seed: Random seed for reproducibility
            use_fred_data: Whether to fetch real market data from FRED API
            lookback_years: Years of historical data to use for parameters
        """
        self.rng = np.random.default_rng(seed)
        self.use_fred_data = use_fred_data
        self.lookback_years = lookback_years
        self._asset_parameters = None
    
    @property
    def asset_parameters(self) -> dict:
        """Get asset parameters, fetching from FRED if needed."""
        if self._asset_parameters is None:
            if self.use_fred_data:
                try:
                    self._asset_parameters = get_market_parameters(self.lookback_years)
                except Exception as e:
                    print(f"FRED API failed: {e}. Using default parameters.")
                    self._asset_parameters = DEFAULT_ASSET_PARAMETERS
            else:
                self._asset_parameters = DEFAULT_ASSET_PARAMETERS
        return self._asset_parameters
    
    def _get_stock_allocation(self, inputs: SimulationInput) -> float:
        """Get stock allocation from custom value or risk profile."""
        if inputs.stock_allocation is not None:
            return inputs.stock_allocation
        return RISK_ALLOCATIONS[inputs.risk_profile]
    
    def _generate_returns(
        self,
        num_simulations: int,
        num_years: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self.asset_parameters
        
        # Generate correlated normal random variables
        # Using Cholesky decomposition for correlation
        correlation = params["correlation"]
        correlation_matrix = np.array([
            [1.0, correlation],
            [correlation, 1.0]
        ])
        cholesky = np.linalg.cholesky(correlation_matrix)
        
        # Generate uncorrelated standard normal samples
        z = self.rng.standard_normal((num_simulations, num_years, 2))
        
        # Apply correlation
        correlated = np.einsum('ijk,lk->ijl', z, cholesky)
        
        # Transform to actual returns using FRED-derived parameters
        stock_returns = (
            params["stocks"]["mean"] +
            params["stocks"]["std"] * correlated[:, :, 0]
        )
        bond_returns = (
            params["bonds"]["mean"] +
            params["bonds"]["std"] * correlated[:, :, 1]
        )
        
        inflation = (
            params["inflation"]["mean"] +
            params["inflation"]["std"] * 
            self.rng.standard_normal((num_simulations, num_years))
        )
        
        return stock_returns, bond_returns, inflation
    
    def _simulate_portfolios(
        self,
        inputs: SimulationInput,
        stock_returns: np.ndarray,
        bond_returns: np.ndarray,
        inflation: np.ndarray,
    ) -> np.ndarray:
        
        num_simulations = inputs.num_simulations
        years_to_retirement = inputs.retirement_age - inputs.current_age
        years_in_retirement = inputs.life_expectancy - inputs.retirement_age
        total_years = years_to_retirement + years_in_retirement
        
        stock_allocation = self._get_stock_allocation(inputs)
        bond_allocation = 1 - stock_allocation
        
        # Initialize portfolio array (includes starting value at year 0)
        portfolios = np.zeros((num_simulations, total_years + 1))
        portfolios[:, 0] = inputs.current_portfolio
        
        # Track cumulative inflation for spending adjustments
        cumulative_inflation = np.ones((num_simulations,))
        
        for year in range(total_years):
            # Calculate portfolio return for this year
            portfolio_return = (
                stock_allocation * stock_returns[:, year] +
                bond_allocation * bond_returns[:, year]
            )
            
            # Update cumulative inflation
            cumulative_inflation *= (1 + inflation[:, year])
            
            # Get current portfolio value
            current_value = portfolios[:, year]
            
            if year < years_to_retirement:
                # Accumulation phase: grow portfolio and add contributions
                # Contributions grow with inflation
                if inputs.inflation_adjusted:
                    contribution = inputs.annual_contribution * cumulative_inflation
                else:
                    contribution = inputs.annual_contribution
                
                new_value = current_value * (1 + portfolio_return) + contribution
            else:
                # Distribution phase: grow portfolio and subtract spending
                if inputs.inflation_adjusted:
                    spending = inputs.annual_spending * cumulative_inflation
                else:
                    spending = inputs.annual_spending
                
                # Add Social Security if applicable
                ss_income = 0
                if (inputs.social_security_age is not None and 
                    inputs.social_security_amount is not None):
                    current_age = inputs.current_age + year
                    if current_age >= inputs.social_security_age:
                        if inputs.inflation_adjusted:
                            # Social Security has COLA adjustments
                            ss_income = inputs.social_security_amount * cumulative_inflation
                        else:
                            ss_income = inputs.social_security_amount
                
                net_withdrawal = spending - ss_income
                new_value = current_value * (1 + portfolio_return) - net_withdrawal
            
            # Portfolio can't go below zero (depleted)
            portfolios[:, year + 1] = np.maximum(new_value, 0)
        
        return portfolios
    
    def _calculate_percentiles(self, values: np.ndarray) -> PercentileOutcome:
        """Calculate percentile statistics for a set of values."""
        return PercentileOutcome(
            p10=float(np.percentile(values, 10)),
            p25=float(np.percentile(values, 25)),
            p50=float(np.percentile(values, 50)),
            p75=float(np.percentile(values, 75)),
            p90=float(np.percentile(values, 90)),
        )
    
    def _calculate_yearly_projections(
        self,
        portfolios: np.ndarray,
        inputs: SimulationInput,
    ) -> list[YearlyProjection]:
        """Generate year-by-year projection data for charts."""
        projections = []
        current_year = datetime.datetime.now().year
        
        for year_idx in range(portfolios.shape[1]):
            age = inputs.current_age + year_idx
            year = current_year + year_idx
            
            values = portfolios[:, year_idx]
            percentiles = self._calculate_percentiles(values)
            probability_solvent = float(np.mean(values > 0))
            
            projections.append(YearlyProjection(
                age=age,
                year=year,
                percentiles=percentiles,
                probability_solvent=probability_solvent,
            ))
        
        return projections
    
    def _calculate_depletion_metrics(
        self,
        portfolios: np.ndarray,
        inputs: SimulationInput,
    ) -> tuple[Optional[int], float, float]:
        """
        Calculate portfolio depletion metrics.
        
        Returns:
            (median_depletion_age, prob_depleted_by_85, prob_depleted_by_90)
        """
        num_simulations = portfolios.shape[0]
        
        # Find depletion year for each simulation
        depleted_mask = portfolios <= 0
        
        # Find first year of depletion for each path
        depletion_years = np.argmax(depleted_mask, axis=1)
        
        # If never depleted, argmax returns 0, need to handle this
        never_depleted = ~np.any(depleted_mask, axis=1)
        depletion_years[never_depleted] = portfolios.shape[1]  # Set to beyond horizon
        
        # Calculate median depletion age (only for paths that deplete)
        depleted_paths = ~never_depleted
        if np.any(depleted_paths):
            depletion_ages = inputs.current_age + depletion_years[depleted_paths]
            median_depletion_age = int(np.median(depletion_ages))
        else:
            median_depletion_age = None
        
        # Calculate probability of depletion by age 85 and 90
        year_at_85 = 85 - inputs.current_age
        year_at_90 = 90 - inputs.current_age
        
        if year_at_85 < portfolios.shape[1]:
            prob_depleted_85 = float(np.mean(portfolios[:, year_at_85] <= 0))
        else:
            prob_depleted_85 = float(np.mean(~never_depleted))
        
        if year_at_90 < portfolios.shape[1]:
            prob_depleted_90 = float(np.mean(portfolios[:, year_at_90] <= 0))
        else:
            prob_depleted_90 = float(np.mean(~never_depleted))
        
        return median_depletion_age, prob_depleted_85, prob_depleted_90
    
    def run_simulation(self, inputs: SimulationInput) -> SimulationResult:
        """
        Run a complete Monte Carlo retirement simulation.
        
        Args:
            inputs: SimulationInput with all user parameters
            
        Returns:
            SimulationResult with probability of success and all metrics
        """
        start_time = time.time()
        
        # Validate inputs
        if inputs.retirement_age <= inputs.current_age:
            raise ValueError("Retirement age must be greater than current age")
        if inputs.life_expectancy <= inputs.retirement_age:
            raise ValueError("Life expectancy must be greater than retirement age")
        
        # Calculate simulation dimensions
        years_to_retirement = inputs.retirement_age - inputs.current_age
        years_in_retirement = inputs.life_expectancy - inputs.retirement_age
        total_years = years_to_retirement + years_in_retirement
        
        # Generate market returns
        stock_returns, bond_returns, inflation = self._generate_returns(
            inputs.num_simulations,
            total_years,
        )
        
        # Run portfolio simulations
        portfolios = self._simulate_portfolios(
            inputs,
            stock_returns,
            bond_returns,
            inflation,
        )
        
        # Calculate success probability (portfolio > 0 at end)
        final_portfolios = portfolios[:, -1]
        success_probability = float(np.mean(final_portfolios > 0))
        
        # Calculate portfolio at retirement
        retirement_idx = years_to_retirement
        portfolio_at_retirement = self._calculate_percentiles(portfolios[:, retirement_idx])
        
        # Calculate portfolio at end
        portfolio_at_end = self._calculate_percentiles(final_portfolios)
        
        # Calculate depletion metrics
        median_depletion_age, prob_85, prob_90 = self._calculate_depletion_metrics(
            portfolios, inputs
        )
        
        # Generate yearly projections
        yearly_projections = self._calculate_yearly_projections(portfolios, inputs)
        
        computation_time = (time.time() - start_time) * 1000
        
        return SimulationResult(
            success_probability=success_probability,
            median_final_portfolio=float(np.median(final_portfolios)),
            portfolio_at_retirement=portfolio_at_retirement,
            portfolio_at_end=portfolio_at_end,
            median_depletion_age=median_depletion_age,
            probability_depleted_by_85=prob_85,
            probability_depleted_by_90=prob_90,
            yearly_projections=yearly_projections,
            inputs=inputs,
            simulations_run=inputs.num_simulations,
            computation_time_ms=computation_time,
        )


# Singleton instance for use across the application
engine = MonteCarloEngine()


def run_simulation(inputs: SimulationInput) -> SimulationResult:
    """Convenience function to run simulation with default engine."""
    return engine.run_simulation(inputs)
