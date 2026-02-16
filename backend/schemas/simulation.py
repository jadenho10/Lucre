from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"      # 30% stocks, 70% bonds
    MODERATE_CONSERVATIVE = "moderate_conservative"  # 40% stocks, 60% bonds
    MODERATE = "moderate"              # 60% stocks, 40% bonds
    MODERATE_AGGRESSIVE = "moderate_aggressive"  # 70% stocks, 30% bonds
    AGGRESSIVE = "aggressive"          # 80% stocks, 20% bonds
    VERY_AGGRESSIVE = "very_aggressive"  # 90% stocks, 10% bonds


class SimulationInput(BaseModel):
    """Input parameters for Monte Carlo retirement simulation"""
    
    # Personal info
    current_age: int = Field(..., ge=18, le=100, description="Current age in years")
    retirement_age: int = Field(..., ge=18, le=100, description="Target retirement age")
    life_expectancy: int = Field(default=95, ge=50, le=120, description="Planning horizon end age")
    
    # Financial inputs
    current_portfolio: float = Field(..., ge=0, description="Current portfolio value in dollars")
    annual_contribution: float = Field(default=0, ge=0, description="Annual savings contribution (pre-retirement)")
    annual_spending: float = Field(..., gt=0, description="Annual spending in retirement (in today's dollars)")
    
    # Risk/allocation
    risk_profile: RiskProfile = Field(default=RiskProfile.MODERATE, description="Investment risk profile")
    stock_allocation: Optional[float] = Field(default=None, ge=0, le=1, description="Custom stock allocation (overrides risk_profile)")
    
    # Simulation parameters
    num_simulations: int = Field(default=10000, ge=1000, le=100000, description="Number of Monte Carlo simulations")
    inflation_adjusted: bool = Field(default=True, description="Whether spending is inflation-adjusted")
    
    # Social Security (optional)
    social_security_age: Optional[int] = Field(default=None, ge=62, le=70, description="Age to start Social Security")
    social_security_amount: Optional[float] = Field(default=None, ge=0, description="Annual Social Security benefit")


class PercentileOutcome(BaseModel):
    """Portfolio values at specific percentiles"""
    p10: float = Field(..., description="10th percentile (pessimistic)")
    p25: float = Field(..., description="25th percentile")
    p50: float = Field(..., description="50th percentile (median)")
    p75: float = Field(..., description="75th percentile")
    p90: float = Field(..., description="90th percentile (optimistic)")


class YearlyProjection(BaseModel):
    """Projection data for a specific year"""
    age: int
    year: int
    percentiles: PercentileOutcome
    probability_solvent: float = Field(..., ge=0, le=1, description="Probability portfolio is still positive")


class SimulationResult(BaseModel):
    """Complete Monte Carlo simulation results"""
    
    # Core metrics
    success_probability: float = Field(..., ge=0, le=1, description="Probability of not running out of money")
    median_final_portfolio: float = Field(..., description="Median ending portfolio value")
    
    # Percentile outcomes at retirement
    portfolio_at_retirement: PercentileOutcome
    
    # Percentile outcomes at end of plan
    portfolio_at_end: PercentileOutcome
    
    # Risk metrics
    median_depletion_age: Optional[int] = Field(default=None, description="Median age of portfolio depletion (if applicable)")
    probability_depleted_by_85: float = Field(..., ge=0, le=1)
    probability_depleted_by_90: float = Field(..., ge=0, le=1)
    
    # Year-by-year projections for charting
    yearly_projections: list[YearlyProjection]
    
    # Input echo for reference
    inputs: SimulationInput
    
    # Metadata
    simulations_run: int
    computation_time_ms: float


class ScenarioComparison(BaseModel):
    """Compare multiple scenarios"""
    base_scenario: SimulationResult
    alternative_scenarios: list[SimulationResult]
