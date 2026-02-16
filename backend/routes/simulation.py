from fastapi import APIRouter, HTTPException
from typing import Optional

from schemas.simulation import (
    SimulationInput,
    SimulationResult,
    ScenarioComparison,
    RiskProfile,
)
from services.monte_carlo import run_simulation, MonteCarloEngine

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/run", response_model=SimulationResult)
async def run_monte_carlo_simulation(inputs: SimulationInput) -> SimulationResult:
    """
    Run a Monte Carlo retirement simulation.
    
    This endpoint simulates thousands of possible market futures to compute
    the probability of achieving retirement goals. Results include:
    
    - **success_probability**: Chance of not running out of money
    - **portfolio_at_retirement**: Portfolio value distribution at retirement age
    - **portfolio_at_end**: Portfolio value distribution at life expectancy
    - **yearly_projections**: Year-by-year percentile data for charting
    - **depletion metrics**: Risk of running out of money at various ages
    """
    try:
        result = run_simulation(inputs)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/compare", response_model=ScenarioComparison)
async def compare_scenarios(
    base: SimulationInput,
    alternatives: list[SimulationInput],
) -> ScenarioComparison:
    """
    Compare multiple retirement scenarios.
    
    Run the base scenario and all alternative scenarios, returning results
    for easy comparison. Useful for "what-if" analysis like:
    
    - What if I retire at 62 vs 65 vs 67?
    - What if I save $500 more per month?
    - What if I reduce spending by 10%?
    """
    try:
        # Use same seed for fair comparison
        engine = MonteCarloEngine(seed=42)
        
        base_result = engine.run_simulation(base)
        alt_results = [engine.run_simulation(alt) for alt in alternatives]
        
        return ScenarioComparison(
            base_scenario=base_result,
            alternative_scenarios=alt_results,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/quick-check")
async def quick_retirement_check(
    current_age: int,
    retirement_age: int,
    current_portfolio: float,
    annual_contribution: float,
    annual_spending: float,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    life_expectancy: int = 95,
) -> dict:
    """
    Quick retirement viability check with minimal parameters.
    
    Returns a simplified result for fast feedback. For detailed results,
    use the /simulation/run endpoint.
    """
    try:
        inputs = SimulationInput(
            current_age=current_age,
            retirement_age=retirement_age,
            life_expectancy=life_expectancy,
            current_portfolio=current_portfolio,
            annual_contribution=annual_contribution,
            annual_spending=annual_spending,
            risk_profile=risk_profile,
            num_simulations=5000,  # Fewer for speed
        )
        
        result = run_simulation(inputs)
        
        return {
            "success_probability": result.success_probability,
            "success_rating": _get_success_rating(result.success_probability),
            "median_portfolio_at_retirement": result.portfolio_at_retirement.p50,
            "median_final_portfolio": result.median_final_portfolio,
            "recommendation": _get_quick_recommendation(result),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_success_rating(probability: float) -> str:
    """Convert probability to human-readable rating."""
    if probability >= 0.95:
        return "Excellent"
    elif probability >= 0.85:
        return "Good"
    elif probability >= 0.75:
        return "Fair"
    elif probability >= 0.60:
        return "Needs Attention"
    else:
        return "At Risk"


def _get_quick_recommendation(result: SimulationResult) -> str:
    """Generate a quick recommendation based on results."""
    prob = result.success_probability
    
    if prob >= 0.90:
        return "Your retirement plan looks solid. Consider whether you could retire earlier or spend more."
    elif prob >= 0.75:
        return "Good foundation, but consider increasing savings or adjusting retirement timeline for more security."
    elif prob >= 0.60:
        return "Your plan has some risk. Consider saving more, working longer, or reducing planned spending."
    else:
        return "Significant adjustments needed. Explore retiring later, saving more, or reducing retirement spending."


@router.get("/risk-profiles")
async def get_risk_profiles() -> dict:
    """
    Get available risk profiles and their allocations.
    
    Returns the mapping of risk profiles to stock/bond allocations.
    """
    from services.monte_carlo import RISK_ALLOCATIONS
    
    profiles = {}
    for profile, stock_pct in RISK_ALLOCATIONS.items():
        profiles[profile.value] = {
            "name": profile.value.replace("_", " ").title(),
            "stock_allocation": stock_pct,
            "bond_allocation": 1 - stock_pct,
            "description": _get_profile_description(profile),
        }
    
    return {"profiles": profiles}


@router.get("/market-parameters")
async def get_market_parameters(lookback_years: int = 30) -> dict:
    """
    Get current market parameters derived from FRED economic data.
    
    Returns real-time market statistics used in Monte Carlo simulations:
    - Stock return mean and standard deviation (from S&P 500)
    - Bond return mean and standard deviation (from Treasury yields)
    - Inflation mean and standard deviation (from CPI)
    - Stock-bond correlation
    
    Parameters are calculated from historical FRED data over the specified
    lookback period.
    """
    try:
        from services.fred_data import get_market_parameters as fetch_params
        params = fetch_params(lookback_years)
        
        return {
            "lookback_years": lookback_years,
            "parameters": {
                "stocks": {
                    "annual_return": params["stocks"]["mean"],
                    "volatility": params["stocks"]["std"],
                    "description": "Real (inflation-adjusted) S&P 500 returns",
                },
                "bonds": {
                    "annual_return": params["bonds"]["mean"],
                    "volatility": params["bonds"]["std"],
                    "description": "Real (inflation-adjusted) 10-year Treasury returns",
                },
                "inflation": {
                    "annual_rate": params["inflation"]["mean"],
                    "volatility": params["inflation"]["std"],
                    "description": "CPI-based inflation rate",
                },
                "correlation": {
                    "stock_bond": params["correlation"],
                    "description": "Correlation between stock and bond returns",
                },
            },
            "source": "Federal Reserve Economic Data (FRED)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


def _get_profile_description(profile: RiskProfile) -> str:
    """Get description for a risk profile."""
    descriptions = {
        RiskProfile.CONSERVATIVE: "Lower risk, focused on capital preservation. Best for those near or in retirement.",
        RiskProfile.MODERATE_CONSERVATIVE: "Slightly higher returns with modest risk. Good for conservative investors with longer horizons.",
        RiskProfile.MODERATE: "Balanced approach between growth and stability. Suitable for most long-term investors.",
        RiskProfile.MODERATE_AGGRESSIVE: "Growth-focused with meaningful risk tolerance. Good for investors with 10+ year horizons.",
        RiskProfile.AGGRESSIVE: "High growth potential with significant volatility. Best for young investors with 20+ year horizons.",
        RiskProfile.VERY_AGGRESSIVE: "Maximum growth focus. Only suitable for investors comfortable with major fluctuations.",
    }
    return descriptions.get(profile, "")
