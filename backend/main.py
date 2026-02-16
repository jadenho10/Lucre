from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.simulation import router as simulation_router
from routes.data import router as data_router

app = FastAPI(
    title="Lucre - Financial Planning API",
    description="AI-powered probabilistic retirement planning with Monte Carlo simulation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simulation_router, prefix="/api")
app.include_router(data_router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "Lucre Financial Planning API",
        "version": "1.0.0",
        "endpoints": {
            "simulation": "/api/simulation/run",
            "quick_check": "/api/simulation/quick-check",
            "compare": "/api/simulation/compare",
            "risk_profiles": "/api/simulation/risk-profiles",
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
