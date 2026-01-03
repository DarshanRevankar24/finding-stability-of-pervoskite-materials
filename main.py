import joblib
import pandas as pd
import numpy as np
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Perovskite Stability Predictor API",
    description="""
    Predict stability of perovskite materials using ML.
    
    **Important Note:** This model was trained on complex multi-element perovskites
    (e.g., Ba2La6Ti8O24, Ba4Pr4Fe8O24) with typical energy_above_hull around 50-150 meV/atom.
    
    Predictions for simple ABX3 perovskites (e.g., BaTiO3, SrTiO3) will appear 
    50-100 meV/atom higher than Materials Project values due to the training data distribution.
    
    For accurate simple perovskite predictions, retrain with ABX3 perovskite data.
    """,
    version="2.1.0"
)

# --- 1. LOAD MODEL & DATABASES ---
try:
    pack = joblib.load('perovskite_model_pack.pkl')
    model = pack['model']
    feature_names = pack['feature_names']
    feature_means = pack['feature_means']
    a_db = pack['a_site_db']
    b_db = pack['b_site_db']
    logger.info(f"✅ Model loaded: {len(feature_names)} features, {len(a_db)} A-site, {len(b_db)} B-site elements")
except Exception as e:
    logger.error(f"❌ CRITICAL: Could not load model pack: {e}")
    model, feature_names, feature_means, a_db, b_db = None, [], {}, {}, {}

# Load atomic database for additional calculations
try:
    with open('atomic_db.json', 'r') as f:
        atomic_db = json.load(f)
    logger.info(f"✅ Atomic database loaded: {len(atomic_db)} elements")
except Exception as e:
    logger.warning(f"⚠️ Could not load atomic_db.json: {e}")
    atomic_db = {}

# --- 2. REQUEST/RESPONSE MODELS ---
class MaterialRequest(BaseModel):
    formula: str
    include_details: Optional[bool] = False
    
    @validator('formula')
    def validate_formula(cls, v):
        if not v or not v.strip():
            raise ValueError("Formula cannot be empty")
        # Check for valid chemical formula pattern
        if not re.match(r'^[A-Z][a-zA-Z0-9\s]*$', v.strip()):
            raise ValueError("Invalid chemical formula format")
        return v.strip()

class StabilityMetrics(BaseModel):
    goldschmidt_tolerance: Optional[float] = None
    octahedral_factor: Optional[float] = None
    formation_energy_estimate: Optional[float] = None

class PredictionResponse(BaseModel):
    formula: str
    energy_above_hull: float
    stability_class: str
    confidence: str
    elements_found: List[str]
    elements_missing: List[str]
    stability_metrics: Optional[StabilityMetrics] = None
    recommendations: Optional[List[str]] = None

# --- 3. UTILITY FUNCTIONS ---
def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse chemical formula into element-count dictionary.
    Handles formats like: BaTiO3, CsPbI3, Ba0.5Sr0.5TiO3
    """
    formula = formula.replace(" ", "")
    # Match element + optional decimal/fraction + optional count
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    composition = {}
    for element, count in matches:
        if element:
            count_val = float(count) if count else 1.0
            composition[element] = composition.get(element, 0) + count_val
    
    return composition

def calculate_goldschmidt_tolerance(composition: Dict[str, int]) -> Optional[float]:
    """
    Calculate Goldschmidt tolerance factor: t = (r_A + r_O) / sqrt(2) * (r_B + r_O)
    Stable perovskites typically have 0.8 < t < 1.0
    """
    try:
        if not atomic_db:
            return None
        
        # Identify A and B site elements (excluding O)
        elements = [el for el in composition.keys() if el != 'O']
        if len(elements) < 2:
            return None
        
        # Get radii from all possible locations in atomic_db
        radii = []
        for el in elements:
            radius = None
            # Check in each site category
            for site in ['A_site_elements', 'B_site_elements', 'X_site_elements']:
                if site in atomic_db and el in atomic_db[site]:
                    radius = atomic_db[site][el].get('radius') or atomic_db[site][el].get('ionic_radius')
                    break
            # Fallback: check if element is at root level (old format)
            if radius is None and el in atomic_db:
                radius = atomic_db[el].get('radius')
            
            if radius is not None:
                radii.append((el, radius))
        
        if len(radii) < 2:
            return None
        
        radii.sort(key=lambda x: x[1], reverse=True)
        r_A = radii[0][1]
        r_B = radii[-1][1]
        
        # Get oxygen radius
        r_O = 0.6  # default
        if 'X_site_elements' in atomic_db and 'O' in atomic_db['X_site_elements']:
            r_O = atomic_db['X_site_elements']['O'].get('radius', 0.6)
        elif 'O' in atomic_db:
            r_O = atomic_db['O'].get('radius', 0.6)
        
        tolerance = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
        return round(tolerance, 3)
    except Exception as e:
        logger.warning(f"Could not calculate tolerance factor: {e}")
        return None

def calculate_octahedral_factor(composition: Dict[str, int]) -> Optional[float]:
    """
    Calculate octahedral factor: μ = r_B / r_O
    Stable structures typically have μ > 0.41
    """
    try:
        if not atomic_db:
            return None
        
        elements = [el for el in composition.keys() if el != 'O']
        if not elements:
            return None
        
        # Get radii from all possible locations
        radii = []
        for el in elements:
            radius = None
            # Check in each site category
            for site in ['A_site_elements', 'B_site_elements', 'X_site_elements']:
                if site in atomic_db and el in atomic_db[site]:
                    radius = atomic_db[site][el].get('radius') or atomic_db[site][el].get('ionic_radius')
                    break
            # Fallback: check if element is at root level (old format)
            if radius is None and el in atomic_db:
                radius = atomic_db[el].get('radius')
            
            if radius is not None:
                radii.append(radius)
        
        if not radii:
            return None
        
        r_B = min(radii)
        
        # Get oxygen radius
        r_O = 0.6  # default
        if 'X_site_elements' in atomic_db and 'O' in atomic_db['X_site_elements']:
            r_O = atomic_db['X_site_elements']['O'].get('radius', 0.6)
        elif 'O' in atomic_db:
            r_O = atomic_db['O'].get('radius', 0.6)
        
        octahedral = r_B / r_O
        return round(octahedral, 3)
    except Exception as e:
        logger.warning(f"Could not calculate octahedral factor: {e}")
        return None

def get_dynamic_features(formula: str) -> tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build feature vector from formula with improved error handling.
    Returns: (features_df, found_elements, missing_elements)
    """
    logger.info(f"Processing formula: {formula}")
    
    try:
        composition = parse_formula(formula)
        logger.info(f"Parsed composition: {composition}")
        
        # Initialize with global means as baseline
        final_features = pd.Series(feature_means.copy())
        
        found_elements = []
        missing_elements = []
        total_weight = 0
        weighted_sum_features = pd.Series(0.0, index=feature_names)

        for element, count in composition.items():
            if element == 'O':
                continue  # Skip oxygen
            
            # Search in both databases
            props = None
            site_type = None
            
            if element in a_db:
                props = pd.Series(a_db[element])
                site_type = "A-site"
            elif element in b_db:
                props = pd.Series(b_db[element])
                site_type = "B-site"
            
            if props is not None:
                # Ensure props matches feature_names exactly
                props = props.reindex(feature_names, fill_value=0)
                weighted_sum_features += props * count
                total_weight += count
                found_elements.append(f"{element} ({site_type})")
                logger.info(f"  ✓ {element}: {site_type}, count={count}")
            else:
                missing_elements.append(element)
                logger.warning(f"  ✗ {element}: Not in database")

        # Calculate weighted average if elements found
        if total_weight > 0:
            final_features = weighted_sum_features / total_weight
            logger.info(f"Features calculated from {len(found_elements)} elements")
        else:
            logger.warning("No elements found in database. Using global average.")

        return pd.DataFrame([final_features])[feature_names], found_elements, missing_elements

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return pd.DataFrame([feature_means])[feature_names], [], []

def classify_stability(energy: float) -> tuple[str, str]:
    """
    Classify stability based on energy above hull.
    Note: This model was trained on complex multi-element perovskites.
    Energy values are higher than simple ABX3 perovskites.
    Returns: (stability_class, confidence)
    """
    if energy < 0:
        return "Highly Stable", "High"
    elif energy < 50:
        return "Stable", "High"
    elif energy < 100:
        return "Moderately Stable", "Medium"
    elif energy < 150:
        return "Marginally Stable", "Medium"
    elif energy < 250:
        return "Metastable", "Low"
    else:
        return "Unstable", "High"

def generate_recommendations(energy: float, metrics: StabilityMetrics, 
                            missing_elements: List[str]) -> List[str]:
    """Generate actionable recommendations based on prediction."""
    recommendations = []
    
    # Add model limitation warning for simple perovskites
    recommendations.append(
        "ℹ️ Note: This model was trained on complex multi-element perovskites. "
        "Predictions for simple ABX3 perovskites may appear ~50-100 meV higher than "
        "expected from Materials Project data."
    )
    
    if missing_elements:
        recommendations.append(
            f"⚠️ Elements not in database: {', '.join(missing_elements)}. "
            "Prediction uses global averages - accuracy may be reduced."
        )
    
    if energy > 150:
        recommendations.append(
            "Consider elemental substitution to improve stability. "
            "Try replacing A-site or B-site cations with similar ionic radius."
        )
    elif energy < 100:
        recommendations.append(
            "✓ Material shows reasonable stability for a complex perovskite system."
        )
    
    if metrics.goldschmidt_tolerance:
        t = metrics.goldschmidt_tolerance
        if t < 0.71:
            recommendations.append(
                f"⚠️ Tolerance factor ({t}) is very low. Non-perovskite structure likely. "
                "Try larger A-site cations or smaller B-site cations."
            )
        elif t < 0.8:
            recommendations.append(
                f"Tolerance factor ({t}) is low. Distorted perovskite or ilmenite structure expected. "
                "Consider larger A-site cations for cubic symmetry."
            )
        elif t <= 1.0:
            recommendations.append(
                f"✓ Tolerance factor ({t}) is in optimal range (0.8-1.0) for cubic perovskite"
            )
        elif t <= 1.13:
            recommendations.append(
                f"✓ Tolerance factor ({t}) indicates hexagonal or layered perovskite structure (1.0-1.13)"
            )
        else:
            recommendations.append(
                f"⚠️ Tolerance factor ({t}) is too high (>1.13). Non-perovskite structure likely. "
                "Consider smaller A-site or larger B-site cations."
            )
    
    if metrics.octahedral_factor:
        mu = metrics.octahedral_factor
        if mu < 0.41:
            recommendations.append(
                f"Octahedral factor ({mu}) is low. B-site cation may be too small."
            )
        else:
            recommendations.append(
                f"✓ Octahedral factor ({mu}) is acceptable (>0.41)"
            )
    
    if energy < 25:
        recommendations.append(
            "✓ Material shows good thermodynamic stability. "
            "Consider experimental synthesis."
        )
    elif energy < 100:
        recommendations.append(
            "Material is moderately stable. May be synthesizable under controlled conditions."
        )
    
    return recommendations if recommendations else ["No specific recommendations."]

# --- 4. API ENDPOINTS ---
@app.get("/")
async def root():
    """API health check and info."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "features": len(feature_names),
        "a_site_elements": len(a_db),
        "b_site_elements": len(b_db),
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model": "loaded" if model is not None else "missing",
        "databases": {
            "feature_means": len(feature_means) > 0,
            "a_site_db": len(a_db) > 0,
            "b_site_db": len(b_db) > 0,
            "atomic_db": len(atomic_db) > 0
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MaterialRequest):
    """
    Predict perovskite stability from chemical formula.
    
    Parameters:
    - formula: Chemical formula (e.g., "BaTiO3", "CsPbI3")
    - include_details: Include additional stability metrics and recommendations
    
    Returns detailed prediction with energy above hull and stability classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features
        input_df, found_elements, missing_elements = get_dynamic_features(request.formula)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        energy = round(float(prediction), 3)
        
        # Classify stability
        stability_class, confidence = classify_stability(energy)
        
        # Calculate additional metrics if requested
        metrics = None
        recommendations = None
        
        if request.include_details:
            composition = parse_formula(request.formula)
            metrics = StabilityMetrics(
                goldschmidt_tolerance=calculate_goldschmidt_tolerance(composition),
                octahedral_factor=calculate_octahedral_factor(composition),
                formation_energy_estimate=energy * -0.032  # Rough conversion
            )
            recommendations = generate_recommendations(energy, metrics, missing_elements)
        
        response = PredictionResponse(
            formula=request.formula,
            energy_above_hull=energy,
            stability_class=stability_class,
            confidence=confidence,
            elements_found=found_elements,
            elements_missing=missing_elements,
            stability_metrics=metrics,
            recommendations=recommendations
        )
        
        logger.info(f"Prediction: {request.formula} -> {energy} meV/atom ({stability_class})")
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(formulas: List[str], include_details: bool = False):
    """
    Predict stability for multiple materials at once.
    
    Parameters:
    - formulas: List of chemical formulas
    - include_details: Include additional metrics for all predictions
    
    Returns list of predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for formula in formulas:
        try:
            request = MaterialRequest(formula=formula, include_details=include_details)
            result = await predict(request)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "formula": formula,
                "error": str(e),
                "status": "failed"
            })
    
    return {"predictions": results, "total": len(formulas), "successful": sum(1 for r in results if "error" not in r)}

@app.get("/database/elements")
async def get_elements():
    """Get list of all elements in the database."""
    return {
        "a_site": list(a_db.keys()) if a_db else [],
        "b_site": list(b_db.keys()) if b_db else [],
        "atomic": list(atomic_db.keys()) if atomic_db else [],
        "total": len(a_db) + len(b_db)
    }

@app.get("/database/features")
async def get_features():
    """Get list of all features used by the model."""
    return {
        "features": feature_names,
        "count": len(feature_names),
        "sample_means": {k: round(v, 3) for k, v in list(feature_means.items())[:10]}
    }

# --- 5. ERROR HANDLERS ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)