{
  "_metadata": {
    "version": "1.0.0",
    "description": "Validation rules and thresholds for perovskite stability prediction",
    "last_updated": "2025-01-04"
  },
  
  "formula_validation": {
    "max_length": 100,
    "min_length": 3,
    "allowed_pattern": "^[A-Z][a-zA-Z0-9.()\\s]*$",
    "forbidden_characters": ["#", "$", "%", "&", "*", "!", "?"],
    "max_elements": 10,
    "min_elements": 2
  },
  
  "stability_thresholds": {
    "highly_stable": {
      "max_energy": 0,
      "description": "Ground state materials",
      "confidence": "high",
      "recommendation": "Excellent candidate for synthesis"
    },
    "stable": {
      "min_energy": 0,
      "max_energy": 50,
      "description": "Thermodynamically stable (complex perovskites)",
      "confidence": "high",
      "recommendation": "Good candidate for synthesis"
    },
    "moderately_stable": {
      "min_energy": 50,
      "max_energy": 100,
      "description": "Moderately stable (typical for multi-element systems)",
      "confidence": "medium",
      "recommendation": "May be synthesizable under controlled conditions"
    },
    "marginally_stable": {
      "min_energy": 100,
      "max_energy": 150,
      "description": "Marginally stable",
      "confidence": "medium",
      "recommendation": "Difficult to synthesize, requires special conditions"
    },
    "metastable": {
      "min_energy": 150,
      "max_energy": 250,
      "description": "Metastable phase",
      "confidence": "low",
      "recommendation": "May exist temporarily, likely to decompose"
    },
    "unstable": {
      "min_energy": 250,
      "max_energy": 1000,
      "description": "Thermodynamically unfavorable",
      "confidence": "high",
      "recommendation": "Not recommended for synthesis"
    }
  },
  
  "tolerance_factor_ranges": {
    "cubic": {
      "min": 0.89,
      "max": 1.0,
      "structure": "Cubic perovskite (Pm-3m)",
      "stability": "optimal"
    },
    "tetragonal_orthorhombic": {
      "min": 0.8,
      "max": 0.89,
      "structure": "Distorted perovskite",
      "stability": "good"
    },
    "ilmenite": {
      "min": 0.71,
      "max": 0.8,
      "structure": "Ilmenite-type",
      "stability": "moderate"
    },
    "hexagonal": {
      "min": 1.0,
      "max": 1.13,
      "structure": "Hexagonal perovskite",
      "stability": "moderate"
    },
    "non_perovskite": {
      "below": 0.71,
      "above": 1.13,
      "structure": "Non-perovskite structures",
      "stability": "poor"
    }
  },
  
  "octahedral_factor_ranges": {
    "stable": {
      "min": 0.41,
      "max": 0.73,
      "description": "B-cation fits well in octahedral site"
    },
    "too_small": {
      "max": 0.41,
      "description": "B-cation too small for octahedral coordination",
      "warning": "May form alternative structures"
    },
    "too_large": {
      "min": 0.73,
      "description": "B-cation too large, coordination change likely",
      "warning": "Non-octahedral coordination probable"
    }
  },
  
  "composition_rules": {
    "perovskite_stoichiometry": {
      "ABX3": {
        "A_site": 1,
        "B_site": 1,
        "X_site": 3,
        "tolerance": 0.1,
        "description": "Standard perovskite"
      },
      "A2BB'X6": {
        "A_site": 2,
        "B_site": 2,
        "X_site": 6,
        "tolerance": 0.1,
        "description": "Double perovskite"
      },
      "ABX4": {
        "A_site": 1,
        "B_site": 1,
        "X_site": 4,
        "tolerance": 0.1,
        "description": "2D layered perovskite"
      }
    },
    "charge_balance": {
      "enabled": true,
      "typical_charges": {
        "A_site": [1, 2, 3],
        "B_site": [2, 3, 4, 5],
        "X_site": [-1, -2]
      }
    }
  },
  
  "prediction_quality": {
    "high_confidence": {
      "min_elements_in_db": 2,
      "max_missing_elements": 0,
      "max_extrapolation_distance": 0.1
    },
    "medium_confidence": {
      "min_elements_in_db": 1,
      "max_missing_elements": 1,
      "max_extrapolation_distance": 0.3
    },
    "low_confidence": {
      "min_elements_in_db": 0,
      "max_missing_elements": 3,
      "max_extrapolation_distance": 0.5,
      "warning": "Prediction uses global averages"
    }
  },
  
  "element_compatibility": {
    "incompatible_pairs": [
      {"A": ["Cs", "Rb", "K"], "B": ["Fe", "Mn"], "reason": "Charge mismatch"},
      {"A": ["La", "Ce", "Nd"], "X": ["I", "Br"], "reason": "Rare earth halides unstable"}
    ],
    "recommended_combinations": [
      {"A": ["Ba", "Sr", "Ca"], "B": ["Ti", "Zr", "Hf"], "X": ["O"], "stability": "high"},
      {"A": ["Cs", "MA", "FA"], "B": ["Pb", "Sn"], "X": ["I", "Br", "Cl"], "stability": "medium"}
    ]
  },
  
  "synthesis_recommendations": {
    "oxide_perovskites": {
      "temperature_range": "1000-1500°C",
      "atmosphere": "Air or oxygen",
      "method": "Solid-state reaction",
      "precautions": ["High temperature required", "Long annealing times"]
    },
    "halide_perovskites": {
      "temperature_range": "100-200°C",
      "atmosphere": "Inert (N2 or Ar)",
      "method": "Solution processing",
      "precautions": ["Moisture sensitive", "Low thermal stability"]
    }
  },
  
  "data_quality_checks": {
    "feature_ranges": {
      "energy_above_hull": {"min": -50, "max": 500},
      "formation_energy": {"min": -10, "max": 5},
      "shannon_radii_AB_avg": {"min": 0.3, "max": 2.0},
      "Density_AB_avg": {"min": 0.5, "max": 25.0}
    },
    "outlier_detection": {
      "enabled": true,
      "method": "z_score",
      "threshold": 3.0
    }
  },
  
  "api_limits": {
    "max_batch_size": 100,
    "rate_limit": {
      "requests_per_minute": 60,
      "requests_per_hour": 1000
    },
    "max_formula_length": 100,
    "timeout_seconds": 30
  }
}