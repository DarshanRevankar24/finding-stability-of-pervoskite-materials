"""
Diagnostic tool to check ML model predictions and feature alignment
"""

import joblib
import pandas as pd
import numpy as np
import json

def diagnose_model():
    print("\n" + "="*70)
    print("ML MODEL DIAGNOSTIC REPORT")
    print("="*70)
    
    # Load model pack
    try:
        pack = joblib.load('perovskite_model_pack.pkl')
        model = pack['model']
        feature_names = pack['feature_names']
        feature_means = pack['feature_means']
        a_db = pack['a_site_db']
        b_db = pack['b_site_db']
        print(f"\n✅ Model pack loaded successfully")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Model info
    print("\n" + "-"*70)
    print("MODEL INFORMATION")
    print("-"*70)
    print(f"Model type: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"A-site elements in model DB: {len(a_db)}")
    print(f"B-site elements in model DB: {len(b_db)}")
    
    # Check which elements are in model databases
    print("\n" + "-"*70)
    print("ELEMENT COVERAGE IN MODEL")
    print("-"*70)
    
    common_elements = {
        'A-site': ['Ba', 'Sr', 'Ca', 'Pb', 'Cs', 'La', 'Nd', 'Y'],
        'B-site': ['Ti', 'Zr', 'V', 'Mn', 'Fe', 'Co', 'Ni', 'Sn', 'Ga', 'Al']
    }
    
    print("\nA-site elements in model database:")
    for el in common_elements['A-site']:
        status = "✅" if el in a_db else "❌"
        print(f"  {status} {el}")
    
    print("\nB-site elements in model database:")
    for el in common_elements['B-site']:
        status = "✅" if el in b_db else "❌"
        print(f"  {status} {el}")
    
    # Sample features
    print("\n" + "-"*70)
    print("SAMPLE FEATURES (first 10)")
    print("-"*70)
    for i, feat in enumerate(feature_names[:10]):
        mean_val = feature_means.get(feat, 0)
        print(f"{i+1:2d}. {feat}: {mean_val:.3f}")
    
    # Test predictions on known materials
    print("\n" + "-"*70)
    print("TEST PREDICTIONS ON KNOWN MATERIALS")
    print("-"*70)
    
    known_materials = [
        ("BaTiO3", 0, "Should be stable ground state"),
        ("SrTiO3", 0, "Should be stable ground state"),
        ("CaTiO3", 0, "Should be stable perovskite"),
        ("PbTiO3", 0, "Should be stable"),
    ]
    
    print("\nFormula          Predicted    Expected    Status")
    print("-" * 55)
    
    for formula, expected_energy, description in known_materials:
        # Simple parsing (assuming ABX3 format)
        if formula in ['BaTiO3', 'SrTiO3', 'CaTiO3', 'PbTiO3']:
            A_el = formula[:2] if formula[1].islower() else formula[0]
            B_el = 'Ti'
            
            # Get features
            features = pd.Series(feature_means)
            total_weight = 0
            weighted_sum = pd.Series(0.0, index=feature_names)
            
            # A-site
            if A_el in a_db:
                props = pd.Series(a_db[A_el]).reindex(feature_names, fill_value=0)
                weighted_sum += props
                total_weight += 1
            
            # B-site
            if B_el in b_db:
                props = pd.Series(b_db[B_el]).reindex(feature_names, fill_value=0)
                weighted_sum += props
                total_weight += 1
            
            if total_weight > 0:
                features = weighted_sum / total_weight
            
            input_df = pd.DataFrame([features])[feature_names]
            prediction = model.predict(input_df)[0]
            
            status = "✅" if abs(prediction - expected_energy) < 50 else "❌"
            print(f"{formula:15s}  {prediction:8.2f}    {expected_energy:8.1f}      {status}")
    
    # Feature importance (if available)
    print("\n" + "-"*70)
    print("MODEL PROPERTIES")
    print("-"*70)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances), 
                            key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most important features:")
        for feat, importance in top_features:
            print(f"  {feat}: {importance:.4f}")
    else:
        print("\n⚠️  Model doesn't expose feature importances")
    
    # Check for data issues
    print("\n" + "-"*70)
    print("POTENTIAL ISSUES")
    print("-"*70)
    
    issues = []
    
    # Check if Ba and Ti are in databases
    if 'Ba' not in a_db:
        issues.append("❌ Ba not in A-site database - predictions will use global average")
    if 'Ti' not in b_db:
        issues.append("❌ Ti not in B-site database - predictions will use global average")
    
    # Check feature alignment
    if 'Ba' in a_db:
        ba_features = set(a_db['Ba'].keys())
        model_features = set(feature_names)
        missing = model_features - ba_features
        if missing:
            issues.append(f"⚠️  Ba features missing {len(missing)} model features")
    
    if issues:
        for issue in issues:
            print(f"\n{issue}")
    else:
        print("\n✅ No obvious issues detected")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if 'Ba' not in a_db or 'Ti' not in b_db:
        print("\n⚠️  CRITICAL: Ba or Ti missing from model databases")
        print("   This means predictions are using GLOBAL AVERAGES, not element-specific properties")
        print("\n   Solutions:")
        print("   1. Retrain model with Ba/Ti included")
        print("   2. Add Ba/Ti to a_db/b_db in the .pkl file")
        print("   3. Check if your training data included these elements")
    else:
        print("\n✅ Key elements present in model databases")
        
        # Check if predictions are reasonable
        test_energy = None
        if 'Ba' in a_db and 'Ti' in b_db:
            features = pd.Series(feature_means)
            ba_props = pd.Series(a_db['Ba']).reindex(feature_names, fill_value=0)
            ti_props = pd.Series(b_db['Ti']).reindex(feature_names, fill_value=0)
            features = (ba_props + ti_props) / 2
            input_df = pd.DataFrame([features])[feature_names]
            test_energy = model.predict(input_df)[0]
        
        if test_energy is not None and test_energy > 50:
            print(f"\n⚠️  Model predicts high energy ({test_energy:.1f} meV/atom) for BaTiO3")
            print("   Possible causes:")
            print("   1. Model trained on different energy reference (check formation_energy vs energy_above_hull)")
            print("   2. Feature scaling issues (check if features need normalization)")
            print("   3. Model trained on subset of perovskites (doesn't generalize to oxides)")
            print("   4. Features in a_db/b_db don't match training data distribution")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")


def compare_features(formula="BaTiO3"):
    """Compare features being used vs expected"""
    print("\n" + "="*70)
    print(f"FEATURE COMPARISON FOR {formula}")
    print("="*70)
    
    pack = joblib.load('perovskite_model_pack.pkl')
    feature_names = pack['feature_names']
    feature_means = pack['feature_means']
    a_db = pack['a_site_db']
    b_db = pack['b_site_db']
    
    # Load feature defaults if available
    try:
        with open('feature_defaults.json', 'r') as f:
            expected_means = json.load(f)
        print("\n✅ Loaded feature_defaults.json for comparison")
    except:
        print("\n⚠️  feature_defaults.json not found")
        expected_means = feature_means
    
    # Get Ba and Ti features
    if 'Ba' in a_db and 'Ti' in b_db:
        ba_features = pd.Series(a_db['Ba']).reindex(feature_names, fill_value=0)
        ti_features = pd.Series(b_db['Ti']).reindex(feature_names, fill_value=0)
        combined = (ba_features + ti_features) / 2
        
        print("\n" + "-"*70)
        print("FEATURE VALUES (first 15)")
        print("-"*70)
        print(f"{'Feature':<40} {'Ba+Ti Avg':>12} {'Global Avg':>12}")
        print("-"*70)
        
        for i, feat in enumerate(feature_names[:15]):
            calc_val = combined.get(feat, 0)
            mean_val = feature_means.get(feat, 0)
            diff_marker = "⚠️" if abs(calc_val - mean_val) > 0.5 * abs(mean_val) else ""
            print(f"{feat:<40} {calc_val:>12.3f} {mean_val:>12.3f} {diff_marker}")
    else:
        print("\n❌ Cannot compare - Ba or Ti not in model databases")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "features":
        formula = sys.argv[2] if len(sys.argv) > 2 else "BaTiO3"
        compare_features(formula)
    else:
        diagnose_model()
        
        print("\n💡 TIP: Run 'python diagnose_model.py features' for detailed feature comparison")