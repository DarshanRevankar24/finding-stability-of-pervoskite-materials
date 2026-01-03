import pandas as pd

# Load your training data
df = pd.read_csv('Perovskite_Stability_with_features.csv')

print("="*70)
print("TRAINING DATA ANALYSIS")
print("="*70)

# Find BaTiO3-related materials
print("\n1. Searching for BaTiO3...")
batio3 = df[df['Material Composition'].str.contains('BaTi', na=False, case=False)]

if len(batio3) > 0:
    print(f"\nFound {len(batio3)} BaTiO3-related entries:")
    print(batio3[['Material Composition', 'energy_above_hull (meV/atom)', 'formation_energy (eV/atom)']].to_string())
else:
    print("❌ No BaTiO3 found in training data")

# Check other titanates
print("\n2. Other titanate perovskites:")
titanates = df[df['Material Composition'].str.contains('Ti.*O', na=False, regex=True)]
titanates_sample = titanates[['Material Composition', 'energy_above_hull (meV/atom)']].head(10)
print(titanates_sample.to_string())

# Energy distribution
print("\n3. Energy distribution in training data:")
print(df['energy_above_hull (meV/atom)'].describe())

print(f"\n4. Materials with energy < 10 meV/atom (stable):")
stable = df[df['energy_above_hull (meV/atom)'] < 10]
print(f"   Count: {len(stable)} out of {len(df)} ({len(stable)/len(df)*100:.1f}%)")
print(stable[['Material Composition', 'energy_above_hull (meV/atom)']].head(10).to_string())

print(f"\n5. Materials with energy 50-150 meV/atom (like BaTiO3 prediction):")
metastable = df[(df['energy_above_hull (meV/atom)'] >= 50) & (df['energy_above_hull (meV/atom)'] <= 150)]
print(f"   Count: {len(metastable)} out of {len(df)} ({len(metastable)/len(df)*100:.1f}%)")
print(metastable[['Material Composition', 'energy_above_hull (meV/atom)']].head(10).to_string())

print("\n" + "="*70)