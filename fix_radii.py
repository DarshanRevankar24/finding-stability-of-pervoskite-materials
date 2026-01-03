import json

# Load current database
with open('atomic_db.json', 'r') as f:
    db = json.load(f)

# Correct ionic radii for perovskites (Shannon radii)
corrections = {
    'A_site_elements': {
        'Ba': {'radius': 1.61, 'ionic_radius': 1.61},  # 12-coord Ba2+
        'Sr': {'radius': 1.44, 'ionic_radius': 1.44},  # 12-coord Sr2+
        'Ca': {'radius': 1.34, 'ionic_radius': 1.34},  # 12-coord Ca2+
        'Pb': {'radius': 1.49, 'ionic_radius': 1.49},  # 12-coord Pb2+
        'Cs': {'radius': 1.88, 'ionic_radius': 1.88},  # 12-coord Cs+
    },
    'B_site_elements': {
        'Ti': {'radius': 0.605, 'ionic_radius': 0.605},  # 6-coord Ti4+
        'Zr': {'radius': 0.72, 'ionic_radius': 0.72},    # 6-coord Zr4+
        'V': {'radius': 0.64, 'ionic_radius': 0.64},     # 6-coord V4+
        'Mn': {'radius': 0.53, 'ionic_radius': 0.53},    # 6-coord Mn4+
        'Fe': {'radius': 0.585, 'ionic_radius': 0.585},  # 6-coord Fe3+
        'Sn': {'radius': 0.69, 'ionic_radius': 0.69},    # 6-coord Sn4+
        'Pb': {'radius': 0.775, 'ionic_radius': 0.775},  # 6-coord Pb4+
    },
    'X_site_elements': {
        'O': {'radius': 1.40, 'ionic_radius': 1.40},   # 6-coord O2-
        'F': {'radius': 1.33, 'ionic_radius': 1.33},   # 6-coord F-
        'Cl': {'radius': 1.81, 'ionic_radius': 1.81},  # 6-coord Cl-
        'Br': {'radius': 1.96, 'ionic_radius': 1.96},  # 6-coord Br-
        'I': {'radius': 2.20, 'ionic_radius': 2.20},   # 6-coord I-
    }
}

# Apply corrections
for site, elements in corrections.items():
    if site in db:
        for element, radii_update in elements.items():
            if element in db[site]:
                db[site][element].update(radii_update)
                print(f"✅ Updated {element}: radius = {radii_update['radius']} Å")

# Save corrected database
with open('atomic_db.json', 'w') as f:
    json.dump(db, f, indent=2)

print("\n✅ Radii corrected! Restart your server.")