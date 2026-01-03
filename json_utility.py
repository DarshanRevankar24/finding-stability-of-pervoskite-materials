"""
Utility functions for managing JSON databases and validation
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class AtomicDatabase:
    """Manager for atomic properties database"""
    
    def __init__(self, db_path: str = "atomic_db.json"):
        self.db_path = Path(db_path)
        self.data = self.load()
    
    def load(self) -> Dict:
        """Load atomic database from JSON file"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Database file not found: {self.db_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {self.db_path}: {e}")
            return {}
    
    def save(self):
        """Save current database to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"✅ Database saved to {self.db_path}")
    
    def get_element(self, symbol: str, site: str = None) -> Optional[Dict]:
        """
        Get element properties
        Args:
            symbol: Chemical symbol (e.g., 'Ba', 'Ti')
            site: 'A', 'B', or 'X' (optional)
        """
        if site:
            site_key = f"{site}_site_elements"
            return self.data.get(site_key, {}).get(symbol)
        
        # Search all sites
        for site_key in ['A_site_elements', 'B_site_elements', 'X_site_elements']:
            if symbol in self.data.get(site_key, {}):
                return self.data[site_key][symbol]
        return None
    
    def add_element(self, symbol: str, properties: Dict, site: str):
        """
        Add new element to database
        Args:
            symbol: Chemical symbol
            properties: Dictionary of element properties
            site: 'A', 'B', or 'X'
        """
        site_key = f"{site}_site_elements"
        if site_key not in self.data:
            self.data[site_key] = {}
        
        self.data[site_key][symbol] = properties
        print(f"✅ Added {symbol} to {site}-site elements")
    
    def list_elements(self, site: str = None) -> List[str]:
        """List all available elements"""
        if site:
            site_key = f"{site}_site_elements"
            return list(self.data.get(site_key, {}).keys())
        
        elements = []
        for site_key in ['A_site_elements', 'B_site_elements', 'X_site_elements']:
            elements.extend(self.data.get(site_key, {}).keys())
        return sorted(set(elements))
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            "A_site_count": len(self.data.get("A_site_elements", {})),
            "B_site_count": len(self.data.get("B_site_elements", {})),
            "X_site_count": len(self.data.get("X_site_elements", {})),
            "organic_count": len(self.data.get("organic_cations", {})),
            "total_elements": sum([
                len(self.data.get("A_site_elements", {})),
                len(self.data.get("B_site_elements", {})),
                len(self.data.get("X_site_elements", {}))
            ])
        }
    
    def validate_element(self, symbol: str, properties: Dict) -> Tuple[bool, List[str]]:
        """Validate element properties"""
        required_fields = ['radius', 'weight', 'en', 'ionization_energy', 
                          'oxidation_states', 'coordination', 'element_type']
        errors = []
        
        for field in required_fields:
            if field not in properties:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'radius' in properties and not (0.3 < properties['radius'] < 3.0):
            errors.append(f"Radius {properties['radius']} outside valid range (0.3-3.0 Å)")
        
        if 'en' in properties and not (0.5 < properties['en'] < 4.5):
            errors.append(f"Electronegativity {properties['en']} outside valid range")
        
        return len(errors) == 0, errors


class ValidationRules:
    """Manager for validation rules"""
    
    def __init__(self, rules_path: str = "validation_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules = self.load()
    
    def load(self) -> Dict:
        """Load validation rules from JSON"""
        try:
            with open(self.rules_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Rules file not found: {self.rules_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {self.rules_path}: {e}")
            return {}
    
    def get_stability_class(self, energy: float) -> Tuple[str, Dict]:
        """Get stability classification for given energy"""
        thresholds = self.rules.get('stability_thresholds', {})
        
        for class_name, criteria in thresholds.items():
            min_e = criteria.get('min_energy', float('-inf'))
            max_e = criteria.get('max_energy', float('inf'))
            if min_e <= energy < max_e:
                return class_name, criteria
        
        return 'unknown', {}
    
    def validate_tolerance_factor(self, t: float) -> Dict:
        """Validate Goldschmidt tolerance factor"""
        ranges = self.rules.get('tolerance_factor_ranges', {})
        
        for range_name, criteria in ranges.items():
            min_t = criteria.get('min', float('-inf'))
            max_t = criteria.get('max', float('inf'))
            if min_t <= t < max_t:
                return {
                    'valid': True,
                    'range': range_name,
                    'structure': criteria.get('structure'),
                    'stability': criteria.get('stability')
                }
        
        # Check special cases
        if t < 0.71:
            return {
                'valid': False,
                'warning': 'Tolerance factor too low for perovskite structure',
                'structure': 'non_perovskite'
            }
        elif t > 1.13:
            return {
                'valid': False,
                'warning': 'Tolerance factor too high for perovskite structure',
                'structure': 'non_perovskite'
            }
        
        return {'valid': False, 'warning': 'Unknown range'}
    
    def validate_octahedral_factor(self, mu: float) -> Dict:
        """Validate octahedral factor"""
        ranges = self.rules.get('octahedral_factor_ranges', {})
        
        if mu < ranges.get('too_small', {}).get('max', 0.41):
            return {
                'valid': False,
                'warning': ranges['too_small']['warning'],
                'description': ranges['too_small']['description']
            }
        elif mu > ranges.get('too_large', {}).get('min', 0.73):
            return {
                'valid': False,
                'warning': ranges['too_large']['warning'],
                'description': ranges['too_large']['description']
            }
        else:
            return {
                'valid': True,
                'description': ranges.get('stable', {}).get('description', '')
            }


def merge_databases(old_path: str, new_path: str, output_path: str):
    """Merge two atomic databases"""
    with open(old_path, 'r') as f:
        old_db = json.load(f)
    
    with open(new_path, 'r') as f:
        new_db = json.load(f)
    
    # Merge logic (new_db takes precedence)
    merged = old_db.copy()
    for key, value in new_db.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✅ Databases merged and saved to {output_path}")


def export_to_csv(db_path: str, output_path: str):
    """Export atomic database to CSV format"""
    import pandas as pd
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    # Flatten structure for CSV
    rows = []
    for site in ['A_site_elements', 'B_site_elements', 'X_site_elements']:
        if site in data:
            for element, props in data[site].items():
                row = {'element': element, 'site': site[0]}
                row.update(props)
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Database exported to {output_path}")


def generate_report(db_path: str = "atomic_db.json"):
    """Generate comprehensive database report"""
    db = AtomicDatabase(db_path)
    stats = db.get_statistics()
    
    print("\n" + "="*60)
    print("ATOMIC DATABASE REPORT")
    print("="*60)
    print(f"\nDatabase: {db.db_path}")
    print(f"Version: {db.data.get('_metadata', {}).get('version', 'Unknown')}")
    print(f"Last Updated: {db.data.get('_metadata', {}).get('last_updated', 'Unknown')}")
    
    print("\n📊 ELEMENT COUNT:")
    print(f"  A-site elements: {stats['A_site_count']}")
    print(f"  B-site elements: {stats['B_site_count']}")
    print(f"  X-site elements: {stats['X_site_count']}")
    print(f"  Organic cations: {stats['organic_count']}")
    print(f"  Total: {stats['total_elements']}")
    
    print("\n🔬 A-SITE ELEMENTS:")
    a_elements = db.list_elements('A')
    print(f"  {', '.join(a_elements)}")
    
    print("\n🔬 B-SITE ELEMENTS:")
    b_elements = db.list_elements('B')
    print(f"  {', '.join(b_elements)}")
    
    print("\n🔬 X-SITE ELEMENTS:")
    x_elements = db.list_elements('X')
    print(f"  {', '.join(x_elements)}")
    
    # Property coverage
    print("\n📋 PROPERTY COVERAGE:")
    sample_element = db.get_element(a_elements[0], 'A') if a_elements else {}
    if sample_element:
        properties = list(sample_element.keys())
        print(f"  Available properties: {', '.join(properties)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python json_utils.py report")
        print("  python json_utils.py export <output.csv>")
        print("  python json_utils.py validate")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "report":
        generate_report()
    
    elif command == "export":
        output = sys.argv[2] if len(sys.argv) > 2 else "atomic_db.csv"
        export_to_csv("atomic_db.json", output)
    
    elif command == "validate":
        db = AtomicDatabase()
        print("\n🔍 VALIDATING DATABASE...")
        
        for site in ['A', 'B', 'X']:
            elements = db.list_elements(site)
            print(f"\n{site}-site ({len(elements)} elements):")
            for el in elements:
                props = db.get_element(el, site)
                valid, errors = db.validate_element(el, props)
                if valid:
                    print(f"  ✅ {el}")
                else:
                    print(f"  ❌ {el}: {'; '.join(errors)}")
    
    else:
        print(f"Unknown command: {command}")