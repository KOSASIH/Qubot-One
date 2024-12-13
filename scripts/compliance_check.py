# scripts/compliance_check.py

import json
import os

def load_compliance_rules(file_path):
    """Load compliance rules from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def check_compliance(data, rules):
    """Check compliance of the data against the rules."""
    compliance_results = {}
    
    for rule in rules:
        field = rule['field']
        expected_value = rule['expected_value']
        
        if field in data:
            compliance_results[field] = (data[field] == expected_value)
        else:
            compliance_results[field] = False  # Field not found

    return compliance_results

def main():
    # Load compliance rules
    rules_file = 'compliance_rules.json'
    if not os.path.exists(rules_file):
        print(f"Compliance rules file '{rules_file}' not found.")
        return

    rules = load_compliance_rules(rules_file)

    # Simulate data to check compliance
    sample_data = {
        'sensor_value': 75,
        'status': 'normal',
        'location': 'warehouse'
    }

    # Check compliance
    results = check_compliance(sample_data, rules)
    print("Compliance Check Results:")
    for field, is_compliant in results.items():
        print(f"{field}: {'Compliant' if is_compliant else 'Non-Compliant'}")

if __name__ == "__main__":
    main()
