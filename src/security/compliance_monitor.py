# src/security/compliance_monitor.py

class ComplianceMonitor:
    def __init__(self):
        """Initialize the compliance monitor."""
        self.compliance_checks = []

    def add_compliance_check(self, check_name, check_function):
        """Add a compliance check.

        Args:
            check_name (str): The name of the compliance check.
            check_function (callable): The function that performs the check.
        """
        self.compliance_checks.append({'name': check_name, 'function': check_function})
        print(f"Compliance check added: {check_name}")

    def run_checks(self):
        """Run all compliance checks and return results.

        Returns:
            dict: Results of the compliance checks.
        """
        results = {}
        for check in self.compliance_checks:
            result = check['function']()
            results[check['name']] = result
            print(f"Ran compliance check: {check['name']} - Result: {result}")
        return results
