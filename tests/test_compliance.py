# tests/test_compliance.py

import unittest
from src.security import ComplianceMonitor  # Assuming you have a ComplianceMonitor class

class TestCompliance(unittest.TestCase):
    def setUp(self):
        self.monitor = ComplianceMonitor()

    def test_add_compliance_check(self):
        def dummy_check():
            return True

        self.monitor.add_compliance_check('Dummy Check', dummy_check)
        results = self.monitor.run_checks()
        self.assertTrue(results['Dummy Check'])

if __name__ == '__main__':
    unittest.main()
