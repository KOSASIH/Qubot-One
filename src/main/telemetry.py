import logging

class Telemetry:
    def __init__(self):
        self.data = []

    def collect_data(self, qubot_id, telemetry_info):
        self.data.append({'qubot_id': qubot_id, 'info': telemetry_info})
        logging.info(f"Telemetry data collected for Qubot {qubot_id}: {telemetry_info}")

    def report_data(self):
        # Placeholder for reporting telemetry data to a server or database
        logging.info("Reporting telemetry data...")
