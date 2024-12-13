# src/security/security_audit.py

import logging

class SecurityAudit:
    def __init__(self, log_file='security_audit.log'):
        """Initialize the security audit logger.

        Args:
            log_file (str): The log file for security audits.
        """
        logging.basicConfig(filename=log_file, level=logging.INFO)
        self.logger = logging.getLogger('SecurityAudit')

    def log_event(self, event_type, message):
        """Log a security event.

        Args:
            event_type (str): The type of event (e.g., 'INFO', 'WARNING', 'ERROR').
            message (str): The message to log.
        """
        if event_type.upper() == 'INFO':
            self.logger.info(message)
        elif event_type.upper() == 'WARNING':
            self.logger.warning(message)
        elif event_type.upper() == 'ERROR':
            self.logger.error(message)
        else:
            raise ValueError("Invalid event type. Use 'INFO', 'WARNING', or 'ERROR'.")
        print(f"Logged {event_type}: {message}")

    def get_logs(self):
        """Retrieve the logs from the audit file.

        Returns:
            list: List of log entries.
        """
        with open('security_audit.log', 'r') as f:
            logs = f.readlines()
        print("Retrieved security audit logs.")
        return logs
