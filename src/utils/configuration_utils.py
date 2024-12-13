# src/utils/configuration_utils.py

import configparser

class ConfigManager:
    @staticmethod
    def read_config(file_path):
        """Read configuration from a file.

        Args:
            file_path (str): Path to the configuration file.

        Returns:
            dict: Configuration settings.
        """
        config = configparser.ConfigParser()
        config.read(file_path)
        settings = {section: dict(config.items(section)) for section in config.sections()}
        print(f"Configuration read from {file_path}.")
        return settings

    @staticmethod
    def write_config(file_path, settings):
        """Write configuration to a file.

        Args:
            file_path (str): Path to the configuration file.
            settings (dict): Configuration settings to write.
        """
        config = configparser.ConfigParser()
        for section, options in settings.items():
            config[section] = options
        with open(file_path, 'w') as configfile:
            config.write(configfile)
        print(f"Configuration written to {file_path}.")
