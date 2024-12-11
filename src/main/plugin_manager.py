import importlib

class PluginManager:
    def __init__(self):
        self.plugins = {}

    def load_plugin(self, plugin_name):
        module = importlib.import_module(plugin_name)
        self.plugins[plugin_name] = module

    def execute_plugin(self, plugin_name, *args, **kwargs):
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].run(*args, **kwargs)
        else:
            raise Exception(f"Plugin {plugin_name} not loaded.")
