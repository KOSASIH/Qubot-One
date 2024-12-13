# examples/custom_plugin_example.py

class CustomPlugin:
    def __init__(self, name):
        self.name = name

    def run(self):
        print(f"Running custom plugin: {self.name}")

def main():
    # Create and run a custom plugin
    plugin = CustomPlugin("MyCustomPlugin")
    plugin.run()

if __name__ == "__main__":
    main()
