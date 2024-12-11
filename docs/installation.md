# Installation Instructions for Qubot-One

Follow the steps below to install Qubot-One on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- Git
- A package manager (e.g., pip)

## Step 1: Clone the Repository

Clone the Qubot-One repository to your local machine:

```bash
1 git clone https://github.com/KOSASIH/Qubot-One.git
2 cd Qubot-One
```

## Step 2: Create a Virtual Environment (Optional)
It is recommended to create a virtual environment to manage dependencies:

```bash
1 python -m venv venv
2 source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Step 3: Install Dependencies
Install the required Python packages using pip:

```bash
1 pip install -r requirements.txt
```

## Step 4: Configuration
Before running the application, you may need to configure certain settings. Open the src/main/config.py file and adjust the parameters as needed.

## Step 5: Run the Application
You can now run the Qubot-One application:

```bash
1 python src/main/main.py
```

## Troubleshooting
If you encounter any issues during installation, please refer to the Troubleshooting Guide or open an issue on our GitHub repository.
