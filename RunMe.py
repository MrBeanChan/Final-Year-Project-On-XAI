import subprocess
import os
# Run pip install command
# Install required libraries
libraries = [
    'pandas',
    'seaborn',
    'scikit-learn',
    'matplotlib',
    'numpy',
    'interpret',
    'shap',
    'xgboost',
    'streamlit'
]

for library in libraries:
    subprocess.check_call(['pip', 'install', library])

#Checking location just in case
current_directory = os.getcwd()


print("All required packages have been installed!")


# Get the current directory of the Python file
current_directory = os.path.dirname(os.path.abspath(__file__))
# Change the working directory
os.chdir(current_directory)
print(current_directory)

# Run GUI
file_name = 'streamlit_app.py'
subprocess.check_call(['streamlit', 'run', file_name])