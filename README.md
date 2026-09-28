# QHemoguard

QHemoguard is an exploratory quantum machine learning project for classifying thallasemia carriers using CBC data. It uses parameterized quantum circuits with Qiskit and compares experiments using selected hemoglobin features and CBC-only features.

## Project contents

- `data_prep.py` - Loads, prepares, rescales, and separates features and labels.
- `pqc.py` - Builds the parameterized quantum circuit.
- `training.py` - Implements training, gradients, and evaluation metrics.
- `notebooks/eda.ipynb` - Exploratory data analysis.
- `notebooks/explanation.ipynb` - Explanation of the approach.
- `notebooks/run_experiment.ipynb` - Runs the experiments.
- `requirements.txt` - Python dependencies.

## Requirements

- Python 3.10 or newer recommended
- Windows, macOS, or Linux
- A local Python environment with enough time for Qiskit simulator experiments

## Setup

From the project directory, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name qhemoguard --display-name "Python (QHemoguard)"
```

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name qhemoguard --display-name "Python (QHemoguard)"
```

## Dataset

The dataset is not included in this repository. It contains medical hemoglobin electrophoresis data and is excluded by `.gitignore`.

To run the notebooks, obtain the dataset through its authorized source and confirm that you are permitted to use it. Place the authorized CSV file at:

```text
data/Hb. Electrophoresis Data from 13-08-2020 to 24-06-2024 Preprocessed Train All with HB.csv
```


## Running the experiments

1. Open the repository in VS Code or start Jupyter:

   ```bash
   jupyter lab
   ```

2. Select the `Python (QHemoguard)` kernel.
3. Run `notebooks/eda.ipynb` for exploratory analysis.
4. Run `notebooks/explanation.ipynb` to review the method.
5. Run `notebooks/run_experiment.ipynb` after placing the authorized dataset in `data/`.

The experiments use local Qiskit Aer simulation by default. Runtime and results can vary because the training process uses simulated measurement shots and randomized seeds.


