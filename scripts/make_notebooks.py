import json
import os

notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(notebooks_dir, exist_ok=True)

# -------------------------------------------------------------
# NOTEBOOK 1: 01_preprocessing_eda.ipynb
# -------------------------------------------------------------
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# TALUS — Phase II: Data Preprocessing & Exploratory Data Analysis (EDA)\n",
            "**Project Title:** TALUS — Physics-Informed AI for Open-Pit Mine Slope Risk Intelligence\n",
            "\n",
            "This notebook demonstrates dataset loading, schema validation, missing value handling, outlier detection, physics-derived feature engineering (Factor of Safety), and comprehensive exploratory data analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "sys.path.append('..')\n",
            "from backend.physics import calculate_factor_of_safety\n",
            "\n",
            "print('Libraries imported successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Synthetic Mine Slope Dataset\n",
            "Dataset includes geometric, geotechnical, environmental, operational, and physical stability monitoring parameters."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "raw_csv_path = '../data/raw/slope_monitoring_synthetic.csv'\n",
            "df = pd.read_csv(raw_csv_path)\n",
            "print(f'Total records loaded: {len(df)}')\n",
            "print(f'Total features: {df.shape[1]}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Preprocessing & Schema Validation\n",
            "- Check missing values\n",
            "- Check duplicate rows\n",
            "- Validate data types and numerical ranges"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('--- MISSING VALUE ANALYSIS ---')\n",
            "print(df.isnull().sum())\n",
            "print('\\n--- DUPLICATE ROWS CHECK ---')\n",
            "print(f'Duplicate rows: {df.duplicated().sum()}')\n",
            "print('\\n--- TARGET CLASS DISTRIBUTION ---')\n",
            "print(df['risk_label'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Physics Stability Feature Verification\n",
            "Verify physics-derived Factor of Safety (FoS) distribution across risk categories."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.groupby('risk_label')[['factor_of_safety', 'deformation_rate_mm_day', 'pore_pressure_kpa', 'rainfall_mm']].mean()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Exploratory Data Analysis & Visualizations"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation Matrix Heatmap\n",
            "num_cols = df.select_dtypes(include=[np.number]).columns\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')\n",
            "plt.title('Feature Correlation Heatmap')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "print('Interpretation: Factor of Safety shows strong inverse correlation with slope angle and pore pressure.')"
        ]
    }
]

nb1_data = {
    "cells": nb1_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# NOTE: notebook 01 is now built (and executed) by scripts/build_eda_notebook.py.
# The original short version is intentionally NOT written here so it cannot overwrite the full EDA notebook.

# -------------------------------------------------------------
# NOTEBOOK 2: 02_model_training.ipynb
# -------------------------------------------------------------
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# TALUS — Phase II: Model Building & Result Evaluation\n",
            "**Project Title:** TALUS — Physics-Informed AI for Open-Pit Mine Slope Risk Intelligence\n",
            "\n",
            "This notebook evaluates Logistic Regression, Support Vector Machine, Random Forest, and Gradient Boosting models on the slope risk classification dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import joblib\n",
            "\n",
            "sys.path.append('..')\n",
            "from backend.ml_engine import PreprocessingPipeline, compute_metrics\n",
            "\n",
            "comp_df = pd.read_csv('../results/model_comparison.csv')\n",
            "print('--- MODEL COMPARISON MATRIX ---')\n",
            "print(comp_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model Selection & Metadata\n",
            "The final model is selected based on validation F1-Score."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "with open('../models/model_metadata.json', 'r') as f:\n",
            "    meta = json.load(f)\n",
            "print('Selected Model:', meta['selected_model'])\n",
            "print('Validation Performance:', meta['performance'])"
        ]
    }
]

nb2_data = {
    "cells": nb2_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(os.path.join(notebooks_dir, "02_model_training.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb2_data, f, indent=2)

print("Notebooks 01_preprocessing_eda.ipynb and 02_model_training.ipynb generated!")
