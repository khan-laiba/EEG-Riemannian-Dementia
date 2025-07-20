# Hybrid EEG Biomarker Fusing Riemannian Geometry and Spectral features for Dementia Diagnosis and Severity Assessment
Laiba Khan, Neuroscience Mentorship Program, Neuroscience Mentorship Program, Lexington, MA, USA


This project implements a machine learning pipeline to analyze resting-state EEG data for the diagnosis of dementia and the assessment of cognitive decline. It uses a hybrid feature engineering approach, combining functional connectivity features derived from Riemannian geometry with traditional spectral power ratios. The analysis is performed on a subject-level basis to ensure robust and clinically relevant results.

The primary script, `main.py`, is capable of performing three distinct tasks:
1.  **Regression:** Predicts the Mini-Mental State Examination (MMSE) score for each participant.
2.  **2-Class Classification:** Differentiates between a combined Dementia group (Alzheimer's Disease + Frontotemporal Dementia) and Healthy Controls.
3.  **3-Class Classification:** Performs differential diagnosis between Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Healthy Controls (CN).

---

## Dataset

This project uses the publicly available OpenNeuro dataset **`ds004504`**.

-   **Title:** A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG.
-   **Citation:** Dimitriadis, S. I., et al. (2023).
-   **Link:** [https://openneuro.org/datasets/ds004504/versions/1.0.8](https://openneuro.org/datasets/ds004504/versions/1.0.8)

### How to Download
To download the dataset, you can use the OpenNeuro command-line interface (CLI):
   - Install the OpenNeuro command-line interface: `pip install openneuro-py`
   - Run the following command in your terminal:
     ```bash
     openneuro-py download --dataset=ds004504
     ```
   - This will download the dataset into a `ds004504` folder in your current directory.

---

## Setup, Installation and Execution

To set up the environment and run this project, follow these steps.

### 1. Prerequisites
- Python 3.8 or newer
- `pip` or `conda` for package management



### 2. Create a Virtual Environment (Recommended)

It is highly recommended to create a dedicated virtual environment to avoid conflicts with other projects.

**Using `conda`:**
```bash
conda create -n eeg_analysis python=3.9
conda activate eeg_analysis
```
#### Install Dependencies

Install all the required libraries using the requirements.txt file provided.

```bash
pip install -r requirements.txt
```

How to Run the Analysis
The entire analysis is controlled from the main.py script.


Open main.py and navigate to the CONFIGURATION section at the top of the file. Modify the task variable to select the desired analysis:

 ```Python
# --- TASK OPTION ---
# Set to 'regression' to predict MMSE score
# Set to '2-class' for Dementia (AD+FTD) vs. CN
# Set to '3-class' for AD vs. FTD vs. CN
task = "2-class"  # <--- CHANGE THIS VALUE
 ```
### 3. Execute the Script

Run the script from your terminal:

```bash
python main.py
```
The script will execute the following steps:

- Load the participant data based on the selected task.
- Gather and preprocess EEG epochs for each subject.
- Engineer subject-level features by combining Riemannian connectivity and spectral power features.
- Run a 5-fold cross-validation using the appropriate model (RandomForestRegressor for regression, BalancedRandomForestClassifier for classification).
- Print a detailed performance report to the console, including all relevant metrics (MAE, Pearson's r, Accuracy, Precision, Recall, F1-Score).
- Generate and save high-quality plots for the results (e.g., confusion matrix, regression scatter plot).
- Perform and plot a feature importance analysis to show which feature groups are most predictive for the selected task.

### 4. File Descriptions
- main.py: The main executable script that contains all the code for data processing, feature engineering, modeling, and evaluation.
- requirements.txt: A file listing all the Python libraries required to run the project.
- README.md: This file, providing an overview of the project and instructions for setup and execution.
- ds004504/: The folder containing the dataset, which must be downloaded and placed in the project root.
