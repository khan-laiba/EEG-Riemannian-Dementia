import os
import glob
import re
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy

# MNE-Python for EEG data handling
from mne.epochs import make_fixed_length_epochs

# pyRiemann for Riemannian geometry
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance

# scikit-learn for machine learning
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator, TransformerMixin

# --- Suppress verbose warnings for a cleaner output ---
warnings.filterwarnings("ignore")
mne.set_log_level("CRITICAL")

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
# Random state for reproducibility
random_state = 9036
N_CHANNELS = 19  # Number of EEG channels

# --- TASK OPTION ---
# Set to 'regression' to predict MMSE score
# Set to '2-class' for Dementia (AD+FTD) vs. CN
# Set to '3-class' for AD vs. FTD vs. CN
task = "3-class"  # <--- SET YOUR DESIRED TASK HERE

data_root = "ds004504"
sfreq = 500
tmin, tmax = 60, 260
epoch_length_sec = 10
overlap_sec = 8

FREQ_BANDS = {
    "Delta": [1.0, 4.0],
    "Theta": [4.0, 8.0],
    "Alpha": [8.0, 13.0],
    "Beta": [13.0, 30.0],
}
BAND_ORDER = ["Delta", "Theta", "Alpha", "Beta"]


# =============================================================================
# --- 2. DATA LOADING & PREPARATION ---
# =============================================================================
print(f"--- Setting up for task: '{task}' ---")
participants_df = pd.read_csv(os.path.join(data_root, "participants.tsv"), sep="\t")

# This dataframe will be configured based on the task
participants_df_target = participants_df[
    participants_df["Group"].isin(["A", "F", "C"])
].copy()

if task == "regression":
    print("Preparing data for MMSE Regression.")
    # For regression, we only need subjects with a valid MMSE score
    participants_df_target["MMSE"] = pd.to_numeric(
        participants_df_target["MMSE"], errors="coerce"
    )
    participants_df_target.dropna(subset=["MMSE"], inplace=True)

elif task == "2-class":
    print("Preparing data for 2-class classification (Dementia vs. CN).")
    group_mapping = {"A": 0, "F": 0, "C": 1}  # AD/FTD -> 0 (Dementia), CN -> 1
    participants_df_target["label"] = participants_df_target["Group"].map(group_mapping)
    target_names_display = ["Dementia", "CN"]

elif task == "3-class":
    print("Preparing data for 3-class classification (AD vs. FTD vs. CN).")
    group_mapping = {"A": 0, "F": 1, "C": 2}
    participants_df_target["label"] = participants_df_target["Group"].map(group_mapping)
    target_names_display = ["AD", "FTD", "CN"]
else:
    raise ValueError(
        "Invalid task specified. Choose 'regression', '2-class', or '3-class'."
    )

participants_df_target.set_index("participant_id", inplace=True)
print(f"\nFound {len(participants_df_target)} participants for the task.")


# --- Helper Functions ---
def get_subject_id_from_path(file_path):
    match = re.search(r"(sub-\d{3})", file_path)
    return match.group(1) if match else None


def load_and_preprocess_eeg(file_path, target_sfreq):
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.filter(1.0, 45.0, fir_design="firwin")
        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq)
        raw.pick_types(eeg=True, exclude="bads")
        return raw if len(raw.ch_names) > 0 else None
    except Exception:
        return None


# =============================================================================
# --- 3. STAGE 1: GATHERING EPOCHS PER SUBJECT ---
# =============================================================================
print("\nStep 1: Gathering all epochs for each subject...")
eeg_files = sorted(
    glob.glob(os.path.join(data_root, "derivatives", "*", "eeg", "*.set"))
)
subject_epochs_dict = {}

for i, file_path in enumerate(eeg_files):
    subject_id = get_subject_id_from_path(file_path)
    if subject_id is None or subject_id not in participants_df_target.index:
        continue

    print(f"Processing ({i+1}/{len(eeg_files)}) {subject_id}...", end="\r")
    raw = load_and_preprocess_eeg(file_path, sfreq)
    if raw and raw.times[-1] >= tmax:
        epochs = make_fixed_length_epochs(
            raw.copy().crop(tmin=tmin, tmax=tmax),
            duration=epoch_length_sec,
            overlap=overlap_sec,
            preload=True,
        )
        if len(epochs) > 0:
            subject_epochs_dict[subject_id] = epochs

print("\nEpoch gathering complete.")


# =============================================================================
# --- 4. STAGE 2: SUBJECT-LEVEL FEATURE ENGINEERING ---
# =============================================================================
print("\nStep 2: Engineering features for each subject...")
X_subjects = []
y_subjects = []
subject_ids_list = []  # Keep track of subjects for whom we have features

tangent_space_transformer = TangentSpace(metric="riemann")
cov_estimator = Covariances(estimator="lwf")

# Fit the TangentSpace transformer on all available data first
all_covs_for_fitting = []
for subject_id, epochs in subject_epochs_dict.items():
    covs = cov_estimator.fit_transform(epochs.get_data())
    all_covs_for_fitting.extend(covs)
if not all_covs_for_fitting:
    raise ValueError("No epochs found to fit the TangentSpace transformer.")
tangent_space_transformer.fit(np.array(all_covs_for_fitting))

for subject_id, epochs in subject_epochs_dict.items():
    # Riemannian Features
    band_mean_covs = [
        mean_covariance(
            cov_estimator.transform(epochs.copy().filter(*FREQ_BANDS[band]).get_data())
        )
        for band in BAND_ORDER
    ]
    riemannian_features = tangent_space_transformer.transform(
        np.array(band_mean_covs)
    ).flatten()

    # Spectral Power Features
    spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds = psds.mean(axis=(0, 1))

    def get_band_power(f_low, f_high):
        idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
        return np.mean(psds[idx_band])

    delta_power = get_band_power(*FREQ_BANDS["Delta"])
    theta_power = get_band_power(*FREQ_BANDS["Theta"])
    alpha_power = get_band_power(*FREQ_BANDS["Alpha"])
    beta_power = get_band_power(*FREQ_BANDS["Beta"])

    theta_alpha_ratio = theta_power / (alpha_power + 1e-10)
    slowing_ratio = (delta_power + theta_power) / (alpha_power + beta_power + 1e-10)

    # Combine features and add to lists
    combined_features = np.concatenate(
        [riemannian_features, np.array([theta_alpha_ratio, slowing_ratio])]
    )
    X_subjects.append(combined_features)
    subject_ids_list.append(subject_id)

    # Append the correct target variable based on the task
    if task == "regression":
        y_subjects.append(participants_df_target.loc[subject_id, "MMSE"])
    else:
        y_subjects.append(participants_df_target.loc[subject_id, "label"])

X_subjects = np.array(X_subjects)
y_subjects = np.array(y_subjects)
print(f"Engineered combined features for {len(X_subjects)} subjects.")


# =============================================================================
# --- 5. PIPELINE DEFINITION AND EVALUATION ---
# =============================================================================
print(f"\nStep 3: Running Cross-Validation for {task} task...")

if task == "regression":
    # --- REGRESSION LOGIC ---
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=random_state, n_estimators=150, n_jobs=-1
                ),
            ),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X_subjects, y_subjects, cv=cv)

    print("\n--- Overall Regression Performance ---")
    mae = mean_absolute_error(y_subjects, y_pred)
    corr, p = scipy.stats.pearsonr(y_subjects, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"Correlation (Pearson's r): {corr:.2f} (p-value: {p:.3f})")

    print("\n--- Generating Visualization ---")
    # Set a clean style with white background
    sns.set_style("whitegrid")
    sns.set(font_scale=2.5)
    plt.figure(figsize=(10, 12))

    # Create a more appealing color palette
    palette = sns.color_palette("viridis")

    # Plot with enhanced visuals
    ax = sns.regplot(
        x=y_subjects,
        y=y_pred,
        scatter_kws={
            "alpha": 0.7,
            "color": "blue",
            "s": 150,  # Larger markers
            "edgecolor": "white",
            "linewidths": 0.5,
        },
        line_kws={"color": "blue", "linewidth": 2.5, "linestyle": "-"},
        ci=95,  # Show confidence interval
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    stats_text = f"Pearson's r = {corr:.2f}, p < {p:.3f}\nMean Absolute Error (MAE) = {mae:.2f}"
    props = dict(boxstyle="round", facecolor=palette[0], alpha=0.2)
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=props,
    )

    plt.xlabel("True MMSE Score")
    plt.ylabel("Predicted MMSE Score")
    plt.title(
        "Prediction of Mini-Mental State Examination (MMSE) Scores",
        pad=10,
    )

    plt.legend(loc="best", frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("mmse_prediction_plot.svg", bbox_inches="tight")

else:
    # --- CLASSIFICATION LOGIC ---
    pipeline = Pipeline(
        [
            (
                "classifier",
                BalancedRandomForestClassifier(
                    random_state=random_state, n_estimators=1000, n_jobs=-1
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    y_preds_overall = np.array([])
    y_tests_overall = np.array([])
    accuracies, balanced_accuracies, conf_matrices, precisions, recalls, f1s = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for fold, (train_index, test_index) in enumerate(cv.split(X_subjects, y_subjects)):
        X_train, X_test = X_subjects[train_index], X_subjects[test_index]
        y_train, y_test = y_subjects[train_index], y_subjects[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_preds_overall = np.append(y_preds_overall, y_pred)
        y_tests_overall = np.append(y_tests_overall, y_test)

        # Calculate all metrics for the current fold
        accuracies.append(accuracy_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))

        # MODIFICATION: Calculate and store precision, recall, and F1
        # Using average='macro' to calculate metrics for each label and find their unweighted mean.
        precisions.append(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        conf_matrices.append(
            confusion_matrix(y_test, y_pred, labels=np.unique(y_subjects))
        )

    # --- MODIFIED RESULTS SUMMARY ---
    print("\n--- Cross-Validation Results Summary ---")
    print(
        f"Mean Accuracy:           {np.mean(accuracies):.2f} +/- {np.std(accuracies):.2f}"
    )
    print(
        f"Mean Balanced Accuracy:  {np.mean(balanced_accuracies):.2f} +/- {np.std(balanced_accuracies):.2f}"
    )
    print(
        f"Mean Precision (Macro):  {np.mean(precisions):.2f} +/- {np.std(precisions):.2f}"
    )
    print(f"Mean Recall (Macro):     {np.mean(recalls):.2f} +/- {np.std(recalls):.2f}")
    print(f"Mean F1-Score (Macro):   {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}")

    print("\n--- Overall Classification Report ---")
    print(
        classification_report(
            y_tests_overall, y_preds_overall, target_names=target_names_display
        )
    )

    print("\n--- Aggregated Confusion Matrix ---")
    avg_conf_matrix = np.sum(conf_matrices, axis=0)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        avg_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names_display,
        yticklabels=target_names_display,
    )
    plt.title(f"Aggregated Confusion Matrix ({task})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_{task}.svg", bbox_inches="tight")


# =============================================================================
# --- 6. FEATURE IMPORTANCE ANALYSIS ---
# =============================================================================
print(f"\nStep 4: Running Feature Importance Analysis for {task} task...")

# --- Create meaningful feature names ---
ts_feature_len = N_CHANNELS * (N_CHANNELS + 1) // 2
feature_names = []
for band in BAND_ORDER:
    feature_names.extend([f"TS_{band}_{i+1}" for i in range(ts_feature_len)])
feature_names.extend(["Theta/Alpha Ratio", "Slowing Ratio"])

# --- Train the final model on all data ---
final_model_pipeline = pipeline.fit(X_subjects, y_subjects)

# --- Extract importances from the model in the pipeline ---
if task == "regression":
    importances = final_model_pipeline.named_steps["regressor"].feature_importances_
else:
    importances = final_model_pipeline.named_steps["classifier"].feature_importances_

# --- Group importances by source ---
feature_importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
)


def get_feature_group(feature_name):
    if "TS_Delta" in feature_name:
        return "Connectivity (Delta)"
    if "TS_Theta" in feature_name:
        return "Connectivity (Theta)"
    if "TS_Alpha" in feature_name:
        return "Connectivity (Alpha)"
    if "TS_Beta" in feature_name:
        return "Connectivity (Beta)"
    return "Spectral Ratios"


feature_importance_df["group"] = feature_importance_df["feature"].apply(
    get_feature_group
)
grouped_importances = (
    feature_importance_df.groupby("group")["importance"]
    .mean()
    .sort_values(ascending=False)
)

# --- Plot the grouped feature importances ---
print("\n--- Generating Feature Importance Visualization ---")
plt.figure(figsize=(8, 7))
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.5)
ax = sns.barplot(
    x=grouped_importances.index, y=grouped_importances.values, palette="viridis"
)
ax.set_title(f"Mean Feature Importance by Group ({task})")
ax.set_ylabel("Mean Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"feature_importance_{task}.svg", bbox_inches="tight")
# =============================================================================
