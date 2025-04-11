import os
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# for clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_data_T1_only(folder_path, rois):
    # Load Excel files
    rsfMRI_info = pd.read_excel("TiMeS_rsfMRI_info.xlsx", engine="openpyxl")  
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")

    # Keep only the first appearance of each subject_id
    regression_info = regression_info.drop_duplicates(subset=["subject_id"], keep="first")

    # Merge on subject_id
    rsfMRI_info = rsfMRI_info.merge(regression_info, on="subject_id", how="left")
    
    # Extract last 4 characters of subject_id
    valid_subjects = rsfMRI_info["subject_id"].astype(str).str[-4:].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    # Store T1 matrices in a list
    t1_matrices = []

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)

        # Get all .mat files containing 'T1'
        t1_files = [f for f in os.listdir(sub_folder) if f.endswith('.mat') and 'T1' in f]

        for mat_file in t1_files:
            mat_file_path = os.path.join(sub_folder, mat_file)
            mat_data = scipy.io.loadmat(mat_file_path)

            if 'CM' in mat_data:
                df = pd.DataFrame(mat_data['CM'])
                df_only_rois = df.iloc[rois, rois]
                t1_matrices.append(df_only_rois)

    return t1_matrices, rsfMRI_full_info, rsfMRI_info, subjects

def get_valid_subjects(folder_path):
    # Keep subjects in subject list that have both T1 and T3
    subjects = [sub for sub in os.listdir(folder_path) if not sub.startswith('.')]
    t1_subjects = [sub for sub in subjects if 'T1' in sub]
    t3_subjects = [sub for sub in subjects if 'T3' in sub]
    valid_subjects = set(t1_subjects) & set(t3_subjects)
    return valid_subjects

def load_data(folder_path, rois, valid_subjects=None):
    # Load Excel files
    rsfMRI_info = pd.read_excel("TiMeS_rsfMRI_info.xlsx", engine="openpyxl")
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")
    rsfMRI_merged_info = regression_info[['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3']]
    
    # Extract last 4 characters of subject_id
    valid_subjects = rsfMRI_info["subject_id"].astype(str).str[-4:].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    # Define session order
    session_order = ['T1', 'T2', 'T3', 'T4']

    # Store matrices: { subject_id: [matrix_T1, matrix_T2, matrix_T3, matrix_T4] }
    subject_matrices = {}

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)
        files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

        # Prepare a session-to-matrix map for this subject
        session_matrices = {session: None for session in session_order}

        for mat_file in files:
            for session in session_order:
                if session in mat_file:
                    mat_file_path = os.path.join(sub_folder, mat_file)
                    mat_data = scipy.io.loadmat(mat_file_path)

                    if 'CM' in mat_data:
                        df = pd.DataFrame(mat_data['CM'])
                        df_only_rois = df.iloc[rois, rois]
                        session_matrices[session] = df_only_rois
                    break  # Don't keep checking once a match is found

        # Keep them in order (T1 to T4)
        ordered_matrices = [session_matrices[session] for session in session_order if session_matrices[session] is not None]
        if ordered_matrices:
            subject_matrices[sub] = ordered_matrices

    rows = []

    for subject_id, matrices in subject_matrices.items():
        row = {"subject_id": subject_id}
        for idx, matrix in enumerate(matrices):
            if idx < len(session_order):
                row[f"{session_order[idx]}_matrix"] = matrix
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Make sure all session columns are present, even if some are missing
    for session in session_order:
        col = f"{session}_matrix"
        if col not in df.columns:
            df[col] = None
    
    # Merge the DataFrame with rsfMRI_info on subject_id
    df = df.merge(rsfMRI_merged_info, on="subject_id", how="left")
    
    return df, rsfMRI_full_info, rsfMRI_info, list(subject_matrices.keys())


def matrices_to_wide_df(subject_matrices):
    # Load Excel files
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_merged_info = regression_info[['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3']]
    
    session_labels = ['T1', 'T2', 'T3', 'T4']
    rows = []

    for subject_id, matrices in subject_matrices.items():
        row = {"subject_id": subject_id}
        for idx, matrix in enumerate(matrices):
            if idx < len(session_labels):
                row[f"{session_labels[idx]}_matrix"] = matrix
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Make sure all session columns are present, even if some are missing
    for session in session_labels:
        col = f"{session}_matrix"
        if col not in df.columns:
            df[col] = None

    # Merge the DataFrame with rsfMRI_info on subject_id
    df = df.merge(rsfMRI_merged_info, on="subject_id", how="left")
    
    return df

'''
DOESN'T WORK CORRECTLY

def load_data_wide(folder_path, rois):
    # Load Excel files
    rsfMRI_info = pd.read_excel("TiMeS_rsfMRI_info.xlsx", engine="openpyxl")  
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")

    # Keep only the first appearance of each StudyID
    regression_info = regression_info.drop_duplicates(subset=["StudyID"], keep="first")

    # Merge on StudyID
    rsfMRI_info = rsfMRI_info.merge(regression_info, on="StudyID", how="left")
    
    # Extract last 4 characters of StudyID
    valid_subjects = rsfMRI_info["StudyID"].astype(str).str[-4:].tolist()

    # Match folders with valid StudyIDs
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    # Define session order
    session_order = ['T1', 'T2', 'T3', 'T4']

    # Store rows for wide-format DataFrame
    data_rows = []

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)
        files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

        # Prepare a dictionary with all session matrices set to None initially
        row = {"subject_id": sub, **{f"{s}_matrix": None for s in session_order}}

        for mat_file in files:
            for session in session_order:
                if session in mat_file:
                    mat_file_path = os.path.join(sub_folder, mat_file)
                    mat_data = scipy.io.loadmat(mat_file_path)

                    if 'CM' in mat_data:
                        df = pd.DataFrame(mat_data['CM'])
                        df_only_rois = df.iloc[rois, rois]
                        row[f"{session}_matrix"] = df_only_rois
                    break  # Session match found; move to next file

        data_rows.append(row)

    # Create the wide-format DataFrame
    df_matrices = pd.DataFrame(data_rows)

    return df_matrices, rsfMRI_full_info, rsfMRI_info, df_matrices['subject_id'].tolist()
'''

def plot_all_subject_matrices(folder_path, rois):
    # Load subjects and metadata
    subject_matrices, rsfMRI_full_info, rsfMRI_info, subjects = load_data(folder_path, rois)

    # Prepare a dictionary with all .mat files per subject
    subject_files = {
        sub: [f for f in os.listdir(os.path.join(folder_path, sub)) if f.endswith('.mat')]
        for sub in subjects
    }

    # Determine the maximum number of columns needed (most .mat files in any subject)
    max_columns = max(len(files) for files in subject_files.values())
    num_rows = len(subjects)

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, max_columns, figsize=(max_columns * 4, num_rows * 4), squeeze=False)

    # Loop through subjects and plot
    for row_idx, (sub, files) in enumerate(subject_files.items()):
        sub_folder = os.path.join(folder_path, sub)

        for col_idx, mat_file in enumerate(files):
            mat_file_path = os.path.join(sub_folder, mat_file)

            # Try to extract session info
            session_info = mat_file[22:24] if len(mat_file) >= 24 else "??"

            # Load .mat file
            mat_data = scipy.io.loadmat(mat_file_path)
            df = pd.DataFrame(mat_data['CM'])

            # Extract ROIs
            df_only_rois = df.iloc[rois, rois]

            # Plot heatmap
            ax = axes[row_idx, col_idx]
            sns.heatmap(df_only_rois, cmap="viridis", annot=False, square=True,
                        xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title(f"{sub} - Session: {session_info}", fontsize=10)

        # Hide unused axes in the row
        for col_idx in range(len(files), max_columns):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

def cluster_and_plot(matrices, numerical_cols_names, categorical_cols_name):
    # Preprocess categorical columns
    matrices[categorical_cols_name] = matrices[categorical_cols_name].fillna('Unknown')  # Handle missing values
    matrices_encoded = pd.get_dummies(matrices[categorical_cols_name], drop_first=True)

    # Extract numerical column
    numerical_cols = matrices[numerical_cols_names].fillna(0).values.reshape(-1, 1)

    # Flatten the upper triangle of T1_matrix
    baseline_matrices = matrices['T1_matrix']

    def flatten_upper(mat):
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat  # ensure it's an array
        return mat[np.triu_indices_from(mat, k=1)]

    X_matrix = np.array([flatten_upper(m) for m in baseline_matrices])

    # Concatenate all features
    X_combined = np.hstack([X_matrix, matrices_encoded.values, numerical_cols])

    # Scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title("Patient Clusters at Baseline")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Cluster")
    plt.show()
