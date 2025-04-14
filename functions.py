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

# for statistical tests
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


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

def load_data_2(folder_path, rois, valid_subjects=None):
    
    # Load Excel files
    rsfMRI_info = pd.read_excel("TiMeS_rsfMRI_info.xlsx", engine="openpyxl")
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")
    rsfMRI_merged_info = regression_info[['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3']]

    # Extract valid subjects from last 4 characters
    valid_subjects = rsfMRI_full_info["subject_id"].astype(str).str[-4:].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    data_rows = []

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)
        files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

        t1_matrix = None
        t3_matrix = None

        for mat_file in files:
            mat_file_path = os.path.join(sub_folder, mat_file)

            # Load the matrix
            mat_data = scipy.io.loadmat(mat_file_path)
            if 'CM' not in mat_data:
                continue

            matrix = pd.DataFrame(mat_data['CM']).iloc[rois, rois]

            if 'T1' in mat_file:
                t1_matrix = matrix
            elif 'T3' in mat_file:
                t3_matrix = matrix

        # Only keep subjects with both T1 and T3
        if t1_matrix is not None and t3_matrix is not None:
            data_rows.append({
                "subject_id": sub,
                "T1_matrix": t1_matrix,
                "T3_matrix": t3_matrix
            })

    # Create dataframe
    df = pd.DataFrame(data_rows)

    # Merge metadata
    df = df.merge(rsfMRI_merged_info, on="subject_id", how="left")

    return df, rsfMRI_full_info, rsfMRI_info, df["subject_id"].tolist()

def load_data_3(folder_path, rois, valid_subjects=None):
    
    # Load Excel files
    rsfMRI_info = pd.read_excel("TiMeS_rsfMRI_info.xlsx", engine="openpyxl")
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")
    rsfMRI_merged_info = regression_info[['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3']]

    # Extract valid subjects from last 4 characters
    valid_subjects = rsfMRI_full_info["subject_id"].astype(str).str[-4:].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    data_rows = []

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)
        files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

        t1_matrix = None
        t4_matrix = None

        for mat_file in files:
            mat_file_path = os.path.join(sub_folder, mat_file)

            # Load the matrix
            mat_data = scipy.io.loadmat(mat_file_path)
            if 'CM' not in mat_data:
                continue

            matrix = pd.DataFrame(mat_data['CM']).iloc[rois, rois]

            if 'T1' in mat_file:
                t1_matrix = matrix
            elif 'T4' in mat_file:
                t4_matrix = matrix

        # Only keep subjects with both T1 and T4
        if t1_matrix is not None and t4_matrix is not None:
            data_rows.append({
                "subject_id": sub,
                "T1_matrix": t1_matrix,
                "T4_matrix": t4_matrix
            })

    # Create dataframe
    df = pd.DataFrame(data_rows)

    # Merge metadata
    df = df.merge(rsfMRI_merged_info, on="subject_id", how="left")

    return df, rsfMRI_full_info, rsfMRI_info, df["subject_id"].tolist()


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

def flatten_upper(mat):
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat  # ensure it's an array
        return mat[np.triu_indices_from(mat, k=1)]

def cluster_and_plot(matrices, numerical_cols_names, categorical_cols_name, clusters = 2):
    # Preprocess categorical columns
    matrices[categorical_cols_name] = matrices[categorical_cols_name].fillna('Unknown')  # Handle missing values
    matrices_encoded = pd.get_dummies(matrices[categorical_cols_name], drop_first=True)

    # Extract numerical column
    numerical_cols = matrices[numerical_cols_names].fillna(0).values.reshape(-1, 1)

    # Flatten the upper triangle of T1_matrix
    baseline_matrices = matrices['T1_matrix']

    X_matrix = np.array([flatten_upper(m) for m in baseline_matrices])

    # Concatenate all features
    X_combined = np.hstack([X_matrix, matrices_encoded.values, numerical_cols])

    # Scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Perform clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
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
    
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {score}")
    
    subject_ids = matrices.loc[baseline_matrices.index, 'subject_id'].tolist()
    
    # Create a DataFrame with subject IDs and their corresponding cluster labels
    subject_cluster_df = pd.DataFrame({'subject_id': subject_ids, 'cluster': labels})

    # Group subjects by cluster label
    grouped_subjects = subject_cluster_df.groupby('cluster')['subject_id'].apply(list).to_dict()

    # Print subjects in each cluster
    for cluster, subjects in grouped_subjects.items():
        print(f"Cluster {cluster}: {subjects}")
    
    # Create a DataFrame mapping subject IDs to their cluster labels
    subject_cluster_df = pd.DataFrame({'subject_id': subject_ids, 'cluster': labels})

    # Merge the cluster DataFrame with the original DataFrame (matrices)
    matrices_with_clusters = matrices.merge(subject_cluster_df, on='subject_id', how='left')
    
    return matrices_with_clusters

def get_sig_matrix(df, tp = 3, alpha=0.05, cluster = False):
    # Create lists of matrices
    t1_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df['T1_matrix']]
    t_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df[f'T{tp}_matrix']]

    # Optionally convert to numpy arrays (shape: [n_subjects, n_rois, n_rois])
    t1_array = np.stack(t1_matrices)
    t_array = np.stack(t_matrices)

    print("shape of T1 matrix: ", np.shape(t1_matrices))
    
    # Paired t-test
    t_stat, p_val = ttest_rel(t1_array, t_array, axis=0)

    # Flatten p-values to 1D if needed
    p_val_flat = p_val.ravel()

    # FDR correction
    reject, p_vals_corrected, _, _ = multipletests(p_val_flat, alpha=alpha, method='holm')

    # Reshape corrected p-values and reject back to original shape if necessary
    p_vals_corrected = p_vals_corrected.reshape(p_val.shape)
    reject = reject.reshape(p_val.shape)

    # Create significant matrix
    significant_matrix = np.zeros_like(p_val, dtype=int)
    significant_matrix[reject] = 1

    plt.figure(figsize=(10, 6))
    sns.heatmap(significant_matrix, cmap='viridis', cbar=True, annot=True, square=True)
    plt.title("Significance Heatmap (FDR-corrected)")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.show()
    
    return significant_matrix, p_vals_corrected, reject


def compare_T1_T_by_cluster(df, rois, tp=3, alpha=0.05, cluster=False):
    n_rois = len(rois)
    results = {}
    if cluster == False:
        return get_sig_matrix(df, tp, alpha)
    
    elif cluster == True:
        for cluster in sorted(df['cluster'].unique()):
            print(f"\nAnalyzing Cluster {cluster}...")

            # Subset to current cluster
            cluster_df = df[df['cluster'] == cluster]

            # Ensure subjects have both T1 and T4
            cluster_df = cluster_df.dropna(subset=['T1_matrix', f'T{tp}_matrix'])

            if cluster_df.empty:
                print(f" - No data for Cluster {cluster}")
                continue

            significant_matrix, p_vals_corrected, reject = get_sig_matrix(cluster_df, tp, alpha)

            # Store results
            results[cluster] = {
                'signif_matrix': significant_matrix,
                'p_corrected': p_vals_corrected,
                'rejected': reject,
            }

            plt.figure(figsize=(10, 6))
            sns.heatmap(significant_matrix, cmap='viridis', cbar=True, annot=True, square=True)
            plt.title("Significance Heatmap (FDR-corrected)")
            plt.xlabel("ROIs")
            plt.ylabel("ROIs")
            plt.tight_layout()
            plt.show()

        return results