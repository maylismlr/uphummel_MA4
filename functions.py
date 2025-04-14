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

def load_data(folder_path, rois, type = 'all'):
    '''
    Load data from .mat files in the specified folder. Either get all matrices (enter type = 'all'), 
    or only T1 matrices (enter type = 't1_only'), or only T1 and T3 matrices matched (enter type = 't1_t3_matched),
    or only T1 and T4 matrices matched (enter type = 't1_t4_matched), or all T1 and T3 matrices (enter type = 't1_t3'),
    or or all T1 and T4 matrices (enter type = 't1_t4').
    '''
    # Load Excel files
    regression_info = pd.read_excel("TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")

    # Keep only the first appearance of each subject_id
    subject_info = regression_info.copy().drop_duplicates(subset=["subject_id"], keep="first")

    # Merge on subject_id
    rsfMRI_full_info = rsfMRI_full_info.merge(subject_info, on="subject_id", how="left")
    rsfMRI_full_info = rsfMRI_full_info[['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3','Gender','Age','Education_level','Combined', 'Bilateral']]
    
    # Extract last 4 characters of subject_id
    valid_subjects = rsfMRI_full_info["subject_id"].astype(str).str[-4:].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    data_rows = []

    for sub in subjects:
        sub_folder = os.path.join(folder_path, sub)
        files = [f for f in os.listdir(sub_folder) if f.endswith('.mat')]

        t1_matrix = None
        t2_matrix = None
        t3_matrix = None
        t4_matrix = None

        for mat_file in files:
            mat_file_path = os.path.join(sub_folder, mat_file)

            # Load the matrix
            mat_data = scipy.io.loadmat(mat_file_path)
            if 'CM' not in mat_data:
                continue

            matrix = pd.DataFrame(mat_data['CM']).iloc[rois, rois]
            matrix = matrix.replace('None', np.nan)

            if 'T1' in mat_file:
                t1_matrix = matrix
            elif 'T2' in mat_file:
                t2_matrix = matrix
            elif 'T3' in mat_file:
                t3_matrix = matrix
            elif 'T4' in mat_file:
                t4_matrix = matrix

        data_rows.append({
            "subject_id": sub,
            "T1_matrix": t1_matrix,
            "T2_matrix": t2_matrix,
            "T3_matrix": t3_matrix,
            "T4_matrix": t4_matrix
        })

    # Create dataframe
    df = pd.DataFrame(data_rows)
    
    # Merge the DataFrame with rsfMRI_info on subject_id
    df = df.merge(rsfMRI_full_info, on="subject_id", how="left")
    
    if type == 't1_only':
        t1_matrices = df.copy().drop(columns=['T2_matrix', 'T3_matrix', 'T4_matrix'])

        return t1_matrices, regression_info, rsfMRI_full_info, subjects
    
    elif type == 't1_t3_matched':
        # keep only rows where both T1 and T3 matrices are not None
        t1_t3_matrices = df.copy().dropna(subset=['T1_matrix', 'T3_matrix'])
        t1_t3_matrices = t1_t3_matrices.drop(columns=['T2_matrix', 'T4_matrix'])
        
        return t1_t3_matrices, regression_info, rsfMRI_full_info, subjects
    
    elif type == 't1_t4_matched':
        # keep only rows where both T1 and T3 matrices are not None
        t1_t4_matrices = df.copy().dropna(subset=['T1_matrix', 'T4_matrix'])
        t1_t4_matrices = t1_t4_matrices.drop(columns=['T2_matrix', 'T3_matrix'])
        
        return t1_t4_matrices, regression_info, rsfMRI_full_info, subjects
    
    elif type == 't1_t3':
        # keep only rows where T1 or T3 matrices are not None
        t1_t3_matrices = df.copy().drop(columns=['T2_matrix', 'T4_matrix'])
        t1_t3_matrices = t1_t3_matrices[~(t1_t3_matrices['T1_matrix'].isna() & t1_t3_matrices['T3_matrix'].isna())]
        
        return t1_t3_matrices, regression_info, rsfMRI_full_info, subjects
    
    elif type == 't1_t4':
        # keep only rows where T1 or T4 matrices are not None
        t1_t4_matrices = df.copy().drop(columns=['T2_matrix', 'T3_matrix'])
        t1_t4_matrices = t1_t4_matrices[~(t1_t4_matrices['T1_matrix'].isna() & t1_t4_matrices['T4_matrix'].isna())]

        return t1_t4_matrices, regression_info, rsfMRI_full_info, subjects
    
    return df, regression_info, rsfMRI_full_info, subjects

# plot the heatmap of the matrices
def plot_all_subject_matrices(subject_matrices, subjects, type='t1_t3'):
    '''
    go through all the subjects and plot the matrices as sns heatmaps, with each row being a subject 
    and each column a timepoint. The timepoints can be modulated based on the input.
    '''
    num_subjects = len(subjects)
    
    if type == 'all':
        timepoints = ['T1', 'T2', 'T3', 'T4']
    elif type == 't1_only':
        timepoints = ['T1']
    elif type == 't1_t3' or type == 't1_t3_matched':
        timepoints = ['T1', 'T3']
    elif type == 't1_t4' or type == 't1_t4_matched':
        timepoints = ['T1', 'T4']
    
    fig, axes = plt.subplots(num_subjects, len(timepoints), figsize=(15, num_subjects * 3))
    
    for i, subject in enumerate(subjects):
        for j, timepoint in enumerate(timepoints):
            ax = axes[i, j] if num_subjects > 1 else axes[j]
            if timepoint in subject_matrices.columns:
                matrix = subject_matrices.loc[subject, timepoint]
                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=False)
                ax.set_title(f'Subject {subject} - {timepoint}')
            else:
                ax.axis('off')  # Hide axes if timepoint is not available
            ax.axis('off')
    
    plt.tight_layout()
    plt.show();
    

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
    plt.show();
    
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
    print(f"shape of T{tp} matrix: ", np.shape(t_matrices))
    
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
    plt.show();
    
    return significant_matrix, p_vals_corrected, reject


def sig_matrix_T1_T(df, tp=3, alpha=0.05, clusters=False):
    results = {}
    if cluster == False:
        return get_sig_matrix(df, tp, alpha)
    
    elif clusters == True:
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

            plt.figure(figsize=(8, 4))
            sns.heatmap(significant_matrix, cmap='viridis', cbar=True, annot=True, square=True)
            plt.title("Significance Heatmap (FDR-corrected)")
            plt.xlabel("ROIs")
            plt.ylabel("ROIs")
            plt.tight_layout()
            plt.show();

        return results
    
def compute_FC_diff(df, tp=3):
    # Extract T1 and T4 matrices
    t1_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df['T1_matrix']]
    t_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df[f'T{tp}_matrix']]

    # Convert to numpy arrays (shape: [n_subjects, n_rois, n_rois])
    t1_array = np.stack(t1_matrices)
    t_array = np.stack(t_matrices)

    # Compute difference
    diff_array = t_array - t1_array
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.heatmap(diff_array.mean(axis=0), cmap='viridis', cbar=True, annot=False, square=True)
    plt.title(f"Mean FC Difference (T{tp} - T1)")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.show();

    return diff_array