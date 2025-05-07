import os
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# for clustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# for statistical tests
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro

# for prettiness <3
from tqdm import tqdm

def load_data(folder_path, rois, type = 'all'):
    '''
    Load data from .mat files in the specified folder. Either get all matrices (enter type = 'all'), 
    or only T1 matrices (enter type = 't1_only'), or only T1 and T3 matrices matched (enter type = 't1_t3_matched),
    or only T1 and T4 matrices matched (enter type = 't1_t4_matched), or all T1 and T3 matrices (enter type = 't1_t3'),
    or or all T1 and T4 matrices (enter type = 't1_t4').
    '''
    # Load Excel files
    regression_info = pd.read_excel("data/TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel("data/TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")

    # Keep only the first appearance of each subject_full_id
    subject_info = regression_info.copy().drop_duplicates(subset=["subject_full_id"], keep="first")

    # Merge on subject_full_id
    rsfMRI_full_info = rsfMRI_full_info.merge(subject_info, on="subject_full_id", how="left")
    rsfMRI_full_info = rsfMRI_full_info[['subject_full_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3','Gender','Age','Education_level','Combined', 'Bilateral']]
    rsfMRI_full_info['subject_id'] = rsfMRI_full_info['subject_full_id'].astype(str).str[-4:] # Extract last 4 characters of subject_id to match with folder names and subjects
    
    # Extract last 4 characters of subject_id
    valid_subjects = rsfMRI_full_info["subject_id"].tolist()

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
    
    # plot mean FC matrices
    plot_mean_FC_matrices(df, rois)
    
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
def plot_all_subject_matrices(subject_matrices, subjects, rois, type='t1_t3'):
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
    
    for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
        for j, timepoint in enumerate(timepoints):
            ax = axes[i, j] if num_subjects > 1 else axes[j]
            if timepoint in subject_matrices.columns:
                matrix = subject_matrices.loc[subject, timepoint]
                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=False, vmin=0, vmax=1, xticklabels=rois, yticklabels=rois)
                ax.set_title(f'Subject {subject} - {timepoint}')
            else:
                ax.axis('off')  # Hide axes if timepoint is not available
            ax.axis('off')

    plt.tight_layout()
    plt.show();
    
def plot_mean_FC_matrices(matrices, rois):
    """
    Plot the mean FC matrix for each timepoint across all subjects.
    Args:
        matrices (pd.DataFrame): DataFrame containing the FC matrices for each subject.
        rois (list): List of ROI indices.
    """
    matrix_columns = ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']

    for timepoint in matrix_columns:
        valid_matrices = matrices[timepoint].dropna()

        if len(valid_matrices) == 0:
            print(f"No data available for {timepoint}")
            continue

        # Stack matrices
        stacked = np.stack([mat.values for mat in valid_matrices])

        # Compute mean
        mean_matrix = np.nanmean(stacked, axis=0)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_matrix, cmap='viridis', cbar=True, xticklabels=rois, yticklabels=rois)
        plt.title(f'Mean FC Matrix of all subjects - {timepoint}')
        plt.xlabel('ROIs')
        plt.ylabel('ROIs')
        plt.show()

    

def flatten_upper(mat):
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat  # ensure it's an array
        return mat[np.triu_indices_from(mat, k=1)]


# ANCIENNE VERSION
def cluster_and_plot(matrices, numerical_cols_names, categorical_cols_name, clusters = 2, plot = True):
    ''' 
    uses Kmeans clustering to cluster the patients based on their T1 matrices and other features.
    Remark: T1_matrices cannot be 'None' here ! So we drop the rows that don't have T1 matrices => a little more data
    than t1_t_matched, but not much.
    '''
    
    matrices = matrices.dropna(subset=['T1_matrix']).copy()  # Handle NaN values
    
    # Preprocess categorical columns
    matrices[categorical_cols_name] = matrices[categorical_cols_name].fillna('Unknown')  # Handle missing values
    matrices_encoded = pd.get_dummies(matrices[categorical_cols_name], drop_first=True)
    print(matrices_encoded.columns)
    
    # Extract numerical column
    numerical_cols = matrices[numerical_cols_names].fillna(0).values # not sure if this is the right way to handle missing values

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

    if plot:
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

# CLUSTERING FUNCTIONS
def compute_feature_importance(features, clusters, feature_names, top_features=10, plot=True):
    """
    Compute feature importance by comparing mean feature values across clusters.

    Args:
        features (np.array): Feature matrix after flattening (no PCA)
        clusters (np.array): Cluster labels
        feature_names (list): Names of the features

    Returns:
        importance_df (pd.DataFrame): Features sorted by absolute mean difference
    """
    df = pd.DataFrame(features, columns=feature_names)
    df['cluster'] = clusters

    mean_by_cluster = df.groupby('cluster').mean()
    if mean_by_cluster.shape[0] != 2:
        print("Warning: more than 2 clusters! Using difference between max and min clusters.")
        diff = mean_by_cluster.max() - mean_by_cluster.min()
    else:
        diff = mean_by_cluster.iloc[0] - mean_by_cluster.iloc[1]

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_difference': diff.abs()
    }).sort_values(by='mean_difference', ascending=False)
    
    if plot:
        plt.figure(figsize=(10,6))
        sns.barplot(data=importance.head(top_features), x='mean_difference', y='feature')
        plt.title("Top 10 discriminative connectivities between clusters")
        plt.xlabel("Absolute Mean Difference")
        plt.show();

    return importance


def flatten_selected_rois(fc_matrix, selected_rois_labels):
    """
    Flatten connectivities from selected ROIs by label (not by index).
    Args:
        fc_matrix (pd.DataFrame): FC matrix with ROI labels as index and columns.
        selected_rois_labels (list): List of ROI labels to select (e.g., [362, 363, ...])
    """
    flattened = []
    for src_label in selected_rois_labels:
        # Get all connections from this ROI to others
        connections = fc_matrix.loc[src_label, :]  # accès par label
        connections = connections.drop(labels=src_label, errors='ignore')  # enlever self-connection
        flattened.append(connections.values)
    return np.concatenate(flattened)


def generate_feature_names(selected_rois_labels, all_roi_labels, numerical_cols, categorical_cols, ohe):
    """
    Generate feature names corresponding to flattened FC features + numerical + categorical features.
    
    Args:
        selected_rois_labels (list): List of selected source ROIs labels.
        all_roi_labels (list): List of all ROI labels (both rows and columns of the FC matrix).
        numerical_cols (list): List of numerical variable names.
        categorical_cols (list): List of categorical variable names.
        ohe (OneHotEncoder): Fitted OneHotEncoder object.

    Returns:
        feature_names (list): List of all feature names.
    """
    roi_connection_names = []
    for src_label in selected_rois_labels:
        for tgt_label in all_roi_labels:
            if src_label != tgt_label:
                roi_connection_names.append(f"{src_label}-{tgt_label}")

    # For numerical features
    numerical_names = list(numerical_cols)

    # For categorical features (after one-hot encoding)
    categorical_names = list(ohe.get_feature_names_out(categorical_cols))

    feature_names = roi_connection_names + numerical_names + categorical_names
    return feature_names


def cluster_subjects(df, selected_rois_labels, matrix_column='T1_matrix', numerical_cols=None, categorical_cols=None, n_components=20, max_clusters=10, random_state=42):
    """
    Full pipeline to flatten FC matrices, combine clinical features, reduce dimensionality, cluster, and return cluster labels.
    Works only on available T1 matrices (subjects missing T1 will be automatically excluded).
    Remark: T1_matrices cannot be 'None' here ! So we drop the rows that don't have T1 matrices => a little more data
    than t1_t_matched, but not much.
    """
    # Remove subjects where the T1_matrix is missing
    df = df[df[matrix_column].notnull()].copy()

    # Get list of all ROI labels from the first FC matrix
    all_roi_labels = df[matrix_column].iloc[0].index.tolist()
    
    # Flatten T1 FC matrices
    print(f"Flattening FC matrices on {len(df)} subjects...")
    flattened_fc = df[matrix_column].apply(lambda mat: flatten_selected_rois(mat, selected_rois_labels))
    fc_features = np.vstack(flattened_fc.values)

    # Prepare numerical features
    if numerical_cols is not None:
        num_features_df = df[numerical_cols].copy()
        if num_features_df.isnull().values.any():
            print("[INFO] Missing numerical values detected. Imputing with column means...")
            imputer = SimpleImputer(strategy='mean')
            num_features = imputer.fit_transform(num_features_df)
        else:
            num_features = num_features_df.values
    else:
        num_features = np.array([]).reshape(len(df), 0)

    # Prepare categorical features
    if categorical_cols is not None and len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna('Unknown').astype(str)  # Handle missing categorical values
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        cat_features = ohe.fit_transform(df[categorical_cols])
    else:
        cat_features = np.array([]).reshape(len(df), 0)

    # Concatenate all features
    all_features = np.hstack([fc_features, num_features, cat_features])
    feature_names = generate_feature_names(selected_rois_labels, all_roi_labels, numerical_cols, categorical_cols, ohe)

    # Standardize
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # PCA reduction
    print("Applying PCA...")
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(all_features_scaled)

    # Find best k (number of clusters)
    silhouette_scores = {}
    print("Testing different cluster numbers...")
    for k in range(2, max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(pca_features)
        score = silhouette_score(pca_features, labels)
        silhouette_scores[k] = score

    # Plot silhouette scores
    plt.figure(figsize=(8,5))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    plt.show();

    # Choose best k
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best number of clusters according to silhouette score: {best_k}")

    # Final KMeans
    kmeans_final = KMeans(n_clusters=best_k, random_state=random_state)
    clusters = kmeans_final.fit_predict(pca_features)
    
    # Attach cluster labels to the cleaned dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters

    return df_with_clusters, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names


# STATISTICAL TESTS
def analyze_matrices(t1_matrices, t_matrices, rois, correction, alpha, label=""):
    '''
    Analyzes the matrices and computes the significance matrix for the function underneath.
    '''
    # Initialize arrays to store t-stats and p-values
    n_rois = len(rois)
    t_stat = np.zeros((n_rois, n_rois))
    p_val = np.ones((n_rois, n_rois))

    # Loop over each cell (i,j)
    for i in tqdm(range(n_rois)):
        for j in range(n_rois):
            t1_values = np.array([mat[i, j] for mat in t1_matrices])
            t_values = np.array([mat[i, j] for mat in t_matrices])

            # Perform independent t-test
            stat, p = ttest_ind(t1_values, t_values, equal_var=True)
            t_stat[i, j] = stat
            p_val[i, j] = p

    # Flatten p-values for correction
    p_val_flat = p_val.ravel()

    if correction:
        reject, p_vals_corrected, _, _ = multipletests(p_val_flat, alpha=alpha, method='fdr_bh')
    else:
        p_vals_corrected = p_val_flat
        reject = np.zeros_like(p_val_flat, dtype=bool)
        reject[p_vals_corrected < alpha] = True

    # Reshape
    p_vals_corrected = p_vals_corrected.reshape(p_val.shape)
    reject = reject.reshape(p_val.shape)

    # Create significance matrix
    significant_matrix = np.zeros_like(p_val, dtype=int)
    significant_matrix[reject] = 1

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(significant_matrix, cmap='viridis', cbar=True, annot=False, square=True, vmin=0, vmax=1, xticklabels=rois, yticklabels=rois)
    plt.title(f"Significance Heatmap {label} (FDR-corrected: {correction})")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.show();

    return significant_matrix, p_vals_corrected, reject

def get_sig_matrix(df, rois, tp=3, correction=True, alpha=0.05, cluster=False):
    '''
    Computes the significance matrix for T1 vs T{tp} matrices, with or without clustering.
    '''

    results = {}

    if not cluster:
        # Whole dataset
        t1_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df['T1_matrix'] if matrix is not None]
        t_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in df[f'T{tp}_matrix'] if matrix is not None]

        print("Shape of T1 matrices:", np.shape(t1_matrices))
        print(f"Shape of T{tp} matrices:", np.shape(t_matrices))

        if not t1_matrices or not t_matrices:
            raise ValueError("No matrices available for T1 or T{tp}.")

        n_rois = len(rois)

        significant_matrix, p_vals_corrected, reject = analyze_matrices(t1_matrices, t_matrices, rois, correction, alpha, label=f"T1 vs T{tp}")

        return significant_matrix, p_vals_corrected, reject

    else:
        # By cluster
        clusters = sorted(df['cluster'].dropna().unique())

        for clust in clusters:
            print(f"\nAnalyzing Cluster {clust}...")

            cluster_df = df[df['cluster'] == clust]
            cluster_df = cluster_df.dropna(subset=['T1_matrix', f'T{tp}_matrix'])

            if cluster_df.empty:
                print(f" - No data for Cluster {clust}")
                continue

            t1_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in cluster_df['T1_matrix']]
            t_matrices = [matrix.values if isinstance(matrix, pd.DataFrame) else matrix for matrix in cluster_df[f'T{tp}_matrix']]

            print(f"Cluster {clust} - Shape of T1 matrices: {np.shape(t1_matrices)}")
            print(f"Cluster {clust} - Shape of T{tp} matrices: {np.shape(t_matrices)}")

            n_rois = len(rois)

            significant_matrix, p_vals_corrected, reject = analyze_matrices(t1_matrices, t_matrices, rois, correction, alpha, label=f"Cluster {clust}: T1 vs T{tp}")

            results[clust] = {
                'significant_matrix': significant_matrix,
                'p_corrected': p_vals_corrected,
                'reject': reject
            }

        return results


'''def sig_matrix_T1_T(df, tp=3, alpha=0.05, clusters=False):
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
            sns.heatmap(significant_matrix, cmap='viridis', cbar=True, annot=True, square=True, vmin=0, vmax=1, xticklabels=rois, yticklabels=rois)
            plt.title("Significance Heatmap (FDR-corrected)")
            plt.xlabel("ROIs")
            plt.ylabel("ROIs")
            plt.tight_layout()
            plt.show();

        return results'''
    
def compute_FC_diff(df, rois, tp=3):
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
    sns.heatmap(diff_array.mean(axis=0), cmap='viridis', cbar=True, annot=False, square=True, vmin=0, vmax=1, xticklabels=rois, yticklabels=rois)
    plt.title(f"Mean FC Difference (T{tp} - T1)")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.show();

    return diff_array

# SUMMARY OF T TESTS
def load_roi_labels(filepath_csv):
    """
    Load ROI labels from a CSV file and return a mapping {Python_index: RegionLongName}.
    
    Args:
        filepath_csv (str): Path to the HCP-MMP1_RegionsCorticesList_379.csv file
    
    Returns:
        roi_mapping (dict): Mapping from 0-based Python indices to RegionLongName
    """
    # Load the CSV file
    roi_df = pd.read_csv(filepath_csv)

    # Check if necessary columns are present
    expected_columns = {'#ID', 'RegionLongName'}
    if not expected_columns.issubset(roi_df.columns):
        raise ValueError(f"The CSV file must contain the columns {expected_columns}. Found columns: {roi_df.columns.tolist()}")

    # Build mapping: Python index (0-based) -> RegionLongName
    roi_mapping = {row['#ID'] - 1: row['RegionLongName'] for _, row in roi_df.iterrows()}

    return roi_mapping


def summarize_significant_differences(p_values_matrix, effect_size_matrix, roi_mapping, cluster_label=None, alpha=0.05):
    """
    Summarize significant differences into a readable DataFrame.

    Args:
        p_values_matrix (np.array): Matrix of p-values
        effect_size_matrix (np.array): Matrix of effect sizes
        roi_mapping (dict): Mapping ROI index -> region name
        cluster_label (int or None): Cluster index. If None, no cluster column is added.
        alpha (float): Significance threshold after correction (default 0.05)

    Returns:
        summary_df (pd.DataFrame): DataFrame of significant results
    """
    n_rois = p_values_matrix.shape[0]
    rows = []

    for i in range(n_rois):
        for j in range(i + 1, n_rois):  # upper triangle only
            p_val = p_values_matrix[i, j]
            if p_val < alpha:
                roi_1 = roi_mapping.get(i, f"ROI_{i}")
                roi_2 = roi_mapping.get(j, f"ROI_{j}")
                effect = effect_size_matrix[i, j]
                entry = {
                    'ROI_1': roi_1,
                    'ROI_2': roi_2,
                    'Comparison': f"{roi_1} - {roi_2}",
                    'p_value': p_val,
                    'effect_size': effect
                }
                if cluster_label is not None:
                    entry['Cluster'] = cluster_label
                rows.append(entry)

    summary_df = pd.DataFrame(rows)

    # Put 'Cluster' first if it exists
    if cluster_label is not None and not summary_df.empty:
        cols = ['Cluster', 'ROI_1', 'ROI_2', 'Comparison', 'p_value', 'effect_size']
        summary_df = summary_df[cols]
    else:
        cols = ['ROI_1', 'ROI_2', 'Comparison', 'p_value', 'effect_size']
        summary_df = summary_df[cols]

    summary_df = summary_df.sort_values(by='p_value', ascending=True).reset_index(drop=True)

    return summary_df

def test_nomality(df, alpha=0.05):
    """
    Perform Shapiro-Wilk test for normality on the data.
    
    Args:
        data (array-like): Data to test for normality.
        alpha (float): Significance level for the test.
    
    Returns:
        bool: True if data is normally distributed, False otherwise.
    """
    df = df.copy().drop(columns=['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix', "subject_full_id",	"TimePoint", "Behavioral_assessment", "MRI", "Gender", "Age", "Education_level", "Lesion_side_old",	"Lesion_side", "Combined", "Bilateral", "Comments", "Stroke_location"])
                        #drop (columns=['subject_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3','
    results = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        stat, p = shapiro(df[col].dropna())
        results[col] = {'W': stat, 'p-value': p}

    # Convert to DataFrame to view nicely
    shapiro_df = pd.DataFrame(results).T
    shapiro_df.index.name = 'Variable'
    shapiro_df.reset_index(inplace=True)

    # Mark whether the variable is normally distributed
    shapiro_df['Normal? (p > 0.05)'] = shapiro_df['p-value'] > 0.05

    print(shapiro_df)
    
    return shapiro_df
