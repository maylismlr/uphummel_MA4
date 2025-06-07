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
from scipy.stats import shapiro, wilcoxon, pearsonr, spearmanr


# for prettiness <3
from tqdm import tqdm



############################################## LOAD DATA FUNCTIONS #################################



def load_excel_data(folder_path, FM_folder_path):
    # Load Excel files
    regression_info = pd.read_excel(f"{folder_path}TiMeS_regression_info_processed.xlsx", engine="openpyxl")
    rsfMRI_full_info = pd.read_excel(f"{folder_path}TiMeS_rsfMRI_full_info.xlsx", engine="openpyxl")
    
    # Keep only the first appearance of each subject_full_id
    subject_info = regression_info.copy().drop_duplicates(subset=["subject_full_id"], keep="first")
    
    # Merge on subject_full_id
    rsfMRI_full_info = rsfMRI_full_info.merge(subject_info, on="subject_full_id", how="left")
    rsfMRI_full_info = rsfMRI_full_info[['subject_full_id', 'Lesion_side', 'Stroke_location', 'lesion_volume_mm3','Gender','Age','Education_level','Combined', 'Bilateral']]
    rsfMRI_full_info['subject_id'] = rsfMRI_full_info['subject_full_id'].astype(str).str[-4:] # Extract last 4 characters of subject_id to match with folder names and subjects
    regression_info['subject_id'] = regression_info['subject_full_id'].astype(str).str[-4:] # Extract last 4 characters of subject_id to match with folder names and subjects

    for file in os.listdir(FM_folder_path):
        if not file.endswith('.xlsx'):
            continue

        file_path = os.path.join(FM_folder_path, file)
        try:
            print(f"Trying to load: {file}")
            df = pd.read_excel(file_path, engine='openpyxl')  # Try reading
            print(f"✅ Successfully loaded: {file}")

            if 'T1' in file:
                FM_T1 = df
            elif 'T2' in file:
                FM_T2 = df
            elif 'T3' in file:
                FM_T3 = df
            elif 'T4' in file:
                FM_T4 = df

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    # Rename 'Patient' to 'subject_id'
    for df in [FM_T1, FM_T2, FM_T3, FM_T4]:
        if 'Patient' in df.columns:
            df.rename(columns={'Patient': 'subject_id'}, inplace=True)
            df.rename(columns={'Fugl.Meyer_affected_TOTAL': 'Fugl_Meyer_contra'}, inplace=True)
            df.rename(columns={'Fugl.Meyer_unaffected_TOTAL': 'Fugl_Meyer_ipsi'}, inplace=True)

    # Replace first character of each ID with 's'
    for df in [FM_T1, FM_T2, FM_T3, FM_T4]:
        if 'subject_id' in df.columns:
            df['subject_id'] = df['subject_id'].astype(str).apply(lambda x: 's' + x[1:])

    FM_T1['TimePoint'] = 'T1'
    FM_T2['TimePoint'] = 'T2'
    FM_T3['TimePoint'] = 'T3'
    FM_T4['TimePoint'] = 'T4'
    
    FM_all = pd.concat([FM_T1, FM_T2, FM_T3, FM_T4], axis=0, ignore_index=True)
    
    # Select relevant columns from FM_all
    FM_subset = FM_all[['subject_id', 'TimePoint', 'Fugl_Meyer_contra', 'Fugl_Meyer_ipsi']]

    # Merge on both 'subject_id' and 'TimePoint'
    merged_df = pd.merge(regression_info, FM_subset, on=['subject_id', 'TimePoint'], how='left')

    return merged_df, rsfMRI_full_info



def load_matrices(folder_path, rsfMRI_full_info, rois, request_type = 'all', plot = False, transform = True):
    '''
    Load data from .mat files in the specified folder. Either get all matrices (enter request_type = 'all'), 
    or only T1 matrices (enter request_type = 't1_only'), or only T1 and T3 matrices matched (enter request_type = 't1_t3_matched),
    or only T1 and T4 matrices matched (enter request_type = 't1_t4_matched), or all T1 and T3 matrices (enter request_type = 't1_t3'),
    or or all T1 and T4 matrices (enter request_type = 't1_t4').
    Returns also yeo matrices for the selected ROIs.
    '''
    
    # Extract last 4 characters of subject_id
    valid_subjects = rsfMRI_full_info["subject_id"].tolist()

    # Match folders with valid subject_id
    subjects = [sub for sub in os.listdir(folder_path) if sub in valid_subjects and not sub.startswith('.')]

    data_rows = []
    
    rois_full = np.arange(0, 379)

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
            
            matrix = pd.DataFrame(mat_data['CM']).iloc[rois_full, rois_full]
            
            matrix = matrix.replace('None', np.nan)
            
            if transform:
                # Apply Fisher Z-transform if needed
                matrix = fisher_z_transform(matrix)

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
    
    subset = False   
    yeo_path = "data/hcp_mmp10_yeo7_modes_indices.csv"

    region_to_yeo = glasser_mapped_to_yeo(yeo_path)

    yeo_mat_all_rois, yeo_labels = compute_all_yeo_matrices_as_dataframe(df, region_to_yeo, rois_full, subset) #get_all_yeo_matrices(df, region_to_yeo, rois_full, subset)
    roi_mapping_yeo = {i: label for i, label in enumerate(yeo_labels)}
    
    if plot:
        # plot mean FC matrices
        plot_mean_FC_matrices(df, rois_full, subset)
        plot_mean_yeo_matrices({col: yeo_mat_all_rois[col].tolist() for col in ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']}, yeo_labels)

    
    subset = True   
    # Merge the DataFrame with rsfMRI_info on subject_id
    df = df.merge(rsfMRI_full_info, on="subject_id", how="left")
    
    # Now we want to work only with our rois:
    df_rois = df.copy()
    for col in ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']:
        df_rois[col] = df[col].apply(lambda mat: mat.iloc[rois, rois] if mat is not None else None)
    yeo_mat_rois, yeo_labels = compute_all_yeo_matrices_as_dataframe(df_rois, region_to_yeo, rois, subset) #get_all_yeo_matrices(df_rois, region_to_yeo, rois, subset)
    
    if plot:
        # plot mean FC matrices
        plot_mean_FC_matrices(df_rois, rois, subset)
        plot_mean_yeo_matrices({col: yeo_mat_rois[col].tolist() for col in ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']}, yeo_labels)

    
    if request_type == 't1_only':            
        t1_matrices = df_rois.copy().drop(columns=['T2_matrix', 'T3_matrix', 'T4_matrix'])
        #yeo_rois_t1 = pd.DataFrame(yeo_mat_all_rois).copy().drop(columns=['T2_matrix', 'T3_matrix', 'T4_matrix'])
        yeo_rois_t1 = yeo_mat_all_rois['T1_matrix']

        return t1_matrices, subjects, yeo_rois_t1, roi_mapping_yeo
    
    elif request_type == 't1_t3_matched':
        # keep only rows where both T1 and T3 matrices are not None
        t1_t3_matrices = df_rois.copy().dropna(subset=['T1_matrix', 'T3_matrix'])
        t1_t3_matrices = t1_t3_matrices.drop(columns=['T2_matrix', 'T4_matrix'])
        #yeo_rois_t1_t3 = dict_of_lists_to_dataframe(yeo_mat_all_rois).drop(columns=['T2_matrix', 'T4_matrix'])
        yeo_rois_t1_t3 = yeo_mat_all_rois.drop(columns=['T2_matrix', 'T4_matrix'])

        
        return t1_t3_matrices, subjects, yeo_rois_t1_t3, roi_mapping_yeo
    
    elif request_type == 't1_t4_matched':
        # keep only rows where both T1 and T3 matrices are not None
        t1_t4_matrices = df_rois.copy().dropna(subset=['T1_matrix', 'T4_matrix'])
        t1_t4_matrices = t1_t4_matrices.drop(columns=['T2_matrix', 'T3_matrix'])
        #yeo_rois_t1_t4 = dict_of_lists_to_dataframe(yeo_mat_all_rois).drop(columns=['T2_matrix', 'T3_matrix'])
        yeo_rois_t1_t4 = yeo_mat_all_rois.drop(columns=['T2_matrix', 'T4_matrix'])

        
        return t1_t4_matrices, subjects, yeo_rois_t1_t4, roi_mapping_yeo
    
    elif request_type == 't1_t3':
        # keep only rows where T1 or T3 matrices are not None
        t1_t3_matrices = df_rois.copy().drop(columns=['T2_matrix', 'T4_matrix'])
        t1_t3_matrices = t1_t3_matrices[~(t1_t3_matrices['T1_matrix'].isna() & t1_t3_matrices['T3_matrix'].isna())]
        #yeo_rois_t1_t3 = dict_of_lists_to_dataframe(yeo_mat_all_rois).drop(columns=['T2_matrix', 'T4_matrix'])
        yeo_rois_t1_t3 = yeo_mat_all_rois.drop(columns=['T2_matrix', 'T4_matrix'])

        
        return t1_t3_matrices, subjects, yeo_rois_t1_t3, roi_mapping_yeo
    
    elif request_type == 't1_t4':
        # keep only rows where T1 or T4 matrices are not None
        t1_t4_matrices = df_rois.copy().drop(columns=['T2_matrix', 'T3_matrix'])
        t1_t4_matrices = t1_t4_matrices[~(t1_t4_matrices['T1_matrix'].isna() & t1_t4_matrices['T4_matrix'].isna())]
        #yeo_rois_t1_t4 = dict_of_lists_to_dataframe(yeo_mat_all_rois).drop(columns=['T2_matrix', 'T3_matrix'])
        yeo_rois_t1_t4 = yeo_mat_all_rois.drop(columns=['T2_matrix', 'T4_matrix'])


        return t1_t4_matrices, subjects, yeo_rois_t1_t4, roi_mapping_yeo
    
    return df_rois, subjects, yeo_mat_all_rois, roi_mapping_yeo



############################################### PLOTTING FUNCTIONS #################################



def plot_all_subject_matrices(subject_matrices, subjects, rois, request_type='t1_t3'):
    '''
    go through all the subjects and plot the matrices as sns heatmaps, with each row being a subject 
    and each column a timepoint. The timepoints can be modulated based on the input.
    '''
    num_subjects = len(subjects)
    
    if request_type == 'all':
        timepoints = ['T1', 'T2', 'T3', 'T4']
    elif request_type == 't1_only':
        timepoints = ['T1']
    elif request_type == 't1_t3' or request_type == 't1_t3_matched':
        timepoints = ['T1', 'T3']
    elif request_type == 't1_t4' or request_type == 't1_t4_matched':
        timepoints = ['T1', 'T4']
    
    fig, axes = plt.subplots(num_subjects, len(timepoints), figsize=(15, num_subjects * 3))
    
    for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
        for j, timepoint in enumerate(timepoints):
            ax = axes[i, j] if num_subjects > 1 else axes[j]
            if timepoint in subject_matrices.columns:
                matrix = subject_matrices.loc[subject, timepoint]
                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=False, vmin=-1, vmax=1, xticklabels=rois, yticklabels=rois)
                ax.set_title(f'Subject {subject} - {timepoint}')
            else:
                ax.axis('off')  # Hide axes if timepoint is not available
            ax.axis('off')

    plt.tight_layout()
    plt.show();
   
    

def plot_mean_FC_matrices(matrices, rois, subset=False):
    """
    Plot the mean FC matrices in a 2x2 grid for T1 to T4 timepoints.

    Args:
        matrices (pd.DataFrame): DataFrame with subject FC matrices.
        rois (list): List of ROI indices.
    """
    matrix_columns = ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, timepoint in enumerate(matrix_columns):
        valid_matrices = matrices[timepoint].dropna()

        if len(valid_matrices) == 0:
            axes[i].axis('off')
            axes[i].set_title(f'No data for {timepoint}')
            continue

        # Stack and average
        stacked = np.stack([mat.values for mat in valid_matrices])
        mean_matrix = np.nanmean(stacked, axis=0)
        

        tick_positions = np.arange(len(rois))
        tick_labels = [str(roi) for roi in rois]
        
        if subset == False:
            tick_positions = [0, len(rois) // 2, len(rois) - 1]
            tick_labels = [str(pos) for pos in tick_positions]

        sns.heatmap(mean_matrix, ax=axes[i], cmap='viridis', cbar=True, vmin=-1, vmax=1)
        axes[i].set_xticks(tick_positions)
        axes[i].set_yticks(tick_positions)
        axes[i].set_xticklabels(tick_labels)
        axes[i].set_yticklabels(tick_labels)
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].tick_params(axis='y', labelrotation=0)
        axes[i].set_title(f'Mean FC Matrix - {timepoint}')
        axes[i].set_xlabel('ROIs')
        axes[i].set_ylabel('ROIs')

    plt.tight_layout()
    plt.show()

    
    
############################################## UTILS FUNCTIONS #################################



def flatten_upper(mat):
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat  # ensure it's an array
        return mat[np.triu_indices_from(mat, k=1)]



def create_roi_hemisphere_map(n_rois=379):
    roi_to_hemi = {}
    for roi in range(n_rois):
        if roi < 180:
            roi_to_hemi[roi] = 'L'
        elif roi < 360:
            roi_to_hemi[roi] = 'R'
        elif 360 <= roi <= 367:
            roi_to_hemi[roi] = 'L'  # Left subcortical
        elif 368 <= roi <= 378:
            roi_to_hemi[roi] = 'R'  # Right subcortical
        elif roi == 378:
            roi_to_hemi[roi] = 'None'  # Cerebellum (not lateralized)
        else:
            roi_to_hemi[roi] = 'Cerebellum'  # Cerebellum (not lateralized)
    return roi_to_hemi



def reorient_t1_t(df, selected_rois, roi_mapping, tp = 3):

    hemi_map = create_roi_hemisphere_map()
    left_rois = [r for r in selected_rois if hemi_map[r] == 'L']
    right_rois = [r for r in selected_rois if hemi_map[r] == 'R']

    aligned_t1 = []
    aligned_t = []
    new_labels = None

    for _, row in df.iterrows():
        mat1 = row['T1_matrix']
        mat = row[f'T{tp}_matrix']
        lesion = row['Lesion_side']
        if mat1 is None or mat is None or lesion not in ['L', 'R']:
            continue

        if lesion == 'L':
            ipsi = left_rois
            contra = right_rois
        else:
            ipsi = right_rois
            contra = left_rois

        roi_order = ipsi + contra
        
        # Cleaned labels directly, no nested reprocessing
        clean = lambda x: x.replace('L_', '').replace('R_', '').replace('_L', '').replace('_R', '')
        cleaned_ipsi_labels = [roi_mapping[r].replace('L_', '').replace('R_', '') + '_ipsi' for r in ipsi]
        cleaned_contra_labels = [roi_mapping[r].replace('L_', '').replace('R_', '') + '_contra' for r in contra]
        cleaned_ipsi_labels = [clean(label) for label in cleaned_ipsi_labels]
        cleaned_contra_labels = [clean(label) for label in cleaned_contra_labels]
        new_labels = cleaned_ipsi_labels + cleaned_contra_labels

        #new_labels = [f"{roi_mapping[r]}_ipsi" for r in ipsi] + [f"{roi_mapping[r]}_contra" for r in contra]
        
        #new_labels = [new_labels[r].replace('L_', '').replace('R_', '') for r in ipsi] + \
        #     [new_labels[r].replace('L_', '').replace('R_', '') for r in contra]


        mat1_re = mat1.loc[roi_order, roi_order].astype(float)
        mat_re = mat.loc[roi_order, roi_order].astype(float)

        # Replace inf/-inf with np.nan
        mat1_re.replace([np.inf, -np.inf], np.nan, inplace=True)
        mat_re.replace([np.inf, -np.inf], np.nan, inplace=True)

        mat1_re.index = mat1_re.columns = new_labels
        mat_re.index = mat_re.columns = new_labels

        aligned_t1.append(mat1_re)
        aligned_t.append(mat_re)
        
        #print("Matrix shape:", mat1_re.shape)
        #print("Number of NaNs in mat1:", mat1_re.isna().sum().sum())
        #print("Number of NaNs in mat :", mat_re.isna().sum().sum())
        #print("Number of aligned subjects:", len(aligned_t1))


    return aligned_t1, aligned_t, new_labels



def dict_of_lists_to_dataframe(data_dict):
    """
    Convert a dictionary of lists to a DataFrame, padding shorter lists with None.
    Because Yeo matrices aren't sorted by subject, and the missing matrices were considered "Missing Values", and therefore 
    we cannot convert them to a DataFrame directly, which we want for further analysis (but we don't need the subjects).
    """
    # Find the maximum list length
    max_len = max(len(v) for v in data_dict.values())

    # Pad each list to the same length using None
    padded_dict = {
        k: v + [None] * (max_len - len(v))
        for k, v in data_dict.items()
    }

    return pd.DataFrame(padded_dict)


def fisher_z_transform(matrix):
    """
    Apply Fisher Z-transformation to a correlation matrix.
    Ignores NaNs and preserves matrix shape.
    """
    if matrix is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        z_matrix = 0.5 * np.log((1 + matrix) / (1 - matrix))
    return z_matrix



def split_by_lesion_side(df):
    """
    Splits the input DataFrame into two DataFrames based on the Lesion_side column.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a 'Lesion_side' column.
    
    Returns:
        df_L (pd.DataFrame): Subjects with left-hemisphere lesions ('L').
        df_R (pd.DataFrame): Subjects with right-hemisphere lesions ('R').
    """
    df_L = df[df['Lesion_side'] == 'L'].copy()
    df_R = df[df['Lesion_side'] == 'R'].copy()
    return df_L, df_R



############################################# CLUSTERING FUNCTIONS #################################



# old version
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
    #print(matrices_encoded.columns)
    
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



############################################ STATISTICAL TESTS ############################################



def analyze_matrices(t1_matrices, t_matrices, correction, alpha, label="", roi_labels=None, matched=False, test = 'ttest_rel'):
    if roi_labels is None:
        roi_labels = np.arange(t1_matrices[0].shape[0])

    n_rois = len(roi_labels)
    t_stat = np.zeros((n_rois, n_rois))
    p_val = np.ones((n_rois, n_rois))

    for i in tqdm(range(n_rois)):
        for j in range(n_rois):
            # Use iloc if matrices are labeled
            if isinstance(t1_matrices[0], pd.DataFrame):
                t1_values = np.array([mat.iloc[i, j] for mat in t1_matrices])
                t_values = np.array([mat.iloc[i, j] for mat in t_matrices])
            else:
                t1_values = np.array([mat[i, j] for mat in t1_matrices])
                t_values = np.array([mat[i, j] for mat in t_matrices])

            if matched:
                if test == 'ttest_rel':
                    stat, p = ttest_rel(t1_values, t_values, nan_policy='omit')
                elif test == 'wilcoxon':
                    stat, p = wilcoxon(t1_values, t_values, alternative='two-sided', nan_policy='omit')
            else:
                    stat, p = ttest_ind(t1_values, t_values, equal_var=True, nan_policy='omit')

            t_stat[i, j] = stat
            p_val[i, j] = p

    # FDR correction
    p_val_flat = p_val.ravel()
    if correction:
        valid_mask = ~np.isnan(p_val_flat)
        p_vals_corrected = np.full_like(p_val_flat, np.nan, dtype=float)
        reject = np.full_like(p_val_flat, False, dtype=bool)

        if np.any(valid_mask):
            reject_valid, p_corr_valid, _, _ = multipletests(p_val_flat[valid_mask], alpha=alpha, method='fdr_bh')
            p_vals_corrected[valid_mask] = p_corr_valid
            reject[valid_mask] = reject_valid

    else:
        p_vals_corrected = p_val_flat
        reject = np.zeros_like(p_val_flat, dtype=bool)
        reject[p_vals_corrected < alpha] = True

    p_vals_corrected = p_vals_corrected.reshape(p_val.shape)
    reject = reject.reshape(p_val.shape)
    significant_matrix = np.zeros_like(p_val, dtype=int)
    significant_matrix[reject] = 1

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        significant_matrix,
        cmap='viridis',
        cbar=True,
        annot=False,
        square=True,
        vmin=0,
        vmax=1,
        xticklabels=roi_labels,
        yticklabels=roi_labels
    )
    plt.title(f"Significance Heatmap {label} (FDR-corrected: {correction})")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return significant_matrix, p_vals_corrected, reject



def test_fc_differences_normality(df, tp=3):
    """
    Test normality (Shapiro-Wilk) of FC differences (T{tp} - T1) for each ROI pair in selected ROIs.
    
    Args:
        df (pd.DataFrame): DataFrame with 'T1_matrix' and f'T{tp}_matrix' columns containing FC matrices.
        rois (list or np.array): List of ROI indices to include (e.g., 33 ROIs).
        tp (int): Timepoint to compare with T1 (default: 3 → T3).
        
    Returns:
        pd.DataFrame: Results with ROI pairs, W-statistics, p-values, and normality flag.
    """
    results = []
    n = len(df)
    first_valid = next(matrix for matrix in df['T1_matrix'] if matrix is not None)
    rois =  np.arange(first_valid.shape[0])
    print(rois)

    for i in tqdm(rois, desc="Testing normality of FC differences"):
        for j in rois:
            if i >= j:
                continue  # Skip lower triangle and diagonal

            diffs = []
            for _, row in df.iterrows():
                t1 = row['T1_matrix']
                tX = row[f'T{tp}_matrix']
                if t1 is not None and tX is not None:
                    val_t1 = t1.iloc[i, j]
                    val_tX = tX.iloc[i, j]
                    if not np.isnan(val_t1) and not np.isnan(val_tX):
                        diffs.append(val_tX - val_t1)

            if len(diffs) >= 3:
                stat, p = shapiro(diffs)
                normal = p > 0.05
                results.append({
                    'ROI_1': i,
                    'ROI_2': j,
                    'W_stat': stat,
                    'p_value': p,
                    'Normal': normal,
                    'n': len(diffs)
                })

    return pd.DataFrame(results)



def get_sig_matrix(df, tp=3, correction=True, alpha=0.05, cluster=False, matched=False, roi_labels=None, contra_ipsi_split=False, selected_rois=None, roi_mapping=None, test = "ttest_rel"):
    '''
    Computes the significance matrix for T1 vs T{tp} matrices.
    Supports clustering and contra/ipsi standardization.
    Supports both matched and independent samples t-tests.
    Args:
        df (pd.DataFrame): DataFrame with 'T1_matrix' and f'T{tp}_matrix' columns containing FC matrices.
        tp (int): Timepoint to compare with T1 (default: 3 → T3).
        correction (bool): Whether to apply FDR correction (default: True).
        alpha (float): Significance level for FDR correction (default: 0.05).
        cluster (bool): Whether to perform clustering on the data (default: False).
        matched (bool): Whether to use paired t-tests (default: False).
        roi_labels (list or np.array): List of ROI labels for the matrices.
        contra_ipsi_split (bool): Whether to split into contra/ipsi matrices.
        selected_rois (list or np.array): List of selected ROIs for contra/ipsi split.
        roi_mapping (dict): Mapping of ROI indices to labels for contra/ipsi split.
    Returns:
        pd.DataFrame: Significant matrix with ROI labels as index and columns.
    '''
    results = {}

    # --------- Contra/Ipsi Comparison Path ---------
    if contra_ipsi_split:
        if selected_rois is None or roi_mapping is None:
            raise ValueError("Must provide `selected_rois` and `roi_mapping` for contra_ipsi_split.")

        # Reorient matrices into standardized ipsi/contra format
        t1_matrices, t_matrices, aligned_labels = reorient_t1_t(df, selected_rois, roi_mapping, tp=tp)
        #print(t1_matrices)

        if not t1_matrices or not t_matrices:
            raise ValueError(f"No valid T1 or T{tp} matrices available after reorientation.")

        sig_matrix, p_corr, reject = analyze_matrices(
            t1_matrices, t_matrices,
            correction=correction,
            alpha=alpha,
            label=f"Ipsi vs Contra FC (T1 vs T{tp})",
            roi_labels=aligned_labels,
            matched=matched, 
            test = test
        )

        return pd.DataFrame(sig_matrix, index=aligned_labels, columns=aligned_labels), \
               pd.DataFrame(p_corr, index=aligned_labels, columns=aligned_labels), \
               pd.DataFrame(reject, index=aligned_labels, columns=aligned_labels), \
               aligned_labels

    # --------- Non-cluster, standard FC path ---------
    elif not cluster:
        first_valid = next(matrix for matrix in df['T1_matrix'] if matrix is not None)
        if roi_labels is None:
            roi_labels = first_valid.index if isinstance(first_valid, pd.DataFrame) else np.arange(first_valid.shape[0])
        
        if isinstance(roi_mapping, dict):
            try:
                roi_labels = [roi_mapping.get(i, f"ROI_{i}") for i in roi_labels]
            except Exception as e:
                print("Warning: Failed to apply roi_mapping to labels:", e)

        t1_matrices = [m for m in df['T1_matrix'] if m is not None]
        t_matrices = [m for m in df[f'T{tp}_matrix'] if m is not None]

        print("Shape of T1 matrices:", np.shape(t1_matrices))
        print(f"Shape of T{tp} matrices:", np.shape(t_matrices))

        sig_matrix, p_corr, reject = analyze_matrices(
            t1_matrices, t_matrices,
            correction=correction,
            alpha=alpha,
            label=f"T1 vs T{tp}",
            roi_labels=roi_labels,
            matched=matched,
            test=test
        )

        return pd.DataFrame(sig_matrix, index=roi_labels, columns=roi_labels), \
               pd.DataFrame(p_corr, index=roi_labels, columns=roi_labels), \
               pd.DataFrame(reject, index=roi_labels, columns=roi_labels)

    # --------- Clustered FC path ---------
    else:
        clusters = sorted(df['cluster'].dropna().unique())
        for clust in clusters:
            print(f"\nAnalyzing Cluster {clust}...")
            cluster_df = df[df['cluster'] == clust].dropna(subset=['T1_matrix', f'T{tp}_matrix'])

            if cluster_df.empty:
                print(f" - No data for Cluster {clust}")
                continue

            t1_matrices = [m for m in cluster_df['T1_matrix']]
            t_matrices = [m for m in cluster_df[f'T{tp}_matrix']]
            roi_labels = t1_matrices[0].index if isinstance(t1_matrices[0], pd.DataFrame) else np.arange(379)

            sig_matrix, p_corr, reject = analyze_matrices(
                t1_matrices, t_matrices,
                correction=correction,
                alpha=alpha,
                label=f"Cluster {clust}: T1 vs T{tp}",
                roi_labels=roi_labels,
                matched=matched,
                test= test
            )

            results[clust] = {
                'significant_matrix': pd.DataFrame(sig_matrix, index=roi_labels, columns=roi_labels),
                'p_corrected': pd.DataFrame(p_corr, index=roi_labels, columns=roi_labels),
                'reject': pd.DataFrame(reject, index=roi_labels, columns=roi_labels)
            }

        return results

    
    
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
    sns.heatmap(diff_array.mean(axis=0), cmap='viridis', cbar=True, annot=False, square=True, vmin=-2, vmax=2, xticklabels=rois, yticklabels=rois)
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
    expected_columns = {'# ID', 'RegionLongName'}
    if not expected_columns.issubset(roi_df.columns):
        raise ValueError(f"The CSV file must contain the columns {expected_columns}. Found columns: {roi_df.columns.tolist()}")

    # Build mapping: Python index (0-based) -> RegionLongName
    roi_mapping = {row['# ID'] - 1: row['RegionLongName'] for _, row in roi_df.iterrows()}

    return roi_mapping



def summarize_significant_differences(p_values_matrix, 
                                      sig_matrix, 
                                      roi_labels, 
                                      effect_size_matrix=None, 
                                      cluster_label=None, 
                                      alpha=0.05, 
                                      filter_striatum=True):
    """
    Summarize significant FC differences.

    Args:
        p_values_matrix (DataFrame or np.array): Matrix of p-values
        sig_matrix (DataFrame or np.array): Binary matrix of significant entries (0/1)
        roi_labels (list or dict): ROI names, e.g. list of strings or index-to-label dict, that has to correspond
        to the case you're studying => eg labels returned from get_significant_matrix if contra_ipsi_split is True, and yeo labels if using yeo!
        effect_size_matrix (optional): Matrix of effect sizes (same shape)
        cluster_label (int or None): Optional cluster index
        alpha (float): Significance threshold
        filter_striatum (bool): Whether to keep only striatal ROI_1s

    Returns:
        pd.DataFrame
    """
    n = sig_matrix.shape[0]
    summary = []


    for i in range(n):
        for j in range(i + 1, n):
            sig = sig_matrix.iloc[i, j] if isinstance(sig_matrix, pd.DataFrame) else sig_matrix[i, j]
            if sig:
                p_val = p_values_matrix.iloc[i, j] if isinstance(p_values_matrix, pd.DataFrame) else p_values_matrix[i, j]
                effect = None
                if effect_size_matrix is not None:
                    effect = effect_size_matrix.iloc[i, j] if isinstance(effect_size_matrix, pd.DataFrame) else effect_size_matrix[i, j]

                # Labels
                if isinstance(roi_labels, dict):
                    roi_1 = roi_labels.get(i, f"ROI_{i}")
                    roi_2 = roi_labels.get(j, f"ROI_{j}")
                else:
                    roi_1 = roi_labels[i]
                    roi_2 = roi_labels[j]

                entry = {
                    "ROI_1": roi_1,
                    "ROI_2": roi_2,
                    "Comparison": f"{roi_1} - {roi_2}",
                    "p_value": p_val
                }

                if effect is not None:
                    entry["effect_size"] = effect
                if cluster_label is not None:
                    entry["Cluster"] = cluster_label

                summary.append(entry)

    df = pd.DataFrame(summary)

    if df.empty:
        return pd.DataFrame(columns=['ROI_1', 'ROI_2', 'Comparison', 'p_value', 'effect_size'])

    # Filter by striatum if needed, so always except for yeo !!
    if filter_striatum:
        striatum_keywords = ['Caudate', 'Putamen', 'Pallidum', 'Accumbens']
        matches = df['ROI_1'].astype(str).str.contains('|'.join(striatum_keywords), case=False, na=False)
        df = df[matches].reset_index(drop=True)


    # Reorder columns
    core_cols = ['ROI_1', 'ROI_2', 'Comparison', 'p_value']
    if 'effect_size' in df.columns:
        core_cols.append('effect_size')
    if 'Cluster' in df.columns:
        core_cols = ['Cluster'] + core_cols

    return df[core_cols].sort_values(by='p_value').reset_index(drop=True)



def test_normality(df, alpha=0.05):
    """
    Perform Shapiro-Wilk test for normality on the data.
    
    Args:
        data (array-like): Data to test for normality.
        alpha (float): Significance level for the test.
    
    Returns:
        bool: True if data is normally distributed, False otherwise.
    """
    df = df.copy().drop(columns=["subject_full_id",	"TimePoint", "Behavioral_assessment", "MRI", "Gender", "Age", "Education_level", "Lesion_side_old",	"Lesion_side", "Combined", "Bilateral", "Comments", "Stroke_location"])
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
    
    return shapiro_df



def motor_longitudinal(regression_info, tp =3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = False):
    """
    Perform Wilcoxon signed-rank test on the specified columns of the task scores, to see if improvements 
    of task scores across timepoints.
    The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples 
    come from the same distribution. 
    """
    
    if split_L_R == True:
        cols_to_keep = ['subject_full_id','TimePoint','Lesion_side'] + regression_info.loc[:, start_col:end_col].columns.tolist()

        regression_info_part = regression_info[cols_to_keep]
        lesion_left = regression_info_part[regression_info_part['Lesion_side'] == 'L']
        lesion_right = regression_info_part[regression_info_part['Lesion_side'] == 'R']
        lesion_both = regression_info_part[regression_info_part['Lesion_side'] == 'R/L']
        lesion_unknown = regression_info_part[~regression_info_part['Lesion_side'].isin(['L', 'R', 'R/L'])]

        score_T1_L = lesion_left[lesion_left.TimePoint == "T1"].copy().dropna()
        score_T_L = lesion_left[lesion_left.TimePoint == f"T{tp}"].copy().dropna()

        score_T1_R = lesion_right[lesion_right.TimePoint == "T1"].copy().dropna()
        score_T_R = lesion_right[lesion_right.TimePoint == f"T{tp}"].copy().dropna()

        # Match score_T1 and score_T3 based on 'subject_full_id'
        common_ids = set(score_T1_L['subject_full_id']).intersection(score_T_L['subject_full_id'])
        score_T1_L_matched = score_T1_L[score_T1_L['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint','Lesion_side'])
        score_T_L_matched = score_T_L[score_T_L['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint','Lesion_side'])
        common_ids = set(score_T1_R['subject_full_id']).intersection(score_T_R['subject_full_id'])
        score_T1_R_matched = score_T1_R[score_T1_R['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint','Lesion_side'])
        score_T_R_matched = score_T_R[score_T_R['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint','Lesion_side'])


        # Run Wilcoxon test across columns (axis=0)
        stat_L, p_L = wilcoxon(score_T1_L_matched, score_T_L_matched, axis=0)
        stat_R, p_R = wilcoxon(score_T1_R_matched, score_T_R_matched, axis=0)

        results_L = []
        results_R = []
        for col in score_T1_L_matched.columns:
            try:
                stat, pval = wilcoxon(score_T1_L_matched[col], score_T_L_matched[col])
                results_L.append({
                    'Task': col, 
                    'n': len(score_T1_L_matched), 
                    'p-value': pval, 
                    'Statistically sig. change between TP': 'Yes' if pval < 0.05 else 'No'
                })
            except ValueError:
                results_L.append({
                    'Task': col, 
                    'n': len(score_T1_L_matched), 
                    'p-value': None, 
                    'Statistically sig. change between TP': 'No'
                })
        for col in score_T1_R_matched.columns:
            try:
                stat, pval = wilcoxon(score_T1_R_matched[col], score_T_R_matched[col])
                results_R.append({
                    'Task': col, 
                    'n': len(score_T1_R_matched), 
                    'p-value': pval, 
                    'Statistically sig. change between TP': 'Yes' if pval < 0.05 else 'No'
                })
            except ValueError:
                results_R.append({
                    'Task': col, 
                    'n': len(score_T1_R_matched), 
                    'p-value': None, 
                    'Statistically sig. change between TP': 'No'
                })
        results_L_df = pd.DataFrame(results_L)
        results_R_df = pd.DataFrame(results_R)
        
        return results_L_df, results_R_df
    
    elif split_L_R == False:
        cols_to_keep = ['subject_full_id','TimePoint'] + regression_info.loc[:, start_col:end_col].columns.tolist()

        regression_info_part = regression_info[cols_to_keep]

        score_T1 = regression_info_part[regression_info_part.TimePoint == "T1"].copy().dropna()
        score_T = regression_info_part[regression_info_part.TimePoint == f"T{tp}"].copy().dropna()

        # Match score_T1 and score_T3 based on 'subject_full_id'
        common_ids = set(score_T1['subject_full_id']).intersection(score_T['subject_full_id'])
        score_T1_matched = score_T1[score_T1['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint'])
        score_T_matched = score_T[score_T['subject_full_id'].isin(common_ids)].set_index('subject_full_id').drop(columns=['TimePoint'])

        # Run Wilcoxon test across columns (axis=0)
        stat, p = wilcoxon(score_T1_matched, score_T_matched, axis=0)

        results = []
        for col in score_T1_matched.columns:
            try:
                stat, pval = wilcoxon(score_T1_matched[col], score_T_matched[col])
                results.append({
                    'Task': col, 
                    'n': len(score_T1_matched), 
                    'p-value': pval, 
                    'Statistically sig. change between TP': 'Yes' if pval < 0.05 else 'No'
                })
            except ValueError:
                results.append({
                    'Task': col, 
                    'n': len(score_T1_matched), 
                    'p-value': None, 
                    'Statistically sig. change between TP': 'No'
                })

        return pd.DataFrame(results)


############################################# YEO FUNCTIONS ################################
    

def glasser_mapped_to_yeo(yeo_path):
    '''
    Maps Glasser regions to Yeo networks.
    '''

    yeo = pd.read_csv(yeo_path, names=['Yeo_Network'])
    yeo['Glasser_Index'] = range(1, 361)
    new_rows = pd.DataFrame({
        'Glasser_Index': range(361, 380),
        'Yeo_Network': 8
    })
    region_to_yeo = pd.concat([yeo, new_rows], ignore_index=True)
    region_to_yeo['Yeo_Network'] = region_to_yeo['Yeo_Network'] - 1
    region_to_yeo['Glasser_Index'] = region_to_yeo['Glasser_Index'] - 1

    # Optional: Yeo network names
    yeo_label_map = {
        0: "Visual", 1: "Somatomotor", 2: "Dorsal Attention", 3: "Ventral Attention",
        4: "Limbic", 5: "Frontoparietal", 6: "Default", 7: "Subcortical"
    }
    region_to_yeo['yeo_name'] = region_to_yeo['Yeo_Network'].map(yeo_label_map)

    return region_to_yeo



def compute_all_yeo_matrices_as_dataframe(matrices_df, region_to_yeo, rois, subset=False):
    """
    Computes subject-wise Yeo network-level FC matrices and returns a DataFrame with subject_id and timepoint matrices.
    
    Args:
        matrices_df (pd.DataFrame): Contains subject_id and T1–T4 FC matrices.
        region_to_yeo (pd.DataFrame): Glasser-to-Yeo mapping with Glasser_Index, Yeo_Network, yeo_name.
        rois (list or np.array): List of ROI indices to include.
        subset (bool): If True, restrict to provided ROIs; else use all.

    Returns:
        df_out (pd.DataFrame): Same format as FC matrix DataFrame, with Yeo-level matrices.
        labels (list): Yeo network labels (e.g., ['Visual', 'Somatomotor', ...])
    """
    timepoints = ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']
    yeo_label_map = region_to_yeo.set_index('Yeo_Network')['yeo_name'].to_dict()
    all_subjects = []

    for _, row in matrices_df.iterrows():
        subject_entry = {'subject_id': row['subject_id']}

        for tp in timepoints:
            mat = row[tp]
            if mat is None:
                subject_entry[tp] = None
                continue

            fc_matrix = mat.to_numpy()

            # Determine ROI mapping
            subset_map = region_to_yeo[region_to_yeo['Glasser_Index'].isin(rois)] if subset else region_to_yeo.copy()
            index_map = {roi: i for i, roi in enumerate(rois)}
            subset_map = subset_map[subset_map['Glasser_Index'].isin(index_map)]

            # Group ROIs by Yeo network
            network_to_indices = subset_map.groupby('Yeo_Network')['Glasser_Index'].apply(list).to_dict()
            key_list = sorted(network_to_indices.keys())
            key_to_idx = {k: i for i, k in enumerate(key_list)}
            num_networks = len(key_list)

            # Create Yeo-level FC matrix
            yeo_fc = np.zeros((num_networks, num_networks))
            for i in key_list:
                for j in key_list:
                    idx_i = [index_map[g] for g in network_to_indices[i]]
                    idx_j = [index_map[g] for g in network_to_indices[j]]
                    submatrix = fc_matrix[np.ix_(idx_i, idx_j)]
                    yeo_fc[key_to_idx[i], key_to_idx[j]] = submatrix.mean()

            subject_entry[tp] = yeo_fc

        all_subjects.append(subject_entry)

    df_out = pd.DataFrame(all_subjects)

    # Ensure alignment with original matrices_df by subject_id
    df_out = df_out.set_index('subject_id').reindex(matrices_df['subject_id']).reset_index()

    # Generate Yeo network labels
    labels = [yeo_label_map[k] if k in yeo_label_map else str(k) for k in key_list]
    #print("type of df_out:", type(df_out))

    return df_out, labels



def plot_mean_yeo_matrices(all_yeo_matrices, labels):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (timepoint, yeo_matrices) in enumerate(all_yeo_matrices.items()):
        if len(yeo_matrices) == 0:
            axes[idx].axis('off')
            axes[idx].set_title(f'No data for {timepoint}')
            continue

        #mean_yeo = np.nanmean(np.stack(yeo_matrices), axis=0)
        #option B:
        # Filter out None and shape-mismatched matrices
        valid_yeo_matrices = [m for m in yeo_matrices if m is not None]
        shape_ref = valid_yeo_matrices[0].shape
        valid_yeo_matrices = [m for m in valid_yeo_matrices if m.shape == shape_ref]

        if len(valid_yeo_matrices) == 0:
            axes[idx].axis('off')
            axes[idx].set_title(f"No valid Yeo matrices for {timepoint}")
            continue

        # Now safely stack and plot
        mean_yeo = np.nanmean(np.stack(valid_yeo_matrices), axis=0)


        sns.heatmap(mean_yeo, ax=axes[idx], cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f",
                    xticklabels=labels, yticklabels=labels)
        axes[idx].set_title(f"Mean Yeo FC - {timepoint}")
        axes[idx].set_xlabel("Yeo Network")
        axes[idx].set_ylabel("Yeo Network")
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



############################################## CORRELATIONS ##################################



'''
def motor_correlation(
    df,
    regression_info,
    tp=3,
    motor_test='nmf_motor',
    corr_type='pearsonr',
    selected_rois_labels=None,
    mask_significant_only=True
):
    # Step 1: Filter rows with valid T1_matrix
    valid_data = df[df["T1_matrix"].apply(lambda x: isinstance(x, pd.DataFrame))].copy()
    if valid_data.empty:
        print("No valid FC matrices found.")
        return None, None

    # Step 2: Get ROI labels from one matrix
    roi_labels = valid_data["T1_matrix"].iloc[0].index.tolist()

    # Step 3: Prepare output matrices
    correlation_matrix = pd.DataFrame(index=roi_labels, columns=roi_labels, dtype=float)
    p_value_matrix = pd.DataFrame(index=roi_labels, columns=roi_labels, dtype=float)

    # Step 4: Merge with behavioral data
    regression_t = regression_info[
        (regression_info["TimePoint"] == f"T{tp}") &
        (regression_info["Behavioral_assessment"] == 1) &
        (regression_info["MRI"] == 1)
    ].copy()

    if motor_test not in regression_t.columns:
        print(f"Motor test '{motor_test}' not found.")
        return None, None

    valid_data = valid_data.merge(
        regression_t[["subject_id", motor_test]],
        on="subject_id"
    )

    if valid_data.empty:
        print("No data after merging with behavioral scores.")
        return None, None

    motor_scores = valid_data[motor_test].values

    # Step 5: Correlation computation
    for i in roi_labels:
        for j in roi_labels:
            fc_values = valid_data["T1_matrix"].apply(lambda mat: mat.loc[i, j]).values
            # Drop NaNs
            stack = np.stack([fc_values, motor_scores], axis=1)
            valid = ~np.isnan(stack).any(axis=1)
            if valid.sum() >= 3:
                if corr_type == "pearsonr":
                    r, p = pearsonr(fc_values[valid], motor_scores[valid])
                elif corr_type == "spearmanr":
                    r, p = spearmanr(fc_values[valid], motor_scores[valid])
                else:
                    raise ValueError(f"Invalid correlation type: {corr_type}")
            else:
                r, p = np.nan, np.nan
            correlation_matrix.loc[i, j] = r
            p_value_matrix.loc[i, j] = p

    # Step 6: Filter + Visualize selected ROIs if specified
    if selected_rois_labels is not None:
        corr_selected = correlation_matrix.loc[selected_rois_labels, :]
        p_val_selected = p_value_matrix.loc[selected_rois_labels, :]

        if mask_significant_only:
            corr_selected = corr_selected.where(p_val_selected < 0.05)

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            corr_selected.astype(float),
            cmap="coolwarm",
            center=0,
            xticklabels=[col + 1 for col in corr_selected.columns],
            yticklabels=[roi + 1 for roi in selected_rois_labels]
        )
        plt.title(f"{corr_type.capitalize()} correlation: FC (T1, selected ROIs) vs {motor_test} at T{tp}")
        plt.xlabel("ROI")
        plt.ylabel("Striatal ROI")
        plt.tight_layout()
        plt.show()
    else:
        # Full matrix heatmap
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            correlation_matrix.astype(float),
            cmap="coolwarm",
            center=0,
            xticklabels=roi_labels,
            yticklabels=roi_labels
        )
        plt.title(f"{corr_type.capitalize()} correlation: FC (T1) vs {motor_test} at T{tp}")
        plt.xlabel("ROI")
        plt.ylabel("ROI")
        plt.tight_layout()
        plt.show()

    return correlation_matrix, p_value_matrix'''



# WORKS FOR IPSI CONTRA SPLIT, BUT NOT FOR YEOMATRIX

def motor_correlation(
    df,
    regression_info,
    tp=3,
    motor_test='nmf_motor',
    corr_type='pearsonr',
    selected_rois_labels=None,
    mask_significant_only=True
):
    """
    Compute correlation between FC matrices and motor scores at a given timepoint,
    with optional FDR correction and selective ROI plotting.

    Returns:
        correlation_matrix (DataFrame): full matrix of correlation coefficients
        p_value_matrix (DataFrame): raw (uncorrected) p-values
        pval_corrected_matrix (DataFrame): FDR-corrected p-values
    """

    # Step 1: Filter valid rows with DataFrame FCs
    valid_data = df[df["T1_matrix"].apply(lambda x: isinstance(x, pd.DataFrame))].copy()
    if valid_data.empty:
        print("No valid FC matrices found.")
        return None, None, None

    # Step 2: Get common ROI labels across all subjects
    common_labels = set(valid_data["T1_matrix"].iloc[0].index)
    for mat in valid_data["T1_matrix"]:
        common_labels &= set(mat.index)

    if not common_labels:
        print("No overlapping ROI labels across subjects.")
        return None, None, None

    # Final list of labels to work with (correlation on full matrix)
    roi_labels = sorted(common_labels)

    # Step 3: Prepare correlation and p-value matrices
    correlation_matrix = pd.DataFrame(index=roi_labels, columns=roi_labels, dtype=float)
    p_value_matrix = pd.DataFrame(index=roi_labels, columns=roi_labels, dtype=float)

    # Step 4: Merge with motor scores
    regression_t = regression_info[
        (regression_info["TimePoint"] == f"T{tp}") &
        (regression_info["Behavioral_assessment"] == 1) &
        (regression_info["MRI"] == 1)
    ].copy()

    if motor_test not in regression_t.columns:
        print(f"Motor test '{motor_test}' not found.")
        return None, None, None

    valid_data = valid_data.merge(
        regression_t[["subject_id", motor_test]],
        on="subject_id"
    )

    if valid_data.empty:
        print("No data after merging with behavioral scores.")
        return None, None, None

    motor_scores = valid_data[motor_test].values

    # Step 5: Loop through ROI pairs
    for i in roi_labels:
        for j in roi_labels:
            fc_values = valid_data["T1_matrix"].apply(
                lambda mat: mat.loc[i, j] if i in mat.index and j in mat.columns else np.nan
            ).values

            valid = ~np.isnan(fc_values) & ~np.isnan(motor_scores)
            if valid.sum() >= 3:
                if corr_type == "pearsonr":
                    r, p = pearsonr(fc_values[valid], motor_scores[valid])
                elif corr_type == "spearmanr":
                    r, p = spearmanr(fc_values[valid], motor_scores[valid])
                else:
                    raise ValueError(f"Invalid correlation type: {corr_type}")
            else:
                r, p = np.nan, np.nan

            correlation_matrix.loc[i, j] = r
            p_value_matrix.loc[i, j] = p

    # Step 5.5: Apply FDR correction
    pval_array = p_value_matrix.values.flatten()
    valid_mask = ~np.isnan(pval_array)

    pvals_to_correct = pval_array[valid_mask]
    _, pvals_corrected, _, _ = multipletests(pvals_to_correct, alpha=0.05, method='fdr_bh')

    pval_corrected_matrix = pd.DataFrame(
        data=np.full_like(p_value_matrix.values, np.nan, dtype=float),
        index=p_value_matrix.index,
        columns=p_value_matrix.columns
    )
    pval_corrected_matrix.values.flat[valid_mask] = pvals_corrected

    # Step 6: Visualization
    if selected_rois_labels is not None:
        matrix_to_plot = correlation_matrix.loc[selected_rois_labels, :]
        pvals_to_use = pval_corrected_matrix.loc[selected_rois_labels, :]
        if mask_significant_only:
            matrix_to_plot = matrix_to_plot.where(pvals_to_use < 0.05)
    else:
        matrix_to_plot = correlation_matrix.copy()
        if mask_significant_only:
            matrix_to_plot = matrix_to_plot.where(pval_corrected_matrix < 0.05)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix_to_plot.astype(float),
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True,
        vmin=-1,
        vmax=1,
        annot=True,
    )
    plt.title(f"{corr_type.capitalize()} correlation (FDR-corrected p<0.05): FC (T1) vs {motor_test} at T{tp}")
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.show()

    return correlation_matrix, p_value_matrix, pval_corrected_matrix




def assign_fugl_ipsi_contra(row):
    if row["Lesion_side"] == "L":
        return pd.Series({
            "Fugl_Meyer_ipsi": row["Fugl_Meyer_left_TOTAL"],
            "Fugl_Meyer_contra": row["Fugl_Meyer_right_TOTAL"]
        })
    elif row["Lesion_side"] == "R":
        return pd.Series({
            "Fugl_Meyer_ipsi": row["Fugl_Meyer_right_TOTAL"],
            "Fugl_Meyer_contra": row["Fugl_Meyer_left_TOTAL"]
        })
    else:
        return pd.Series({
            "Fugl_Meyer_ipsi": np.nan,
            "Fugl_Meyer_contra": np.nan
        })



def switch_contra_ipsi_df(df, regression_info, rois, tp=3, roi_mapping=None):
    #regression_info[["Fugl_Meyer_ipsi", "Fugl_Meyer_contra"]] = regression_info.apply(assign_fugl_ipsi_contra, axis=1)
    matrices_contra_ipsi, _, roi_labels = reorient_t1_t(df, rois, roi_mapping=roi_mapping, tp=3)

    # Match valid subjects (same number as aligned_t1)
    valid_subject_ids = []

    for i, row in df.iterrows():
        mat1 = row['T1_matrix']
        mat = row[f'T{tp}_matrix']  # or whatever T you chose
        lesion = row['Lesion_side']
        if mat1 is None or mat is None or lesion not in ['L', 'R']:
            continue
        valid_subject_ids.append(row['subject_id'])

    # Now create a new DataFrame
    df_aligned = pd.DataFrame({
        "subject_id": valid_subject_ids,
        "T1_matrix": matrices_contra_ipsi  # reoriented FC matrices
    })
    
    return df_aligned, regression_info