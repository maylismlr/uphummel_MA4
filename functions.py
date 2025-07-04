import os
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

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
from scipy.stats import shapiro, wilcoxon, pearsonr, spearmanr, jarque_bera

# for regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import linregress
from scipy.stats import zscore
from scipy.stats import skew
from statsmodels.stats.stattools import durbin_watson 
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import probplot
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss

# for prettiness <3
from tqdm import tqdm

# for modularity
import bct



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


'''
def plot_all_subject_matrices(subject_matrices, subjects, rois, request_type='t1_t3'):
    """
    go through all the subjects and plot the matrices as sns heatmaps, with each row being a subject 
    and each column a timepoint. The timepoints can be modulated based on the input.
    """
    num_subjects = len(subjects)
    print("Subjects:", subjects)
    print("Number of subjects:", len(subjects))

    
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
                print(f"Plotting subject: {subject}, timepoint: {timepoint}")
                print(f"Matrix shape: {matrix.shape}")

                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=False, vmin=-1, vmax=1, xticklabels=rois, yticklabels=rois)
                ax.set_title(f'Subject {subject} - {timepoint}')
            else:
                ax.axis('off')  # Hide axes if timepoint is not available
            ax.axis('off')

    plt.tight_layout()
    plt.show();
'''
def plot_all_subject_matrices(subject_matrices, subjects, rois, request_type='t1_t3'):

    if request_type == 'all':
        timepoints = ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']
    elif request_type == 't1_only':
        timepoints = ['T1_matrix']
    elif request_type in ['t1_t3', 't1_t3_matched']:
        timepoints = ['T1_matrix', 'T3_matrix']
    elif request_type in ['t1_t4', 't1_t4_matched']:
        timepoints = ['T1_matrix', 'T4_matrix']
    else:
        raise ValueError(f"Unknown request_type: {request_type}")

    num_subjects = len(subjects)
    num_timepoints = len(timepoints)

    fig, axes = plt.subplots(num_subjects, num_timepoints, figsize=(5 * num_timepoints, 3 * num_subjects))

    # Normalize axes to 2D array for consistent indexing
    if num_subjects == 1 and num_timepoints == 1:
        axes = np.array([[axes]])
    elif num_subjects == 1:
        axes = np.expand_dims(axes, axis=0)  # shape (1, num_timepoints)
    elif num_timepoints == 1:
        axes = np.expand_dims(axes, axis=1)  # shape (num_subjects, 1)

    for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
        for j, timepoint in enumerate(timepoints):
            ax = axes[i, j]
            
            row = subject_matrices[subject_matrices["subject_id"] == subject]
            if not row.empty and timepoint in row.columns:
                matrix = row.iloc[0][timepoint]
                if matrix is None or not hasattr(matrix, "shape") or len(matrix.shape) != 2:
                    ax.axis('off')
                    continue
                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=False, vmin=-1, vmax=1,
                            xticklabels=rois, yticklabels=rois)
                ax.set_title(f'Subject {subject} - {timepoint}')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.show()
   

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
        elif 368 <= roi < 378:
            roi_to_hemi[roi] = 'R'  # Right subcortical
        elif roi == 378:
            roi_to_hemi[roi] = 'None'  # Brain stem (not lateralized)
       # else:
        #    roi_to_hemi[roi] = 'Cerebellum'  # Brain stem (not lateralized)
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



def keep_selected_rows(mat, selected_rois):
    if mat is None:
        return None
    return mat.loc[selected_rois, :]



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
    Perform Shapiro-Wilk test for normality on all numerical columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        alpha (float): Significance level for normality (default = 0.05).

    Returns:
        pd.DataFrame: Test statistics and normality decision per variable.
    """
    results = {}
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) >= 3:  # Shapiro requires at least 3 values
            stat, p = shapiro(data)
            results[col] = {'W': stat, 'p-value': p}
        else:
            results[col] = {'W': None, 'p-value': None}

    shapiro_df = pd.DataFrame(results).T
    shapiro_df.index.name = 'Variable'
    shapiro_df.reset_index(inplace=True)
    shapiro_df['Normal? (p > alpha)'] = shapiro_df['p-value'] > alpha

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
        region_to_yeo (pd.DataFrame): Glasser-to-Yeo mapping with Glasser_Index, Yeo_Network, yeo_name or network name.
        rois (list or np.array): List of ROI indices to include.
        subset (bool): If True, restrict to provided ROIs; else use all.

    Returns:
        df_out (pd.DataFrame): Same format as FC matrix DataFrame, with Yeo-level matrices (as DataFrames).
        labels (list): Yeo network labels from the **last subject** processed (usually stable).
    """
    timepoints = ['T1_matrix', 'T2_matrix', 'T3_matrix', 'T4_matrix']
    yeo_label_map = region_to_yeo.set_index('Yeo_Network')['yeo_name'].to_dict()
    all_subjects = []
    final_labels = None  # Will hold the labels from last subject processed

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

            # Labels for this subject's matrix
            labels = [yeo_label_map[k] if k in yeo_label_map else str(k) for k in key_list]
            final_labels = labels  # update global labels with the latest one

            # Create Yeo-level FC matrix
            yeo_fc = np.zeros((num_networks, num_networks))
            for i in key_list:
                for j in key_list:
                    idx_i = [index_map[g] for g in network_to_indices[i]]
                    idx_j = [index_map[g] for g in network_to_indices[j]]
                    submatrix = fc_matrix[np.ix_(idx_i, idx_j)]
                    yeo_fc[key_to_idx[i], key_to_idx[j]] = submatrix.mean()

            yeo_df = pd.DataFrame(yeo_fc, index=labels, columns=labels)
            subject_entry[tp] = yeo_df

        all_subjects.append(subject_entry)

    df_out = pd.DataFrame(all_subjects)
    df_out = df_out.set_index('subject_id').reindex(matrices_df['subject_id']).reset_index()

    return df_out, final_labels

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



def switch_contra_ipsi_df(df, rois, tp=3, roi_mapping=None):
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
    
    return df_aligned



def check_corr(df_aligned, regression_T1, region1, region2, tp=3, motor_test='Fugl_Meyer_ipsi', corr_type='pearsonr'):
    # Step 1: Merge to ensure alignment
    merged = df_aligned.merge(regression_T1[["subject_id", motor_test]], on="subject_id")

    # Step 2: Extract FC values
    fc_values = merged["T1_matrix"].apply(
        lambda mat: mat.loc[region1, region2] if region1 in mat.index and region2 in mat.columns else np.nan
    ).to_numpy()

    motor_scores = merged[motor_test].to_numpy()

    # Step 3: Clean and correlate
    valid = ~np.isnan(fc_values) & ~np.isnan(motor_scores)
    if valid.sum() < 3:
        print("Not enough valid data to compute correlation.")
        return None

    if corr_type == 'pearsonr':
        corr, pval = pearsonr(fc_values[valid], motor_scores[valid])
        print(f"Pearson r = {corr:.3f}, p = {pval:.3f}")
        
        # Plot with regression line
        plt.scatter(fc_values[valid], motor_scores[valid], label="Data")
        slope, intercept, _, _, _ = linregress(fc_values[valid], motor_scores[valid])
        x_vals = np.linspace(min(fc_values[valid]), max(fc_values[valid]), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='red', label=f"r = {corr:.2f}, p = {pval:.3f}")
        plt.xlabel(f"{region1} x {region2} (T1_matrix)")
        plt.ylabel(f"{motor_test} (T{tp})")
        plt.title("Pearson Correlation with Fit Line")
        plt.legend()
        plt.show()
        
        return None

    elif corr_type == 'spearmanr':
        corr, pval = spearmanr(fc_values[valid], motor_scores[valid])
        print(f"Spearman r = {corr:.3f}, p = {pval:.3f}")
            # Step 4: Plot
        plt.scatter(fc_values[valid], motor_scores[valid])
        plt.plot([], [], ' ', label=f"r = {corr:.2f}, p = {pval:.3f}")
        plt.xlabel(f"{region1} x {region2} (T1_matrix)")
        plt.ylabel(f"{motor_test} (T{tp})")
        plt.title(f"{corr_type.capitalize()} Correlation")
        plt.legend()
        plt.show()
        
        return None

    else:
        raise ValueError("Unsupported correlation type. Use 'pearsonr' or 'spearmanr'.")



def check_corr_cleaned(df_aligned, regression_T1, region1, region2, tp=3, motor_test='Fugl_Meyer_ipsi', corr_type='pearsonr', z_thresh=3.0):
    # Merge datasets
    merged = df_aligned.merge(regression_T1[["subject_id", motor_test]], on="subject_id")

    # Extract FC values
    fc_values = merged["T1_matrix"].apply(
        lambda mat: mat.loc[region1, region2] if region1 in mat.index and region2 in mat.columns else np.nan
    ).to_numpy()

    motor_scores = merged[motor_test].to_numpy()

    # Clean and remove NaNs
    valid = ~np.isnan(fc_values) & ~np.isnan(motor_scores)
    fc_values_clean = fc_values[valid]
    motor_scores_clean = motor_scores[valid]

    # Set quantile thresholds
    lower_q = 0.05
    upper_q = 0.95

    # Calculate quantile bounds
    mod_low, mod_high = np.quantile(fc_values_clean, [lower_q, upper_q])
    motor_low, motor_high = np.quantile(fc_values_clean, [lower_q, upper_q])

    # Keep only values within the quantile range for both variables
    non_outliers = (
        (fc_values_clean >= mod_low) & (fc_values_clean <= mod_high) &
        (fc_values_clean >= motor_low) & (fc_values_clean <= motor_high)
    )

    # Filter out outliers
    fc_filtered = fc_values_clean[non_outliers]
    motor_filtered = motor_scores_clean[non_outliers]

    if len(fc_filtered) < 3:
        print("Not enough data points after outlier removal.")
        return None

    # Compute correlation
    if corr_type == 'pearsonr':
        corr, pval = pearsonr(fc_filtered, motor_filtered)
        print(f"Pearson r = {corr:.3f}, p = {pval:.3f}")
        
        # Plot
        plt.scatter(fc_filtered, motor_filtered, label="Data")
        slope, intercept, _, _, _ = linregress(fc_filtered, motor_filtered)
        x_vals = np.linspace(min(fc_filtered), max(fc_filtered), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='red', label=f"r = {corr:.2f}, p = {pval:.3f}")
        plt.xlabel(f"{region1} x {region2} (T1_matrix)")
        plt.ylabel(f"{motor_test} (T{tp})")
        plt.title("Pearson Correlation with Outliers Removed")
        plt.legend()
        plt.show()
        
    elif corr_type == 'spearmanr':
        corr, pval = spearmanr(fc_filtered, motor_filtered)
        print(f"Spearman r = {corr:.3f}, p = {pval:.3f}")
        
        plt.scatter(fc_filtered, motor_filtered, label="Data")
        
        # Add correlation info as a legend entry
        plt.plot([], [], ' ', label=f"r = {corr:.2f}, p = {pval:.3f}")  # invisible line just for legend
        plt.xlabel(f"{region1} x {region2} (T1_matrix)")
        plt.ylabel(f"{motor_test} (T{tp})")
        plt.title("Spearman Correlation with Outliers Removed")
        plt.legend()
        plt.show()

    else:
        raise ValueError("Unsupported correlation type. Use 'pearsonr' or 'spearmanr'.")

    return corr, pval



def metrics_correlation(
    modularity_df,
    regression_info,
    tp=3,
    motor_test='nmf_motor',
    corr_type='pearsonr',
    metric = 'Modularity'
):
    """
    Correlate modularity values at a specific timepoint with motor test scores.

    Args:
        modularity_df (DataFrame): Must contain modularity values in columns like 'T1_matrix', 'T2_matrix', etc.
        regression_info (DataFrame): Behavioral data with 'TimePoint', 'subject_id', and motor score columns.
        tp (int): Timepoint to use (e.g., 1, 3, 4).
        motor_test (str): Column name of the motor score to correlate.
        corr_type (str): 'pearsonr' or 'spearmanr'.

    Returns:
        r (float), p (float): Correlation coefficient and p-value.
    """
    # Match timepoint column name in modularity_df
    tp_col = f"T{tp}_matrix"

    if tp_col not in modularity_df.columns:
        print(f"{tp_col} not found in modularity_df")
        return None, None

    # Filter behavioral data
    regression_t = regression_info[
        (regression_info["TimePoint"] == f"T{tp}") &
        (regression_info["Behavioral_assessment"] == 1) &
        (regression_info["MRI"] == 1)
    ].copy()

    if motor_test not in regression_t.columns:
        print(f"Motor test '{motor_test}' not found in regression_info.")
        return None, None

    df = modularity_df.copy()
    df["subject_id"] = df["subject_id"].astype(str)  # ensure string
    regression_t["subject_id"] = regression_t["subject_id"].astype(str)

    merged = df.merge(regression_t[["subject_id", motor_test]], on="subject_id")


    modularity_scores = merged[tp_col].values
    motor_scores = merged[motor_test].values

    # Remove NaNs
    valid = ~np.isnan(modularity_scores) & ~np.isnan(motor_scores)

    if valid.sum() < 3:
        print("Not enough valid data points for correlation.")
        return None, None

    # Perform correlation
    if corr_type == 'pearsonr':
        r, p = pearsonr(modularity_scores[valid], motor_scores[valid])
    elif corr_type == 'spearmanr':
        r, p = spearmanr(modularity_scores[valid], motor_scores[valid])
    else:
        raise ValueError("corr_type must be 'pearsonr' or 'spearmanr'")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.regplot(x=modularity_scores[valid], y=motor_scores[valid])
    plt.xlabel(f"{metric} (T{tp})")
    plt.ylabel(f"{motor_test} (T{tp})")
    plt.title(f"{corr_type.capitalize()} correlation: {metric} vs {motor_test} at T{tp}\n"
              f"r = {r:.2f}, p = {p:.4f}")
    plt.tight_layout()
    plt.show()

    return r, p



def metrics_correlation_cleaned(
    modularity_df,
    regression_info,
    tp=3,
    motor_test='nmf_motor',
    corr_type='pearsonr',
    metric='Modularity',
    z_thresh=3.0
):
    """
    Correlate modularity values at a specific timepoint with motor test scores, removing outliers.

    Args:
        modularity_df (DataFrame): Must contain modularity values in columns like 'T1_matrix', 'T2_matrix', etc.
        regression_info (DataFrame): Behavioral data with 'TimePoint', 'subject_id', and motor score columns.
        tp (int): Timepoint to use (e.g., 1, 3, 4).
        motor_test (str): Column name of the motor score to correlate.
        corr_type (str): 'pearsonr' or 'spearmanr'.
        z_thresh (float): Z-score threshold to detect outliers.

    Returns:
        r (float), p (float): Correlation coefficient and p-value.
    """
    tp_col = f"T{tp}_matrix"

    if tp_col not in modularity_df.columns:
        print(f"{tp_col} not found in modularity_df")
        return None, None

    regression_t = regression_info[
        (regression_info["TimePoint"] == f"T{tp}") &
        (regression_info["Behavioral_assessment"] == 1) &
        (regression_info["MRI"] == 1)
    ].copy()

    if motor_test not in regression_t.columns:
        print(f"Motor test '{motor_test}' not found in regression_info.")
        return None, None

    df = modularity_df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    regression_t["subject_id"] = regression_t["subject_id"].astype(str)

    merged = df.merge(regression_t[["subject_id", motor_test]], on="subject_id")

    modularity_scores = merged[tp_col].values
    motor_scores = merged[motor_test].values

    # Remove NaNs
    valid = ~np.isnan(modularity_scores) & ~np.isnan(motor_scores)
    modularity_clean = modularity_scores[valid]
    motor_clean = motor_scores[valid]

    # Set quantile thresholds
    lower_q = 0.05
    upper_q = 0.95

    # Calculate quantile bounds
    mod_low, mod_high = np.quantile(modularity_clean, [lower_q, upper_q])
    motor_low, motor_high = np.quantile(motor_clean, [lower_q, upper_q])

    # Keep only values within the quantile range for both variables
    non_outliers = (
        (modularity_clean >= mod_low) & (modularity_clean <= mod_high) &
        (motor_clean >= motor_low) & (motor_clean <= motor_high)
    )


    mod_final = modularity_clean[non_outliers]
    motor_final = motor_clean[non_outliers]

    if len(mod_final) < 3:
        print("Not enough valid data points after outlier removal.")
        return None, None

    # Correlation
    if corr_type == 'pearsonr':
        r, p = pearsonr(mod_final, motor_final)
    elif corr_type == 'spearmanr':
        r, p = spearmanr(mod_final, motor_final)
    else:
        raise ValueError("corr_type must be 'pearsonr' or 'spearmanr'")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.regplot(x=mod_final, y=motor_final)
    plt.xlabel(f"{metric} (T{tp})")
    plt.ylabel(f"{motor_test} (T{tp})")
    plt.title(f"{corr_type.capitalize()}: {metric} vs {motor_test} at T{tp}, no outliers\n"
              f"r = {r:.2f}, p = {p:.4f}")
    plt.tight_layout()
    plt.show()

    return r, p



######################################### MODULARITY COMPUTATION ############################



def compute_modularity(fc_df):
    if fc_df is None or not isinstance(fc_df, pd.DataFrame):
        return np.nan

    try:
        W = fc_df.replace([np.inf, -np.inf], 0).to_numpy()

        # Optional: match expected size
        W = W[:359, :359]

        # Remove negative weights (as done in the paper)
        W[W < 0] = 0

        # Compute modularity
        _, Q = bct.modularity_louvain_und(W, seed=42)
        
        return Q
    except Exception as e:
        print(f"Error: {e}")
        return np.nan



############################################## HEMISPHERIC SYMETRY ################################



def check_network_symmetry(region_to_yeo, region_to_hemi, roi_indices):
    sub_yeo = region_to_yeo[region_to_yeo['Glasser_Index'].isin(roi_indices)]
    sub_yeo = sub_yeo.set_index('Glasser_Index')  # << key fix
    sub_hemi = region_to_hemi.loc[roi_indices]
    combined = sub_yeo.merge(sub_hemi, left_index=True, right_index=True)
    counts = combined.groupby(['Yeo_Network', 'Hemisphere']).size()
    #print("Yeo-Hemisphere counts:\n", counts)
    return counts



def compute_system_segregation(fc_df, region_to_yeo):
    segregation_scores = []
    for idx in range(fc_df.shape[0]):
        region_network = region_to_yeo.loc[idx, 'Yeo_Network']
        same_net = [i for i in range(fc_df.shape[0]) if region_to_yeo.loc[i, 'Yeo_Network'] == region_network and i != idx]
        diff_net = [i for i in range(fc_df.shape[0]) if region_to_yeo.loc[i, 'Yeo_Network'] != region_network]

        within_vals = fc_df.iloc[idx, same_net].values if same_net else np.nan
        between_vals = fc_df.iloc[idx, diff_net].values if diff_net else np.nan

        within = np.nanmean(within_vals)
        between = np.nanmean(between_vals)
        #print(f"within: {within}, between: {between}")
        if np.isnan(within) or np.isnan(between) or between == 0:
            segregation_scores.append(np.nan)
        else:
            segregation_scores.append(within / between)
    return segregation_scores



def compute_hemispheric_symmetry(seg_scores, region_to_hemi):
    # Ensure alignment
    df = pd.DataFrame({'score': seg_scores})
    df['hemi'] = region_to_hemi['Hemisphere'].values

    # Filter and drop NaNs/infs
    df = df[df['hemi'].isin(['L', 'R'])]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    left = df[df['hemi'] == 'L']['score'].values
    right = df[df['hemi'] == 'R']['score'].values

    min_len = min(len(left), len(right))
    if min_len < 5:  # optional sanity threshold
        return np.nan
    return pearsonr(left[:min_len], right[:min_len])[0]

def compute_symmetry_from_fc_df(fc_df, region_to_yeo, region_to_hemi):
    results = []

    for _, row in fc_df.iterrows():
        subject = row['subject_id']

        for timepoint in ['T1_matrix', 'T3_matrix', 'T4_matrix']:
            fc_matrix = row[timepoint]
            if fc_matrix is None:
                continue

            # Exclude brainstem (index 378)
            roi_indices = [i for i in fc_matrix.index if i != 378]
            fc_matrix = fc_matrix.loc[roi_indices, roi_indices]

            # Subset metadata
            sub_yeo = region_to_yeo.set_index('Glasser_Index').loc[roi_indices].copy()
            sub_hemi = region_to_hemi.loc[roi_indices].copy()

            # Sanity check: filter only L/R
            sub_hemi = sub_hemi[sub_hemi['Hemisphere'].isin(['L', 'R'])]
            valid_indices = sub_hemi.index.intersection(sub_yeo.index)
            fc_matrix = fc_matrix.loc[valid_indices, valid_indices]
            sub_yeo = sub_yeo.loc[valid_indices]
            sub_hemi = sub_hemi.loc[valid_indices]

            # Reindex for clean downstream use
            roi_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
            fc_matrix_reindexed = fc_matrix.rename(index=roi_map, columns=roi_map)
            sub_yeo.index = range(len(valid_indices))
            sub_hemi.index = range(len(valid_indices))


            # Compute segregation and symmetry
            seg_scores = compute_system_segregation(fc_matrix_reindexed, sub_yeo)
            valid_left = sum((~np.isnan(seg_scores)) & (sub_hemi['Hemisphere'] == 'L').values)
            valid_right = sum((~np.isnan(seg_scores)) & (sub_hemi['Hemisphere'] == 'R').values)
            #print(f"[{subject} - {timepoint}] valid L: {valid_left}, valid R: {valid_right}")
            symmetry = compute_hemispheric_symmetry(seg_scores, sub_hemi)

            results.append({
                'subject_id': subject,
                'timepoint': timepoint.replace('_matrix', ''),
                'hemispheric_symmetry': symmetry
            })

    return pd.DataFrame(results)



######################################## HOMOTOPY #########################################



def compute_mean_homotopic_fc(fc_df, pairs):
    values = [fc_df.loc[i, j] for i, j in pairs]
    return pd.Series(values).mean()



######################################## REGRESSION #########################################

def run_Ridge_with_RFE(X_df_clean, y, param_grid, motor_score_name=None, apply_log_transform=True, verbose=True):
    """
    Runs Ridge regression with Recursive Feature Elimination (RFE) and performs 
    grid search to select the optimal number of features based on cross-validated R² score.
    
    This function includes comprehensive data validation and handles skewed data with 
    appropriate log transformations.

    Parameters
    ----------
    X_df_clean : pandas.DataFrame
        DataFrame of shape (n_samples, n_features) containing preprocessed and labeled input features.
    y : array-like of shape (n_samples,)
        Target variable (e.g., motor scores).
    param_grid : dict
        Dictionary defining the grid of 'rfe__n_features_to_select' values to search over.
    motor_score_name : str, optional
        Name of the motor score for logging purposes (e.g., 'nmf_motor', 'Fugl_Meyer_ipsi').
    apply_log_transform : bool, default=True
        Whether to apply log transformation based on data skewness.
    verbose : bool, default=True
        Whether to print detailed information about data validation and transformations.

    Returns
    -------
    selected_feature_names : list of str
        List of ROIxROI feature names selected by the best RFE model.
    grid : sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object containing the best estimator and CV results.
    y_transformed : array-like
        The transformed target variable (original if no transformation applied).
    transformation_info : dict
        Information about the applied transformations.

    Notes
    -----
    The pipeline used is:
        StandardScaler → RFE(Ridge) → Ridge

    Cross-validation strategy: RepeatedKFold(n_splits=5, n_repeats=10)
    Scoring metric: R² (coefficient of determination)
    """
    
    # ===== DATA VALIDATION =====
    if verbose:
        print(f"=== Data Validation for {motor_score_name or 'target variable'} ===")
    
    # Check input types and shapes
    if not isinstance(X_df_clean, pd.DataFrame):
        raise TypeError("X_df_clean must be a pandas DataFrame")
    
    if not isinstance(y, (np.ndarray, pd.Series, list)):
        raise TypeError("y must be an array-like object")
    
    # Convert y to numpy array for consistency
    y = np.array(y)
    
    # Check for matching sample sizes
    if len(y) != len(X_df_clean):
        raise ValueError(f"Sample size mismatch: X has {len(X_df_clean)} samples, y has {len(y)} samples")
    
    # Check for sufficient samples
    if len(y) < 10:
        raise ValueError(f"Insufficient samples: {len(y)} samples. Need at least 10 for reliable cross-validation.")
    
    # Check for missing values
    if np.any(np.isnan(y)):
        raise ValueError("Target variable contains NaN values")
    
    if X_df_clean.isnull().any().any():
        raise ValueError("Feature matrix contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(y)):
        raise ValueError("Target variable contains infinite values")
    
    if np.any(np.isinf(X_df_clean.values)):
        raise ValueError("Feature matrix contains infinite values")
    
    # Check for constant target variable
    if np.std(y) == 0:
        raise ValueError("Target variable is constant (zero variance)")
    
    # Check for constant features
    constant_features = X_df_clean.columns[X_df_clean.std() == 0].tolist()
    if constant_features:
        if verbose:
            print(f"Warning: {len(constant_features)} constant features found and will be removed")
        X_df_clean = X_df_clean.drop(columns=constant_features)
    
    # Check for sufficient features after removing constants
    if X_df_clean.shape[1] < 2:
        raise ValueError(f"Insufficient features: {X_df_clean.shape[1]} features remaining after removing constants")
    
    # ===== DATA DISTRIBUTION ANALYSIS =====
    if verbose:
        print(f"\n=== Data Distribution Analysis ===")
        print(f"Sample size: {len(y)}")
        print(f"Number of features: {X_df_clean.shape[1]}")
        print(f"Target variable range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Target variable mean: {y.mean():.3f}")
        print(f"Target variable std: {y.std():.3f}")
    
    # Calculate skewness
    y_skewness = skew(y)
    
    if verbose:
        print(f"Target variable skewness: {y_skewness:.3f}")
        if abs(y_skewness) > 1:
            print(f"Data is {'right' if y_skewness > 0 else 'left'}-skewed (|skewness| > 1)")
    
    # ===== LOG TRANSFORMATION HANDLING =====
    y_transformed = y.copy()
    transformation_info = {
        'original_skewness': y_skewness,
        'transformation_applied': None,
        'transformation_params': {}
    }
    
    if apply_log_transform and abs(y_skewness) > 1:
        # Determine transformation based on skewness and motor score type
        if motor_score_name and 'nmf' in motor_score_name.lower():
            # nmf_motor is typically right-skewed (many low values, few high values)
            if y_skewness > 1:
                # Apply log transformation for right-skewed data
                y_transformed = np.log1p(y)  # log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
                if verbose:
                    print(f"Applied log1p(y) transformation for right-skewed nmf_motor data")
            else:
                if verbose:
                    print(f"No transformation applied: nmf_motor data not sufficiently right-skewed")
        
        elif motor_score_name and 'fugl' in motor_score_name.lower():
            # Fugl Meyer is typically left-skewed (many high values, few low values)
            if y_skewness < -1:
                # Apply log transformation for left-skewed data
                y_transformed = np.log1p(y.max() - y)  # log1p(max - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
                if verbose:
                    print(f"Applied log1p(max - y) transformation for left-skewed Fugl Meyer data")
            else:
                if verbose:
                    print(f"No transformation applied: Fugl Meyer data not sufficiently left-skewed")
        
        else:
            # Generic transformation based on skewness
            if y_skewness > 1:
                y_transformed = np.log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
                if verbose:
                    print(f"Applied log1p(y) transformation for right-skewed data")
            elif y_skewness < -1:
                y_transformed = np.log1p(y.max() - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
                if verbose:
                    print(f"Applied log1p(max - y) transformation for left-skewed data")
    
    # Check transformed data
    if np.any(np.isnan(y_transformed)) or np.any(np.isinf(y_transformed)):
        raise ValueError("Transformation resulted in NaN or infinite values")
    
    if np.std(y_transformed) == 0:
        raise ValueError("Transformed target variable is constant (zero variance)")
    
    if verbose:
        transformed_skewness = skew(y_transformed)
        print(f"Transformed data skewness: {transformed_skewness:.3f}")
        print(f"Transformation applied: {transformation_info['transformation_applied']}")
    
    # ===== PARAMETER GRID VALIDATION =====
    if verbose:
        print(f"\n=== Parameter Grid Validation ===")
    
    max_features = X_df_clean.shape[1]
    if 'rfe__n_features_to_select' in param_grid:
        feature_counts = param_grid['rfe__n_features_to_select']
        # Ensure no feature count exceeds available features
        valid_feature_counts = [f for f in feature_counts if f <= max_features]
        if len(valid_feature_counts) != len(feature_counts):
            removed_counts = [f for f in feature_counts if f > max_features]
            if verbose:
                print(f"Warning: Removed invalid feature counts: {removed_counts}")
            param_grid['rfe__n_features_to_select'] = valid_feature_counts
        
        if not valid_feature_counts:
            raise ValueError(f"No valid feature counts in parameter grid. Max available: {max_features}")
        
        if verbose:
            print(f"Valid feature counts: {valid_feature_counts}")
    
    # ===== MODEL TRAINING =====
    if verbose:
        print(f"\n=== Model Training ===")
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(estimator=Ridge(alpha=1.0))),
        ("ridge", Ridge(alpha=1.0))
    ])
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    try:
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
        grid.fit(X_df_clean, y_transformed)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")
    
    # ===== RESULTS ANALYSIS =====
    if verbose:
        print(f"\n=== Results ===")
        print(f"Best number of features: {grid.best_params_['rfe__n_features_to_select']}")
        print(f"Best cross-validated R² score: {grid.best_score_:.3f}")
        
        # Check for convergence issues
        if grid.best_score_ < 0:
            print(f"Warning: Negative R² score ({grid.best_score_:.3f}) indicates poor model performance")
        
        if grid.best_score_ < 0.1:
            print(f"Warning: Low R² score ({grid.best_score_:.3f}) suggests limited predictive power")

    # ===== FEATURE INTERPRETATION =====
    best_rfe = grid.best_estimator_.named_steps['rfe']
    support = best_rfe.support_

    # Get selected feature names
    selected_indices = [i for i, keep in enumerate(support) if keep]
    all_feature_names = X_df_clean.columns.tolist()
    selected_feature_names = [all_feature_names[i] for i in selected_indices]

    best_ridge = grid.best_estimator_.named_steps['ridge']
    coefficients = best_ridge.coef_

    # Match coefficients with feature names
    feature_importance = pd.DataFrame({
        "Feature": selected_feature_names,
        "Weight": coefficients
    }).sort_values(by="Weight", key=np.abs, ascending=False)

    if verbose:
        print(f"\nTop predictive features:")
        print(feature_importance.head(10))
        
        # Check for potential overfitting
        if len(selected_feature_names) > len(y) * 0.1:  # More than 10% of sample size
            print(f"Warning: {len(selected_feature_names)} features selected for {len(y)} samples - risk of overfitting")
    
    return selected_feature_names, grid, y_transformed, transformation_info



def reverse_transform_predictions(y_pred_transformed, transformation_info):
    """
    Reverse transform predictions back to original scale.
    
    Parameters
    ----------
    y_pred_transformed : array-like
        Predictions in transformed space.
    transformation_info : dict
        Information about the applied transformation.
        
    Returns
    -------
    array-like
        Predictions in original scale.
    """
    if transformation_info['transformation_applied'] == 'log1p':
        return np.expm1(y_pred_transformed)
    elif transformation_info['transformation_applied'] == 'log1p_reverse':
        max_value = transformation_info['transformation_params']['max_value']
        return max_value - np.expm1(y_pred_transformed)
    else:
        return y_pred_transformed


def preprocess_data_for_regression(df_aligned, regression_info, tp=3, motor_score='Fugl_Meyer_ipsi', striatum_labels=None, param_grid=None):
    
        
    t1_t3_t4_matched_sel = df_aligned.copy()

    for col in ['T1_matrix', 'T3_matrix', 'T_matrix']:
        if col in df_aligned.columns:
            t1_t3_t4_matched_sel[col] = df_aligned[col].apply(lambda mat: keep_selected_rows(mat, striatum_labels))

    selected_col = 'T1_matrix'
    
    regression_t = regression_info[
        (regression_info["TimePoint"] == f"T{tp}") &
        (regression_info["Behavioral_assessment"] == 1) &
        (regression_info["MRI"] == 1)
    ].copy()
    
    valid_data = t1_t3_t4_matched_sel.merge(
        regression_t[["subject_id", "nmf_motor", "Fugl_Meyer_contra", "Fugl_Meyer_ipsi"]],
        on="subject_id"
    )
    
    X = np.array([fc.values.flatten() for fc in valid_data[selected_col]]) 

    # Example matrix from one subject
    example_fc = valid_data[selected_col].iloc[0]  # shape (8, 33)

    # Extract row and column labels
    row_labels = example_fc.index.tolist()
    col_labels = example_fc.columns.tolist()

    # Create ROIxROI feature names
    feature_names = [f"{row}|{col}" for row in row_labels for col in col_labels]

    X_df = pd.DataFrame(X, columns=feature_names)  # No scaling
    X_df_clean = X_df.dropna(axis=1)
    
    '''if param_grid is None:
        param_grid = {
        "rfe__n_features_to_select": [5, 10, 20, 24, 30, 40, 80, 160, X_df_clean.shape[1]]
        }'''
    
    param_grid = {
        "rfe__n_features_to_select": [5, 10, 20, 24, 30, 40, 80, 160, X_df_clean.shape[1]]
    }
    
    y = valid_data[motor_score].values
    
    return X_df_clean, y, param_grid



def run_whole_pipeline(df_aligned, regression_info, tp=3, motor_score='Fugl_Meyer_ipsi', striatum_labels=None, apply_log_transform=True, verbose=True):
    """
    Run the whole pipeline for feature selection and model training with comprehensive validation.
    
    Parameters
    ----------
    df_aligned : pandas.DataFrame
        DataFrame containing aligned functional connectivity matrices.
    regression_info : pandas.DataFrame
        DataFrame containing regression information and motor scores.
    tp : int, default=3
        Time point to analyze (1, 2, 3, or 4).
    motor_score : str, default='Fugl_Meyer_ipsi'
        Name of the motor score column to predict.
    striatum_labels : list, optional
        List of striatum ROI labels to include in the analysis.
    apply_log_transform : bool, default=True
        Whether to apply log transformation based on data skewness.
    verbose : bool, default=True
        Whether to print detailed information about the pipeline.
    
    Returns
    -------
    dict
        Dictionary containing results including selected features, model, and performance metrics.
    """
    
    if verbose:
        print(f"=== Running Whole Pipeline ===")
        print(f"Time point: T{tp}")
        print(f"Motor score: {motor_score}")
        print(f"Striatum labels: {len(striatum_labels) if striatum_labels else 'None'}")
    
    # Step 1: Preprocess data
    try:
        X_df_clean, y, param_grid = preprocess_data_for_regression(df_aligned, regression_info, tp, motor_score, striatum_labels)
        if verbose:
            print(f"Data preprocessing completed: {X_df_clean.shape[0]} samples, {X_df_clean.shape[1]} features")
    except Exception as e:
        raise RuntimeError(f"Data preprocessing failed: {str(e)}")
    
    # Step 2: Run Ridge regression with RFE
    try:
        selected_feature_names, grid, y_transformed, transformation_info = run_Ridge_with_RFE(
            X_df_clean, y, param_grid, 
            motor_score_name=motor_score,
            apply_log_transform=apply_log_transform,
            verbose=verbose
        )
        if verbose:
            print(f"Model training completed successfully")
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")
    
    # Step 3: Predict with the best model on all data
    try:
        y_pred_transformed = grid.best_estimator_.predict(X_df_clean)
        
        # Reverse transform predictions if transformation was applied
        y_pred = reverse_transform_predictions(y_pred_transformed, transformation_info)
        
        if verbose:
            print(f"Predictions generated successfully")
    except Exception as e:
        raise RuntimeError(f"Prediction generation failed: {str(e)}")
    
    # Step 4: Evaluate performance
    try:
        # Evaluate on transformed space for consistency
        r2_transformed = r2_score(y_transformed, y_pred_transformed)
        mse_transformed = mean_squared_error(y_transformed, y_pred_transformed)
        mae_transformed = mean_absolute_error(y_transformed, y_pred_transformed)
        
        # Also evaluate on original space
        r2_original = r2_score(y, y_pred)
        mse_original = mean_squared_error(y, y_pred)
        mae_original = mean_absolute_error(y, y_pred)
        
        if verbose:
            print(f"\n=== Performance Metrics ===")
            print(f"Transformed space - R²: {r2_transformed:.3f}, MSE: {mse_transformed:.3f}, MAE: {mae_transformed:.3f}")
            print(f"Original space - R²: {r2_original:.3f}, MSE: {mse_original:.3f}, MAE: {mae_original:.3f}")
    except Exception as e:
        raise RuntimeError(f"Performance evaluation failed: {str(e)}")
    
    # Step 5: Create visualizations
    if verbose:
        try:
            print(f"\n=== Creating Visualizations ===")
            
            # Plot 1: True vs Predicted (original space)
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(y, y_pred, edgecolor='k', alpha=0.7)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel("True motor scores")
            plt.ylabel("Predicted motor scores")
            plt.title(f"True vs. Predicted ({motor_score})")
            plt.grid(True, alpha=0.3)
            
            # Plot 2: True vs Predicted (transformed space)
            plt.subplot(1, 2, 2)
            plt.scatter(y_transformed, y_pred_transformed, edgecolor='k', alpha=0.7)
            plt.plot([y_transformed.min(), y_transformed.max()], 
                    [y_transformed.min(), y_transformed.max()], 'r--', lw=2)
            plt.xlabel("True motor scores (transformed)")
            plt.ylabel("Predicted motor scores (transformed)")
            plt.title(f"True vs. Predicted (transformed)")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Plot 3: Distribution comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.histplot(y, kde=True, bins=20)
            plt.title(f"Original {motor_score} Distribution")
            plt.xlabel("Motor scores")
            
            plt.subplot(1, 3, 2)
            sns.histplot(y_transformed, kde=True, bins=20)
            plt.title(f"Transformed {motor_score} Distribution")
            plt.xlabel("Motor scores (transformed)")
            
            plt.subplot(1, 3, 3)
            sns.histplot(y_pred, kde=True, bins=20, color='orange', alpha=0.7)
            plt.title(f"Predicted {motor_score} Distribution")
            plt.xlabel("Predicted motor scores")
            
            plt.tight_layout()
            plt.show()
            
            # Plot 4: Residuals
            plt.figure(figsize=(12, 5))
            
            residuals_original = y - y_pred
            residuals_transformed = y_transformed - y_pred_transformed
            
            plt.subplot(1, 2, 1)
            plt.scatter(y_pred, residuals_original, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs Predicted (Original)")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.scatter(y_pred_transformed, residuals_transformed, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted values (transformed)")
            plt.ylabel("Residuals (transformed)")
            plt.title("Residuals vs Predicted (Transformed)")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: Visualization creation failed: {str(e)}")
    
    # Step 6: Compile results
    results = {
        'selected_features': selected_feature_names,
        'model': grid,
        'transformation_info': transformation_info,
        'performance': {
            'transformed_space': {
                'r2': r2_transformed,
                'mse': mse_transformed,
                'mae': mae_transformed
            },
            'original_space': {
                'r2': r2_original,
                'mse': mse_original,
                'mae': mae_original
            }
        },
        'predictions': {
            'y_true': y,
            'y_pred': y_pred,
            'y_true_transformed': y_transformed,
            'y_pred_transformed': y_pred_transformed
        },
        'data_info': {
            'n_samples': len(y),
            'n_features': X_df_clean.shape[1],
            'n_selected_features': len(selected_feature_names),
            'motor_score': motor_score,
            'time_point': tp
        }
    }
    
    if verbose:
        print(f"\n=== Pipeline Summary ===")
        print(f"Selected {len(selected_feature_names)} features from {X_df_clean.shape[1]} total features")
        print(f"Best R² score: {grid.best_score_:.3f}")
        print(f"Transformation applied: {transformation_info['transformation_applied']}")
        print(f"Pipeline completed successfully!")
    
    return results



def demonstrate_improved_regression_pipeline(df_aligned, regression_info, striatum_labels=None):
    """
    Demonstrate the improved regression pipeline with comprehensive validation and log transformations.
    
    This function shows how to use the improved run_Ridge_with_RFE and run_whole_pipeline functions
    with proper data validation and handling of skewed data.
    
    Parameters
    ----------
    df_aligned : pandas.DataFrame
        DataFrame containing aligned functional connectivity matrices.
    regression_info : pandas.DataFrame
        DataFrame containing regression information and motor scores.
    striatum_labels : list, optional
        List of striatum ROI labels to include in the analysis.
    
    Returns
    -------
    dict
        Dictionary containing results for all motor scores and time points.
    """
    
    print("=== Demonstrating Improved Regression Pipeline ===")
    print("This pipeline includes:")
    print("1. Comprehensive data validation")
    print("2. Automatic skewness detection")
    print("3. Appropriate log transformations for skewed data")
    print("4. Proper error handling and informative messages")
    print("5. Detailed performance metrics and visualizations")
    print()
    
    # Define motor scores to analyze
    motor_scores = ['nmf_motor', 'Fugl_Meyer_ipsi', 'Fugl_Meyer_contra']
    time_points = [3, 4]  # T3 and T4
    
    all_results = {}
    
    for motor_score in motor_scores:
        print(f"\n{'='*60}")
        print(f"Analyzing {motor_score}")
        print(f"{'='*60}")
        
        all_results[motor_score] = {}
        
        for tp in time_points:
            print(f"\n--- Time Point T{tp} ---")
            
            try:
                # Run the improved pipeline
                results = run_whole_pipeline(
                    df_aligned=df_aligned,
                    regression_info=regression_info,
                    tp=tp,
                    motor_score=motor_score,
                    striatum_labels=striatum_labels,
                    apply_log_transform=True,
                    verbose=True
                )
                
                all_results[motor_score][f'T{tp}'] = results
                
                # Print summary
                print(f"\nSummary for {motor_score} at T{tp}:")
                print(f"  - Original R²: {results['performance']['original_space']['r2']:.3f}")
                print(f"  - Transformed R²: {results['performance']['transformed_space']['r2']:.3f}")
                print(f"  - Features selected: {results['data_info']['n_selected_features']}")
                print(f"  - Transformation: {results['transformation_info']['transformation_applied']}")
                
            except Exception as e:
                print(f"Error analyzing {motor_score} at T{tp}: {str(e)}")
                all_results[motor_score][f'T{tp}'] = {'error': str(e)}
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for motor_score in motor_scores:
        print(f"\n{motor_score}:")
        for tp in time_points:
            tp_key = f'T{tp}'
            if tp_key in all_results[motor_score]:
                if 'error' in all_results[motor_score][tp_key]:
                    print(f"  T{tp}: ERROR - {all_results[motor_score][tp_key]['error']}")
                else:
                    r2_orig = all_results[motor_score][tp_key]['performance']['original_space']['r2']
                    r2_trans = all_results[motor_score][tp_key]['performance']['transformed_space']['r2']
                    transform = all_results[motor_score][tp_key]['transformation_info']['transformation_applied']
                    n_features = all_results[motor_score][tp_key]['data_info']['n_selected_features']
                    print(f"  T{tp}: R²={r2_orig:.3f} (orig), R²={r2_trans:.3f} (trans), {transform}, {n_features} features")
    
    return all_results



def run_neural_network_regression(X_df_clean, y, motor_score_name=None, apply_log_transform=True, verbose=True, 
                                 hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, 
                                 create_plots=False, save_plots_dir=None):
    """
    Runs Neural Network regression with comprehensive data validation and log transformations.
    
    Parameters
    ----------
    X_df_clean : pandas.DataFrame
        DataFrame of shape (n_samples, n_features) containing preprocessed features.
    y : array-like of shape (n_samples,)
        Target variable (e.g., motor scores).
    motor_score_name : str, optional
        Name of the motor score for logging purposes.
    apply_log_transform : bool, default=True
        Whether to apply log transformation based on data skewness.
    verbose : bool, default=True
        Whether to print detailed information.
    hidden_layer_sizes : tuple, default=(100, 50)
        Hidden layer sizes for the neural network.
    max_iter : int, default=1000
        Maximum iterations for training.
    random_state : int, default=42
        Random state for reproducibility.
    create_plots : bool, default=False
        Whether to create diagnostic plots.
    save_plots_dir : str, optional
        Directory to save plots if create_plots=True.
    
    Returns
    -------
    model : sklearn.neural_network.MLPRegressor
        Fitted neural network model.
    y_transformed : array-like
        The transformed target variable.
    transformation_info : dict
        Information about the applied transformations.
    r2_transformed : float
        R^2 score on the transformed scale.
    r2_original : float
        R^2 score on the original scale (if possible).
    y_pred : array-like
        Predictions on original scale (if possible) or transformed scale.
    y_pred_transformed : array-like
        Predictions on transformed scale.
    """
    
    # ===== DATA VALIDATION (same as Ridge function) =====
    if verbose:
        print(f"=== Neural Network Regression for {motor_score_name or 'target variable'} ===")
    
    # Check input types and shapes
    if not isinstance(X_df_clean, pd.DataFrame):
        raise TypeError("X_df_clean must be a pandas DataFrame")
    
    if not isinstance(y, (np.ndarray, pd.Series, list)):
        raise TypeError("y must be an array-like object")
    
    y = np.array(y)
    
    # Check for matching sample sizes
    if len(y) != len(X_df_clean):
        raise ValueError(f"Sample size mismatch: X has {len(X_df_clean)} samples, y has {len(y)} samples")
    
    # Check for sufficient samples
    if len(y) < 10:
        raise ValueError(f"Insufficient samples: {len(y)} samples. Need at least 10 for reliable training.")
    
    # Check for missing values
    if np.any(np.isnan(y)) or X_df_clean.isnull().any().any():
        raise ValueError("Data contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(y)) or np.any(np.isinf(X_df_clean.values)):
        raise ValueError("Data contains infinite values")
    
    # Check for constant target variable
    if np.std(y) == 0:
        raise ValueError("Target variable is constant (zero variance)")
    
    # Remove constant features
    constant_features = X_df_clean.columns[X_df_clean.std() == 0].tolist()
    if constant_features:
        if verbose:
            print(f"Warning: {len(constant_features)} constant features found and will be removed")
        X_df_clean = X_df_clean.drop(columns=constant_features)
    
    if X_df_clean.shape[1] < 2:
        raise ValueError(f"Insufficient features: {X_df_clean.shape[1]} features remaining")
    
    # ===== DATA DISTRIBUTION ANALYSIS =====
    if verbose:
        print(f"\n=== Data Distribution Analysis ===")
        print(f"Sample size: {len(y)}")
        print(f"Number of features: {X_df_clean.shape[1]}")
        print(f"Target variable range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Target variable mean: {y.mean():.3f}")
        print(f"Target variable std: {y.std():.3f}")
    
    # Calculate skewness
    y_skewness = skew(y)
    
    if verbose:
        print(f"Target variable skewness: {y_skewness:.3f}")
        if abs(y_skewness) > 1:
            print(f"Data is {'right' if y_skewness > 0 else 'left'}-skewed (|skewness| > 1)")
    
    # ===== LOG TRANSFORMATION HANDLING =====
    y_transformed = y.copy()
    transformation_info = {
        'original_skewness': y_skewness,
        'transformation_applied': None,
        'transformation_params': {}
    }
    
    if apply_log_transform and abs(y_skewness) > 1:
        # Determine transformation based on skewness and motor score type
        if motor_score_name and 'nmf' in motor_score_name.lower():
            # nmf_motor is typically right-skewed (many low values, few high values)
            if y_skewness > 1:
                y_transformed = np.log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
                if verbose:
                    print(f"Applied log1p(y) transformation for right-skewed nmf_motor data")
        
        elif motor_score_name and 'fugl' in motor_score_name.lower():
            # Fugl Meyer is typically left-skewed (many high values, few low values)
            if y_skewness < -1:
                y_transformed = np.log1p(y.max() - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
                if verbose:
                    print(f"Applied log1p(max - y) transformation for left-skewed Fugl Meyer data")
        
        else:
            # Generic transformation based on skewness
            if y_skewness > 1:
                y_transformed = np.log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
            elif y_skewness < -1:
                y_transformed = np.log1p(y.max() - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
    
    # Check transformed data
    if np.any(np.isnan(y_transformed)) or np.any(np.isinf(y_transformed)):
        raise ValueError("Transformation resulted in NaN or infinite values")
    
    if np.std(y_transformed) == 0:
        raise ValueError("Transformed target variable is constant (zero variance)")
    
    if verbose:
        transformed_skewness = skew(y_transformed)
        print(f"Transformed data skewness: {transformed_skewness:.3f}")
        print(f"Transformation applied: {transformation_info['transformation_applied']}")
    
    # ===== MODEL TRAINING =====
    if verbose:
        print(f"\n=== Neural Network Training ===")
    
    # Create pipeline with scaling and neural network
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])
    
    # Cross-validation
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    
    try:
        scores = cross_val_score(pipe, X_df_clean, y_transformed, cv=cv, scoring='r2')
        if verbose:
            print(f"Cross-validation R² scores: {scores}")
            print(f"Mean CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Fit the model on all data
        pipe.fit(X_df_clean, y_transformed)
        model = pipe
        
        # Predict on training data
        y_pred_transformed = model.predict(X_df_clean)
        # R^2 on transformed scale
        r2_transformed = r2_score(y_transformed, y_pred_transformed)
        # R^2 on original scale (if possible)
        try:
            y_pred = reverse_transform_predictions(y_pred_transformed, transformation_info)
            r2_original = r2_score(y, y_pred)
        except Exception:
            y_pred = y_pred_transformed
            r2_original = None
        if verbose:
            print(f"R² on transformed scale: {r2_transformed:.3f}")
            if r2_original is not None:
                print(f"R² on original scale: {r2_original:.3f}")
            else:
                print("R² on original scale: Could not compute (reverse transform failed)")
        
        # Permutation importance for top 5 predictors
        result = permutation_importance(model, X_df_clean, y_transformed, n_repeats=10, random_state=random_state, scoring='r2')
        importances = result.importances_mean
        feature_names = X_df_clean.columns
        top_indices = importances.argsort()[::-1][:5]
        print("Top 5 predictors (by permutation importance):")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        
        if verbose:
            print(f"Neural network training completed successfully")
        
        # ===== CREATE PLOTS IF REQUESTED =====
        if create_plots:
            if verbose:
                print(f"\n=== Creating Diagnostic Plots ===")
            
            # Create diagnostic report
            diagnostic_report = create_neural_network_diagnostic_report(
                model, X_df_clean, y, y_transformed, transformation_info, 
                motor_score_name, save_dir=save_plots_dir
            )
            
            if verbose:
                print("Diagnostic plots created successfully!")
            
    except Exception as e:
        raise RuntimeError(f"Neural network training failed: {str(e)}")
    
    return model, y_transformed, transformation_info, r2_transformed, r2_original, y_pred, y_pred_transformed



def run_random_forest_regression(X_df_clean, y, motor_score_name=None, apply_log_transform=True, verbose=True,
                                n_estimators=100, max_depth=None, random_state=42):
    """
    Runs Random Forest regression with comprehensive data validation and log transformations.
    
    Parameters
    ----------
    X_df_clean : pandas.DataFrame
        DataFrame of shape (n_samples, n_features) containing preprocessed features.
    y : array-like of shape (n_samples,)
        Target variable (e.g., motor scores).
    motor_score_name : str, optional
        Name of the motor score for logging purposes.
    apply_log_transform : bool, default=True
        Whether to apply log transformation based on data skewness.
    verbose : bool, default=True
        Whether to print detailed information.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    model : sklearn.ensemble.RandomForestRegressor
        Fitted random forest model.
    y_transformed : array-like
        The transformed target variable.
    transformation_info : dict
        Information about the applied transformations.
    feature_importance : pandas.DataFrame
        Feature importance scores.
    """
    
    # ===== DATA VALIDATION (same as other functions) =====
    if verbose:
        print(f"=== Random Forest Regression for {motor_score_name or 'target variable'} ===")
    
    # Check input types and shapes
    if not isinstance(X_df_clean, pd.DataFrame):
        raise TypeError("X_df_clean must be a pandas DataFrame")
    
    if not isinstance(y, (np.ndarray, pd.Series, list)):
        raise TypeError("y must be an array-like object")
    
    y = np.array(y)
    
    # Check for matching sample sizes
    if len(y) != len(X_df_clean):
        raise ValueError(f"Sample size mismatch: X has {len(X_df_clean)} samples, y has {len(y)} samples")
    
    # Check for sufficient samples
    if len(y) < 10:
        raise ValueError(f"Insufficient samples: {len(y)} samples. Need at least 10 for reliable training.")
    
    # Check for missing values
    if np.any(np.isnan(y)) or X_df_clean.isnull().any().any():
        raise ValueError("Data contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(y)) or np.any(np.isinf(X_df_clean.values)):
        raise ValueError("Data contains infinite values")
    
    # Check for constant target variable
    if np.std(y) == 0:
        raise ValueError("Target variable is constant (zero variance)")
    
    # Remove constant features
    constant_features = X_df_clean.columns[X_df_clean.std() == 0].tolist()
    if constant_features:
        if verbose:
            print(f"Warning: {len(constant_features)} constant features found and will be removed")
        X_df_clean = X_df_clean.drop(columns=constant_features)
    
    if X_df_clean.shape[1] < 2:
        raise ValueError(f"Insufficient features: {X_df_clean.shape[1]} features remaining")
    
    # ===== DATA DISTRIBUTION ANALYSIS =====
    if verbose:
        print(f"\n=== Data Distribution Analysis ===")
        print(f"Sample size: {len(y)}")
        print(f"Number of features: {X_df_clean.shape[1]}")
        print(f"Target variable range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Target variable mean: {y.mean():.3f}")
        print(f"Target variable std: {y.std():.3f}")
    
    # Calculate skewness
    y_skewness = skew(y)
    
    if verbose:
        print(f"Target variable skewness: {y_skewness:.3f}")
        if abs(y_skewness) > 1:
            print(f"Data is {'right' if y_skewness > 0 else 'left'}-skewed (|skewness| > 1)")
    
    # ===== LOG TRANSFORMATION HANDLING =====
    y_transformed = y.copy()
    transformation_info = {
        'original_skewness': y_skewness,
        'transformation_applied': None,
        'transformation_params': {}
    }
    
    if apply_log_transform and abs(y_skewness) > 1:
        # Determine transformation based on skewness and motor score type
        if motor_score_name and 'nmf' in motor_score_name.lower():
            # nmf_motor is typically right-skewed (many low values, few high values)
            if y_skewness > 1:
                y_transformed = np.log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
                if verbose:
                    print(f"Applied log1p(y) transformation for right-skewed nmf_motor data")
        
        elif motor_score_name and 'fugl' in motor_score_name.lower():
            # Fugl Meyer is typically left-skewed (many high values, few low values)
            if y_skewness < -1:
                y_transformed = np.log1p(y.max() - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
                if verbose:
                    print(f"Applied log1p(max - y) transformation for left-skewed Fugl Meyer data")
        
        else:
            # Generic transformation based on skewness
            if y_skewness > 1:
                y_transformed = np.log1p(y)
                transformation_info.update({
                    'transformation_applied': 'log1p',
                    'transformation_params': {}
                })
            elif y_skewness < -1:
                y_transformed = np.log1p(y.max() - y)
                transformation_info.update({
                    'transformation_applied': 'log1p_reverse',
                    'transformation_params': {'max_value': y.max()}
                })
    
    # Check transformed data
    if np.any(np.isnan(y_transformed)) or np.any(np.isinf(y_transformed)):
        raise ValueError("Transformation resulted in NaN or infinite values")
    
    if np.std(y_transformed) == 0:
        raise ValueError("Transformed target variable is constant (zero variance)")
    
    if verbose:
        transformed_skewness = skew(y_transformed)
        print(f"Transformed data skewness: {transformed_skewness:.3f}")
        print(f"Transformation applied: {transformation_info['transformation_applied']}")
    
    # ===== MODEL TRAINING =====
    if verbose:
        print(f"\n=== Random Forest Training ===")
    
    # Create random forest model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    
    try:
        scores = cross_val_score(rf_model, X_df_clean, y_transformed, cv=cv, scoring='r2')
        if verbose:
            print(f"Cross-validation R² scores: {scores}")
            print(f"Mean CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Fit the model on all data
        rf_model.fit(X_df_clean, y_transformed)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_df_clean.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        if verbose:
            print(f"Random forest training completed successfully")
            print(f"Top 10 most important features:")
            print(feature_importance.head(10))
            
    except Exception as e:
        raise RuntimeError(f"Random forest training failed: {str(e)}")
    
    return rf_model, y_transformed, transformation_info, feature_importance



def check_regression_assumptions(y_true, y_pred, y_transformed=None, y_pred_transformed=None, 
                                transformation_info=None, verbose=True, alpha=0.05):
    """
    Comprehensive check of regression assumptions including normality, homoscedasticity, 
    linearity, and independence of residuals.
    
    Parameters
    ----------
    y_true : array-like
        True target values (original scale).
    y_pred : array-like
        Predicted target values (original scale).
    y_transformed : array-like, optional
        True target values in transformed scale.
    y_pred_transformed : array-like, optional
        Predicted target values in transformed scale.
    transformation_info : dict, optional
        Information about applied transformations.
    verbose : bool, default=True
        Whether to print detailed results and create plots.
    alpha : float, default=0.05
        Significance level for statistical tests.
    
    Returns
    -------
    dict
        Dictionary containing assumption test results and recommendations.
    """
    
    if verbose:
        print("=== REGRESSION ASSUMPTIONS CHECK ===")
        print("This function checks the following assumptions:")
        print("1. Normality of residuals")
        print("2. Homoscedasticity (constant variance)")
        print("3. Linearity of relationship")
        print("4. Independence of residuals")
        print("5. Absence of influential outliers")
        print()
    
    results = {
        'assumptions': {},
        'recommendations': [],
        'transformation_info': transformation_info
    }
    
    # Calculate residuals in both original and transformed spaces
    residuals_original = y_true - y_pred
    
    if y_transformed is not None and y_pred_transformed is not None:
        residuals_transformed = y_transformed - y_pred_transformed
        use_transformed = True
    else:
        residuals_transformed = None
        use_transformed = False
    
    # ===== 1. NORMALITY OF RESIDUALS =====
    if verbose:
        print("1. NORMALITY OF RESIDUALS")
        print("-" * 30)
    
    # Shapiro-Wilk test for normality
    
    # Test original residuals
    shapiro_stat_orig, shapiro_p_orig = shapiro(residuals_original)
    jarque_bera_stat_orig, jarque_bera_p_orig = jarque_bera(residuals_original)
    
    normality_original = {
        'shapiro_wilk': {'statistic': shapiro_stat_orig, 'p_value': shapiro_p_orig, 'is_normal': shapiro_p_orig > alpha},
        'jarque_bera': {'statistic': jarque_bera_stat_orig, 'p_value': jarque_bera_p_orig, 'is_normal': jarque_bera_p_orig > alpha}
    }
    
    if verbose:
        print(f"Original scale residuals:")
        print(f"  Shapiro-Wilk: statistic={shapiro_stat_orig:.3f}, p={shapiro_p_orig:.3f}, normal={shapiro_p_orig > alpha}")
        print(f"  Jarque-Bera: statistic={jarque_bera_stat_orig:.3f}, p={jarque_bera_p_orig:.3f}, normal={jarque_bera_p_orig > alpha}")
    
    # Test transformed residuals if available
    if use_transformed:
        shapiro_stat_trans, shapiro_p_trans = shapiro(residuals_transformed)
        jarque_bera_stat_trans, jarque_bera_p_trans = jarque_bera(residuals_transformed)
        
        normality_transformed = {
            'shapiro_wilk': {'statistic': shapiro_stat_trans, 'p_value': shapiro_p_trans, 'is_normal': shapiro_p_trans > alpha},
            'jarque_bera': {'statistic': jarque_bera_stat_trans, 'p_value': jarque_bera_p_trans, 'is_normal': jarque_bera_p_trans > alpha}
        }
        
        if verbose:
            print(f"Transformed scale residuals:")
            print(f"  Shapiro-Wilk: statistic={shapiro_stat_trans:.3f}, p={shapiro_p_trans:.3f}, normal={shapiro_p_trans > alpha}")
            print(f"  Jarque-Bera: statistic={jarque_bera_stat_trans:.3f}, p={jarque_bera_p_trans:.3f}, normal={jarque_bera_p_trans > alpha}")
    
    results['assumptions']['normality'] = {
        'original': normality_original,
        'transformed': normality_transformed if use_transformed else None
    }
    
    # ===== 2. HOMOSCEDASTICITY (CONSTANT VARIANCE) =====
    if verbose:
        print("\n2. HOMOSCEDASTICITY (CONSTANT VARIANCE)")
        print("-" * 40)
    
    # Breusch-Pagan test for homoscedasticity
    # Test correlation between residuals and predicted values
    spearman_corr_orig, spearman_p_orig = spearmanr(y_pred, np.abs(residuals_original))
    
    homoscedasticity_original = {
        'spearman_correlation': {'correlation': spearman_corr_orig, 'p_value': spearman_p_orig, 'is_homoscedastic': spearman_p_orig > alpha}
    }
    
    if verbose:
        print(f"Original scale:")
        print(f"  Spearman correlation (residuals vs predicted): {spearman_corr_orig:.3f}, p={spearman_p_orig:.3f}")
        print(f"  Homoscedastic: {spearman_p_orig > alpha}")
    
    if use_transformed:
        spearman_corr_trans, spearman_p_trans = spearmanr(y_pred_transformed, np.abs(residuals_transformed))
        
        homoscedasticity_transformed = {
            'spearman_correlation': {'correlation': spearman_corr_trans, 'p_value': spearman_p_trans, 'is_homoscedastic': spearman_p_trans > alpha}
        }
        
        if verbose:
            print(f"Transformed scale:")
            print(f"  Spearman correlation (residuals vs predicted): {spearman_corr_trans:.3f}, p={spearman_p_trans:.3f}")
            print(f"  Homoscedastic: {spearman_p_trans > alpha}")
    
    results['assumptions']['homoscedasticity'] = {
        'original': homoscedasticity_original,
        'transformed': homoscedasticity_transformed if use_transformed else None
    }
    
    # ===== 3. LINEARITY =====
    if verbose:
        print("\n3. LINEARITY")
        print("-" * 15)
    
    # Test linearity using correlation between predicted and actual values
    pearson_corr_orig, pearson_p_orig = pearsonr(y_true, y_pred)
    spearman_corr_orig_lin, spearman_p_orig_lin = spearmanr(y_true, y_pred)
    
    linearity_original = {
        'pearson_correlation': {'correlation': pearson_corr_orig, 'p_value': pearson_p_orig},
        'spearman_correlation': {'correlation': spearman_corr_orig_lin, 'p_value': spearman_p_orig_lin}
    }
    
    if verbose:
        print(f"Original scale:")
        print(f"  Pearson correlation (true vs predicted): {pearson_corr_orig:.3f}, p={pearson_p_orig:.3f}")
        print(f"  Spearman correlation (true vs predicted): {spearman_corr_orig_lin:.3f}, p={spearman_p_orig_lin:.3f}")
    
    if use_transformed:
        pearson_corr_trans, pearson_p_trans = pearsonr(y_transformed, y_pred_transformed)
        spearman_corr_trans_lin, spearman_p_trans_lin = spearmanr(y_transformed, y_pred_transformed)
        
        linearity_transformed = {
            'pearson_correlation': {'correlation': pearson_corr_trans, 'p_value': pearson_p_trans},
            'spearman_correlation': {'correlation': spearman_corr_trans_lin, 'p_value': spearman_p_trans_lin}
        }
        
        if verbose:
            print(f"Transformed scale:")
            print(f"  Pearson correlation (true vs predicted): {pearson_corr_trans:.3f}, p={pearson_p_trans:.3f}")
            print(f"  Spearman correlation (true vs predicted): {spearman_corr_trans_lin:.3f}, p={spearman_p_trans_lin:.3f}")
    
    results['assumptions']['linearity'] = {
        'original': linearity_original,
        'transformed': linearity_transformed if use_transformed else None
    }
    
    # ===== 4. INDEPENDENCE OF RESIDUALS =====
    if verbose:
        print("\n4. INDEPENDENCE OF RESIDUALS")
        print("-" * 30)
    
    # Durbin-Watson test for autocorrelation
    
    dw_stat_orig = durbin_watson(residuals_original)
    
    # Durbin-Watson interpretation: 1.5-2.5 is good, <1.5 or >2.5 indicates autocorrelation
    is_independent_orig = 1.5 <= dw_stat_orig <= 2.5
    
    independence_original = {
        'durbin_watson': {'statistic': dw_stat_orig, 'is_independent': is_independent_orig}
    }
    
    if verbose:
        print(f"Original scale:")
        print(f"  Durbin-Watson statistic: {dw_stat_orig:.3f}")
        print(f"  Independent residuals: {is_independent_orig} (1.5-2.5 is good)")
    
    if use_transformed:
        dw_stat_trans = durbin_watson(residuals_transformed)
        is_independent_trans = 1.5 <= dw_stat_trans <= 2.5
        
        independence_transformed = {
            'durbin_watson': {'statistic': dw_stat_trans, 'is_independent': is_independent_trans}
        }
        
        if verbose:
            print(f"Transformed scale:")
            print(f"  Durbin-Watson statistic: {dw_stat_trans:.3f}")
            print(f"  Independent residuals: {is_independent_trans} (1.5-2.5 is good)")
    
    results['assumptions']['independence'] = {
        'original': independence_original,
        'transformed': independence_transformed if use_transformed else None
    }
    
    # ===== 5. OUTLIERS AND INFLUENTIAL POINTS =====
    if verbose:
        print("\n5. OUTLIERS AND INFLUENTIAL POINTS")
        print("-" * 35)
    
    # Calculate standardized residuals
    std_residuals_orig = residuals_original / np.std(residuals_original)
    outliers_orig = np.abs(std_residuals_orig) > 3  # 3 standard deviations
    
    outliers_original = {
        'n_outliers': np.sum(outliers_orig),
        'outlier_percentage': np.mean(outliers_orig) * 100,
        'outlier_indices': np.where(outliers_orig)[0].tolist()
    }
    
    if verbose:
        print(f"Original scale:")
        print(f"  Number of outliers (|residual| > 3σ): {outliers_original['n_outliers']}")
        print(f"  Outlier percentage: {outliers_original['outlier_percentage']:.1f}%")
    
    if use_transformed:
        std_residuals_trans = residuals_transformed / np.std(residuals_transformed)
        outliers_trans = np.abs(std_residuals_trans) > 3
        
        outliers_transformed = {
            'n_outliers': np.sum(outliers_trans),
            'outlier_percentage': np.mean(outliers_trans) * 100,
            'outlier_indices': np.where(outliers_trans)[0].tolist()
        }
        
        if verbose:
            print(f"Transformed scale:")
            print(f"  Number of outliers (|residual| > 3σ): {outliers_transformed['n_outliers']}")
            print(f"  Outlier percentage: {outliers_transformed['outlier_percentage']:.1f}%")
    
    results['assumptions']['outliers'] = {
        'original': outliers_original,
        'transformed': outliers_transformed if use_transformed else None
    }
    
    # ===== GENERATE RECOMMENDATIONS =====
    recommendations = []
    
    # Normality recommendations
    if not normality_original['shapiro_wilk']['is_normal']:
        recommendations.append("Original scale residuals are not normally distributed. Consider log transformation.")
    
    if use_transformed and not normality_transformed['shapiro_wilk']['is_normal']:
        recommendations.append("Transformed scale residuals are still not normally distributed. Consider other transformations.")
    
    # Homoscedasticity recommendations
    if not homoscedasticity_original['spearman_correlation']['is_homoscedastic']:
        recommendations.append("Heteroscedasticity detected in original scale. Consider transformation or weighted regression.")
    
    # Independence recommendations
    if not independence_original['durbin_watson']['is_independent']:
        recommendations.append("Residuals show autocorrelation. Consider time-series methods or check for missing variables.")
    
    # Outlier recommendations
    if outliers_original['outlier_percentage'] > 5:
        recommendations.append(f"High number of outliers ({outliers_original['outlier_percentage']:.1f}%). Consider robust regression methods.")
    
    # Overall assessment
    if verbose:
        print("\n=== OVERALL ASSESSMENT ===")
        print("-" * 25)
        
        # Count violated assumptions
        violations_orig = 0
        if not normality_original['shapiro_wilk']['is_normal']: violations_orig += 1
        if not homoscedasticity_original['spearman_correlation']['is_homoscedastic']: violations_orig += 1
        if not independence_original['durbin_watson']['is_independent']: violations_orig += 1
        if outliers_original['outlier_percentage'] > 5: violations_orig += 1
        
        if violations_orig == 0:
            print("✅ All major assumptions are satisfied in original scale!")
        elif violations_orig <= 2:
            print(f"⚠️  {violations_orig} assumption(s) violated in original scale. Model may still be useful.")
        else:
            print(f"❌ {violations_orig} assumption(s) violated in original scale. Consider model alternatives.")
        
        if use_transformed:
            violations_trans = 0
            if not normality_transformed['shapiro_wilk']['is_normal']: violations_trans += 1
            if not homoscedasticity_transformed['spearman_correlation']['is_homoscedastic']: violations_trans += 1
            if not independence_transformed['durbin_watson']['is_independent']: violations_trans += 1
            if outliers_transformed['outlier_percentage'] > 5: violations_trans += 1
            
            if violations_trans == 0:
                print("✅ All major assumptions are satisfied in transformed scale!")
            elif violations_trans <= 2:
                print(f"⚠️  {violations_trans} assumption(s) violated in transformed scale. Model may still be useful.")
            else:
                print(f"❌ {violations_trans} assumption(s) violated in transformed scale. Consider model alternatives.")
    
    results['recommendations'] = recommendations
    
    # ===== CREATE VISUALIZATIONS =====
    if verbose:
        try:
            print("\n=== CREATING DIAGNOSTIC PLOTS ===")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Residuals vs Predicted (Original)
            axes[0, 0].scatter(y_pred, residuals_original, alpha=0.7)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted (Original)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Q-Q Plot (Original)
            probplot(residuals_original, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Original)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Histogram of Residuals (Original)
            axes[0, 2].hist(residuals_original, bins=20, alpha=0.7, density=True)
            axes[0, 2].set_xlabel('Residuals')
            axes[0, 2].set_ylabel('Density')
            axes[0, 2].set_title('Residuals Distribution (Original)')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Residuals vs Predicted (Transformed) - if available
            if use_transformed:
                axes[1, 0].scatter(y_pred_transformed, residuals_transformed, alpha=0.7)
                axes[1, 0].axhline(y=0, color='r', linestyle='--')
                axes[1, 0].set_xlabel('Predicted Values (Transformed)')
                axes[1, 0].set_ylabel('Residuals (Transformed)')
                axes[1, 0].set_title('Residuals vs Predicted (Transformed)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 5. Q-Q Plot (Transformed)
                probplot(residuals_transformed, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot (Transformed)')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 6. Histogram of Residuals (Transformed)
                axes[1, 2].hist(residuals_transformed, bins=20, alpha=0.7, density=True)
                axes[1, 2].set_xlabel('Residuals (Transformed)')
                axes[1, 2].set_ylabel('Density')
                axes[1, 2].set_title('Residuals Distribution (Transformed)')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                # Hide transformed plots if not available
                for i in range(3):
                    axes[1, i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create diagnostic plots: {str(e)}")
    
    return results



def transform_target_if_skewed(y, motor_score_name=None, apply_log_transform=True, verbose=True):
    """
    Detects skew in a target variable and applies log transformation if appropriate.

    Parameters
    ----------
    y : array-like
        Original target variable.
    motor_score_name : str, optional
        Used to guide expected skew direction (e.g., 'nmf' or 'fugl').
    apply_log_transform : bool, default=True
        Whether to apply transformation if skewed.
    verbose : bool, default=True
        Whether to print transformation info.

    Returns
    -------
    y_transformed : np.ndarray
        Transformed target (or original if no transform applied).
    transformation_info : dict
        Information about the applied transformation.
    """
    y = np.array(y)
    y_skewness = skew(y)

    transformation_info = {
        'original_skewness': y_skewness,
        'transformation_applied': None,
        'transformation_params': {}
    }

    y_transformed = y.copy()

    if apply_log_transform and abs(y_skewness) > 1:
        if motor_score_name and 'nmf' in motor_score_name.lower() and y_skewness > 1:
            y_transformed = np.log1p(y)
            transformation_info.update({
                'transformation_applied': 'log1p',
                'transformation_params': {}
            })
        elif motor_score_name and 'fugl' in motor_score_name.lower() and y_skewness < -1:
            y_transformed = np.log1p(y.max() - y)
            transformation_info.update({
                'transformation_applied': 'log1p_reverse',
                'transformation_params': {'max_value': y.max()}
            })
        elif y_skewness > 1:
            y_transformed = np.log1p(y)
            transformation_info.update({
                'transformation_applied': 'log1p',
                'transformation_params': {}
            })
        elif y_skewness < -1:
            y_transformed = np.log1p(y.max() - y)
            transformation_info.update({
                'transformation_applied': 'log1p_reverse',
                'transformation_params': {'max_value': y.max()}
            })

    if verbose:
        print(f"Original skewness: {y_skewness:.3f}")
        print(f"Transformation applied: {transformation_info['transformation_applied'] or 'None'}")

    return y_transformed, transformation_info



def visualize_neural_network_regression(model, X_df_clean, y, y_transformed, transformation_info, 
                                       motor_score_name=None, figsize=(15, 12), save_path=None):
    """
    Comprehensive visualization of neural network regression results.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted neural network model (pipeline with scaler and MLPRegressor).
    X_df_clean : pandas.DataFrame
        Feature matrix used for training.
    y : array-like
        Original target variable.
    y_transformed : array-like
        Transformed target variable used for training.
    transformation_info : dict
        Information about applied transformations.
    motor_score_name : str, optional
        Name of the motor score for plot titles.
    figsize : tuple, default=(15, 12)
        Figure size for the plots.
    save_path : str, optional
        Path to save the figure (e.g., 'neural_network_plots.png').
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure with all plots.
    """
    
    # Get predictions
    y_pred_transformed = model.predict(X_df_clean)
    
    # Try to get predictions on original scale
    try:
        y_pred = reverse_transform_predictions(y_pred_transformed, transformation_info)
        use_original_scale = True
    except Exception:
        y_pred = y_pred_transformed
        use_original_scale = False
    
    # Calculate residuals
    residuals_transformed = y_transformed - y_pred_transformed
    if use_original_scale:
        residuals_original = y - y_pred
    else:
        residuals_original = residuals_transformed
    
    # Calculate R² scores
    r2_transformed = r2_score(y_transformed, y_pred_transformed)
    if use_original_scale:
        r2_original = r2_score(y, y_pred)
    else:
        r2_original = r2_transformed
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Neural Network Regression Results for {motor_score_name or "Target Variable"}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Prediction vs Actual Plot (Transformed Scale)
    ax1 = axes[0, 0]
    ax1.scatter(y_transformed, y_pred_transformed, alpha=0.6, color='steelblue', s=50)
    
    # Add regression line
    min_val = min(y_transformed.min(), y_pred_transformed.min())
    max_val = max(y_transformed.max(), y_pred_transformed.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add trend line
    z = np.polyfit(y_transformed, y_pred_transformed, 1)
    p = np.poly1d(z)
    ax1.plot(y_transformed, p(y_transformed), "g-", linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Actual Values (Transformed)')
    ax1.set_ylabel('Predicted Values (Transformed)')
    ax1.set_title(f'Prediction vs Actual (Transformed)\nR² = {r2_transformed:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction vs Actual Plot (Original Scale) - if possible
    ax2 = axes[0, 1]
    if use_original_scale:
        ax2.scatter(y, y_pred, alpha=0.6, color='darkgreen', s=50)
        
        # Add regression line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(y, y_pred, 1)
        p = np.poly1d(z)
        ax2.plot(y, p(y), "g-", linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
        
        ax2.set_xlabel('Actual Values (Original)')
        ax2.set_ylabel('Predicted Values (Original)')
        ax2.set_title(f'Prediction vs Actual (Original)\nR² = {r2_original:.3f}')
    else:
        ax2.text(0.5, 0.5, 'Original scale\nnot available\n(due to transform)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Prediction vs Actual (Original)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals Plot (Transformed Scale)
    ax3 = axes[1, 0]
    ax3.scatter(y_pred_transformed, residuals_transformed, alpha=0.6, color='orange', s=50)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Values (Transformed)')
    ax3.set_ylabel('Residuals (Transformed)')
    ax3.set_title('Residuals vs Predicted (Transformed)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals Plot (Original Scale) - if possible
    ax4 = axes[1, 1]
    if use_original_scale:
        ax4.scatter(y_pred, residuals_original, alpha=0.6, color='purple', s=50)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted Values (Original)')
        ax4.set_ylabel('Residuals (Original)')
        ax4.set_title('Residuals vs Predicted (Original)')
    else:
        ax4.text(0.5, 0.5, 'Original scale\nnot available\n(due to transform)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Residuals vs Predicted (Original)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals Distribution
    ax5 = axes[2, 0]
    ax5.hist(residuals_transformed, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals (Transformed)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Residuals Distribution (Transformed)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature Importance (Top 10)
    ax6 = axes[2, 1]
    
    # Get permutation importance
    result = permutation_importance(model, X_df_clean, y_transformed, n_repeats=10, random_state=42, scoring='r2')
    importances = result.importances_mean
    feature_names = X_df_clean.columns
    
    # Get top 10 features
    top_indices = importances.argsort()[::-1][:10]
    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax6.barh(y_pos, top_importances, color='coral', alpha=0.7)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(top_features)
    ax6.set_xlabel('Permutation Importance')
    ax6.set_title('Top 10 Feature Importance')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, v in enumerate(top_importances):
        ax6.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig



def plot_neural_network_learning_curves(model, X_df_clean, y_transformed, cv_splits=5, 
                                       motor_score_name=None, figsize=(12, 5), save_path=None):
    """
    Plot learning curves for neural network to assess overfitting/underfitting.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted neural network model.
    X_df_clean : pandas.DataFrame
        Feature matrix used for training.
    y_transformed : array-like
        Transformed target variable used for training.
    cv_splits : int, default=5
        Number of cross-validation splits.
    motor_score_name : str, optional
        Name of the motor score for plot titles.
    figsize : tuple, default=(12, 5)
        Figure size for the plots.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure with learning curves.
    """
    
    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_df_clean, y_transformed, 
        cv=cv_splits, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2',
        random_state=42
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Neural Network Learning Curves for {motor_score_name or "Target Variable"}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Learning curves
    ax1.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    ax1.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation Score')
    ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax1.set_xlabel('Training Examples')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gap between training and validation
    gap = train_mean - val_mean
    ax2.plot(train_sizes, gap, 'o-', color='green', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Training Examples')
    ax2.set_ylabel('Training Score - CV Score')
    ax2.set_title('Overfitting Gap')
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation text
    if gap[-1] > 0.1:
        interpretation = "High overfitting detected"
        color = 'red'
    elif gap[-1] > 0.05:
        interpretation = "Moderate overfitting"
        color = 'orange'
    else:
        interpretation = "Good generalization"
        color = 'green'
    
    ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=12, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    return fig



def create_neural_network_diagnostic_report(model, X_df_clean, y, y_transformed, transformation_info, 
                                           motor_score_name=None, save_dir=None):
    """
    Create a comprehensive diagnostic report for neural network regression.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted neural network model.
    X_df_clean : pandas.DataFrame
        Feature matrix used for training.
    y : array-like
        Original target variable.
    y_transformed : array-like
        Transformed target variable used for training.
    transformation_info : dict
        Information about applied transformations.
    motor_score_name : str, optional
        Name of the motor score for report titles.
    save_dir : str, optional
        Directory to save the report figures.
    
    Returns
    -------
    dict
        Dictionary containing all diagnostic metrics and figures.
    """
    
    # Get predictions
    y_pred_transformed = model.predict(X_df_clean)
    
    # Try to get predictions on original scale
    try:
        y_pred = reverse_transform_predictions(y_pred_transformed, transformation_info)
        use_original_scale = True
    except Exception:
        y_pred = y_pred_transformed
        use_original_scale = False
    
    # Calculate metrics
    metrics = {
        'r2_transformed': r2_score(y_transformed, y_pred_transformed),
        'mse_transformed': mean_squared_error(y_transformed, y_pred_transformed),
        'mae_transformed': mean_absolute_error(y_transformed, y_pred_transformed),
        'rmse_transformed': np.sqrt(mean_squared_error(y_transformed, y_pred_transformed))
    }
    
    if use_original_scale:
        metrics.update({
            'r2_original': r2_score(y, y_pred),
            'mse_original': mean_squared_error(y, y_pred),
            'mae_original': mean_absolute_error(y, y_pred),
            'rmse_original': np.sqrt(mean_squared_error(y, y_pred))
        })
    
    # Create visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = f"nn_diagnostic_{motor_score_name or 'target'}"
        
        # Main diagnostic plots
        fig1 = visualize_neural_network_regression(
            model, X_df_clean, y, y_transformed, transformation_info, 
            motor_score_name, save_path=f"{save_dir}/{base_name}_main_plots.png"
        )
        
        # Learning curves
        fig2 = plot_neural_network_learning_curves(
            model, X_df_clean, y_transformed, 
            motor_score_name, save_path=f"{save_dir}/{base_name}_learning_curves.png"
        )
        
        # Regression assumptions
        fig3 = check_regression_assumptions(
            y, y_pred, y_transformed, y_pred_transformed, transformation_info, 
            verbose=False, alpha=0.05
        )
        if fig3:
            fig3.savefig(f"{save_dir}/{base_name}_assumptions.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"NEURAL NETWORK DIAGNOSTIC REPORT")
    print(f"Target Variable: {motor_score_name or 'Unknown'}")
    print(f"{'='*60}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"R² (transformed): {metrics['r2_transformed']:.4f}")
    print(f"MSE (transformed): {metrics['mse_transformed']:.4f}")
    print(f"MAE (transformed): {metrics['mae_transformed']:.4f}")
    print(f"RMSE (transformed): {metrics['rmse_transformed']:.4f}")
    
    if use_original_scale:
        print(f"R² (original): {metrics['r2_original']:.4f}")
        print(f"MSE (original): {metrics['mse_original']:.4f}")
        print(f"MAE (original): {metrics['mae_original']:.4f}")
        print(f"RMSE (original): {metrics['rmse_original']:.4f}")
    
    print(f"\nTRANSFORMATION INFO:")
    print(f"Original skewness: {transformation_info.get('original_skewness', 'N/A'):.3f}")
    print(f"Transformation applied: {transformation_info.get('transformation_applied', 'None')}")
    
    print(f"\nMODEL INFO:")
    print(f"Number of features: {X_df_clean.shape[1]}")
    print(f"Number of samples: {len(y)}")
    
    # Get model architecture info
    mlp = model.named_steps['mlp']
    print(f"Hidden layers: {mlp.hidden_layer_sizes}")
    print(f"Activation: {mlp.activation}")
    print(f"Max iterations: {mlp.max_iter}")
    
    return {
        'metrics': metrics,
        'transformation_info': transformation_info,
        'use_original_scale': use_original_scale,
        'figures': [fig1, fig2] if save_dir else None
    }

############################################# CLASSIFICATION FUNCTIONS #############################################

def nagelkerke_r2(y_true, y_pred_proba, y_null_proba):
    """
    Compute Nagelkerke's pseudo-R² for binary classification,
    robust to test sets that contain only one class.
    """
    N = len(y_true)
    try:
        logL_model = -log_loss(y_true, y_pred_proba, labels=[0, 1], normalize=False)
        logL_null = -log_loss(y_true, y_null_proba, labels=[0, 1], normalize=False)
    except ValueError:
        return None  # In case log_loss fails despite labels arg

    r2 = (1 - np.exp(-2 * (logL_model - logL_null) / N)) / (1 - np.exp(-2 * logL_null / N))
    return r2



def run_classification_with_rfe_tracking(X_df, y, n_splits=100, test_size=0.1, min_accuracy_threshold=0.9):
    """
    Classification with RFE and repeated subsampling based on Adhikari et al., 2021.
    Also computes average Nagelkerke R² over successful splits.

    Parameters
    ----------
    X_df : pd.DataFrame
        Z-scored input features.
    y : np.array
        Target binary class labels (0 or 1).
    n_splits : int
        Number of train/test splits to repeat.
    test_size : float
        Proportion of data to hold out for testing.
    min_accuracy_threshold : float
        Minimum accuracy required to retain feature set and compute R².

    Returns
    -------
    feature_ranking_counts : pd.Series
        How often each feature was selected across good splits.
    r2_scores : list
        List of Nagelkerke R² values from each successful split.
    """
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    feature_counts = defaultdict(int)
    r2_scores = []

    for train_idx, test_idx in sss.split(X_df, y):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit MLR with RFE
        base_model = LogisticRegression(max_iter=1000, solver='lbfgs')  # Let multi_class be automatic
        rfe = RFE(estimator=base_model, n_features_to_select=1, step=1)
        rfe.fit(X_train, y_train)

        # Get feature ranking order (best to worst)
        ranking_order = np.argsort(rfe.ranking_)

        # Track accuracy and R² as we reduce features
        best_features = []
        best_accuracy = 0
        best_r2 = None

        for i in range(1, len(ranking_order) + 1):
            selected = X_train.columns[ranking_order[:i]]
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_train[selected], y_train)
            y_pred = model.predict(X_test[selected])
            acc = accuracy_score(y_test, y_pred)

            if acc >= min_accuracy_threshold and acc >= best_accuracy:
                best_accuracy = acc
                best_features = selected.tolist()  # Convert to plain list

                # Compute Nagelkerke R²
                y_pred_proba = model.predict_proba(X_test[selected])
                y_null_proba = np.tile(np.bincount(y_train) / len(y_train), (len(y_test), 1))
                best_r2 = nagelkerke_r2(y_test, y_pred_proba, y_null_proba)

        if len(best_features) > 0:
            for feat in best_features:
                feature_counts[feat] += 1
            if best_r2 is not None:
                r2_scores.append(best_r2)


    feature_ranking_counts = pd.Series(feature_counts).sort_values(ascending=False)
    return feature_ranking_counts, r2_scores

def run_simple_classification(X_df, y, n_splits=100, test_size=0.1, min_accuracy_threshold=0.9):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    r2_scores = []
    acc_scores = []

    for train_idx, test_idx in sss.split(X_df, y):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        if acc >= min_accuracy_threshold:
            y_pred_proba = model.predict_proba(X_test)
            y_null_proba = np.tile(np.bincount(y_train) / len(y_train), (len(y_test), 1))
            r2 = nagelkerke_r2(y_test, y_pred_proba, y_null_proba)
            r2_scores.append(r2)
            acc_scores.append(acc)

    return acc_scores, r2_scores