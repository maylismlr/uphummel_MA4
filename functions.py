import os
import pandas as pd
import scipy.io

def load_data_T1_only(folder_path, rois):
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


def load_data(folder_path, rois):
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

    return subject_matrices, rsfMRI_full_info, rsfMRI_info, list(subject_matrices.keys())

def matrices_to_wide_df(subject_matrices):
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

    return df

import os
import pandas as pd
import scipy.io

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
