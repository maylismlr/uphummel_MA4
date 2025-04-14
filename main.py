import functions
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main(type = 'all', cluster = False, num_clusters = 2):
    # Folder containing the data
    folder_path = "FC_matrices_times_wp11/"

    # keep only ROIS
    rois = [363, 364, 365, 368, 372, 373, 374, 377, 379, 361, 370, 362, 371, 10, 11, 12, 54, 56, 78, 96, 190, 191, 192, 234, 236, 258, 276, 8, 9, 51, 52, 53, 188, 189, 231, 232, 233]
    rois = [roi - 1 for roi in rois]
    
    # categorical and numerical columns
    categorical_cols = ['Lesion_side', 'Stroke_location','Gender','Age','Education_level','Combined', 'Bilateral']
    numerical_cols = ['lesion_volume_mm3']


    if type == 'all':
        # Load the data
        all_matrices, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(all_matrices, subjects, folder_path, "all_matrices")
        
        if cluster == True:
            all_matrices_clustered = functions.cluster_and_plot(all_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)

    
    elif type == 't1_only':
        t1_matrices, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_matrices, subjects, folder_path, "all_matrices")
        
        if cluster == True:
            t1_t3_matrices_clustered = functions.cluster_and_plot(t1_t3_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
        

    elif type == 't1_t3':
        t1_t3_matrices, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type='t1_t3')
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matrices, subjects, folder_path, "all_matrices")
        
        # plot the significant differences between the matrices
        print("Plotting significant differences...")
        significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t3_matrices, alpha=0.05, cluster=cluster)
        
        if cluster == True:
            t1_t3_matrices_clustered = functions.cluster_and_plot(t1_t3_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t3_matrices_clustered, alpha=0.05, cluster=cluster)

    elif type == 't1_t4':
        t1_t4_matrices, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type='t1_t4')
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matrices, subjects, folder_path, "all_matrices")
        
        # plot the significant differences between the matrices
        print("Plotting significant differences...")
        significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t4_matrices, alpha=0.05, cluster=cluster)
        
        if cluster == True:
            t1_t4_matrices_clustered = functions.cluster_and_plot(t1_t4_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t4_matrices_clustered, alpha=0.05, cluster=cluster)
        
    elif type == 't1_t3_matched':
        t1_t3_matched, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type='t1_t3_matched')
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matched, subjects, folder_path, "all_matrices")
        
        # plot the significant differences between the matrices
        print("Plotting significant differences...")
        significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t3_matched, alpha=0.05, cluster=cluster)
    
        if cluster == True:
            t1_t3_matched_clustered = functions.cluster_and_plot(t1_t3_matched, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t3_matched_clustered, alpha=0.05, cluster=cluster)
    
    elif type == 't1_t4_matched':
        t1_t4_matched, regression_info, rsfMRI_full_info, subjects = functions.load_data(folder_path, rois, type='t1_t4_matched')
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matched, subjects, folder_path, "all_matrices")
        
        # plot the significant differences between the matrices
        print("Plotting significant differences...")
        significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t4_matched, alpha=0.05, cluster=cluster)
        
        if cluster == True:
            t1_t4_matched_clustered = functions.cluster_and_plot(t1_t4_matched, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.sig_matrix_T1_T(t1_t4_matched_clustered, alpha=0.05, cluster=cluster)
    
    else:
        raise ValueError("Invalid type. Choose from 'all', 't1_only', 't1_t3', 't1_t4', 't1_t3_matched', or 't1_t4_matched'.")


if __name__ == "__main__":
    # Run the main function
    main(type='all', cluster=True)