import functions
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main(type = 'all', cluster = False, num_clusters = 2, correction = False, alpha = 0.05, split_L_R = False):
    # Folder containing the data
    folder_path = "FC_matrices_times_wp11/"
    atlas_file_path = "data/HCP-MMP1_RegionsCorticesList_379.csv"

    # keep only ROIS
    rois = [363, 364, 365, 368, 372, 373, 374, 377, 379, 361, 370, 362, 371, 12, 54, 56, 78, 96, 192, 234, 236, 258, 276, 8, 9, 51, 52, 53, 188, 189, 231, 232, 233]
    rois = [roi - 1 for roi in rois]
    selected_rois_labels = [362, 363, 364, 367, 371, 372, 373, 376] 
    roi_mapping = functions.load_roi_labels(atlas_file_path)

    
    # categorical and numerical columns
    categorical_cols = ['Lesion_side', 'Stroke_location','Combined', 'Bilateral']
    numerical_cols = ['lesion_volume_mm3','Gender','Age','Education_level']
    
    regression_info, rsfMRI_full_info = functions.load_excel_data(folder_path)

    if type == 'all':
        # Load the data
        all_matrices, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(all_matrices, subjects, type = type)
        
        # perform longitudinal analysis on tasks
        if split_L_R == True:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
            print(task_results_t4)
        
        if split_L_R == False:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
            print(task_results_t4)
        
        if cluster == False:
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(all_matrices, rois, correction=correction, alpha=alpha, cluster=cluster)

        elif cluster == True:
            all_matrices_clustered = functions.cluster_and_plot(all_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            all_matrices_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                all_matrices, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
            results = functions.get_sig_matrix(all_matrices_clustered_v2, rois, correction=correction, alpha=alpha, cluster=cluster)        

    
    elif type == 't1_only':
        t1_matrices, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_matrices, subjects, type = type, rois=rois)
        
        
        if cluster == True: # not sure if this is important to know ...
            t1_matrices_clustered = functions.cluster_and_plot(t1_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            t1_matrices_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                t1_matrices, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
               

    elif type == 't1_t3':
        t1_t3_matrices, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matrices, subjects, rois=rois, type = type)
        
        # perform longitudinal analysis on tasks
        if split_L_R == True:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
        
        if split_L_R == False:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
        
        if cluster == False:
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t3_matrices, rois, correction=correction, alpha=alpha, cluster=cluster)
        
        if cluster == True:
            t1_t3_matrices_clustered = functions.cluster_and_plot(t1_t3_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            t1_t3_matrices_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                t1_t3_matrices, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            results = functions.get_sig_matrix(t1_t3_matrices_clustered_v2, rois, correction=correction, alpha=alpha, cluster=cluster)


    elif type == 't1_t4':
        t1_t4_matrices, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matrices, subjects, rois=rois, type = type)
        
        # perform longitudinal analysis on tasks
        if split_L_R == True:
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t4)
        
        if split_L_R == False:
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t4)
        
        if cluster == False:
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t4_matrices, rois, correction=correction, alpha=alpha, cluster=cluster)
        
        elif cluster == True:
            t1_t4_matrices_clustered = functions.cluster_and_plot(t1_t3_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            t1_t4_matrices_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                t1_t4_matrices, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            results = functions.get_sig_matrix(t1_t4_matrices_clustered_v2, rois, correction=correction, alpha=alpha, cluster=cluster)
      
        
    elif type == 't1_t3_matched':
        t1_t3_matched, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matched, subjects, rois=rois, type = type)
        
        # perform longitudinal analysis on tasks
        if split_L_R == True:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
            
        if split_L_R == False:
            task_results_t3 = functions.motor_longitudinal(regression_info, tp = 3, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t3)
            
        if cluster == False:
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t3_matched, rois, correction=correction, alpha=alpha, cluster=cluster)
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected,
                significant_matrix,
                roi_mapping,
                alpha=alpha
            )
            
        
        elif cluster == True:
            t1_t3_matched_clustered = functions.cluster_and_plot(t1_t3_matched, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            t1_t3_matched_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                t1_t3_matched, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            results = functions.get_sig_matrix(t1_t3_matched_clustered_v2, rois, correction=correction, alpha=alpha, cluster=cluster)
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix,
                    effect_size_matrix,
                    roi_mapping,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))


    
    elif type == 't1_t4_matched':
        t1_t4_matched, subjects = functions.load_matrices(folder_path, rsfMRI_full_info, rois, type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matched, subjects, rois=rois, type = type)
        
        # perform longitudinal analysis on tasks
        if split_L_R == True:
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t4)
        
        if split_L_R == False:
            task_results_t4 = functions.motor_longitudinal(regression_info, tp = 4, start_col='FAB_abstraction', end_col='nmf_motor', split_L_R = split_L_R)
            print(task_results_t4)
        
        if cluster == False:
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t4_matched, rois, correction=correction, alpha=alpha, cluster=cluster)
        
        elif cluster == True:
            t1_t4_matched_clustered = functions.cluster_and_plot(t1_t3_matched, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
            t1_t4_matched_clustered_v2, clusters, silhouette_scores, pca_features, scaler, pca, all_features, feature_names = functions.cluster_subjects(
                t1_t4_matched, 
                selected_rois_labels, 
                matrix_column='T1_matrix', 
                numerical_cols=numerical_cols, 
                categorical_cols=categorical_cols
            )
            importance_df = functions.compute_feature_importance(all_features, clusters, feature_names)
            
            # plot the significant differences between the matrices
            print("Plotting significant differences...")
            results = functions.get_sig_matrix(t1_t4_matched_clustered_v2, rois, correction=correction, alpha=alpha, cluster=cluster)

    
    else:
        raise ValueError("Invalid type. Choose from 'all', 't1_only', 't1_t3', 't1_t4', 't1_t3_matched', or 't1_t4_matched'.")
    
    return None


if __name__ == "__main__":
    # Run the main function
    main(type='t1_t3', cluster=True, num_clusters=2, split_L_R=True)