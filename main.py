import functions
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main(rois, request_type = 'all', cluster = False, num_clusters = 2, correction = False, alpha = 0.05, split_L_R = False, plot = True, subset = False):
    # Folder containing the data
    mat_folder_path = "FC_matrices_times_wp11/"
    excel_folder_path = "data/"
    atlas_file_path = "data/HCP-MMP1_RegionsCorticesList_379.csv"

    # ROIS
    selected_rois_labels = [362, 363, 364, 367, 371, 372, 373, 376] 
    roi_mapping_glasser = functions.load_roi_labels(atlas_file_path)
    print(roi_mapping_glasser)

    
    # categorical and numerical columns
    categorical_cols = ['Lesion_side', 'Stroke_location','Combined', 'Bilateral']
    numerical_cols = ['lesion_volume_mm3','Gender','Age','Education_level']
    excel_folder_path = "data/"
    
    FM_folder_path = "data/Raw_MissingDataImputed/"
    '''regression_info, rsfMRI_full_info = functions.load_excel_data(excel_folder_path, FM_folder_path)
    folder_path = "FC_matrices_times_wp11/"

    rois = [363, 364, 365, 368, 372, 373, 374, 377, 379, 361, 370, 362, 371, 12, 54, 56, 78, 96, 192, 234, 236, 258, 276, 8, 9, 51, 52, 53, 188, 189, 231, 232, 233]
    rois = [roi - 1 for roi in rois]
    rois_sub = rois
    rois_full = np.arange(0, 379)
    roi_mapping = functions.load_roi_labels("data/HCP-MMP1_RegionsCorticesList_379.csv")'''
    regression_info, rsfMRI_full_info = functions.load_excel_data(excel_folder_path, FM_folder_path)

    if request_type == 'all':
        # Load the data
        all_matrices, subjects, yeo_all_rois, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(all_matrices, subjects, request_type = request_type)
        
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
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(all_matrices, correction=correction, alpha=alpha, cluster=cluster) #for tp = 3
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected.values,
                significant_matrix,
                roi_mapping_glasser,
                alpha=alpha
            )
            print(f"Top significant connections for request_type {request_type}:")
            print(summary.head(10))

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
            results = functions.get_sig_matrix(all_matrices_clustered_v2, correction=correction, alpha=alpha, cluster=cluster) #for tp = 3
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))

    
    elif request_type == 't1_only':
        t1_matrices, subjects, yeo_rois_t1, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_matrices, subjects, request_type = request_type, rois=rois)
        
        
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
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))
               

    elif request_type == 't1_t3':
        print("doing load matrices")
        t1_t3_matrices, subjects, yeo_rois_t1_t3, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matrices, subjects, rois=rois, request_type = request_type)
        
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
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t3_matrices, correction=correction, alpha=alpha, cluster=cluster)
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected.values,
                significant_matrix,
                roi_mapping_glasser,
                alpha=alpha
            )
            
            print(f"Top significant connections (Glasser atlas) for request_type {request_type}:")
            print(summary.head(10))
            
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(yeo_rois_t1_t3, correction=False, alpha=0.05, cluster=False)

            summary = functions.summarize_significant_differences(
                            p_vals_corrected.values,
                            significant_matrix,
                            roi_mapping_yeo,
                            alpha=0.05
                        )
            
            print(f"Top significant connections (Yeo atlas) for request_type {request_type}:")
            print(summary.head(10))
        
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
            results = functions.get_sig_matrix(t1_t3_matrices_clustered_v2, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections (Glasser atlas) for cluster {clust}:")
                print(summary.head(10))


    elif request_type == 't1_t4':
        t1_t4_matrices, subjects, yeo_rois_t1_t4, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matrices, subjects, rois=rois, request_type = request_type)
        
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
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t4_matrices, tp=4, correction=correction, alpha=alpha, cluster=cluster)
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected.values,
                significant_matrix,
                roi_mapping_glasser,
                alpha=alpha
            )
            
            print(f"Top significant connections for request_type {request_type}:")
            print(summary.head(10))
            
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(yeo_rois_t1_t4, tp=4, correction=False, alpha=0.05, cluster=False)

            summary = functions.summarize_significant_differences(
                            p_vals_corrected.values,
                            significant_matrix,
                            roi_mapping_yeo,
                            alpha=0.05
                        )
            
            print(f"Top significant connections (Yeo atlas) for request_type {request_type}:")
            print(summary.head(10))
        
        elif cluster == True:
            t1_t4_matrices_clustered = functions.cluster_and_plot(t1_t4_matrices, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
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
            results = functions.get_sig_matrix(t1_t4_matrices_clustered_v2, tp=4, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))
      
        
    elif request_type == 't1_t3_matched':
        t1_t3_matched, subjects, yeo_rois_t1_t3, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t3_matched, subjects, rois=rois, request_type = request_type)
        
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
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t3_matched, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected.values,
                significant_matrix,
                roi_mapping_glasser,
                alpha=alpha
            )
            
            print(f"Top significant connections for request_type {request_type}:")
            print(summary.head(10)) 
            
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(yeo_rois_t1_t3, correction=False, alpha=0.05, cluster=False, matched=True)

            summary = functions.summarize_significant_differences(
                            p_vals_corrected.values,
                            significant_matrix,
                            roi_mapping_yeo,
                            alpha=0.05
                        )
            
            print(f"Top significant connections (Yeo atlas) for request_type {request_type}:")
            print(summary.head(10))
        
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
            results = functions.get_sig_matrix(t1_t3_matched_clustered_v2, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))


    
    elif request_type == 't1_t4_matched':
        t1_t4_matched, subjects, yeo_rois_t1_t4, roi_mapping_yeo = functions.load_matrices(mat_folder_path, rsfMRI_full_info, rois, request_type)
        
        # plot the heatmaps of the FC matrices
        print("Plotting all matrices...")
        functions.plot_all_subject_matrices(t1_t4_matched, subjects, rois=rois, request_type = request_type)
        
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
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(t1_t4_matched, tp=4, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            summary = functions.summarize_significant_differences(
                p_vals_corrected.values,
                significant_matrix,
                roi_mapping_glasser,
                alpha=alpha
            )
            
            print(f"Top significant connections for request_type {request_type}:")
            print(summary.head(10))
            
            significant_matrix, p_vals_corrected, reject = functions.get_sig_matrix(yeo_rois_t1_t4, tp=4, correction=False, alpha=0.05, cluster=False, matched=True)

            summary = functions.summarize_significant_differences(
                            p_vals_corrected.values,
                            significant_matrix,
                            roi_mapping_yeo,
                            alpha=0.05
                        )
            
            print(f"Top significant connections (Yeo atlas) for request_type {request_type}:")
            print(summary.head(10))
        
        elif cluster == True:
            t1_t4_matched_clustered = functions.cluster_and_plot(t1_t4_matched, numerical_cols_names= numerical_cols, categorical_cols_name=categorical_cols, clusters=num_clusters)
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
            results = functions.get_sig_matrix(t1_t4_matched_clustered_v2, tp=4, correction=correction, alpha=alpha, cluster=cluster, matched=True)
            
            for clust in results.keys():
                p_values_matrix = results[clust]['p_corrected']
                effect_size_matrix = results[clust]['significant_matrix']  # Attention: ici il faut être sûr que c'est bien l'effect size

                summary = functions.summarize_significant_differences(
                    p_values_matrix.values,
                    effect_size_matrix,
                    roi_mapping_glasser,
                    cluster_label=clust
                )

                print(f"Top significant connections for cluster {clust}:")
                print(summary.head(10))


    
    else:
        raise ValueError("Invalid request_type. Choose from 'all', 't1_only', 't1_t3', 't1_t4', 't1_t3_matched', or 't1_t4_matched'.")
    
    return None


if __name__ == "__main__":
    # Run the main function
    rois = [363, 364, 365, 368, 372, 373, 374, 377, 379, 361, 370, 362, 371, 12, 54, 56, 78, 96, 192, 234, 236, 258, 276, 8, 9, 51, 52, 53, 188, 189, 231, 232, 233]
    rois = [roi - 1 for roi in rois]
    main(rois, request_type='t1_t4', cluster=True, num_clusters=2, split_L_R=False, correction=True)