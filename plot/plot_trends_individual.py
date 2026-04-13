
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import to_rgb

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os
import argparse
import itertools

# If plot font needs to be Times New Roman
# matplotlib.rcParams['font.family'] = 'Times New Roman'


def rescale_acc(raw_acc):
    diags = np.diagonal(raw_acc)
    return (raw_acc - diags) * 100 / diags


def load_distance_results(distance_type, spatial_splits, is_satclip=False, model_name="resnet50", resolution=None):

    assert distance_type in ['cosine', 'ot', 'arc'], "Options for distance_type: ['cosine', 'ot', 'arc']"
    assert is_satclip in [True, False], "is_satclip must be a boolean variable"
    assert model_name in ['resnet50'], "Options for distance models: ['resnet50']"
    if is_satclip:
        assert resolution in [10, 40], "Options for resolution: [10, 40] for is_satclip=True"

    num_domains_dict = {'continent': 5,
                        'country_top40': 40,
                        'checkerboard_10': 10}
    num_domains = num_domains_dict[spatial_splits]
    domain_pair_indices = list(itertools.product(range(num_domains), range(num_domains)))

    ROOT_DIR = os.path.join("..", "..", "00_results")
    if distance_type == 'arc':
        DISTANCE_DIR = os.path.join(ROOT_DIR, "arc_distances", f"{spatial_splits}")
    else:
        embedding_dir_name = "satclip_embedding_results" if is_satclip else "img_embedding_results"
        distance_dir_name = "ot_distances" if distance_type == 'ot' else "cosine_distances"
        model_dir_name = f"{model_name}_l{resolution}" if is_satclip else f"{model_name}"
        DISTANCE_DIR = os.path.join(ROOT_DIR, embedding_dir_name, distance_dir_name, f"{model_dir_name}", f"{spatial_splits}")

    SUMMARY_DIR = os.path.join(DISTANCE_DIR, "summary_raw")

    if distance_type == 'arc':
        distance_filename = f"{distance_type}_distance"
        summary_id_filename = "avg"
        
        summary_filename_base = f"arc_distance_{spatial_splits}_{summary_id_filename}_summary"
        summary_filepath = os.path.join(SUMMARY_DIR, f"{summary_filename_base}.npy")
    else:
        distance_filename = f"{distance_type}_distance"
        model_filename = f"{model_name}_l{resolution}" if is_satclip else f"{model_name}"
        summary_id_filename = "avg" if distance_type == 'cosine' else f"eps0.01_linearOT"

        summary_filename_base = f"{distance_filename}_{model_filename}_{spatial_splits}_{summary_id_filename}_summary"
        summary_filepath = os.path.join(SUMMARY_DIR, f"{summary_filename_base}.npy")
    
    dist_type_id = 'Cosine Dist' if distance_type == 'cosine' else 'OT Dist' if distance_type == 'ot' else 'Arc Dist'
    dist_type = f"{dist_type_id}" if distance_type == 'arc' else \
                    f"{dist_type_id} - {model_name.capitalize()} (L={resolution})" if is_satclip else \
                    f"{dist_type_id} - {model_name.capitalize()}"
    dist_filename_str = f"{distance_type}_dist" if distance_type == 'arc' else \
                            f"{distance_type}_dist_{model_name}_l{resolution}" if is_satclip else \
                            f"{distance_type}_dist_{model_name}"
    color_map = {('cosine', False): 'green',
                 ('cosine', True): 'springgreen',
                 ('ot', False): 'cornflowerblue',
                 ('ot', True): 'turquoise',
                 ('arc', False): 'orange'}
    distance_summary = np.load(summary_filepath)
    distance_df = pd.DataFrame([{'src_domain_idx': src_domain_idx,
                                 'tgt_domain_idx': tgt_domain_idx,
                                 'dist_value': distance_summary[src_domain_idx, tgt_domain_idx],
                                 'dist_type': dist_type,
                                 'is_self_pair': src_domain_idx == tgt_domain_idx,
                                 'dist_filename_str': dist_filename_str,
                                 'color': color_map[(distance_type, is_satclip)]}
                                 for src_domain_idx, tgt_domain_idx in domain_pair_indices])
    return distance_df

def load_domain_adaptation_results(finetune_size, model_name, spatial_splits, rescale_acc_flag):

    assert finetune_size in [0, 5, 10], "Options for finetune settings: [0, 5, 10]"
    assert model_name in ['resnet50', 'densenet121'], "Options for domain adaptation models: ['resnet50', 'densenet121']"

    num_domains_dict = {'continent': 5,
                        'country_top40': 40,
                        'checkerboard_10': 10}
    num_domains = num_domains_dict[spatial_splits]
    domain_pair_indices = list(itertools.product(range(num_domains), range(num_domains)))

    task_name = "zeroshot_eval" if finetune_size == 0 else f"{finetune_size}shot_eval"
    finetune_info = f"ft{finetune_size}shot"
    acc_name_in_file = f"top1acc"
    agg_func_name = "avg"

    ROOT_DIR = os.path.join("..", "..", "00_results", "domain_adaptation_results", "test_results")
    SUMMARY_DIR = os.path.join(ROOT_DIR, f"{task_name}", f"{spatial_splits}", f"{model_name}")

    summary_filename_base = f"{task_name}_{spatial_splits}_{model_name}_{acc_name_in_file}_{agg_func_name}_summary" if finetune_size == 0 else \
                            f"{task_name}_{spatial_splits}_{model_name}_{finetune_info}_{acc_name_in_file}_{agg_func_name}_summary"
    summary_filepath = os.path.join(SUMMARY_DIR, f"{summary_filename_base}.npy")
    domain_adaptation_summary = np.load(summary_filepath)
    if rescale_acc_flag:
        domain_adaptation_summary = rescale_acc(domain_adaptation_summary)

    transfer_setting = f"{finetune_size}-shot {model_name.capitalize()}"
    domain_adaptation_df = pd.DataFrame([{'src_domain_idx': src_domain_idx,
                                          'tgt_domain_idx': tgt_domain_idx,
                                          'acc_value': domain_adaptation_summary[src_domain_idx, tgt_domain_idx],
                                          'transfer_setting': transfer_setting,
                                          'adaptation_filename_str': f"{finetune_size}shot_{model_name}"}
                                          for src_domain_idx, tgt_domain_idx in domain_pair_indices])
    return domain_adaptation_df

def plot_trends(df, filename, title_name,
                x_title, y_title,
                figsize=(10, 10)):

    scatter_x = 'dist_value'
    scatter_y = 'acc_value'
    color = df['color'].unique()[0]
    scatter_color = tuple(0.7 * c for c in to_rgb(color))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    rho, p_value = spearmanr(df[scatter_x], df[scatter_y])
    linear_reg = LinearRegression().fit(df[[scatter_x]], df[scatter_y])
    y_pred = linear_reg.predict(df[[scatter_x]])
    r2 = r2_score(df[scatter_y], y_pred)

    label_text = r"$\rho$" + f": {rho:.4f},\n" + r"$p$" + f"-value: {p_value:.4f},\n" + \
                    r"$\mathcal{R}^2$" + f": {r2:.4f}"

    alpha_value = 0.3 if len(df) > 50 else 0.6
    ax.scatter(df[scatter_x], df[scatter_y], alpha=alpha_value, color=scatter_color)
    sns.regplot(data=df, x=scatter_x, y=scatter_y, ci=None,
                scatter=False, fit_reg=True, ax=ax, label=label_text, color=color)

    ax.tick_params('both', labelsize=14)
    ax.set_xlabel(f"{x_title}", fontsize=16)
    ax.set_ylabel(f"{y_title}", fontsize=16)
    ax.legend(fontsize=16)

    ax.set_title(title_name, fontsize=14)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()



def main(spatial_splits, include_self_pair, rescale_acc_flag, mask_outlier_domains):

    PLOT_DIR = os.path.join(".", f"trend_plots_by_dist_type_{spatial_splits}")
    for dir_path in [PLOT_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    distance_df = pd.concat([load_distance_results('cosine', spatial_splits, is_satclip=False, model_name='resnet50'),
                             load_distance_results('cosine', spatial_splits, is_satclip=True, model_name='resnet50', resolution=10),
                             load_distance_results('cosine', spatial_splits, is_satclip=True, model_name='resnet50', resolution=40),
                             load_distance_results('ot', spatial_splits, is_satclip=False, model_name='resnet50'),
                             load_distance_results('ot', spatial_splits, is_satclip=True, model_name='resnet50', resolution=10),
                             load_distance_results('ot', spatial_splits, is_satclip=True, model_name='resnet50', resolution=40),
                             load_distance_results('arc', spatial_splits)])

    finetune_size_list = [0, 5, 10] if spatial_splits == 'continent' else [0, 5]
    domain_adaptation_df = pd.concat([load_domain_adaptation_results(finetune_size, model_name, spatial_splits, rescale_acc_flag)
                                        for finetune_size in finetune_size_list for model_name in ['resnet50', 'densenet121']])

    combined_df = pd.merge(distance_df, domain_adaptation_df, how='inner', on=['src_domain_idx', 'tgt_domain_idx'])
    if not include_self_pair:
        combined_df = combined_df[~combined_df['is_self_pair']]

    exclude_domain_idx_list = []
    if mask_outlier_domains:
        if spatial_splits == 'continent':
            exclude_domain_idx_list = [4]
        elif spatial_splits == 'country_top40':
            return
        elif spatial_splits == 'checkerboard_10':
            exclude_domain_idx_list = [5, 6]
        for exclude_domain_idx in exclude_domain_idx_list:
            combined_df = combined_df[(combined_df['src_domain_idx'] != exclude_domain_idx) & 
                                      (combined_df['tgt_domain_idx'] != exclude_domain_idx)]

    for dist_type in combined_df['dist_type'].unique():
        for transfer_setting in combined_df['transfer_setting'].unique():
            df_subset = combined_df[(combined_df['dist_type'] == dist_type) & (combined_df['transfer_setting'] == transfer_setting)]

            title_name = f"Domain Adaptation Performance vs. Domain Distances Trend Plot\n" + \
                            f"Spatial Splits: {spatial_splits.capitalize()},\n" + \
                            f"Domain Adaptaion Setting: {transfer_setting},\n" + \
                            f"Distance Type: {dist_type}"

            if mask_outlier_domains:
                title_name += f"\nExclude outlier domains: {', '.join([str(i) for i in exclude_domain_idx_list])}"
            
            x_title = "Domain Distance"
            y_title = "Relative Change in Test Accuracy (%)" if rescale_acc_flag else "Domain Adaptaion Test Accuracy"

            dist_type_str = df_subset['dist_filename_str'].unique()[0]
            domain_adaptation_str = df_subset['adaptation_filename_str'].unique()[0]
            include_self_pair_str = "inc-self" if include_self_pair else "exc-self"
            rescale_acc_str = "rescale_acc" if rescale_acc_flag else "raw_acc"
            mask_domain_str = f"_mask{','.join([str(i) for i in exclude_domain_idx_list])}" if mask_outlier_domains else ""
            plot_filename = f"trend_plots_{spatial_splits}_{dist_type_str}_{domain_adaptation_str}_{include_self_pair_str}_{rescale_acc_str}{mask_domain_str}.png"
            plot_filepath = os.path.join(PLOT_DIR, plot_filename)

            plot_trends(df_subset, plot_filepath, title_name, x_title, y_title)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--spatial_splits', type=str, required=True, choices=['continent', 'country_top40', 'checkerboard_10'], help='Spatial splits for domain definition')
    # parser.add_argument('--include_self_pair', type=bool, default=True, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--rescale_acc_flag', type=bool, default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    for include_self_pair in [False]:
        for rescale_acc_flag in [True]:
            for mask_domains_with_few_data in [False, True]:
                main(args.spatial_splits, include_self_pair, rescale_acc_flag, mask_domains_with_few_data)

