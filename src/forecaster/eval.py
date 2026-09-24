from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from epiweeks import Week
import yaml
import argparse
from sklearn import linear_model
import csv
import os 

from forecaster.utils import pickle_load, pickle_save
from forecaster.metrics import rmse, norm_rmse, mape
from forecaster.online_training import get_params


def concat(array1, array2):
    return np.concatenate((array1, array2))


def round3(x):
    return round(x, 3)


def prepare_scores(base_pred, target_region, ahead, oss=False):
    scores = []
    y_preds = []
    y_trues = []
    ahead_idx = ahead - 1
    for i in range(len(base_pred)):
        predictions, addition_infos = base_pred[i]
        y, _, _, _, _ = addition_infos[target_region]
        y_pred = predictions[target_region]
        y_trues.append(y[ahead_idx])
        y_preds.append(y_pred[ahead_idx])
        if oss:
            scores.append(y[ahead_idx] - y_pred[ahead_idx])
        else:
            scores.append(np.abs(y_pred[ahead_idx] - y[ahead_idx]))
        if round3(y[ahead_idx]) == -9:
            scores[-1] = -1e5
    scores = np.array(scores)
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    return scores, y_preds, y_trues


def WriteRows2CSV(rows:list, output_path:Path):
    with open(output_path, 'w+') as f: 
        csv_writer = csv.writer(f)  
        csv_writer.writerows(rows)


def get_mse(cp_params, base_pred_results_all):
    aheads = cp_params['aheads']
    exp_id = cp_params['exp_id']

    results = {}
    regions = cp_params['regions']
    for region in regions:
        results[region] = {}
        for ahead in aheads:
            results[region][ahead] = []

    for _, base_pred_results in base_pred_results_all.items():
        base_pred = base_pred_results['base_pred']
        regions = list(base_pred[0][0].keys())
        for region in tqdm(regions):
            for ahead in aheads:
                scores, y_preds, y_trues = prepare_scores(base_pred, region, ahead)
                mse = np.mean((y_preds - y_trues)**2)**0.5
                results[region][ahead].append(mse)

    avg_results = {}
    for region in regions:
        avg_results[region] = {}
        for ahead in aheads:
            avg_results[region][ahead] = np.mean(results[region][ahead])
    return avg_results


def get_mse_single_file(regions, aheads, base_pred, output_file, verbose=True, plot=False):
    results = {}
    with open(output_file, 'w') as f:  # Open the file for writing
        for region in regions:
            results[region] = {}
            for ahead in aheads:
                results[region][ahead] = None

        base_pred = base_pred['base_pred']
        regions = list(base_pred[0][0].keys())
        f.write(f'Regions: {regions}\n')  # Write regions to the file
        for region in regions:
            for ahead in aheads:
                _, y_preds, y_trues = prepare_scores(base_pred, region, ahead)
                if plot:
                    plt.plot(y_preds, label='y_preds')
                    plt.plot(y_trues, label='y_trues')
                    plt.legend()
                    # plt.show()
                    # plt.savefig('./results.png')
                rmse_val = rmse(y_preds, y_trues)
                nrmse_val = norm_rmse(y_preds, y_trues)
                mape_val = mape(y_preds, y_trues)

                results[region][ahead] = (rmse_val, nrmse_val, mape_val)
                output = f'{region}, {ahead}: rmse: {rmse_val}, nrmse: {nrmse_val}, mape: {mape_val}\n'
                f.write(output)  # Write to the file
                if verbose:
                    print(output.strip())  # Optional: Print to the console for debugging
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    parser.add_argument('--multiple', '-m', action='store_true')
    args = parser.parse_args()

    input_file = args.input
    cp_params = get_params(input_file=input_file)

    try:
        regions = cp_params['regions']
        aheads = cp_params['aheads']
    except:
        regions = ['X']
        aheads = [1, 2, 3, 4]

    output_dir = os.path.join(
        "../../results/base_pred",
        cp_params["model_name"],
        cp_params["data_name"]
    )

    if cp_params['model_variant'] is not None:
        base_pred_file = os.path.join(
            output_dir,
            f"saved_pred_{cp_params['data_id']}_"
            f"model_{cp_params['model_name']}_"
            f"{cp_params['model_variant']}_"
            f"{cp_params['dataset']}_"
            f"seed{cp_params['seed']}.pickle"
        )
    else:
        base_pred_file = os.path.join(
            output_dir,
            f"saved_pred_{cp_params['data_id']}_"
            f"model_{cp_params['model_name']}_"
            f"{cp_params['dataset']}_"
            f"seed{cp_params['seed']}.pickle"
        )

    if args.multiple:
        base_pred_results_all = pickle_load(base_pred_file, version5=True)
        _ = get_mse(cp_params, base_pred_results_all)

    else:
        base_pred = pickle_load(base_pred_file, version5=True)
        print(len(base_pred['base_pred']))

        output_file = os.path.join(
            output_dir,
            f"results_{cp_params['data_id']}_"
            f"model_{cp_params['model_name']}_"
            f"{cp_params['dataset']}_"
            f"seed{cp_params['seed']}.text"
        )

        print(base_pred_file)
        _ = get_mse_single_file(
            regions,
            aheads,
            base_pred,
            output_file
        )


if __name__ == '__main__':
    main()