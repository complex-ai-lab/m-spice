import numpy as np
import pandas as pd
from utils import pickle_load, pickle_save
import torch

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

def make_forecast_df(base_pred_results, regions=None, aheads=[1,2,3,4], seed=None):
    rows = []

    base_pred = base_pred_results["base_pred"]

    if regions is None:
        regions = list(base_pred[0][0].keys())

    for region in regions:
        for ahead in aheads:
            scores, y_preds, y_trues = prepare_scores(
                base_pred,
                region,
                ahead
            )

            for t_idx, (score, pred, true) in enumerate(zip(scores, y_preds, y_trues)):
                if score == -1e5:
                    continue

                rows.append({
                    "region": region,
                    "seed": seed,
                    "ahead": ahead,
                    "t_idx": t_idx,
                    "y_pred": float(pred),
                    "y_true": float(true),
                    "score": float(abs(true - pred)),
                })

    return pd.DataFrame(rows)

def run_simple_aci(
    forecast_df,
    alpha=0.1,
    lr=0.008,
    burnin=5,
):
    rows = []

    group_cols = ["region", "ahead", "seed"]

    for (region, ahead, seed), g in forecast_df.groupby(group_cols):
        g = g.sort_values("t_idx").reset_index(drop=True).copy()

        q = 0.0

        for i, row in g.iterrows():
            y_pred = row["y_pred"]
            y_true = row["y_true"]

            lower = y_pred - q
            upper = y_pred + q

            covered = int(lower <= y_true <= upper)
            score = abs(y_true - y_pred)

            # only evaluate after burn-in
            eval_flag = i >= burnin

            rows.append({
                "region": region,
                "ahead": ahead,
                "seed": seed,
                "t_idx": row["t_idx"],
                "y_pred": y_pred,
                "y_true": y_true,
                "score": score,
                "q": q,
                "lower90": lower,
                "upper90": upper,
                "covered": covered,
                "eval": eval_flag,
            })

            # update q using current observed coverage
            q = q + lr * (alpha - covered)
            q = max(q, 0.0)

    return pd.DataFrame(rows)

def run_aci_lab_style(
    forecast_df,
    alpha=0.1,
    lr=0.008,
    burnin=5,
    window=20,
):
    rows = []
    group_cols = ["region", "ahead", "seed"]

    for (region, ahead, seed), g in forecast_df.groupby(group_cols, dropna=False):
        g = g.sort_values("t_idx").reset_index(drop=True).copy()

        alpha_t = alpha
        past_scores = []

        for i, row in g.iterrows():
            y_pred = row["y_pred"]
            y_true = row["y_true"]

            if len(past_scores) < burnin:
                q = np.nan
                lower = np.nan
                upper = np.nan
                covered = np.nan
                eval_flag = False

            else:
                recent_scores = np.array(past_scores[-window:])

                # ACI interval radius from past residuals
                q = np.quantile(recent_scores, 1 - alpha_t)

                # q = np.quantile(past_scores, 0.9)
                q = max(q, 0)

                lower = y_pred - q
                upper = y_pred + q

                covered = int(lower <= y_true <= upper)
                eval_flag = True

                # ACI update
                err_t = 1 - covered
                alpha_t = alpha_t + lr * (alpha - err_t)

                # keep alpha_t valid
                alpha_t = min(max(alpha_t, 1e-4), 0.999)

            score = abs(y_true - y_pred)

            rows.append({
                "region": region,
                "ahead": ahead,
                "seed": seed,
                "t_idx": row["t_idx"],
                "y_pred": y_pred,
                "y_true": y_true,
                "score": score,
                "q": q,
                "alpha_t": alpha_t,
                "lower90": lower,
                "upper90": upper,
                "covered": covered,
                "eval": eval_flag,
            })

            # add current score only AFTER interval/evaluation
            past_scores.append(score)

    return pd.DataFrame(rows)

def interval_score(y, lower, upper, alpha):
    score = upper - lower

    if y < lower:
        score += (2 / alpha) * (lower - y)
    elif y > upper:
        score += (2 / alpha) * (y - upper)

    return score


def weighted_interval_score(y, median, intervals):
    """
    intervals = [(alpha, lower, upper)]
    For 90% interval: alpha = 0.1
    """
    K = len(intervals)

    wis = 0.5 * abs(y - median)

    for alpha, lower, upper in intervals:
        wis += (alpha / 2) * interval_score(y, lower, upper, alpha)

    wis /= (K + 0.5)

    return wis

def compute_wis_from_aci_df(aci_df):
    results = {}

    eval_df = aci_df[aci_df["eval"]].copy()

    for ahead, g in eval_df.groupby("ahead"):
        wis_vals = []

        for _, row in g.iterrows():
            intervals = [
                (0.1, row["lower90"], row["upper90"])
            ]

            wis_vals.append(
                weighted_interval_score(
                    y=row["y_true"],
                    median=row["y_pred"],
                    intervals=intervals
                )
            )

        results[ahead] = np.mean(wis_vals)
        print(f"W{ahead}: WIS = {results[ahead]:.4f}")

    overall_wis = np.mean(list(results.values()))
    print(f"\nOverall WIS = {overall_wis:.4f}")

    return results, overall_wis
    
def get_wis_values_from_aci_df(aci_df, alpha=0.1):
    eval_df = aci_df[aci_df["eval"]].copy()

    rows = []

    for _, row in eval_df.iterrows():

        wis = weighted_interval_score(
            y=row["y_true"],
            median=row["y_pred"],
            intervals=[
                (alpha, row["lower90"], row["upper90"])
            ]
        )

        rows.append({
            "region": row["region"],
            "ahead": row["ahead"],
            "seed": row["seed"],
            "t_idx": row["t_idx"],
            "wis": wis,
        })

    return pd.DataFrame(rows)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Built with CUDA:", torch.backends.cuda.is_built())

base_pred_file = '../../results/base_pred/ts_img_jointattn/covid_all/saved_pred_3_model_ts_img_jointattn_best_model_00_11_ts_pe_more_epochs_multimodal_covid_seed5.pickle'
base_pred = pickle_load(base_pred_file, version5=True)
seed_files = {
    5: "../../results/base_pred/ts_img_jointattn/percent_ili_all/saved_pred_3_model_ts_img_jointattn_best_model_00_11_ts_pe_more_epochs_multimodal_covid_seed5.pickle",
    17: "../../results/base_pred/ts_img_jointattn/percent_ili_all/saved_pred_3_model_ts_img_jointattn_best_model_00_11_ts_pe_more_epochs_multimodal_covid_seed17.pickle",
    33: "../../results/base_pred/ts_img_jointattn/percent_ili_all/saved_pred_3_model_ts_img_jointattn_best_model_00_11_ts_pe_more_epochs_multimodal_covid_seed33.pickle",
}

forecast_dfs = []

for seed, fname in seed_files.items():

    base_pred = pickle_load(fname, version5=True)
    # with open(fname, "rb") as f:
    #     obj = pickle.load(f)

    # print(type(obj))

    cur_df = make_forecast_df(
        base_pred_results=base_pred,
        seed=seed
    )

    forecast_dfs.append(cur_df)

forecast_df = pd.concat(
    forecast_dfs,
    ignore_index=True
)

print(forecast_df.shape)
print(forecast_df["seed"].value_counts())

# forecast_df = make_forecast_df(base_pred, seed=5)

forecast_df = forecast_df[
    (forecast_df["y_true"].round(3) != -9) &
    (forecast_df["y_pred"].round(3) != -9)
].copy()

print(forecast_df)
print(forecast_df.groupby("ahead")["score"].describe())

aci_df = run_aci_lab_style(
    forecast_df,
    alpha=0.1,
    lr=0.008,
    burnin=5,
    window=20
)
print(aci_df["alpha_t"].describe())
print(aci_df["q"].describe())
# Compute Cov90:
eval_df = aci_df[aci_df["eval"]]
print(eval_df["q"].describe())
print(
    eval_df.groupby("ahead")["q"].describe()
)
for ahead in [1,2,3,4]:
    g = eval_df[eval_df["ahead"] == ahead]

    empirical_cov = np.mean(g["score"] <= g["q"])

    print(
        f"W{ahead}:",
        "stored =", g["covered"].mean(),
        "recomputed =", empirical_cov
    )

print('Mean: ', 
    np.mean(
        eval_df["score"] <= eval_df["q"]
    )
)
for ahead in [1,2,3,4]:
    g = eval_df[eval_df["ahead"] == ahead]

    print(
        f"W{ahead}",
        "score q90:", np.quantile(g["score"], 0.9),
        "q q90:", np.quantile(g["q"], 0.9),
    )
cov90_by_horizon = eval_df.groupby("ahead")["covered"].mean()
overall_cov90 = eval_df["covered"].mean()

print(cov90_by_horizon)
print("Overall Cov90:", overall_cov90)

# Compute WIS:
model_wis_by_horizon, model_overall_wis = compute_wis_from_aci_df(aci_df)
print("Model WIS by horizon: ", model_wis_by_horizon)
print("Model Overall WIS: ", model_overall_wis)
# rwis = model_overall_wis / persistence_overall_wis
# print("rWIS:", rwis)

# Individual model WIS values
model_wis_df = get_wis_values_from_aci_df(aci_df)

persist_wis_df = pd.read_csv("covid_persistence_wis_df.csv")
persist_wis_df_filtered = persist_wis_df.copy()

persist_wis_df_filtered = persist_wis_df_filtered[
    persist_wis_df_filtered["region"] != "DC"
].copy()

persist_wis_df_filtered = persist_wis_df_filtered[
    persist_wis_df_filtered["t_idx"].between(5, 23)
].copy()

print("model keys:", model_wis_df[["region","ahead","t_idx"]].drop_duplicates().shape)
print("persist keys:", persist_wis_df_filtered[["region","ahead","t_idx"]].drop_duplicates().shape)

merged = model_wis_df.merge(
    persist_wis_df_filtered,
    on=["region", "ahead", "t_idx"],
    suffixes=("_model", "_persist")
)

print("merged:", merged.shape)

# Drop rows where persistence WIS is zero to avoid division/aggregation issues
# merged = merged[merged["wis_persist"] > 0].copy()

# Stable rWIS: ratio of aggregate WIS over matched forecasts
rwis_ratio_of_means = (
    merged["wis_model"].sum()
    / merged["wis_persist"].sum()
)

rwis_by_horizon = (
    merged.groupby("ahead")["wis_model"].sum()
    / merged.groupby("ahead")["wis_persist"].sum()
)

print("rWIS by horizon:")
print(rwis_by_horizon)

print("Overall rWIS ratio-of-means:")
print(rwis_ratio_of_means)

