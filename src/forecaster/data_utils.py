import numpy as np
import torch
from epiweeks import Week
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from PIL import Image

fips_to_region = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '12': 'FL', '13': 'GA',
    '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
    '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO',
    '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ',
    '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC',
    '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT',
    '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY',
    '11': 'DC', '72': 'PR', 'US': 'US'
}

def region_from_fips(fips_code: str) -> str:
    """
    Look up the state/region abbreviation for a given FIPS code.
    Falls back to the input if the code isn’t in the map.
    """
    return fips_to_region.get(fips_code, fips_code)

def convert_to_epiweek(x):
    """ convert string of the format YYYYww to epiweek object """
    return Week.fromstring(str(x))


def epiweek_sub(week1, week2, max_attempts=500):
    # assumes week1 is no less than week2, week1 - week2
    for i in range(max_attempts):
        if week2 + i == week1:
            return i
    return -1

def subtract_epiweek(epiweek_str, lag=3):
    # epiweek_str like "202240"
    w = Week.fromstring(epiweek_str)
    w_lag = w - lag   # subtract 3 weeks
    return f"{w_lag.year}{w_lag.week:02d}"

def lag_record_id(rec_id, lag=3):
    epiweek_str, fips = rec_id.split("_")
    epi_lag = subtract_epiweek(epiweek_str, lag)
    return f"{epi_lag}_{fips}"
   
def shift_np_array(data, offset=10):
    """Shift an array along second axis and pad with zeros."""
    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        new_data[:, i] = shift(data[:, i], offset, cval=0)
    return new_data


def load_ground_truth_before_test(data_file, test_week, region, length_before_test=10):
    test_week = convert_to_epiweek(test_week)
    df = pd.read_csv(data_file, low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)

    df = df[(df["epiweek"] <= test_week) & (df["epiweek"] >= test_week - length_before_test + 1)]
    df = df.ffill()
    df = df.bfill()

    df = df.fillna(0)
    target = df.loc[:, ['flu_hospitalizations']].values

    return target

def load_df(params, region, start_week_data, pred_week, smooth=False):
    """ load data and subset to desired region and epiweeks"""

    # load and clean data
    data_file = os.path.join(params['input_files']['parent_dir'],
                             params['input_files']['weekly_data'])
    df = pd.read_csv(data_file, low_memory=False)
    df = df.ffill()
    df = df.bfill()

    df = df.fillna(0)
    
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    df = df[(df["epiweek"] <= pred_week) & (df["epiweek"] >= start_week_data)]

    # smooth all features
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode='full')[:-w + 1]

    # if target variable not in features, add it
    if params['target'] not in params['data_features']:
        params['data_features'].append(params['target'])

    # smooth all features
    if smooth:
        # add not smoothed true
        df['test_gt'] = df[params['target']].copy()
        for feature in params['data_features']:
            df.loc[:, feature] = moving_average(df.loc[:, feature].values, params['data_params']['smooth_window'])
        return df[params['data_features']+['test_gt']]
    
    df["weekid"] = df["epiweek"].apply(lambda x: x.cdcformat())
    
    # return subset of data
    return df[params['data_features']+['weekid']]

def load_df_stl(params, region, start_week_data, pred_week, smooth=False):
    """ load data and subset to desired region and epiweeks"""

    # load and clean data
    data_file = os.path.join(params['input_files']['parent_dir'],
                             params['input_files']['weekly_data'])
    df = pd.read_csv(data_file, low_memory=False)

    # subset data using init parameters
    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)
    
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    df = df[(df["epiweek"] <= pred_week) & (df["epiweek"] >= start_week_data)]

    # smooth all features
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode='full')[:-w + 1]

    # if target variable not in features, add it
    if params['target'] not in params['data_features']:
        params['data_features'].append(params['target'])

    # smooth all features
    if smooth:
        # add not smoothed true
        df['test_gt'] = df[params['target']].copy()
        for feature in params['data_features']:
            df.loc[:, feature] = moving_average(df.loc[:, feature].values, params['data_params']['smooth_window'])
        return df[params['data_features']+['test_gt']]
    
    df["weekid"] = df["epiweek"].apply(lambda x: x.cdcformat())
    
    # return subset of data
    return df[params['data_features']+['weekid']], df['seasonal'].to_numpy().reshape(-1, 1), df['trend'].to_numpy().reshape(-1, 1), df['resid'].to_numpy().reshape(-1, 1)

def get_state_train_data_stl(params, region, smooth=False):
    """ get processed dataframe of data + target as array """

    # convert to epiweeks
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    last_train_week = convert_to_epiweek(params['last_train_time'])

    # load data
    df, s, t, r = load_df_stl(params, region, start_week, last_train_week, smooth)

    # select target
    target = df.loc[:, [params['target']]].values

    return df[params['data_features']+['weekid']], target, s, t, r

def get_state_test_data_stl(params, region, pred_week, smooth=False):
    """ get processed dataframe of data + target as array"""
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    pred_week = convert_to_epiweek(pred_week)
    weeks_ahead = params['weeks_ahead']

    # import smoothed dataframe
    df, s, t, r = load_df_stl(params, region, start_week, pred_week + weeks_ahead, smooth).tail(weeks_ahead)

    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values

    return df[params['data_features']+['weekid']], target, s, t, r

def get_state_test_data_xy_stl(params, region, pred_week, x_length, smooth=False, remove_weeks_after_test=True):
    weeks_ahead = params['weeks_ahead']
    start_week = convert_to_epiweek(params['data_params']['start_time'])

    # y
    test_week = convert_to_epiweek(params['test_time'])
    avail_weeks = epiweek_sub(test_week, pred_week)
    clipped_weeks_ahead = min(avail_weeks, weeks_ahead) if remove_weeks_after_test else weeks_ahead
    df, s, t, r = load_df_stl(params, region, start_week, pred_week + clipped_weeks_ahead, smooth) #.tail(clipped_weeks_ahead)
    df = df.tail(clipped_weeks_ahead)
    s = s[-clipped_weeks_ahead:]
    t = t[-clipped_weeks_ahead:]
    r = r[-clipped_weeks_ahead:]

    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values

    def pad_col(v, W, pad_value=-9.0):
        v = np.asarray(v, dtype=np.float32).reshape(-1, 1)
        out = np.full((W, 1), pad_value, dtype=np.float32)
        L = min(v.shape[0], W)
        out[:L, 0] = v[:L, 0]
        return out

    target = pad_col(target, weeks_ahead)   # sum/original
    s = pad_col(s, weeks_ahead)
    t = pad_col(t, weeks_ahead)
    r = pad_col(r, weeks_ahead)
    
    # x
    df, _, _, _= load_df_stl(params, region, start_week, pred_week, smooth)
    df = df.tail(x_length)

    return df[params['data_features']+['weekid']], target, s, t, r

def load_df_multimodal(params, region, start_week_data, pred_week, smooth=False):
    """ load data and subset to desired region and epiweeks"""

    # load and clean data
    data_file = os.path.join(params['input_files']['parent_dir'],
                             params['input_files']['weekly_data'])
    df = pd.read_csv(data_file, low_memory=False)

    df = df.ffill()
    df = df.bfill()

    df = df.fillna(0)
    
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    df = df[(df["epiweek"] <= pred_week) & (df["epiweek"] >= start_week_data)]

    # smooth all features
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode='full')[:-w + 1]

    # if target variable not in features, add it
    if params['target'] not in params['data_features']:
        params['data_features'].append(params['target'])

    # smooth all features
    if smooth:
        # add not smoothed true
        df['test_gt'] = df[params['target']].copy()
        for feature in params['data_features']:
            df.loc[:, feature] = moving_average(df.loc[:, feature].values, params['data_params']['smooth_window'])
        return df[params['data_features']+['test_gt']]

    df["weekid"] = df["epiweek"].apply(lambda x: x.cdcformat())
    a = df[params['data_features']+['weekid']]
    df['record_id'] = df['fips']+"_"+df['epiweek'].astype(str).astype(str)
    b = df[['record_id']]

    return a,b
    
def get_state_train_data_multimodal(params, region, smooth=False):
    """ get processed dataframe of data + target as array """

    # convert to epiweeks
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    last_train_week = convert_to_epiweek(params['last_train_time'])

    # load data
    df, recs = load_df_multimodal(params, region, start_week, last_train_week, smooth)

    # select target
    target = df.loc[:, [params['target']]].values

    return df[params['data_features']+['weekid']], target , recs


def get_state_test_data_multimodal(params, region, pred_week, smooth=False):
    """ get processed dataframe of data + target as array"""
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    pred_week = convert_to_epiweek(pred_week)
    weeks_ahead = params['weeks_ahead']

    # import smoothed dataframe
    df = load_df(params, region, start_week, pred_week + weeks_ahead, smooth).tail(weeks_ahead)

    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values

    return df[params['data_features']+['weekid']], target


def get_state_test_data_xy_multimodal(params, region, pred_week, x_length, smooth=False, remove_weeks_after_test=True):
    weeks_ahead = params['weeks_ahead']
    start_week = convert_to_epiweek(params['data_params']['start_time'])

    # y
    test_week = convert_to_epiweek(params['test_time'])
    avail_weeks = epiweek_sub(test_week, pred_week)
    clipped_weeks_ahead = min(avail_weeks, weeks_ahead) if remove_weeks_after_test else weeks_ahead
    df,recs = load_df_multimodal(params, region, start_week, pred_week + clipped_weeks_ahead, smooth)[0].tail(clipped_weeks_ahead),load_df_multimodal(params, region, start_week, pred_week + clipped_weeks_ahead, smooth)[1]
    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values
    
    # pad if clipped weeks ahead is smaller than weeks ahead
    tmp_target = [-9] * weeks_ahead
    for i in range(target.shape[0]):
        tmp_target[i] = target[i, 0]
    target = np.array(tmp_target).reshape(-1, 1)
    
    # x
    df, recs = load_df_multimodal(params, region, start_week, pred_week, smooth)
    df = df.tail(x_length)
    recs = np.array([recs.tail(x_length)['record_id'].iloc[0]])
    # print(df[params['data_features']+['weekid']].shape, recs.shape)
    return df[params['data_features']+['weekid']], target, recs


def get_state_train_data(params, region, smooth=False):
    """ get processed dataframe of data + target as array """

    # convert to epiweeks
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    last_train_week = convert_to_epiweek(params['last_train_time'])

    # load data
    df = load_df(params, region, start_week, last_train_week, smooth)

    # select target
    target = df.loc[:, [params['target']]].values

    return df[params['data_features']+['weekid']], target


def get_state_test_data(params, region, pred_week, smooth=False):
    """ get processed dataframe of data + target as array"""
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    pred_week = convert_to_epiweek(pred_week)
    weeks_ahead = params['weeks_ahead']

    # import smoothed dataframe
    df = load_df(params, region, start_week, pred_week + weeks_ahead, smooth).tail(weeks_ahead)

    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values

    return df[params['data_features']+['weekid']], target


def get_state_test_data_xy(params, region, pred_week, x_length, smooth=False, remove_weeks_after_test=True):
    weeks_ahead = params['weeks_ahead']
    start_week = convert_to_epiweek(params['data_params']['start_time'])

    # y
    test_week = convert_to_epiweek(params['test_time'])
    avail_weeks = epiweek_sub(test_week, pred_week)
    clipped_weeks_ahead = min(avail_weeks, weeks_ahead) if remove_weeks_after_test else weeks_ahead
    df = load_df(params, region, start_week, pred_week + clipped_weeks_ahead, smooth).tail(clipped_weeks_ahead)
    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values
    
    # pad if clipped weeks ahead is smaller than weeks ahead
    tmp_target = [-9] * weeks_ahead
    for i in range(target.shape[0]):
        tmp_target[i] = target[i, 0]
    target = np.array(tmp_target).reshape(-1, 1)
    
    # x
    df = load_df(params, region, start_week, pred_week, smooth)
    df = df.tail(x_length)

    return df[params['data_features']+['weekid']], target


def pad_sequence(seqs, batch_first=True, padding_value=0, max_length=None):
    """
        Pads a list of sequences to the same length
        Input:
            seqs: list of sequences
            batch_first: if True, output is (batch, seq_len, ...)
                            else, output is (seq_len, batch, ...)
            padding_value: value to pad with
    """
    max_len = max(len(seq) for seq in seqs)
    if max_length is not None:
        max_len = max_length
    if batch_first:
        padded_seqs = np.full((len(seqs), max_len, *seqs[0].shape[1:]),
                              padding_value,
                              dtype=seqs[0].dtype)
    else:
        padded_seqs = np.full((max_len, len(seqs), *seqs[0].shape[1:]),
                              padding_value,
                              dtype=seqs[0].dtype)

    for i, seq in enumerate(seqs):
        if batch_first:
            padded_seqs[i, :len(seq)] = seq
        else:
            padded_seqs[:len(seq), i] = seq

    return padded_seqs.astype(np.float32)


def create_window_seqs(x, y, min_sequence_length, weeks_ahead, pad_value):
    """
    Creates windows of fixed size with appended zeros
    Input:
        x: features [n_samples, n_features]
        y: targets, [n_samples, 1]
        min_sequence_length: minimum length of sequence
        weeks_ahead: number of weeks ahead to predict
        pad_value: value to pad with
    """
    seqs, mask_seqs, targets, mask_ys = [], [], [], []
    for idx in range(min_sequence_length, x.shape[0] + 1, 1):
        # Sequences
        seqs.append(x[:idx, :])
        # mask_seqs.append(np.ones(idx))
        mask_seqs.append(np.zeros(idx))

        # Targets
        y_val = y[idx:idx + weeks_ahead]
        y_ = np.ones((weeks_ahead, y_val.shape[1])) * pad_value
        y_[:y_val.shape[0], :] = y_val
        mask_y = np.zeros(weeks_ahead)
        mask_y[:len(y_val)] = 1
        targets.append(y_)
        mask_ys.append(mask_y)

    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask_seqs = pad_sequence(mask_seqs, batch_first=True, padding_value=-np.inf)
    ys = pad_sequence(targets, batch_first=True, padding_value=pad_value)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0)

    return seqs, mask_seqs, ys, mask_ys


def create_fixed_window_seqs_multimodal(x, y, recs, sequence_length, weeks_ahead, pad_value):
    seqs, mask_seqs, ys, mask_ys, r = [], [], [], [], []
    for idx in range(sequence_length, x.shape[0] + 1, 1):
        # Sequences
        seqs.append(x[idx-sequence_length:idx, :])
        mask_seqs.append(np.zeros(sequence_length, dtype=float))
        r.append(recs['record_id'].iloc[idx-sequence_length])
        # Targets
        y_val = y[idx:idx + weeks_ahead]
        y_ = np.ones((weeks_ahead, y_val.shape[1])) * pad_value
        y_[:y_val.shape[0], :] = y_val
        mask_y = np.zeros(weeks_ahead)
        mask_y[:len(y_val)] = 1
        ys.append(y_)
        mask_ys.append(mask_y)
    seqs = np.array(seqs, dtype=float)
    mask_seqs = np.array(mask_seqs)
    r = np.array(r)
    ys = pad_sequence(ys, batch_first=True, padding_value=pad_value)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0)
    return seqs, mask_seqs, ys, mask_ys, r

def create_fixed_window_seqs(x, y, sequence_length, weeks_ahead, pad_value):
    seqs, mask_seqs, ys, mask_ys = [], [], [], []
    for idx in range(sequence_length, x.shape[0] + 1, 1):
        # Sequences
        seqs.append(x[idx-sequence_length:idx, :])
        mask_seqs.append(np.zeros(sequence_length, dtype=float))

        # Targets
        y_val = y[idx:idx + weeks_ahead]
        y_ = np.ones((weeks_ahead, y_val.shape[1])) * pad_value
        y_[:y_val.shape[0], :] = y_val
        mask_y = np.zeros(weeks_ahead)
        mask_y[:len(y_val)] = 1
        ys.append(y_)
        mask_ys.append(mask_y)
    seqs = np.array(seqs, dtype=float)
    mask_seqs = np.array(mask_seqs)
    ys = pad_sequence(ys, batch_first=True, padding_value=pad_value)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0)
    return seqs, mask_seqs, ys, mask_ys

def split_seqs(xs, mask_xs, ys, mask_ys, weeks_ahead, cal_num, test_num):
    """Split one sequences into train, calibration and test set."""
    total_num = len(xs)
    train_num = total_num - cal_num - test_num

    def split_seqs_helper(start, end):
        return xs[start:end], mask_xs[start:end], ys[start:end], mask_ys[start:end]

    trains = split_seqs_helper(0, train_num)
    cals = split_seqs_helper(train_num, train_num + cal_num)
    tests = split_seqs_helper(train_num + cal_num, train_num + cal_num + test_num)
    
    return trains, cals, tests

def prepare_ds_multimodal(xs, xs_masks, ys, ys_masks,recs, regions, metas, model_name,temp_dir, hum_dir, test=False, return_ds=True, with_week_id=True):
    regions = np.array(regions, dtype="str").tolist()
    metas = np.concatenate(metas, axis=0)
    
    xs = np.concatenate(xs, axis=0)
    xs_masks = np.concatenate(xs_masks, axis=0)
    #  add a dimension for the number of features
    xs_masks = np.expand_dims(xs_masks, 2)
    recs = np.concatenate(recs, axis=0)
    
    if not test:
        ys = np.concatenate(ys, axis=0)
        ys_masks = np.concatenate(ys_masks, axis=0)
        ys_masks = np.expand_dims(ys_masks, 2)
        ys_masks = np.array(ys_masks)
    else:
        ys = np.ones((xs.shape[0], 2))
        ys_masks = np.ones((xs.shape[0], 2))

    if return_ds:
        if model_name == 'seq2seq':
            dataset = SeqData(regions, metas, xs, xs_masks, ys, ys_masks, with_week_id)
        elif model_name in ["ts_img_jointattn"]:
            dataset = SeqDataMMoEMapsPNG(regions, metas, xs, xs_masks, ys, ys_masks, recs, temp_dir, hum_dir, with_week_id)
        return dataset
    return xs, xs_masks, ys, ys_masks, regions, metas

def prepare_ds(xs, xs_masks, ys, ys_masks, regions, metas, model_name, titers = None, test=False, return_ds=True, with_week_id=True):
    regions = np.array(regions, dtype="str").tolist()
    metas = np.concatenate(metas, axis=0)
    
    xs = np.concatenate(xs, axis=0)
    xs_masks = np.concatenate(xs_masks, axis=0)
    #  add a dimension for the number of features
    xs_masks = np.expand_dims(xs_masks, 2)
    
    
    if not test:
        ys = np.concatenate(ys, axis=0)
        ys_masks = np.concatenate(ys_masks, axis=0)
        ys_masks = np.expand_dims(ys_masks, 2)
        ys_masks = np.array(ys_masks)
    else:
        ys = np.ones((xs.shape[0], 2))
        ys_masks = np.ones((xs.shape[0], 2))

    if return_ds:
        dataset = SeqData(regions, metas, xs, xs_masks, ys, ys_masks, with_week_id)
        return dataset
    return xs, xs_masks, ys, ys_masks, regions, metas



# ====== DATALOADERS =========
class SeqDataMMoEMapsPNG(torch.utils.data.Dataset):
    def __init__(
        self,
        region,
        metas,
        X,
        mask_X,
        y,
        mask_y,
        recs,
        temp_dir,
        covid_dir,
        with_week_id: bool = True,
        img_in_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
    ):
        """
        PNG-based multimodal dataset.

        Args:
            region: [N]
            metas: [N, meta_dim]
            X: (N, T, F+1) if with_week_id else (N, T, F)
            mask_X: (N, T)
            y: (N, W, out_dim)
            mask_y: (N, W)
            recs: per-window record IDs; can be scalar or per-time-step IDs
            temp_dir: directory with temperature PNGs
            covid_dir: directory with covid PNGs (reusing hum_dir slot)
            with_week_id: if True, last feature in X is week_id
            img_in_channels: 1 (covid only) or 2 (temp+covid)
        """
        self.region = region
        self.metas = metas
        self.X = X
        self.mask_X = mask_X
        self.y = y
        self.mask_y = mask_y
        self.recs = recs
        self.temp_dir = temp_dir
        self.covid_dir = covid_dir
        self.with_week_id = with_week_id
        self.img_in_channels = img_in_channels
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return self.X.shape[0]


    def _load_single_map(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((self.img_height, self.img_width), dtype=np.float32)
        img = Image.open(path).convert("L")
        if img.size != (self.img_width, self.img_height):
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32, copy=True)  # forces a real copy
        return arr


    def _canonicalize_epi_for_covid(self, rec_str: str) -> str:
        """
        Map any epiweek (YYYYWW_) to the canonical epi-year 2023-2024
        for COVID heatmaps.

        Example:
        - '202020_19' -> '202320_19'
        - '202004_19' -> '202404_19'
        """
        # expect something like '202043_19' or '202043'
        region, epi_part = rec_str.split("_", 1)
  
        # robust guard
        if len(epi_part) < 6:
            # if format is weird, just return original
            return rec_str

        year = int(epi_part[:4])
        week = int(epi_part[4:6])

        ######
        ####### Skipping this for the time begin to test model #######
        ######
        # Map week to canonical epi-year:
        # Weeks 20–53 -> 2023, Weeks 1–19 -> 2024
        if 19 <= week <= 53:
            new_year = 2023
            if week == 53:
                return f"{region}_202352"
        else:
            new_year = 2024

        new_epi = f"{new_year}{week:02d}"
        # new_epi = f"{year}{week:02d}"

        return f"{region}_{new_epi}"


    def __getitem__(self, idx):
        # ----- tabular + meta -----
        region = self.region[idx]
        meta = self.metas[idx]

        if self.with_week_id:
            features = self.X[idx, :, :-1]   # [Tin, F]
            week_id = self.X[idx, :, -1]     # [Tin]
        else:
            features = self.X[idx]
            week_id = None

        mask_X = self.mask_X[idx]           # [Tin]
        label = self.y[idx]                 # [W, ...]
        mask_y = self.mask_y[idx]           # [W]

        # ----- rec IDs -----
        rec_ids = self.recs[idx]

        rec_ids = np.atleast_1d(self.recs[idx])
        # print(rec_ids.shape)
        Tin = features.shape[0]

        # parse numeric region key from the scalar window id like "17_202228"
        region_key = None
        s0 = str(rec_ids[0])
        if "_" in s0:
            region_key = s0.split("_", 1)[0]   # e.g., "17"

        imgs = []
        for t in range(Tin):
            if len(rec_ids) == 1:
                # build per-timestep rec_str using region_key + week_id[t]
                if week_id is None:
                    raise ValueError("with_week_id must be True when rec_ids is scalar.")
                if region_key is None:
                    raise ValueError(f"Cannot parse region_key from rec_ids[0]={rec_ids[0]}")
                epi = int(week_id[t])  # YYYYWW
                rec_str = f"{region_key}_{epi:06d}"
            else:
                # if rec_ids already provides per-timestep ids, use them directly
                rec_str = str(rec_ids[t])

            temp_path = os.path.join(self.temp_dir, f"temperature_{rec_str}.png")
            covid_rec_str = self._canonicalize_epi_for_covid(rec_str)
            covid_path = os.path.join(self.covid_dir, f"cch_{covid_rec_str}.png")

            temp_map = self._load_single_map(temp_path)
            covid_map = self._load_single_map(covid_path)
            # covid_map = np.zeros_like(temp_map)

            if self.img_in_channels == 1:
                img = covid_map[None, ...]
            else:
                img = np.stack([temp_map, covid_map], axis=0)

            imgs.append(img)

        maps = np.stack(imgs, axis=0).astype(np.float32)  # [Tin,C,H,W]

        meta = torch.as_tensor(meta, dtype=torch.float32)
        features = torch.as_tensor(features, dtype=torch.float32)
        mask_X = torch.as_tensor(mask_X, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.float32)
        mask_y = torch.as_tensor(mask_y, dtype=torch.float32)
        maps = torch.as_tensor(maps, dtype=torch.float32)



        if self.with_week_id:
            week_id = torch.as_tensor(week_id, dtype=torch.float32)
            return (
                region,
                meta,
                features,   # [Tin, F]
                mask_X,     # [Tin]
                label,      # [W, ...]
                mask_y,     # [W]
                week_id,    # [Tin]
                maps,       # [Tin, C, H, W]
            )
        else:
            return (
                region,
                meta,
                features,
                mask_X,
                label,
                mask_y,
                maps,
            )


class SeqData(torch.utils.data.Dataset):
    def __init__(self, region, metas, X, mask_X, y, mask_y, with_week_id=True, titers=None):
        self.region = region
        self.metas = metas
        self.X = X
        self.mask_X = mask_X
        self.y = y
        self.mask_y = mask_y
        self.with_week_id = with_week_id
        self.titers=titers

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.with_week_id:
            return (self.region[idx], self.metas[idx, :], self.X[idx, :, :-1], self.mask_X[idx], self.y[idx], self.mask_y[idx], self.X[idx, :, -1])
        return (self.region[idx], self.metas[idx, :], self.X[idx, :, :], self.mask_X[idx], self.y[idx], self.mask_y[idx], self.mask_y[idx])


