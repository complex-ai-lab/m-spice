import torch.nn as nn
import torch
from epiweeks import Week
import yaml
import numpy as np
import random
import argparse
from tqdm import tqdm
import os

from forecaster.utils import ForecasterTrainer, EarlyStopping, pickle_save, decode_onehot, last_nonzero
from forecaster.jointattntransformer import TsEncoderThenJointAttn
from forecaster.simplebasemodels import ArimaWrapper
from forecaster.baseline import GRUBaseline, TransformerBaseline
from forecaster.load_covid import prepare_data, prepare_region_fine_tuning_data, prepare_data_multimodal

import warnings
warnings.filterwarnings("ignore")

#############################
# dataset related variables #
#############################

# new mapping: FIPS → state abbreviation
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

# TODO: Make use of ot_params
def map_data_params_file(ot_params):
    data_params_file = '../../setup/covid_mortality_3.yaml'
    return data_params_file


def assign_last_train_time(params, last_train_time):
    dataset_name = params['dataset']
    if dataset_name == 'covid' or dataset_name == 'multimodal_covid':
        params['last_train_time'] = Week.fromstring(last_train_time).cdcformat()
    return params


def prepare_dataloader_helper(params, power_df):
    print(params['dataset'])
    if params['dataset'] == 'covid':
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_data(params)
    elif params['dataset'] == 'multimodal_covid':
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_data_multimodal(params)
    return train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length

def model_init(params, metas_dim, x_dim, device, seq_length):
    model = None
    if params['model_name'] == 'colagnn':
        from forecaster.baseline import ColaGNNWrapper
        model = ColaGNNWrapper(params=params, device=device, seq_length=seq_length)
    
    if params["model_name"] == "cnnrnnres":
        from forecaster.baseline import CNNRNNResOnlineWrapper
        model = CNNRNNResOnlineWrapper(params=params, device=device, seq_length=seq_length)

    if params['model_name'] == 'transformer':
        model = TransformerBaseline(
            input_dim=x_dim - 1,
            meta_dim=metas_dim,
            hidden_dim=params['model_parameters']['hidden_dim'],
            num_layers=params['model_parameters'].get('num_layers', 2),
            num_heads=params['model_parameters'].get('num_heads', 4),
            weeks_ahead=params['weeks_ahead'],
            dropout=params['model_parameters'].get('dropout', 0.1),
            pool=params.get('pool', 'last'),
        )
    if params['model_name'] == 'ts_img_jointattn':
        model = TsEncoderThenJointAttn(
            x_dim=x_dim - 1,
            meta_dim=metas_dim,
            d_model=params['model_parameters'].get('hidden_dim', 128),
            weeks_ahead=params['weeks_ahead'],
            ts_layers=params['model_parameters'].get('ts_layers', 2),
            ts_heads=params['model_parameters'].get('ts_heads', 4),
            joint_layers=params['model_parameters'].get('joint_layers', 1),
            joint_heads=params['model_parameters'].get('joint_heads', 4),
            dropout=params['model_parameters'].get('dropout', 0.1),
            pool=params.get('pool', 'last'),
        )

    if params['model_name'] == 'gru':
        model = GRUBaseline(
            input_dim=x_dim - 1,
            hidden_dim=params['model_parameters'].get('hidden_dim', 128),
            horizon=params['weeks_ahead'],
            meta_dim=metas_dim,                  
            meta_emb_dim=params['model_parameters'].get('meta_emb_dim', 16),
        )
    if params['model_name'] == 'arima':
        model = ArimaWrapper(aheads=params['weeks_ahead'], target_idx=params["data_features"].index(params["target"])) #params['target_idx'])
    return model


def region_fine_tuning(params, model_state_dict, target_region, all_dataloaders, seq_length):
    device = torch.device(params['device'])
    train_dataloader, val_dataloader, _, x_dim, _ = all_dataloaders[target_region]
    metas_dim = len(params['regions'])
    
    # load pretrained model states
    model = model_init(params, metas_dim, x_dim, device, seq_length)
    
    model = model.to(device)
    model.load_state_dict(model_state_dict)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

    # create loss function
    loss_fn = nn.MSELoss()

    # create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=params['training_parameters']['gamma'],
        verbose=False)

    # create early stopping
    early_stopping = EarlyStopping(
        patience=50, verbose=False)

    # create trainer
    trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

    # train model
    for epoch in range(params['rft_epochs']):
        trainer.train(train_dataloader, epoch)
        val_loss = trainer.evaluate(val_dataloader, epoch)
        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    
    model.load_state_dict(early_stopping.model_state_dict)
    return model


def forecast(model, model_name, test_dataloader, device, is_test, true_scale, ys_scalers):
    model.eval()
    predictions = {}
    addition_info = {}
    with torch.no_grad():
        for batch in test_dataloader:
            if model_name not in ['ts_img_jointattn']:
                regions, meta, x, x_mask, y, y_mask, weekid = batch
                regionid = decode_onehot(meta)
                x_mask = x_mask.type(torch.float)
                regionid = decode_onehot(meta)
                weekid = last_nonzero(weekid)

            if model_name in ['ts_img_jointattn']:
                regions, meta, x, x_mask, y, y_mask, weekid, maps = batch
                regionid = decode_onehot(meta)
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)
                # y = y.to(device)
                y_mask = y_mask.to(device)
                maps = maps.to(device)
 
                # forward pass
                y_pred = model.forward(x, x_mask, meta, maps).cpu().numpy() 

            if model_name == 'transformer':
                # send to device
                regionid = regionid.to(device)
                weekid = weekid.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)
                meta = meta.to(device)
                # forward pass
                # y_pred = model.forward(x, x_mask, regionid, weekid).unsqueeze(-1).cpu().numpy()
                y_pred = model.forward(x, x_mask, meta).unsqueeze(-1).cpu().numpy()

                emb = np.zeros(len(regions))
            
            if model_name == 'arima':
                y_pred = model.forward(x)[:, :, None]
                emb = np.zeros(len(regions))


            if model_name == 'gru':
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)  

                y_pred = model.forward(x, x_mask, meta)   # (B, H, 1)
                y_pred = y_pred.detach().cpu().numpy()
            
            if is_test:
                y = np.zeros((len(regions), len(y_pred[0])))
            else:
                y = y.numpy()
                y = y[:, :, 0] 
            
            meta = meta.cpu().numpy()
            # use scaler to inverse transform
            for i, region in enumerate(regions):
                if true_scale:
                    predictions[region] = ys_scalers[region].inverse_transform(y_pred[i]).reshape(-1)
                    y_in_true_scale = ys_scalers[region].inverse_transform(y[i].reshape(-1, 1)).reshape(-1)
                    addition_info[region] = (y_in_true_scale, y_mask[i], x[i], x_mask[i], weekid[i])
                else:
                    predictions[region] = y_pred[i].reshape(-1)
                    addition_info[region] = (y[i], y_mask[i], x[i], x_mask[i], weekid[i])
    return predictions, addition_info


def train_and_forcast(last_train_time, params, pretrained_model_state, train=True, power_df=None):
    params = assign_last_train_time(params, last_train_time)
    # device = torch.device(params['device'])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    true_scale = params['true_scale']

    is_test = False
    if params['test_time'] == params['last_train_time']:
        is_test = True
    
    train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_dataloader_helper(params, power_df)
    metas_dim = len(params['regions'])
    
    skip_training = False
    if params['model_name'] in ['arima', 'colagnn', 'cnnrnnres']:
        skip_training = True

    # params['target_idx'] = 0 
    model = model_init(params, metas_dim, x_dim, device, seq_length)

    if params["model_name"] == "colagnn":
        print("in here")
        if pretrained_model_state is not None:
            model.load_state_dict(pretrained_model_state)

        current_epiweek = int(params["last_train_time"])
        upto_index = model.epiweek_to_rowidx.get(current_epiweek, model.max_row)

        if train or pretrained_model_state is None:
            epochs = 75 
            if params["week_retrain"] == False and pretrained_model_state is not None:
                epochs = 75 
            model.train(upto_index=upto_index, epochs=epochs)

        pretrained_model_state = model.state_dict()

        predictions = model.predict_dict(upto_index=upto_index)
        H = params["weeks_ahead"]
        t0 = upto_index
        max_t = model.Y_full.shape[0] - 1

        addition_info = {}
        for i, r in enumerate(model.node_order):
            ys = []
            for h in range(1, H + 1):
                t = t0 + h
                if t > max_t:
                    break
                ys.append(float(model.Y_full[t, i]))
            y_true = np.array(ys, dtype=np.float32)
            addition_info[r] = (y_true, None, None, None, None)

        return predictions, addition_info, pretrained_model_state, model


    if params["model_name"] == "cnnrnnres":
        # restore weights
        if pretrained_model_state is not None:
            model.model.load_state_dict(pretrained_model_state)

        # map epiweek -> row index
        current_epiweek = int(params["last_train_time"])
        upto_index = model.epiweek_to_rowidx.get(current_epiweek, model.max_row)

        fine_tune_epochs = 75

        # run online step: returns (H, m)
        preds_Hm = model.online_step(
            current_t=upto_index,
            fine_tune_epochs=fine_tune_epochs
        )

        # save state
        pretrained_model_state = model.model.state_dict()

        # format predictions: dict[region] -> (H,)
        predictions = {}
        for i, r in enumerate(model.node_order):
            predictions[r] = preds_Hm[:, i].astype(np.float32)

        H = params["weeks_ahead"]
        t0 = upto_index
        max_t = model.Y_full.shape[0] - 1

        addition_info = {}
        for i, r in enumerate(model.node_order):
            ys = []
            for h in range(1, H + 1):
                t = t0 + h
                if t > max_t:
                    break
                ys.append(float(model.Y_full[t, i]))
            y_true = np.array(ys, dtype=np.float32)
            addition_info[r] = (y_true, None, None, None, None)

        return predictions, addition_info, pretrained_model_state, model


    if not skip_training:
        model = model.to(device)

        if train or pretrained_model_state is None: # Model we create is first trained and then later finetuned
            epochs = params['training_parameters']['epochs']
            epochs = 75
            if params['week_retrain'] == False and pretrained_model_state is not None:
                model.load_state_dict(pretrained_model_state)
                epochs = params['week_retrain_epochs']
            
            if params["model_name"] != 'ts_img_jointattn':
                # optimizer for baselines models:
                optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])
            else:
                # optimizer for M-SPICE model:
                optimizer = torch.optim.Adam(
                                (p for p in model.parameters() if p.requires_grad),
                                lr=params['training_parameters']['lr']
                            ) # freezing img_encoder

            # create loss function
            loss_fn = nn.MSELoss()

            # create scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=params['training_parameters']['gamma'],
                verbose=False)

            # create early stopping
            early_stopping = EarlyStopping(
                patience=100, verbose=False)

            # create trainer
            trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

            # train model
            for epoch in range(epochs):
                trainer.train(train_dataloader, epoch)
                val_loss = trainer.evaluate(val_dataloader, epoch)
                scheduler.step()
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break

            pretrained_model_state = early_stopping.model_state_dict


        model.load_state_dict(pretrained_model_state)
    
    rft_models = {}
    predictions = {}
    addition_info = {}
    
    # fine-tuning for each state
    if params['region_fine_tuning'] == True:
        all_dataloaders = prepare_region_fine_tuning_data(params)
        for region in params['regions']:
            rft_models[region] = region_fine_tuning(params, pretrained_model_state, region, all_dataloaders, seq_length)
        for region in params['regions']:
            rft_model = rft_models[region]
            _, _, region_test_dataloader, _, _ = all_dataloaders[region]
            region_predictions, region_addition_info = forecast(rft_model, params['model_name'], region_test_dataloader, device, is_test, true_scale, ys_scalers)
            predictions[region] = region_predictions[region]
            addition_info[region] = region_addition_info[region]
    else:
        predictions, addition_info = forecast(model, params['model_name'], test_dataloader, device, is_test, true_scale, ys_scalers)
    return predictions, addition_info, pretrained_model_state, model


def get_params(input_file='1'):
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            ot_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    data_params_file = '../../setup/covid_mortality_3.yaml' # base settings
    if ot_params['dataset']:
        data_params_file = map_data_params_file(ot_params)
    
    with open(data_params_file, 'r') as stream:
        try:
            task_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    # Start with base task/data parameters 
    params = task_params.copy()
    
    if 'training_params' in ot_params:
        for key, value in ot_params['training_params'].items():
            params['training_params'][key] = value
    if 'model_params' in ot_params:
        for key, value in ot_params['model_params'].items():
            params['model_params'][key] = value
    
    # overwrite using online training params
    for key, value in ot_params.items():
        if key == 'training_params' or key == 'model_params':
            continue
        params[key] = value

    if params['dataset'] == 'covid' or params['dataset'] == 'multimodal_covid':
        params['data_params']['start_time'] = Week.fromstring(params['data_params']['start_time']).cdcformat()
        params['test_time'] = Week.fromstring(str(params['test_time'])).cdcformat()
    
    if params['week_retrain'] == False:
        params['week_retrain_period'] = params['total_steps']
    
    print('Paramaters loading success.')
    
    return params


def run_online_training(params):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    base_pred = []
    test_pred = None
    pretrained_model_state = None 
    
    if params['dataset'] == 'covid' or params['dataset'] == 'multimodal_covid':
        starting_week = str(params['pred_starting_time'])
        test_week = str(params['test_time'])
        total_weeks_number = int(params['total_steps'])

        for i in tqdm(range(total_weeks_number)):
            if i%params['week_retrain_period'] == 0:
                train = True
            else:
                train = False
            current_week = (Week.fromstring(starting_week) + i).cdcformat()
            if current_week != test_week:
                predictions, addition_infos, pretrained_model_state, final_model = train_and_forcast(current_week, params, pretrained_model_state, train=train)
                base_pred.append((predictions, addition_infos))
            if current_week == test_week:
                test_pred, _, pretrained_model_state, final_model  = train_and_forcast(current_week, params, pretrained_model_state)
                break
    
    results = {
        'params': params,
        'base_pred': base_pred,
        'test_pred': test_pred 
    }
    return results, final_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    args = parser.parse_args()

    input_file = args.input
    params = get_params(input_file)

    print(params)
    data_id = int(params['data_id'])
    print(params['week_retrain_period'])

    results, final_model = run_online_training(params)

    # Example:
    # ../../results/base_pred/ts_img_jointattn/covid/
    output_dir = os.path.join(
        "../../results/base_pred",
        params["model_name"],
        params["data_name"]
    )

    os.makedirs(output_dir, exist_ok=True)

    pred_save_path = os.path.join(
        output_dir,
        f"saved_pred_{data_id}_model_{params['model_name']}_"
        f"{params['model_variant']}_{params['dataset']}_"
        f"seed{params['seed']}.pickle"
    )

    pickle_save(pred_save_path, results)

    model_save_path = os.path.join(
        output_dir,
        f"final_model_{params['model_name']}_"
        f"{params['model_variant']}_{params['dataset']}_"
        f"seed{params['seed']}_"
        f"start{params['pred_starting_time']}.pt"
    )

    if params['model_name'] not in ["arima"]:
        torch.save(final_model.state_dict(), model_save_path)

    print(f"Saved predictions to: {pred_save_path}")
    print(f"Saved final model to: {model_save_path}")


if __name__ == '__main__':
    main()