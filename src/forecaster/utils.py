import yaml
import torch
import numpy as np
import pickle
# import pickle5


def pickle_save(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(fname, version5=False):
    # if version5:
    #     with open(fname, 'rb') as handle:
    #         return pickle5.load(handle)
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


def load_yaml_params(params_file_path):
    with open(f'{params_file_path}', 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    return params


def decode_onehot(oh_encoding):
    """oh encoding is in the shape of (N, F), where N is data number, F is dimension."""
    ids = torch.argmax(oh_encoding, dim=1)
    ids.to(torch.int)
    return ids


def last_nonzero(data):
    """get last non zero element in a (N, F) tensor."""
    new_data = torch.zeros((data.size()[0]))
    for i in range(data.size()[0]):
        idx = torch.max(torch.nonzero(data[i]))
        new_data[i] = data[i, idx]
    new_data = new_data.type(torch.int)
    return new_data

def freeze_module_(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

def assert_frozen_(m: torch.nn.Module, name="module"):
    for p in m.parameters():
        if p.requires_grad:
            raise RuntimeError(f"{name} is NOT frozen (requires_grad=True).")


class ForecasterTrainer:

    def __init__(self, model, model_name, optimizer, loss_fn, device):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader, epoch):
        """
            Trains the model for one epoch on the given data.
            """
        self.model.train()

        # If your model has an image encoder attribute, freeze it every epoch. #DGG freezing img_encoder
        if hasattr(self.model, "img_encoder"):
            freeze_module_(self.model.img_encoder)
            assert_frozen_(self.model.img_encoder, "img_encoder")

        losses = []
        for batch in dataloader:
            # get data
                
            if self.model_name not in ["ts_img_jointattn"]: #DGG -> Make this more modular
                region, meta, x, x_mask, y, y_mask, weekid = batch #DGG
                regionid = decode_onehot(meta)
                # print("x:", x.shape)
                # print("x_mask:", x_mask.shape, "unique:", torch.unique(x_mask))


            if self.model_name in ['gru']:
                # send to device
                meta = meta.to(self.device)
                x = x.to(self.device)
                x_mask = x_mask.to(self.device)
                y = y.to(self.device)
                y_mask = y_mask.to(self.device)

                # forward pass
                y_pred = self.model.forward(x, x_mask, meta)
     

            if self.model_name in ['ts_img_jointattn']:
                region, meta, x, x_mask, y, y_mask, weekid, maps = batch
                regionid = decode_onehot(meta)
                meta = meta.to(self.device)
                x      = x.to(self.device)                 # [B, Tin, x_dim]
                x_mask = x_mask.to(self.device) # [B, Tin]
                y      = y.to(self.device)                 # [B, W, 1]
                y_mask = y_mask.to(self.device)            # [B, W]
                maps    = maps.to(self.device)              # [B, Tin, C, H, W]

                y_pred = self.model.forward(x, x_mask, meta, maps)

            if self.model_name == 'transformer':
                # send to device
                # weekid = last_nonzero(weekid)
                # regionid = regionid.to(self.device)
                # weekid = weekid.to(self.device)
                x = x.to(self.device)
                x_mask = x_mask.to(self.device)
                y = y.to(self.device)
                y_mask = y_mask.to(self.device)
                meta = meta.to(self.device)

                # forward pass
                # y_pred = self.model.forward(x, x_mask, regionid, weekid).unsqueeze(-1)
                y_pred = self.model.forward(x, x_mask, meta).unsqueeze(-1)

            loss = self.loss_fn(y_pred * y_mask, y * y_mask)
                
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
            self.optimizer.step()

            # store loss
            losses.append(loss.item())

        # log epoch statistics
        mean_loss = np.mean(losses)
        print(f"Epoch {epoch}: train loss {mean_loss}")
        # logger.log_scalar('train_loss', mean_loss, epoch)


    def evaluate(self, dataloader, epoch):
        """
            Evaluates the model on the given data.
            """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                # get data
                if self.model_name not in ["ts_img_jointattn"]: 
                    # print("batch shape: ", batch.shape)
                    region, meta, x, x_mask, y, y_mask, weekid = batch
                    regionid = decode_onehot(meta)

                if self.model_name in ['gru']:
                    # send to device
                    meta = meta.to(self.device)
                    x = x.to(self.device)
                    x_mask = x_mask.to(self.device)
                    y = y.to(self.device)
                    y_mask = y_mask.to(self.device)

                    # forward pass
                    y_pred = self.model.forward(x, x_mask, meta)

                if self.model_name in ["ts_img_jointattn"]:
                    region, meta, x, x_mask, y, y_mask, weekid, maps = batch
                    regionid = decode_onehot(meta)
                    meta = meta.to(self.device)
                    x      = x.to(self.device)                 # [B, Tin, x_dim]
                    x_mask = x_mask.to(self.device) # [B, Tin, 1]
                    y      = y.to(self.device)                 # [B, W, 1]
                    y_mask = y_mask.to(self.device)            # [B, W]
                    maps    = maps.to(self.device)              # [B, Tin, C, H, W]

                    y_pred = self.model.forward(x, x_mask, meta, maps)

                if self.model_name == 'transformer':
                    # send to device
                    # weekid = last_nonzero(weekid)
                    # regionid = regionid.to(self.device)
                    # weekid = weekid.to(self.device)
                    x = x.to(self.device)
                    x_mask = x_mask.to(self.device)
                    y = y.to(self.device)
                    y_mask = y_mask.to(self.device)
                    meta = meta.to(self.device)

                    # forward pass
                    # y_pred = self.model.forward(x, x_mask, regionid, weekid).unsqueeze(-1)
                    y_pred = self.model.forward(x, x_mask, meta).unsqueeze(-1)

                loss = self.loss_fn(y_pred * y_mask, y * y_mask)
                # store loss
                losses.append(loss.item())

        # log epoch statistics
        mean_loss = np.mean(losses)
        # print(f"Epoch {epoch}: val loss {mean_loss}")

        return mean_loss


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        # torch.save(model.state_dict(), './models/checkpoint.pt')
        self.model_state_dict = model.state_dict()
        self.val_loss_min = val_loss