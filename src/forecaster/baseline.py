import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from forecaster.cola_utils import normalize_adj2, sparse_mx_to_torch_sparse_tensor
from torch.nn.modules.module import Module
import torch.nn.init as init
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ===== TS Baselines ======
class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, horizon, meta_dim=None, meta_emb_dim=16):
        super().__init__()
        self.horizon = horizon
        self.use_meta = meta_dim is not None
        if self.use_meta:
            self.meta_proj = nn.Linear(meta_dim, meta_emb_dim)
            gru_in = input_dim + meta_emb_dim
        else:
            gru_in = input_dim

        self.gru = nn.GRU(gru_in, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x, x_mask=None, meta=None, **kwargs):
        if self.use_meta:
            m = self.meta_proj(meta)                      # (B, E)
            m = m.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
            x = torch.cat([x, m], dim=-1)                 # (B, T, F+E)

        _, h = self.gru(x)
        y = self.fc(h.squeeze(0))        # (B, H)
        return y.unsqueeze(-1)           # (B, H, 1)

class PositionalEncoding(nn.Module):
    """From: https://anonymous.4open.science/r/EmbedTS-3F5D/src/embedts/models/transformers/layers.py
    
    Positional encoding as described in "Attention is all you need"
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
   
    Sinusoidal positional encoding for inputs of shape (T, B, D)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Encoder-only Transformer baseline (no decoder, no week IDs).

    Inputs:
      - x: (B, T, F)
      - meta: (B, R) one-hot region metadata

    Output:
      - (B, H) multi-horizon prediction
    """

    def __init__(
        self,
        input_dim: int,
        meta_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        weeks_ahead: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
        pool: str = "last",  # "last" or "mean"
    ):
        super().__init__()
        if pool not in {"last", "mean"}:
            raise ValueError(f"pool must be one of ['last','mean'], got {pool}")

        self.pool = pool
        self.hidden_dim = hidden_dim
        self.weeks_ahead = weeks_ahead

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=False,  # (T, B, D)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, weeks_ahead),
        )

    def forward(self, x, x_mask, meta, week_ids=None):
        """
        Kept signature-compatible with your pipeline:
          - x_mask ignored (all sequences same length)
          - week_ids ignored (not used)

        Args:
            x: (B, T, F)
            meta: (B, meta_dim) one-hot
        Returns:
            (B, H)
        """
        # project inputs
        h = self.in_proj(x)                 # (B, T, D)

        # add region conditioning (broadcast across time)
        meta = meta.float()
        meta_emb = self.meta_proj(meta).unsqueeze(1)  # (B, 1, D)
        h = h + meta_emb                              # (B, T, D)

        # transformer encoder
        h = h.permute(1, 0, 2)         # (T, B, D)
        h = self.pos_encoder(h)
        enc = self.encoder(h)          # (T, B, D)
        enc = enc.permute(1, 0, 2)     # (B, T, D)

        # pool and predict
        pooled = enc[:, -1, :] if self.pool == "last" else enc.mean(dim=1)  # (B, D)
        out = self.head(pooled)                                            # (B, H)
        return out

# ===== Epi Models =======

class _GraphWindowDataset(Dataset):
    def __init__(self, Y_all, window, horizon):
        self.Y = torch.tensor(Y_all, dtype=torch.float32)
        self.window = window
        self.horizon = horizon
        self.T, self.m = self.Y.shape
        self.idxs = list(range(window, self.T - horizon + 1))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.Y[t - self.window:t, :]          # (window, m)
        y = self.Y[t + self.horizon - 1, :]       # (m,)
        return x, y


def _freeze_graph_params(model):
    freeze_names = {"W1","W2","V","b1","bv","Wb","wb"}
    for name, p in model.named_parameters():
        if any(name.endswith(k) or f".{k}" in name for k in freeze_names):
            p.requires_grad = False


class ColaGNNWrapper:
    """
    Self-contained wrapper: uses cola_gnn from the same file.
    Loads saved Y/A/node_order/epiweeks, trains up to current epiweek row,
    returns predictions as dict[region] -> (H,)
    """

    def __init__(self, params, device, seq_length):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.aheads = params["weeks_ahead"]
        self.window = seq_length

        # --- file paths (can also move these to YAML) ---
        self.target_matrix_path =  "../../data/colagnn_inputs/cola_target_matrix_covid_hosp.txt" #"../../data/cola_target_matrix_percent_ili.txt"  #"../../data/cola_target_matrix_covid_hosp.txt" #"../../data/cola_target_matrix_percent_ili.txt" 
        self.adj_matrix_path    =  "../../data/colagnn_inputs/state_adj_matrix.txt"
        self.node_order_path    =  "../../data/colagnn_inputs/cola_nodes_order.csv"
        self.epiweeks_path      =  "../../data/colagnn_inputs/cola_epiweeks.csv"

        # --- load artifacts ---
        self.Y_full = np.loadtxt(self.target_matrix_path, delimiter=",").astype(np.float32)  # (T,m)
        self.A      = np.loadtxt(self.adj_matrix_path, delimiter=",").astype(np.float32)     # (m,m)
        # self.node_order = pd.read_csv(self.node_order_path, header=None).iloc[:, 0].tolist()
        self.node_order_path    =  "../../data/colagnn_inputs/cola_nodes_order.csv"
        self.node_order = pd.read_csv(self.node_order_path).iloc[:, 0].astype(str).tolist()

        self.scaler = StandardScaler()
        self.scaler_fitted = False

        epi_df = pd.read_csv(self.epiweeks_path)
        self.epiweek_to_rowidx = {int(w): i for i, w in enumerate(epi_df["epiweek"].tolist())}
        self.max_row = self.Y_full.shape[0] - 1

        assert self.Y_full.shape[1] == self.A.shape[0] == len(self.node_order)

        # --- build the data object expected by cola_gnn ---
        class _Data: 
            pass

        self.data = _Data()
        self.data.m = self.A.shape[0]
        self.data.d = 0  # or 1 — not used by cola_gnn forward, but required

        orig_adj = torch.tensor(self.A, dtype=torch.float32)
        self.data.orig_adj = orig_adj.to(self.device)

        # cola_gnn expects data.adj too (it normalizes inside, but still references this field)
        self.data.adj = self.data.orig_adj.clone()

        # --- build the args object expected by cola_gnn (hidden here) ---
        class _Args: pass
        self.args = _Args()
        self.args.cuda = torch.cuda.is_available()
        self.args.window = self.window
        self.args.dropout = params["training_parameters"].get("dropout", 0.1)
        self.args.n_hidden = params["model_parameters"].get("hidden_dim", 64)
        self.args.k = params.get("k", 10)
        self.args.rnn_model = params.get("colagnn_rnn", "GRU")
        self.args.n_layer = params["model_parameters"].get("num_layers", 1)
        self.args.bi = False
        self.args.horizon = 1  # overridden per model

        self.lr = params["training_parameters"].get("lr", 1e-3)
        self.freeze_graph = params.get("colagnn_freeze_graph", True)

        # --- one model per horizon ---
        self.models = nn.ModuleDict()
        self.opts = {}
        for h in range(1, self.aheads + 1):
            args_h = copy.deepcopy(self.args)
            args_h.horizon = h

            model = cola_gnn(args_h, self.data).to(self.device)   # <-- uses cola_gnn from same file

            if self.freeze_graph:
                _freeze_graph_params(model)

            self.models[str(h)] = model
            self.opts[str(h)] = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr
            )

        self.loss_fn = nn.MSELoss()

    def _Y_upto(self, upto_index: int):
        upto_index = int(min(max(upto_index, 0), self.max_row))
        return self.Y_full[:upto_index + 1, :]

    def train(self, upto_index: int, epochs: int = 10, batch_size: int = 32, grad_clip: float = 1.0):
        Y = self._Y_upto(upto_index)
        # ---- scale targets (fit once) ----
        if not self.scaler_fitted:
            # Y_log = np.log1p(Y)          # strongly recommended for flu counts
            # self.scaler.fit(Y_log)
            self.scaler.fit(Y)
            self.scaler_fitted = True

        # Y_scaled = self.scaler.transform(np.log1p(Y))
        Y_scaled = self.scaler.transform(Y)


        for h in range(1, self.aheads + 1):
            ds = _GraphWindowDataset(Y_scaled, window=self.window, horizon=h)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

            model = self.models[str(h)]
            opt = self.opts[str(h)]
            model.train()
            total_steps = epochs * len(dl)
            pbar = tqdm(total=total_steps, desc=f"ColaGNN | h={h}", leave=True)

            for _ in range(epochs):
                for x, y in dl:
                    x = x.to(self.device)  # (B, window, m)
                    y = y.to(self.device)  # (B, m)
                    opt.zero_grad(set_to_none=True)
                    yhat, _ = model(x)
                    loss = self.loss_fn(yhat, y)
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    pbar.update(1)
            pbar.close()


    @torch.no_grad()
    def forward(self, upto_index: int):
        """
        Returns (m, H) numpy predictions aligned with node_order
        """
        Y = self._Y_upto(upto_index)
        # transform inputs to match training space
        # Y_scaled = self.scaler.transform(np.log1p(Y))   # (T, m)
        if not self.scaler_fitted:
            # Y_log = np.log1p(Y)          # strongly recommended for flu counts
            # self.scaler.fit(Y_log)
            self.scaler.fit(Y)
            self.scaler_fitted = True
        Y_scaled = self.scaler.transform(Y) 

        x_win = Y_scaled[-self.window:, :]              # (window, m)

        x = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(self.device)

        preds = []
        for h in range(1, self.aheads + 1):
            model = self.models[str(h)]
            model.eval()
            yhat, _ = model(x)  # (1, m)
            preds.append(yhat.squeeze(0).detach().cpu().numpy())

        pred_mat = np.stack(preds, axis=0)  # (H, m)
        return pred_mat.T                   # (m, H)

    def predict_dict(self, upto_index: int):
        pred_scaled = self.forward(upto_index)  # (m, H)
        # pred_scaled: model output in scaled log space
        pred = self.scaler.inverse_transform(pred_scaled.T).T
        # pred_log = self.scaler.inverse_transform(pred_scaled.T).T
        # pred = np.expm1(pred_log)
        return {r: pred[i, :] for i, r in enumerate(self.node_order)}

    def state_dict(self):
        return {str(h): self.models[str(h)].state_dict() for h in range(1, self.aheads + 1)}

    def load_state_dict(self, sd):
        for h in range(1, self.aheads + 1):
            self.models[str(h)].load_state_dict(sd[str(h)], strict=True)


class cola_gnn(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.x_h = 1 
        self.f_h = data.m   
        self.m = data.m  
        self.d = data.d 
        self.w = args.window
        self.h = args.horizon
        self.adj = data.adj
        self.o_adj = data.orig_adj
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.n_spatial = 10  
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
 
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError (' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)  

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)  
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        ''' 
        b, w, m = x.size()
        orig_x = x 
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)

        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i:i+1].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = self.adj.repeat(b,1)
        adjs = adjs.view(b,self.m, self.m)

        # print (self.adj)
        # print (adjs)

        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c) 
        adj = a_mx

        x = r_l  
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        out = torch.cat((out_spatial, out_temporal),dim=-1)
        out = self.out(out)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]

        return out, None



class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        # feature: (B, N, Fin) or (B, N, K)
        support = torch.matmul(feature, self.weight)   # (B, N, Fout)
        output = torch.matmul(adj, support)            # (B, N, Fout)
        return output + self.bias if self.bias is not None else output




# ------------- CNNRNN-res baseline ----------
        
class CNNRNNRes(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.ratio = args.ratio
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN

        self.GRU1 = nn.GRU(self.m, self.hidR)  # batch_first=False

        self.residual_window = args.residual_window

        self.mask_mat = Parameter(torch.empty(self.m, self.m))
        nn.init.xavier_normal_(self.mask_mat)

        # store adj as buffer so it moves with .to(device)
        self.register_buffer("adj", torch.as_tensor(data.adj, dtype=torch.float32))

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)

        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1)

        self.output = None
        if args.output_fun == "sigmoid":
            self.output = torch.sigmoid
        elif args.output_fun == "tanh":
            self.output = torch.tanh

    def forward(self, x):
        # x: (B, P, m)
        masked_adj = self.adj * self.mask_mat         # (m,m)
        x = x.matmul(masked_adj)                      # (B,P,m)

        r = x.permute(1, 0, 2).contiguous()           # (P,B,m)
        _, h = self.GRU1(r)                           # h: (1,B,hidR)
        h = self.dropout(h.squeeze(0))                # (B,hidR)

        res = self.linear1(h)                         # (B,m)

        if self.residual_window > 0:
            z = x[:, -self.residual_window:, :]       # (B,resW,m)
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)  # (B*m,resW)
            z = self.residual(z).view(-1, self.m)     # (B,m)
            res = res * self.ratio + z

        if self.output is not None:
            res = self.output(res).float()

        return res

@torch.no_grad()
def rollout_autoreg(model, x0: torch.Tensor, H: int):
    """
    x0: (B,P,m) -> returns (B,H,m)
    """
    model.eval()
    cur = x0.clone()
    preds = []
    for _ in range(H):
        y1 = model(cur)                          # (B,m)
        preds.append(y1.unsqueeze(1))            # (B,1,m)
        cur = torch.cat([cur[:, 1:, :], y1.unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1)




class RollingWindowDataset(Dataset):
    def __init__(self, Y: np.ndarray, window: int):
        self.Y = Y.astype(np.float32)   # (T,m)
        self.P = window

    def __len__(self):
        return max(0, self.Y.shape[0] - self.P)

    def __getitem__(self, idx):
        t = idx + self.P
        x = self.Y[t-self.P:t]          # (P,m)
        y = self.Y[t]                   # (m,)
        return torch.from_numpy(x), torch.from_numpy(y)



class CNNRNNResOnlineWrapper:
    def __init__(self, params, device, seq_length=None):
        """
        params: dict from YAML (same as in online_training.py)
        device: torch.device or str
        seq_length: unused (kept for compatibility with model_init signature)
        """
        self.params = params
        self.device = torch.device(device if isinstance(device, str) else device)

        # --- required from params ---
        self.P = int(params["data_params"]["window"]) if "data_params" in params and "window" in params["data_params"] else int(params.get("window", params["training_parameters"].get("window", 17)))
        # Prefer explicit window in params; but you likely already have params['data_params']['window'] or params['window']
        # If your YAML uses params['data_params']['window'], keep that. Otherwise set params['window'].

        self.H = int(params["weeks_ahead"])

        # --- load ColaGNN artifacts (reuse) ---
        # You can also move these paths into params if you want.
        self.Y_full = np.loadtxt("../../data/colagnn_inputs/cola_target_matrix_percent_ili.txt" , delimiter=",").astype(np.float32) #"../../data/cola_target_matrix_flu_hosp.txt" #"../../data/cola_target_matrix_percent_ili.txt" 
        self.A = np.loadtxt("../../data/colagnn_inputs/state_adj_matrix.txt", delimiter=",").astype(np.float32)

        # --- node order + epiweeks (for mapping epiweek->row idx) ---
        epiweeks = pd.read_csv("../../data/colagnn_inputs/cola_epiweeks.csv")['epiweek'].astype(int).tolist()
        self.node_order_path    =  "../../data/colagnn_inputs/cola_nodes_order.csv"
        self.node_order = pd.read_csv(self.node_order_path).iloc[:, 0].astype(str).tolist()

        self.epiweek_to_rowidx = {ew: i for i, ew in enumerate(epiweeks)}
        self.max_row = len(epiweeks) - 1

        # --- build args-like object for CNNRNNRes ---
        # CNNRNNRes expects attributes: ratio, window, hidRNN, residual_window, dropout, output_fun
        # optimizer expects lr, wd, batch_size
        class Args: 
            pass
        args = Args()

        args.ratio = params.get("ratio", 1.0)
        args.window = self.P
        args.hidRNN = params.get("hidRNN", params.get("model_parameters", {}).get("hidRNN", 64))
        args.residual_window = params.get("residual_window", 4)
        args.dropout = params.get("training_parameters", {}).get("dropout", params.get("model_parameters", {}).get("dropout", 0.1))
        args.output_fun = params.get("output_fun", "none")

        args.lr = params["training_parameters"]["lr"]
        args.wd = params["training_parameters"].get("wd", 0.0)
        args.batch_size = params["training_parameters"]["batch_size"]

        self.args = args  # for loader batch_size etc.

        # --- make data object expected by model ---
        class _Data: 
            pass
        data = _Data()
        data.m = self.Y_full.shape[1]
        data.adj = self.A

        self.model = CNNRNNRes(args, data).to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
        )
        self.loss_fn = nn.MSELoss()

    def _loader(self, Y_hist: np.ndarray, shuffle=True):
        ds = RollingWindowDataset(Y_hist, window=self.P)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=False
        )

    def online_step(self, current_t: int, fine_tune_epochs: int = 1):
        """
        Train on Y_full[:current_t] and predict horizons starting at current_t.
        current_t is a row index into Y_full (NOT an epiweek string).
        returns: (H, m)
        """
        assert current_t >= self.P
        assert current_t <= self.Y_full.shape[0]

        # --- train (only if epochs > 0) ---
        if fine_tune_epochs > 0:
            Y_hist = self.Y_full[:current_t]
            loader = self._loader(Y_hist, shuffle=True)

            self.model.train()
            for _ in range(fine_tune_epochs):
                for xb, yb in loader:
                    xb = xb.to(self.device)   # (B,P,m)
                    yb = yb.to(self.device)   # (B,m)

                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)

                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()

        # --- predict next H ---
        x0 = torch.from_numpy(self.Y_full[current_t-self.P:current_t]).unsqueeze(0).to(self.device)
        preds = rollout_autoreg(self.model, x0, H=self.H)  # (1,H,m)
        return preds.squeeze(0).detach().cpu().numpy()
