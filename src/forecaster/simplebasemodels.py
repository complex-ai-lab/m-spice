import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA

class ArimaWrapper:
    def __init__(self, aheads, target_idx) -> None:
        self.aheads = aheads
        self.target_idx = target_idx
    
    def train(self, dataloader):
        pass

    def forward(self, input_data):
        """Input: torch.Tensor in the shape of (batch_size x features x seq length)."""
        input_data = input_data.numpy()[:, :, self.target_idx]
        batch_size = input_data.shape[0]
        predictions = []
        for i in range(batch_size):
            current_input_data = input_data[i, :]
            regr = ARIMA(current_input_data, order=(3,1,0))
            regr = regr.fit()
            current_pred = regr.forecast(steps=self.aheads)      # length H
            predictions.append(np.asarray(current_pred, dtype=np.float32))

        return np.array(predictions)
    
    def eval(self):
        pass

    
    