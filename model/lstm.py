from transformers import AutoModel
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch


class LSTMLayer(nn.Module):
    """
    Bidirectional LSTM -> dropout -> activation -> linear layer
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(LSTMLayer, self).__init__()
        self.use_activation = use_activation
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=2, dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.tanh = nn.Tanh()

    @autocast()
    def forward(self, x):  # W(tanh(lstm(x)))+b
        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        lstm_output, (last_hidden, last_cell) = self.lstm(x)
        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        x = self.dropout(cat_hidden)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class LSTMModel(CustomModel):
    """
    pretrained model통과 후 classification 하기 전 LSTM 레이어를 통과하도록 추가한 모델

    데이터 -> Pretrained 모델 -> lstm -> activation -> output_proj(batch_size, 30) -> return
    ver 2) 데이터 -> Pretrained 모델 -> lstm -> dense -> activation -> output_proj(batch_size, 30) -> return
    """

    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        self.model = AutoModel.from_pretrained(self.model_name, hidden_dropout_prob=conf.train.dropout, attention_probs_dropout_prob=conf.train.dropout)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        self.lstm = LSTMLayer(self.hidden_dim, self.num_labels)

    @autocast()
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        logits = self.lstm(output[0])
        return logits


class FCLayer(nn.Module):  # fully connected layer
    """
    RBERT emask를 위한 Fully Connected layer
    데이터 -> BERT 모델 -> emask 평균 -> FC layer -> 분류(FC layer)
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):  # W(tanh(x))+b
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)
