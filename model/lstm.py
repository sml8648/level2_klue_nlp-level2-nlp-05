from transformers import AutoModel
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast


class LSTMModel(nn.Module):
    """
    pretrained model통과 후 classification 하기 전 LSTM 레이어를 통과하도록 추가한 모델

    데이터 -> Pretrained 모델 -> lstm -> activation -> output_proj(batch_size, 30) -> return
    ver 2) 데이터 -> Pretrained 모델 -> lstm -> dense -> activation -> output_proj(batch_size, 30) -> return
    """

    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        self.model = AutoModel.from_pretrained(self.model_name, hidden_dropout_prob=conf.train.dropout, attention_probs_dropout_prob=conf.train.dropout)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2, num_layers=2, dropout=conf.train.dropout, batch_first=True, bidirectional=True)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim, self.num_labels)

    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        lstm_output, (last_hidden, last_cell) = self.lstm(output)
        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        x = self.activation(cat_hidden)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.conf.train.rdrop:
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        logits2 = self.process(input_ids, attention_mask, token_type_ids)
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)

        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss
