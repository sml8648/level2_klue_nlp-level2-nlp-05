from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, RobertaForSequenceClassification
from torch import nn
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch

class Model(nn.Module):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[conf.train.loss]

    @autocast()  
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            if(self.conf.train.rdrop):
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return outputs

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)
        
        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss


class CustomModel(nn.Module):
    def __init__(self, conf, new_vocab_size):
        super(CustomModel, self).__init__()
        self.num_labels = 30
        self.model_name = conf.model.model_name
        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        self.activation = torch.tanh
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.dropout = nn.Dropout(conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim * 4, self.num_labels)

    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Add custom layers
        features = outputs[0]  # outputs[0]=last hidden state
        x = features[:, 0, :] # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
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

            if(self.conf.train.rdrop):
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


class LSTMModel(CustomModel):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        self.model = AutoModel.from_pretrained(
                        self.model_name, 
                        hidden_dropout_prob=conf.train.dropout,
                        attention_probs_dropout_prob=conf.train.dropout
                    )
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim//2, num_layers=2, dropout=conf.train.dropout,
                            batch_first=True, bidirectional=True)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim, self.num_labels)
    
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output= self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids)[0] 
        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        lstm_output, (last_hidden, last_cell)= self.lstm(output)
        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)
        x = self.activation(cat_hidden)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class AuxiliaryModel(CustomModel):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        self.activation = torch.tanh
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.dropout = nn.Dropout(conf.train.dropout)
        self.binary_classification = nn.Linear(self.hidden_dim * 4, 2)
        self.classification0 = nn.Linear(self.hidden_dim * 4, self.num_labels)
        self.classification1 = nn.Linear(self.hidden_dim * 4, self.num_labels)
        self.weight = [0.5, 0.5]
    
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Add custom layers
        features = outputs[0]  # outputs[0]=last hidden state
        x = features[:, 0, :] # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)

        binary_logits = self.binary_classification(x)
        if(torch.argmax(binary_logits) == 0):
            logits = self.classification0(x)
        else:
            logits = self.classification1(x)

        return binary_logits, logits

    @autocast() 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        binary_logits, logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            binary_labels = torch.tensor([i if i==0 else 1 for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0]*binary_loss + self.weight[1]+loss

            if(self.conf.train.rdrop):
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids)
        binary_labels = torch.tensor([i if i==0 else 1 for i in labels], device="cuda")
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)
        
        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss

        binary_ce_loss = 0.5 * (self.loss_fct(binary_logits, binary_labels.view(-1)) + self.loss_fct(binary_logits2, binary_labels.view(-1)))
        binary_kl_loss = loss_module.compute_kl_loss(binary_logits, binary_logits2)
        # carefully choose hyper-parameters
        binary_loss = binary_ce_loss + alpha * binary_kl_loss
        return self.weight[0]*binary_loss + self.weight[1]*loss