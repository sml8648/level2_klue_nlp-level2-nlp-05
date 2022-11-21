from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, RobertaForSequenceClassification
from transformers import BertPreTrainedModel, AutoModel
from torch import nn
import torch
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
        self.classification = nn.Linear(self.hidden_dim * 4, self.num_labels)
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
        logits = self.classification(x)
        
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

## https://github.com/monologg/R-BERT/blob/master/model.py 사용
class FCLayer(nn.Module):       #fully connected layer
    '''
        RBERT emask를 위한 Fully Connected layer
        데이터 -> BERT 모델 -> emask 평균 -> FC layer -> 분류(FC layer)
    '''
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):          # W(tanh(x))+b
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

#RBERT
class CustomRBERT(BertPreTrainedModel):
    '''
        RBERT model
        데이터 -> BERT 모델 -> emask 평균 -> FClayer
        -> (hidden size, e1, e2, e3, e4 mask concat) -> 분류(FC layer)
    '''
    def __init__(self, config, conf, new_vocab_size):
        super(CustomRBERT, self).__init__(config)
        self.num_labels = 30
        self.config = config
        self.model_name = conf.model.model_name
        self.loss_fct = loss_module.loss_config[conf.train.loss]
        self.model = AutoModel.from_pretrained(self.model_name,config = self.config) 
        self.model.resize_token_embeddings(new_vocab_size)

        #cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, 0.1)
        #entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, 0.1)
        #entity type 토큰 FC layer
        #self.entity_type_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, 0.1)
        #concat 후 FC layer
        self.label_classifier = FCLayer(
            self.config.hidden_size * 5,
            self.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):  #엔티티 안의 토큰들의 임베딩 평균
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e3_h = self.entity_average(sequence_output, e3_mask)
        e4_h = self.entity_average(sequence_output, e4_mask)

        # Concat -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        #e3와 e4는 어떻게 할까?(fc layer 써야하나? e1,e1와 같은거로? 다른거로?
        e3_h = self.entity_fc_layer(e3_h)
        e4_h = self.entity_fc_layer(e4_h)

        #concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)