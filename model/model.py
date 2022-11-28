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
    '''
        pretrained model통과 후 classification하는 레이어를 커스텀할 수 있도록 구성한 모델
        activation function이나 dense레이어 개수/크기를 바꾸거나,
        classification할 때 CLS말고 다른 hidden state값도 사용할 수 있다.

        config.train.rdrop=True, config.train.dropout=0.2 추가

        데이터 -> Pretrained 모델 -> dense -> activation -> output_proj(batch_size, 30) -> return
    '''
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


class LSTMLayer(nn.Module):   
    '''
        Bidirectional LSTM -> dropout -> activation -> linear layer
    '''
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(LSTMLayer, self).__init__()
        self.use_activation = use_activation
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=2, dropout=dropout_rate,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim*2, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):          # W(tanh(lstm(x)))+b
        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        lstm_output, (last_hidden, last_cell)= self.lstm(x)
        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)
        x = self.dropout(cat_hidden)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class LSTMModel(CustomModel):
    '''
        pretrained model통과 후 classification 하기 전 LSTM 레이어를 통과하도록 추가한 모델

        데이터 -> Pretrained 모델 -> lstm -> activation -> output_proj(batch_size, 30) -> return
        ver 2) 데이터 -> Pretrained 모델 -> lstm -> dense -> activation -> output_proj(batch_size, 30) -> return
    '''
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

        self.lstm = LSTMLayer(self.hidden_dim, self.num_labels)
    
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output= self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids)[0] 
        logits = self.lstm(output[0])
        return logits


class AuxiliaryModel(CustomModel):
    '''
        binary label인지를 분류하는 binary_classification task를 추가한 모델
        binary_classifier에서 나온 logit(batch_size, 2)에 argmax를 취해 0인지 1인지 판별 후
        0이라면 label_classifier_0 레이어에, 1이라면 label_classifier_1 레이어에 넣어 각각 logit을 판단한다.
        그 후 binary_classifier의 loss와 label_classifier의 loss를 더해서 backpropagation -> binary_classifier과 classifier가 모두 학습.

        데이터 -> Pretrained 모델 -> binary_classifier(batch_size,2) -> label_classifier_0/label_classifier_1 -> add loss -> return
    '''
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

        self.binary_classifier = FCLayer(self.hidden_dim, 2)
        self.label_classifier_0 = FCLayer(self.hidden_dim, self.num_labels, False)
        self.label_classifier_1 = FCLayer(self.hidden_dim, self.num_labels, False)
        self.weight = [0.5, 0.5]
    
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Add custom layers
        x= outputs[1]  # take <s> token (equiv. to [CLS])  (batch_size, hidden_dim)
        
        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits,1)
        for i,l in enumerate(binary_labels.tolist()):
            if(l == 0):
                logits.append(self.label_classifier_0(x[i,:]))
            else:
                logits.append(self.label_classifier_1(x[i,:]))
            
        logits = torch.stack(logits,0)
        return binary_logits, logits    # (batch_size, 2), (batch_size, 30)

    @autocast() 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        binary_logits, logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            binary_labels = torch.tensor([0 if l==0 else 1 for l in labels], device="cuda") # (batch_size,)
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


class AuxiliaryModel2(CustomModel):
    '''
        binary label인지를 분류하는 binary_classification task를 추가한 AuxiliaryModel에서 binary classifier label을 0,1,2 3개로 분류하도록 변경한 것
        0은 no_relation, 1은 org, 2는 per
        0이라면 label_classifier_0에 넣고 1일때는 label_classifier_1, 2일때는 label_classifier_2에 넣는다. 그 후는 AuxiliaryModel과 같음.

        데이터 -> Pretrained 모델 -> binary_classifier(batch_size,3) -> label_classifier_0/label_classifier_1/label_classifier_2 -> add loss -> return
    '''
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

        self.binary_classifier = FCLayer(self.hidden_dim, 3, 0.1)
        self.label_classifier_0 = FCLayer(self.hidden_dim, self.num_labels, 0.1)
        self.label_classifier_1 = FCLayer(self.hidden_dim, self.num_labels, 0.1)
        self.label_classifier_2 = FCLayer(self.hidden_dim, self.num_labels, 0.1)
        self.weight = [0.3, 0.7]
        self.weight2 = [0.8, 0.2]
    
    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Add custom layers
        x = outputs[1]  # take <s> token (equiv. to [CLS])  (batch_size, hidden_dim)
        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits,1)
        for i,l in enumerate(binary_labels.tolist()):
            if(l == 0):
                logits.append(self.label_classifier_0(x[i,:]))
            elif(l == 0):
                logits.append(self.label_classifier_1(x[i,:]))
            else:
                logits.append(self.label_classifier_2(x[i,:]))
            
        logits = torch.stack(logits,0)
        return binary_logits, logits    # (batch_size, 2), (batch_size, 30)

    @autocast() 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        binary_logits, logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            dic = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 7: 1, 9: 1, 18: 1, 19: 1, 20: 1, 22: 1, 28: 1, 4: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 21: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 29: 2}
            binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits, binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0]*binary_loss + self.weight[1]+loss

            if(self.conf.train.rdrop):
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids)
        dic = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 7: 1, 9: 1, 18: 1, 19: 1, 20: 1, 22: 1, 28: 1, 4: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 21: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 29: 2}
        binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
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


class AuxiliaryModelWithRBERT(AuxiliaryModel):
    '''
        RBert에서 사용하는 entity token의 average값을 cls와 concat해서 binary_classification, label_classification에 사용한다.
    '''
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

        #cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        #entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        #entity type 토큰 FC layer
        #self.entity_type_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        
        self.binary_classifier = FCLayer(self.hidden_dim * 5, 2, 0.1)
        self.label_classifier_0 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_1 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.weight = [0.5, 0.5]
    
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

    def get_classifier_input(self, input_ids, attention_mask, token_type_ids=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # last hidden state
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

        #e3와 e4는 어떻게 할까?(fc layer 써야하나? e1,e1와 같은거로? 다른거로?-> nouse

        #concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)  # (batch_size, hidden_dim * 5)
        return concat_h  
        
    def process(self, input_ids, attention_mask, token_type_ids=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        # Extract outputs from the body
        x = self.get_classifier_input(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask) 
        # get classifier input for RBERT

        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits,1)
        for i,l in enumerate(binary_labels.tolist()):
            if(l == 0):
                logits.append(self.label_classifier_0(x[i,:]))
            else:
                logits.append(self.label_classifier_1(x[i,:]))
            
        logits = torch.stack(logits,0)
        return binary_logits, logits    # (batch_size, 2), (batch_size, 30)


    @autocast() 
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        binary_logits, logits = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            binary_labels = torch.tensor([i if i==0 else 1 for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0]*binary_loss + self.weight[1]+loss

            if(self.conf.train.rdrop):
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
            return loss, logits
        return logits


    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
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


class AuxiliaryModel2WithRBERT(AuxiliaryModelWithRBERT):
    '''
        RBert에서 사용하는 entity token의 average값을 cls와 concat해서 binary_classification, label_classification에 사용한다.
    '''
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

        #cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        #entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        #entity type 토큰 FC layer
        #self.entity_type_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        
        self.binary_classifier = FCLayer(self.hidden_dim * 5, 3, 0.1)
        self.label_classifier_0 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_1 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_2 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.weight = [0.5, 0.5]
        #self.weight2 = [0.8, 0.2]
        
    def process(self, input_ids, attention_mask, token_type_ids=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        # Extract outputs from the body
        x = self.get_classifier_input(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask) 
        # get classifier input for RBERT

        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits,1)
        for i,l in enumerate(binary_labels.tolist()):
            if(l == 0):
                logits.append(self.label_classifier_0(x[i,:]))
            elif(l == 1):
                logits.append(self.label_classifier_1(x[i,:]))
            else:
                logits.append(self.label_classifier_2(x[i,:]))
            
        logits = torch.stack(logits,0)
        return binary_logits, logits    # (batch_size, 2), (batch_size, 30)


    @autocast() 
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        binary_logits, logits = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            dic = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 7: 1, 9: 1, 18: 1, 19: 1, 20: 1, 22: 1, 28: 1, 4: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 21: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 29: 2}
            binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits.view(-1, 3), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0]*binary_loss + self.weight[1]+loss

            if(self.conf.train.rdrop):
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
            return loss, logits
        return logits


    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
        dic = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 7: 1, 9: 1, 18: 1, 19: 1, 20: 1, 22: 1, 28: 1, 4: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 21: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 29: 2}
        binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
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




class RECENTModel(nn.Module):
    '''
        subject_enitity, object_entity의 pair종류(head_id)에 따라 나올 수 있는 label을 제한해서 classification
        logit은 데이터로더에서 head_id의 정보를 받아 해당하는 것만 모아서 구하고 
        loss는 모든 classifier에 대해 계산 -> 정답 label이 classifier의 output 후보에 없으면 0으로 판별하는데 0에는 가중치를 적게 줌.
    '''
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]
        self.label_ids = {0: [0, 1, 2, 3, 5, 7, 19, 20, 28], 1: [0, 1, 2, 3, 5, 7, 19, 20, 28], 2: [0, 5, 7, 18, 19, 20, 22], 3: [0, 1, 2, 3, 5, 7, 19, 20],
                    4: [0, 1, 2, 3, 5, 7, 19, 20, 28], 5: [0, 1, 2, 3, 5, 7, 9, 20], 6: [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 27],
                    7: [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 29], 8: [0, 4, 6, 10, 11, 14, 15, 17, 21, 24, 25, 26, 27],
                    9: [0, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 21, 23, 24, 26, 27, 28, 29], 10: [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27, 29],
                    11: [0, 4, 6, 10, 12, 15, 21, 24, 25]}
        
        self.classifiers = nn.ModuleList()
        for i in range(12):
            self.classifiers.append(FCLayer(self.hidden_dim, len(self.label_ids[i]), 0.0))

    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        features = outputs[1]  # (batch_size, num_head_labels)
        logits = {}
        for i in range(12):
            head = self.classifiers[i]
            logits[i] = head(features)
        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        head_ids = attention_mask[:,0].clone().detach()
        attention_mask[:,0] = 1
        head_logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # first, scatter
        scattered = []
        bsz = input_ids.shape[0]
        for logit, ind in zip(head_logits.values(), self.label_ids.values()):
            z = torch.ones(bsz, self.num_labels,
                        device=logit.device, dtype=logit.dtype) * 1e-07
            ind = torch.tensor(ind, device=input_ids.device).repeat(bsz, 1)
            scattered.append(z.scatter(1, ind, logit))
        del z, ind, logit

        # second gather
        cat_logits = torch.cat(
            [tensor.view(bsz, 1, -1) for tensor in scattered],
            dim=1,
        )
        del scattered
        ind = head_ids.detach().view(-1, 1, 1)
        ind = ind.repeat(1, 1, self.num_labels)
        logits = cat_logits.gather(-2, ind).squeeze()
        del ind, cat_logits
        torch.cuda.empty_cache()

        has_labels = labels is not None and labels.shape[-1] != 1
        loss = None
        if has_labels:
            for ix in range(12):
                candidates = self.label_ids[ix]
                n_labels = len(self.label_ids[ix])
                weight = torch.tensor([0.05] + [1] * (n_labels - 1),
                                    device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
                label = torch.tensor([candidates.index(l) if l in candidates else 0 for l in labels], device='cuda')
                head_loss = loss_fct(head_logits[ix].view(-1, n_labels), label.view(-1))
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss
            return loss, logits
    
        return logits


## https://github.com/monologg/R-BERT/blob/master/model.py 사용
class FCLayer(nn.Module):       #fully connected layer
    '''
        RBERT emask를 위한 Fully Connected layer
        데이터 -> BERT 모델 -> emask 평균 -> FC layer -> 분류(FC layer)
    '''
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
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
class CustomRBERT(nn.Module):
    '''
        RBERT model
        데이터 -> BERT 모델 -> emask 평균 -> FClayerf
        -> (hidden size, e1, e2, e3, e4 mask concat) -> 분류(FC layer)
    '''
    def __init__(self, conf, new_vocab_size):
        super(CustomRBERT, self).__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name) 
        self.model.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        #cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        #entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        #entity type 토큰 FC layer
        #self.entity_type_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        #concat 후 FC layer
        self.label_classifier = FCLayer(
            self.config.hidden_size * 5,
            self.num_labels,
            conf.train.dropout,
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

    def process(self, input_ids, attention_mask, token_type_ids=None,
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

        #e3와 e4는 어떻게 할까?(fc layer 써야하나? e1,e1와 같은거로? 다른거로?-> nouse

        #concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # (hidden_states), (attentions)
    
    @autocast() 
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                e1_mask=e1_mask, e2_mask=e2_mask, e3_mask=e3_mask, e4_mask=e4_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if(self.conf.train.rdrop):
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
            
            outputs = (loss,) + outputs
        return outputs

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        outputs2 = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                e1_mask=e1_mask, e2_mask=e2_mask, e3_mask=e3_mask, e4_mask=e4_mask)
        logits2 = outputs2[0]
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits2.view(-1, self.num_labels)
        
        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss
