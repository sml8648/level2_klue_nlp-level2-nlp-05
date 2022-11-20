from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers import BertPreTrainedModel, AutoModel
from torch import nn
import torch
import model.loss as loss_module


class Model(nn.Module):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.model_name = conf.model.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

        self.model.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[conf.train.loss]

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits


class CustomModel(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModel, self).__init__()
        self.num_labels = 30
        self.model_name = checkpoint
        self.config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
        return loss, logits


## https://github.com/monologg/R-BERT/blob/master/model.py 사용
class FCLayer(nn.Module):       #fully connected layer
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