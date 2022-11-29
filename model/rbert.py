from transformers import AutoConfig, AutoModel
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast


class CustomRBERT(nn.Module):
    """
    RBERT model
    데이터 -> BERT 모델 -> emask 평균 -> FClayerf
    -> (hidden size, e1, e2, e3, e4 mask concat) -> 분류(FC layer)
    """

    def __init__(self, conf, new_vocab_size):
        super(CustomRBERT, self).__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        # cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        # entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        # entity type 토큰 FC layer
        # self.entity_type_fc_layer = FCLayer(self.config.hidden_size, self.config.hidden_size, conf.train.dropout)
        # concat 후 FC layer
        self.label_classifier = FCLayer(
            self.config.hidden_size * 5,
            self.num_labels,
            conf.train.dropout,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):  # 엔티티 안의 토큰들의 임베딩 평균
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, max_seq_len]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def process(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
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

        # concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # (hidden_states), (attentions)

    @autocast()
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, e1_mask=e1_mask, e2_mask=e2_mask, e3_mask=e3_mask, e4_mask=e4_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.conf.train.rdrop:
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)

            outputs = (loss,) + outputs
        return outputs

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        outputs2 = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, e1_mask=e1_mask, e2_mask=e2_mask, e3_mask=e3_mask, e4_mask=e4_mask)
        logits2 = outputs2[0]
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits2.view(-1, self.num_labels)

        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss


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
