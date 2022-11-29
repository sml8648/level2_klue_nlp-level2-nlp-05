from transformers import AutoModel
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast
from model.model import FCLayer


class AuxiliaryModel(nn.Module):
    """
    binary label을 분류하는 binary_classification task를 추가한 모델
    binary_classifier에서 나온 logit(batch_size, 2)에 argmax를 취해 0인지 1인지 판별 후
    0이라면 label_classifier_0 레이어에, 1이라면 label_classifier_1 레이어에 넣어 각각 logit을 판단한다.
    그 후 binary_classifier의 loss와 label_classifier의 loss를 더해서 backpropagation -> binary_classifier과 classifier가 모두 학습.

    데이터 -> Pretrained 모델 -> binary_classifier(batch_size,2) -> label_classifier_0/label_classifier_1 -> add loss -> return
    """

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

        self.binary_classifier = FCLayer(self.hidden_dim, 2)
        self.label_classifier_0 = FCLayer(self.hidden_dim, self.num_labels, use_activation=False)
        self.label_classifier_1 = FCLayer(self.hidden_dim, self.num_labels, use_activation=False)
        self.weight = [0.5, 0.5]

    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Add custom layers
        x = outputs[1]  # take <s> token (equiv. to [CLS])  (batch_size, hidden_dim)

        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits, 1)
        for idx, label in enumerate(binary_labels.tolist()):
            if label == 0:
                logits.append(self.label_classifier_0(x[idx, :]))
            else:
                logits.append(self.label_classifier_1(x[idx, :]))

        logits = torch.stack(logits, 0)
        return binary_logits, logits  # (batch_size, 2), (batch_size, 30)

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        binary_logits, logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            binary_labels = torch.tensor([0 if l == 0 else 1 for l in labels], device="cuda")  # (batch_size,)
            binary_loss = loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0] * binary_loss + self.weight[1] + loss

            if self.conf.train.rdrop:
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids)
        binary_labels = torch.tensor([i if i == 0 else 1 for i in labels], device="cuda")
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
        return self.weight[0] * binary_loss + self.weight[1] * loss


class AuxiliaryModel2(nn.Module):
    """
    binary label을 분류하는 binary_classification task를 추가한 AuxiliaryModel에서 binary classifier label을 0,1,2 3개로 분류하도록 변경한 것
    0은 no_relation, 1은 org, 2는 per
    0이라면 label_classifier_0에 넣고 1일때는 label_classifier_1, 2일때는 label_classifier_2에 넣는다. 그 후는 AuxiliaryModel과 같음.

    데이터 -> Pretrained 모델 -> binary_classifier(batch_size, 3) -> label_classifier_0/label_classifier_1/label_classifier_2 -> add loss -> return
    """

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
        binary_labels = torch.argmax(binary_logits, 1)
        for i, l in enumerate(binary_labels.tolist()):
            if l == 0:
                logits.append(self.label_classifier_0(x[i, :]))
            elif l == 1:
                logits.append(self.label_classifier_1(x[i, :]))
            else:
                logits.append(self.label_classifier_2(x[i, :]))

        logits = torch.stack(logits, 0)
        return binary_logits, logits  # (batch_size, 2), (batch_size, 30)

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        binary_logits, logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            dic = {
                0: 0,
                1: 1,
                2: 1,
                3: 1,
                5: 1,
                7: 1,
                9: 1,
                18: 1,
                19: 1,
                20: 1,
                22: 1,
                28: 1,
                4: 2,
                6: 2,
                8: 2,
                10: 2,
                11: 2,
                12: 2,
                13: 2,
                14: 2,
                15: 2,
                16: 2,
                17: 2,
                21: 2,
                23: 2,
                24: 2,
                25: 2,
                26: 2,
                27: 2,
                29: 2,
            }
            binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits, binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0] * binary_loss + self.weight[1] + loss

            if self.conf.train.rdrop:
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids)
        dic = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            5: 1,
            7: 1,
            9: 1,
            18: 1,
            19: 1,
            20: 1,
            22: 1,
            28: 1,
            4: 2,
            6: 2,
            8: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            21: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 2,
            27: 2,
            29: 2,
        }
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
        return self.weight[0] * binary_loss + self.weight[1] * loss


class AuxiliaryModelWithRBERT(AuxiliaryModel):
    """
    RBert에서 사용하는 entity token의 average값을 cls와 concat해서 binary_classification, label_classification에 사용한다.
    """

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

        # cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        # entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        # entity type 토큰 FC layer
        # self.entity_type_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)

        self.binary_classifier = FCLayer(self.hidden_dim * 5, 2, 0.1)
        self.label_classifier_0 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_1 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.weight = [0.5, 0.5]

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

    def get_classifier_input(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
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

        # concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)  # (batch_size, hidden_dim * 5)
        return concat_h

    def process(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        # Extract outputs from the body
        x = self.get_classifier_input(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
        # get classifier input for RBERT

        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits, 1)
        for i, l in enumerate(binary_labels.tolist()):
            if l == 0:
                logits.append(self.label_classifier_0(x[i, :]))
            else:
                logits.append(self.label_classifier_1(x[i, :]))

        logits = torch.stack(logits, 0)
        return binary_logits, logits  # (batch_size, 2), (batch_size, 30)

    @autocast()
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        binary_logits, logits = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            binary_labels = torch.tensor([i if i == 0 else 1 for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0] * binary_loss + self.weight[1] + loss

            if self.conf.train.rdrop:
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
        binary_labels = torch.tensor([i if i == 0 else 1 for i in labels], device="cuda")
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
        return self.weight[0] * binary_loss + self.weight[1] * loss

    def get_classifier_input(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
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

        # e3와 e4는 어떻게 할까?(fc layer 써야하나? e1,e1와 같은거로? 다른거로?-> nouse

        # concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)  # (batch_size, hidden_dim * 5)
        return concat_h

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


class AuxiliaryModel2WithRBERT(AuxiliaryModel):
    """
    RBert에서 사용하는 entity token의 average값을 cls와 concat해서 binary_classification, label_classification에 사용한다.
    """

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

        # cls 토큰 FC layer
        self.cls_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        # entity 토큰 FC layer
        self.entity_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)
        # entity type 토큰 FC layer
        # self.entity_type_fc_layer = FCLayer(self.hidden_dim, self.hidden_dim, 0.1)

        self.binary_classifier = FCLayer(self.hidden_dim * 5, 3, 0.1)
        self.label_classifier_0 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_1 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.label_classifier_2 = FCLayer(self.hidden_dim * 5, self.num_labels, 0.1)
        self.weight = [0.5, 0.5]
        # self.weight2 = [0.8, 0.2]

    def process(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        # Extract outputs from the body
        x = self.get_classifier_input(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
        # get classifier input for RBERT

        logits = []
        binary_logits = self.binary_classifier(x)
        binary_labels = torch.argmax(binary_logits, 1)
        for i, l in enumerate(binary_labels.tolist()):
            if l == 0:
                logits.append(self.label_classifier_0(x[i, :]))
            elif l == 1:
                logits.append(self.label_classifier_1(x[i, :]))
            else:
                logits.append(self.label_classifier_2(x[i, :]))

        logits = torch.stack(logits, 0)
        return binary_logits, logits  # (batch_size, 2), (batch_size, 30)

    @autocast()
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        binary_logits, logits = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            dic = {
                0: 0,
                1: 1,
                2: 1,
                3: 1,
                5: 1,
                7: 1,
                9: 1,
                18: 1,
                19: 1,
                20: 1,
                22: 1,
                28: 1,
                4: 2,
                6: 2,
                8: 2,
                10: 2,
                11: 2,
                12: 2,
                13: 2,
                14: 2,
                15: 2,
                16: 2,
                17: 2,
                21: 2,
                23: 2,
                24: 2,
                25: 2,
                26: 2,
                27: 2,
                29: 2,
            }
            binary_labels = torch.tensor([dic[i.item()] for i in labels], device="cuda")
            binary_loss = loss_fct(binary_logits.view(-1, 3), binary_labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.weight[0] * binary_loss + self.weight[1] + loss

            if self.conf.train.rdrop:
                loss = self.rdrop(binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
            return loss, logits
        return logits

    def rdrop(self, binary_logits, logits, labels, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask, alpha=0.1):
        binary_logits2, logits2 = self.process(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, e3_mask, e4_mask)
        dic = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            5: 1,
            7: 1,
            9: 1,
            18: 1,
            19: 1,
            20: 1,
            22: 1,
            28: 1,
            4: 2,
            6: 2,
            8: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            21: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 2,
            27: 2,
            29: 2,
        }
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
        return self.weight[0] * binary_loss + self.weight[1] * loss

    def get_classifier_input(self, input_ids, attention_mask, token_type_ids=None, e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
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

        # e3와 e4는 어떻게 할까?(fc layer 써야하나? e1,e1와 같은거로? 다른거로?-> nouse

        # concat 후 분류
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)  # (batch_size, hidden_dim * 5)
        return concat_h

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
