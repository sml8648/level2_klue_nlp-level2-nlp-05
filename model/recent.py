from transformers import AutoModel
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch


class RECENTModel(nn.Module):
    """
    subject_enitity, object_entity의 pair종류(head_id)에 따라 나올 수 있는 label을 제한해서 classification
    logit은 데이터로더에서 head_id의 정보를 받아 해당하는 것만 모아서 구하고
    loss는 모든 classifier에 대해 계산 -> 정답 label이 classifier의 output 후보에 없으면 0으로 판별하는데 0에는 가중치를 적게 줌.
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
        self.label_ids = {
            0: [0, 1, 2, 3, 5, 7, 19, 20, 28],
            1: [0, 1, 2, 3, 5, 7, 19, 20, 28],
            2: [0, 5, 7, 18, 19, 20, 22],
            3: [0, 1, 2, 3, 5, 7, 19, 20],
            4: [0, 1, 2, 3, 5, 7, 19, 20, 28],
            5: [0, 1, 2, 3, 5, 7, 9, 20],
            6: [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 27],
            7: [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 29],
            8: [0, 4, 6, 10, 11, 14, 15, 17, 21, 24, 25, 26, 27],
            9: [0, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 21, 23, 24, 26, 27, 28, 29],
            10: [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27, 29],
            11: [0, 4, 6, 10, 12, 15, 21, 24, 25],
        }

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

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        head_ids = attention_mask[:, 0].clone().detach()
        attention_mask[:, 0] = 1
        head_logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # first, scatter
        scattered = []
        bsz = input_ids.shape[0]
        for logit, ind in zip(head_logits.values(), self.label_ids.values()):
            z = torch.ones(bsz, self.num_labels, device=logit.device, dtype=logit.dtype) * 1e-07
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
                weight = torch.tensor([0.05] + [1] * (n_labels - 1), device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
                label = torch.tensor([candidates.index(l) if l in candidates else 0 for l in labels], device="cuda")
                head_loss = loss_fct(head_logits[ix].view(-1, n_labels), label.view(-1))
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss
            return loss, logits

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
