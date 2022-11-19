from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from torch import nn
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
        return outputs


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
