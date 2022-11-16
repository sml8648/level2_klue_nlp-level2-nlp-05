from transformers import PreTrainedModel
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
import model.loss as loss_module
from model.loss import FocalLoss

class Model(nn.Module):
  def __init__(self, args, conf, new_vocab_size):
    super().__init__()
    self.num_labels = 30
    self.model_name = conf.model.model_name
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
    
    self.model.resize_token_embeddings(new_vocab_size)
    # self.loss_fct = loss_module.loss_config[conf.train.loss]
    self.loss_fct = FocalLoss()
    
  def forward(self, input_ids=None, attention_mask=None, labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    loss = None
    if labels is not None:
      loss_fct = self.loss_fct
      # print(logits.view(-1, self.num_labels).size())
      # print(labels.view(-1).size())
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return loss, logits


class CustomModel(nn.Module):
  def __init__(self, checkpoint): 
    super(CustomModel, self).__init__() 
    self.num_labels = 30 
    self.model_name = checkpoint
    self.config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint, config=self.config)
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    # return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
    return loss, logits