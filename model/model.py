from transformers import PreTrainedModel
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
# https://huggingface.co/docs/transformers/custom_models#sending-the-code-to-the-hub
# https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd

class CustomModel(nn.Module):
  def __init__(self, checkpoint, num_labels): 
    super(CustomModel, self).__init__() 
    self.num_labels = num_labels 
    self.model_name = checkpoint
    self.config = AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)

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
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)