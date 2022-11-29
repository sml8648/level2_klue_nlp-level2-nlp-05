from transformers import AutoConfig, AutoModel

from transformers.models.roberta.modeling_roberta import (
    RobertaEncoder,
    RobertaEmbeddings,
    RobertaPooler,
    RobertaPreTrainedModel,
)
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch


class RobertaEmbeddingsWithEntity(RobertaEmbeddings):
    """Roberta Embedding with entity embedding"""

    def __init__(self, config):
        super().__init__(config)
        self.entity_embeddings = nn.Embedding(2, config.hidden_size)
        self.config = config

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # entitiy_embeddings 추가 @GinToFim
        if entity_ids is not None:
            entity_embeddings = self.entity_embeddings(entity_ids)
            embeddings += entity_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaModelWithEntity(RobertaPreTrainedModel):
    """
    Roberta Model with entity embedding

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddingsWithEntity(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    ### 이게 필요한가??? ###
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    ### 이게 필요한가??? ###

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,  # entity id 추가
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaForSequenceClassificationWithEntity(nn.Module):
    """
    Roberta model with entity embedding for sequence classification + with LSTM classifier
    """

    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name

        self.config = transformers.AutoConfig.from_pretrained(self.model_name)
        self.roberta = RobertaModelWithEntity.from_pretrained(self.model_name)
        self.classifier = RobertaClassificationHeadLSTM(self.config, self.conf)  # 기존 classifier를 LSTM으로 수정
        self.roberta.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[self.conf.train.loss]

        # Initialize weights and apply final processing
        # self.post_init()

    @autocast()
    def process(self, input_ids=None, attention_mask=None, entity_ids=None):
        # Extract outputs from the body
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.process(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.conf.train.rdrop:
                loss = self.rdrop(logits, labels, input_ids, attention_mask, entity_ids)
            return loss, logits
        return logits

    def rdrop(self, logits, labels, input_ids, attention_mask, entity_ids, alpha=0.1):
        logits2 = self.process(input_ids, attention_mask, entity_ids)
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)

        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, conf):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.hidden_dim = config.hidden_size

        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadLSTM(nn.Module):
    """Head for sentence-level classification tasks with LSTM"""

    def __init__(self, config, conf):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.hidden_dim = config.hidden_size

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim // 2, num_layers=2, dropout=conf.train.dropout, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(self.conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim * 4, self.num_labels)

    def forward(self, features, **kwargs):
        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        lstm_output, (last_hidden, last_cell) = self.lstm(features)
        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        x = self.dropout(cat_hidden)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
