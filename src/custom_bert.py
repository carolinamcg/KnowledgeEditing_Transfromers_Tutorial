"""PyTorch Custom BERT model."""

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F

#from transformers.utils import cached_path

logger = logging.getLogger(__name__)

#PRETRAINED_MODEL_ARCHIVE_MAP = {
#    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
#    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
#    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
#    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
#    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
#    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
#    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
#}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):

    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                            "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size # = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask #attention mask serves to set atten probs=0 for padding tokens

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer) #(bs, n_heads, n_tokens, dim_per_head)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #(bs, n_tokens, n_heads, dim_per_head)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) #((bs, n_tokens, config_hidden_size)

        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) #Wproj layer: merges the values of each concatenated head in the context layer
        hidden_states = self.dropout(hidden_states) #attention dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor) #RC + LayerNorm
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, att_score = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_score


class BertIntermediate(nn.Module):
    '''
    This is the first "half" of the transformer's FFN.
    The first layer projecting it to a higher dimensional space (normally, 768 -> 3072) and then the activation function.
    My interpretation/hyptothesis is that the output of this layer is measuring the weight/connection between each input 
    contextual token and each one of the 3072 contextual features ("nslot"). 
    The activation function eliminates the non-correlated ones (weight < 0)
    '''
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act #activation

    def forward(self, hidden_states, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # [batch, max_len, nslot]
        if tmp_score is not None: #When we are at the FFN module of a target layer (layer which neuron's weight we are modifying
            #to compute the attribution scores)
            #the gradients are computed wrt this tmp_score, a matrix of (batch, config.intermediate_size), where each row corresponds
            #to a modified version of the pre-trained fnn_weights of this layer, or the target/masked token.
            #Then we'll get the gradients wrt to each of these weights versions to compute the full attribution score
            hidden_states[:, tgt_pos, :] = tmp_score
        if imp_op == 'return': #operation = return the output for the wanted token (tgt_pos) and 
                                #contextual feature(s), aka Knowledge neuron(s) (pos)
            imp_weights = []
        if imp_pos is not None:
            for layer, pos in imp_pos:
                if imp_op == 'remove': #operation = erase the relation by setting knowledge neuron to 0 for the target token (tgt_pos)
                    hidden_states[:, tgt_pos, pos] = 0.0
                if imp_op == 'enhance': #operation = enhance this neuron's activation for the target token (tgt_pos) 
                    hidden_states[:, tgt_pos, pos] *= 2.0
                if imp_op == 'return':
                    imp_weights.append(hidden_states[0, tgt_pos, pos].item())

        if imp_op == 'return':
            return hidden_states, imp_weights
        else:
            return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        '''
        This is the second half of the FFN module: the second Linear layer projecting the "contextual feature-based token vectors"
        to the model's hidden size (normally, 768).
        Followed by the rest of the traditional transformer's architecture: Dropout -> RC -> LayerNorm
        '''
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        attention_output, att_score = self.attention(hidden_states, attention_mask)
        if imp_op == 'return':
            intermediate_output, imp_weights = self.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        else:
            intermediate_output = self.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        layer_output = self.output(intermediate_output, attention_output)
        if imp_op == 'return':
            return layer_output, intermediate_output, imp_weights
        else:
            return layer_output, intermediate_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, tgt_layer=None, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        '''
        im_pos: (layer_index, neuron_index)
        '''
        all_encoder_layers = []
        ffn_weights = None
        if imp_op == 'return':
            imp_weights = []
        for layer_index, layer_module in enumerate(self.layer):
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index] #take the target neurons idxs for this layer
            else:
                imp_pos_at_this_layer = None
            if imp_op == 'return':
                if tgt_layer == layer_index:
                    #ffn_weights=output of the intermediate (1st half) of this layer's FFN module
                    hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    #NOTE: tmp_score is not passed to the non_target layers
                    hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                imp_weights.extend(imp_weights_l)
            else:
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
        all_encoder_layers.append(hidden_states) #output of each layer
        if imp_op == 'return':
            return all_encoder_layers, ffn_weights, imp_weights
        else:
            return all_encoder_layers, ffn_weights #returns the ffn_weights of the target layer only


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        '''
        bert_model_embedding_weights = self.bert.embeddings.word_embeddings.weight
        So, self.decoder is a de_embeddings layer. It transposes the word_embeddings to transform
        the output vectors into a distibution over the vocabulary
        '''
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config) #dense, activation, layernorm

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), #hidden_size (768) = in_features
                                bert_model_embedding_weights.size(0), #n_vocab = out_features
                                bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PreTrainedBertModel(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, dir_path='', state_dict=None, cache_dir=None, *inputs, **kwargs):
        #if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
        #    archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        #else:
        archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        #try:
        resolved_archive_file = os.path.join(dir_path, archive_file) + "/" #cached_path(archive_file, cache_dir=cache_dir) 
        # except FileNotFoundError:
        #     logger.error(
        #         "Model name '{}' was not found in model name list ({}). "
        #         "We assumed '{}' was a path or url but couldn't find any file "
        #         "associated to this path or url.".format(
        #             pretrained_model_name,
        #             ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
        #             archive_file))
        #     return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        try:
            if os.path.isdir(resolved_archive_file):
                serialization_dir = resolved_archive_file
        except FileNotFoundError:
            logger.error("PLEASE, load the folder with the pre-trained model first!!!")
            return None
            # if resolved_archive_file is a link, not a dir. This doesn't work, so I removed it from happenning
            # Extract archive to temp dir
            # tempdir = tempfile.mkdtemp()
            # logger.info("extracting archive file {} to temp dir {}".format(
            #     resolved_archive_file, tempdir))
            # with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            #     archive.extractall(tempdir)
            # serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {}) #version of the loaded pre-trained model
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, tgt_pos=None, tgt_layer=None, tmp_score=None, imp_pos=None, imp_op=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if imp_op == 'return':
            encoded_layers, ffn_weights, imp_weights = self.encoder(embedding_output,
                                        extended_attention_mask,
                                        tgt_layer=tgt_layer,
                                        tgt_pos=tgt_pos,
                                        tmp_score=tmp_score,
                                        imp_pos=imp_pos,
                                        imp_op=imp_op
                                        )
        else:
            encoded_layers, ffn_weights = self.encoder(embedding_output,
                                        extended_attention_mask,
                                        tgt_layer=tgt_layer,
                                        tgt_pos=tgt_pos,
                                        tmp_score=tmp_score,
                                        imp_pos=imp_pos,
                                        imp_op=imp_op
                                        )
        sequence_output = encoded_layers[-1] #last layer's output
        if imp_op == 'return':
            return sequence_output, ffn_weights, imp_weights
        else:
            return sequence_output, ffn_weights


class BertForMaskedLM(PreTrainedBertModel):

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None, imp_pos=None, imp_op=None):
        '''
        tmp_score: ffn_weights of a specific target layer computed previously (with model(..., tmp_score=None)). We need this to compute
        the gradient wrt to these weights for the attribution scores
        '''
        if tmp_score is not None:
            batch_size = tmp_score.shape[0] #batch here is the number of approximate steps to calculate the attribution score
                        #each idx in batch corresponds to changes pre-trained weights of this target_layer for the target_pos (Masked token) 
            input_ids = input_ids.repeat(batch_size, 1)
            token_type_ids = token_type_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
        if imp_op == 'return':
            last_hidden, ffn_weights, imp_weights = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size), (n_imp_pos)
        else:
            last_hidden, ffn_weights = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size)
       
        last_hidden = last_hidden[:, tgt_pos, :]  # (batch, hidden_size) -> target pos is the masked one
        ffn_weights = ffn_weights[:, tgt_pos, :]  # (batch, ffn_size)
        tgt_logits = self.cls(last_hidden)  # (batch, n_vocab) -> distribution over vocab for the tgt_pos
        tgt_prob = F.softmax(tgt_logits, dim=1)  # (batch, n_vocab)

        ## NOTE: ffn_weights are not actual layer's weights, they're the output of the GELU(Linear(x)) in FFN module.
        ## the tmp_score with scaled weights that is used to compute gradients is also not actual weights, but this (scaled)
        ## output of a target transformer's layer, previously computed when infering model(input_ids)

        if imp_op == 'return':
            return imp_weights
        else:
            if tmp_score is None:
                # return ffn_weights at a layer and the final logits at the [MASK] position
                return ffn_weights, tgt_logits
            else:
                # return final probabilities and grad at a layer at the [MASK] position
                gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, tgt_label]), tmp_score) 
                # torch.unbind(tgt_prob[:, tgt_label]) -> tuple of 20 tgt_prob[tgt_label] tensor, one per each tgt_layer scaled weights
                # torch.autograd.grad(outputs, inputs) -> Computes and returns the sum of gradients of outputs with respect to 
                # the inputs, by taking the saved gradients along inference. This computes the gradient from the softmax prob of the target label (wrt) tmp_score

                # What happens when computing this grads?
                # for each output tensor value in torch.unbind(tgt_prob[:, tgt_label]), autograd is going to compute the gradients wrt
                # to tmp_score matrix (shape=(20, 3072)). As each output tensor value is only related in the graph to the corresponding indexed
                # row of tmp_score, the grad for each output, e.g.: torch.unbind(tgt_prob[:, tgt_label])[0], is going to be = to a matrix of
                # all 0's (cause the grads don't exist), except for gradient[0, :], that will have the gradients from output to tmp_score[0, :]
                # This is done for each if the 20 output tensors and then summed altogether in the same matrix.
                # So the final gradient matrix (shape=(20, 3072)) has, per row, the grads of the output to each correspondent scaled weight in tmp_score

                # Why is gradient[0, :] not all 0, since tmp_score[0, :] is set to all 0?
                # Because the final component in the chain rule is act'(tmp_score) = gelu'(tmp_score) wrt tmp_score, which is = 0.5, 
                # when tmp_score is 0
                return tgt_prob, gradient[0]