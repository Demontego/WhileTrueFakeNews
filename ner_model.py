import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification

class BiaffineLayer(torch.nn.Module):
    def __init__(self, deps_dim : int, heads_dim : int, output_dim : int): # output_dim == n_deps
        """
        Implements a biaffine layer from [Dozat, Manning, 2016].

        Args:
            deps_dim: the dimension of dependency states,
            heads_dim: the dimension of head_states,
            output_dim: the output dimension
        """
        super(BiaffineLayer, self).__init__()
        self.deps_dim = deps_dim
        self.heads_dim = heads_dim
        self.output_dim = output_dim
        self.kernel = torch.nn.Parameter(torch.zeros((deps_dim, heads_dim * output_dim)))
        self.first_bias = torch.nn.Parameter(torch.zeros((deps_dim, output_dim)))
        self.second_bias = torch.nn.Parameter(torch.zeros((heads_dim, output_dim)))
        self.label_bias = torch.nn.Parameter(torch.zeros((output_dim)))

        torch.nn.init.xavier_normal_(self.kernel)
        torch.nn.init.xavier_normal_(self.kernel)
        torch.nn.init.xavier_normal_(self.kernel)
        torch.nn.init.constant_(self.label_bias, 0.01)

    def forward(self, deps : torch.Tensor, heads : torch.Tensor) -> torch.Tensor:
        """
        Implements a biaffine layer from [Dozat, Manning, 2016].

        Args:
            deps: the 3D-tensor of dependency states,
            heads: the 3D-tensor of head states,

        Returns:
            `answer` the output 3D-tensor

        """
        input_shape = [x for x in deps.size()]
        first_input = deps.reshape((-1, self.deps_dim))
        # shape : (B * L, D1)
        second_input = heads.reshape((-1, self.heads_dim))
        # shape : (B * L, D2)

        first = torch.matmul(first_input, self.kernel)
        # shape : (B * L, D2 * H)
        first = first.reshape((-1, self.heads_dim, self.output_dim))
        # shape : (B * L, D2, H)
        answer = torch.einsum('bdh,bd->bh', first, second_input)
        # shape : (B * L, H)
        answer = answer + torch.matmul(first_input, self.first_bias)
        answer = answer + torch.matmul(second_input, self.second_bias)
        answer = answer + self.label_bias
        answer = answer.reshape(input_shape[:-1] + [self.output_dim])
        # shape : (B, L, H)
        return answer

class NERWrapper(torch.nn.Module):
    def __init__(self, n_tags, biaffine_size):
        super(NERWrapper, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base")
        self.bert_config = self.bert.roberta.config
        self.bert = self.bert.roberta
        self.res_dense = torch.nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.res_dense_1 = torch.nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)

        # self.biaffine_attention = BiaffineAttention(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.biaffine_layer = BiaffineLayer(self.bert_config.hidden_size, self.bert_config.hidden_size, biaffine_size)
        self.classifier = torch.nn.Linear(biaffine_size, n_tags)

        self.prelu = torch.nn.PReLU()
        self.prelu_1 = torch.nn.PReLU()

        torch.nn.init.xavier_normal_(self.res_dense.weight)
        torch.nn.init.xavier_normal_(self.res_dense_1.weight)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        torch.nn.init.constant_(self.res_dense.bias, 0.01)
        torch.nn.init.constant_(self.res_dense_1.bias, 0.01)
        torch.nn.init.constant_(self.classifier.bias, 0.01)


    def forward(self, input_ids, attention_mask):
        out = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.res_dense.forward(self.prelu(out))
        logits_res = self.res_dense_1.forward(logits)
        logits = logits + logits_res
        biaffine_res = self.biaffine_layer(out, logits)
        logits = self.classifier.forward(self.prelu_1(biaffine_res))
        return logits

    def __call__(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)
