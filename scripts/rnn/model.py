import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.encoder = nn.Embedding(len(config.vocab), config.hidden_size)

        self.lstm = nn.RNN(
            input_size=config.hidden_size, 
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers, 
            dropout=config.dropout,
            batch_first=True
        )

        # Linear decoder
        self.decoder = nn.Linear(config.hidden_size, len(config.vocab), bias=False)
        
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        x, _ = self.lstm(x)

        # Reference https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1765
        if self.config.tie_word_embeddings: x = x * (self.embed_dim**-0.5)
        
        # Decode the outputs
        x = self.decoder(x)  # (B, seq_len, d_model) -> (B, seq_len, d_output)

        return x