import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.encoder = nn.Embedding(len(config.in_vocab), config.hidden_size)

        # no dropout if num_hidden_layers=1
        if config.num_hidden_layers == 1:
            self.model = nn.RNN(
                input_size=config.hidden_size, 
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers, 
                batch_first=True
            )
        else:
            self.model = nn.RNN(
                input_size=config.hidden_size, 
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers, 
                batch_first=True,
                dropout=config.dropout,
            )

        # Linear decoder
        self.decoder = nn.Linear(config.hidden_size, len(config.out_vocab), bias=False)
        
        self.apply(self._init_weights)

        # print shapes, param_count for each parameter matrix, then total param_count
        print("Encoder weight shape:", self.encoder.weight.shape, "param count:", self.encoder.weight.numel())
        for i in range(config.num_hidden_layers):
            print(f"RNN layer {i} weight_ih shape:", self.model.weight_ih_l0.shape, "param count:", self.model.weight_ih_l0.numel())
            print(f"RNN layer {i} weight_hh shape:", self.model.weight_hh_l0.shape, "param count:", self.model.weight_hh_l0.numel())
        print("Decoder weight shape:", self.decoder.weight.shape, "param count:", self.decoder.weight.numel())
        total_params = sum(p.numel() for p in self.parameters())
        print("Total param count:", total_params)

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
        x, _ = self.model(x)        
        x = self.decoder(x)  # (B, seq_len, d_model) -> (B, seq_len, d_output)

        return x