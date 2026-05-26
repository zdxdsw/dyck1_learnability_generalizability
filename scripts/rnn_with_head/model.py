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

        # Avoid exec() here. It's unsafe. Use setattr / getattr. 
        for supervision_tag in ["S2", "S3"]:
            if supervision_tag in config.task:
                vocab_name = f"out_{supervision_tag}_vocab"
                assert hasattr(config, vocab_name), f"{vocab_name} is required for {supervision_tag} head"
                setattr(
                    self,
                    f"{supervision_tag}_head",
                    nn.Linear(config.hidden_size, len(getattr(config, vocab_name)), bias=False),
                )
    
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            print(name, param.shape, param.numel())
        
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
        result = {'logits': self.decoder(x)}  # (B, seq_len, d_model) -> (B, seq_len, d_output)

        for supervision_tag in ["S2", "S3"]:
            if supervision_tag in self.config.task:
                head = getattr(self, f"{supervision_tag}_head")
                result[f"{supervision_tag}_logits"] = head(x)
        
        return result