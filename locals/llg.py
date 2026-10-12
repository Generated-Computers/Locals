"""llguidance as an mlx-lm logits processor: the classical constrained-decoding
stack (one grammar matcher per request, a token mask computed on the CPU each
step), used as the baseline backend."""


class LLGProcessor:
    def __init__(self, llg_tok, grammar):
        import llguidance
        import llguidance.mlx as llmlx
        import llguidance.numpy as llnp

        self.m = llguidance.LLMatcher(llg_tok, grammar)
        self.mask = llnp.allocate_token_bitmask(1, llg_tok.vocab_size)
        self.fill = llnp.fill_next_token_bitmask
        self.apply = llmlx.apply_token_bitmask
        self.first = True

    def __call__(self, tokens, logits):
        if self.first:
            self.first = False  # context ends with the last prompt token
        else:
            self.m.consume_token(int(tokens[-1].item()))
        self.fill(self.m, self.mask, 0)
        return self.apply(logits, self.mask)
