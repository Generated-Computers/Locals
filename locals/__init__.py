"""locals: exact constrained decoding for MLX on Apple silicon.

Compile a JSON schema, a regular expression or a tool-call format into a
grammar for a tokenizer, then decode under it: forced text is written without
running the model, the mask lives on the GPU, recursive schemas run on a small
stack, outputs stay valid under a token limit, and many requests decode
together in one batch.

    import locals

    rt = locals.load("mlx-community/Qwen3.5-4B-MLX-4bit")    # or the short names "qwen", "gemma"
    print(rt.generate("Extract the person from: ...", schema=SCHEMA).text)

    # or as a grammar library, in any decode loop
    grammar = locals.GrammarCompiler(tokenizer).compile_json_schema(SCHEMA)
    matcher = grammar.matcher()

    # or as a logits processor for mlx-lm
    from mlx_lm import generate
    generate(model, tokenizer, prompt, logits_processors=[locals.logits_processor(grammar)])

    # or as mlx-lm, faster: the same functions on a patched model
    from locals.lm import load, generate, batch_generate
    model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")   # any mlx-lm model; or: locals.patch(model)

Serving (OpenAI-compatible, continuous batching for concurrent agents):
``locals code DIR`` opens OpenCode on a project with everything set up;
``locals serve`` runs the server alone; see ``locals --help``.
"""

from .api import CompiledGrammar, GrammarCompiler, Matcher, Vocabulary, load, logits_processor
from .engine import ConstraintExhausted, Engine, GenResult
from .runtime import Runtime

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "load",
    "Runtime",
    "Engine",
    "GenResult",
    "ConstraintExhausted",
    "Vocabulary",
    "GrammarCompiler",
    "CompiledGrammar",
    "Matcher",
    "logits_processor",
    "patch",
    "BatchService",
    "Server",
]


def __getattr__(name):
    # Serving pieces are imported on first use so that ``import locals`` stays light.
    if name == "patch":
        from .lm import patch

        return patch
    if name == "BatchService":
        from .service import BatchService

        return BatchService
    if name == "Server":
        from .server import Server

        return Server
    raise AttributeError(f"module 'locals' has no attribute {name!r}")
