# Constrained decoding and tool-call formats

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

Every grammar the engine decodes under goes through the same front end (to a regex, or regexes that call each other, then to a byte automaton). On the token-table path the automaton is then crossed with the vocabulary. With --walk that product is never built: a Metal kernel recomputes the current state's row at each step (`walk.py`, `runtime.Runtime.compile_walk`). A JSON schema, a regular expression or a model's tool-call format becomes a regular expression. A recursive schema instead becomes a set of regular expressions that call each other. That regex becomes a byte-level automaton. The automaton is then crossed with the tokenizer's vocabulary, which gives, for every automaton state, the tokens that are allowed and the state each one leads to.

The decode loop keeps the automaton state on the GPU and masks the logits with one gather. Text that the grammar forces is fed to the model without being sampled.

The theory behind this part, its proofs and its full measurements are in the paper [Breaking the Space Barrier and its Application to Language Model Inference](https://arxiv.org/abs/2610.09139) (Arip Asadulaev, 2026).

## Module map

| Module | Role |
|---|---|
| `grammar.py` | JSON Schema to a grammar of regexes (`compile_grammar`, `is_recursive`); `json_safe_schema` rewrites a schema so the regex compilers always produce valid JSON |
| `regex.py` | Regex parser, Thompson NFA over UTF-8 bytes, subset construction (`determinize`), trimming and Moore minimization (`trim_and_minimize`); entry point `compile_dfa` |
| `vocab.py` | `TokenVocab` (the byte string of every token id), `ByteTrie` (the vocabulary as a level-ordered trie), `FORMAT_TOKENS` |
| `constraint.py` | `compile_constraint`: DFA x trie gives the token table (`CompiledConstraint`); `forced_chains` |
| `forced.py` | `ForcedPlanner`: turns forced byte chains into canonical token spans; boundary cover rule; healing |
| `engine.py` | The `Controller` interface, `LocalsController` (a token-table constraint on the GPU), the `Engine` decode loops, `_TableCache` |
| `budget.py` | `budget_tables`: shortest completions per state, used for budget-aware masking |
| `walk.py` | `WalkConstraint` / `WalkController`: no token table; a Metal kernel walks every token through the automaton at each step |
| `pda.py` | Recursive schemas: `build_pda` (a deterministic pushdown automaton) and `token_tables` (token tables with a symbolic stack) |
| `pdactl.py` | `PDAController`: the pushdown state and its stack, kept on the GPU |
| `formats.py` | Tool-call formats per model family, `detect`, `thinking_style`, `reasoning_effort` mapping |
| `oai.py` | OpenAI messages to template messages, `reply_regex`, `parse_reply`, `ReplyStream` |
| `api.py`, `__init__.py` | The library API: `Vocabulary`, `GrammarCompiler`, `CompiledGrammar`, `Matcher`, `logits_processor`, `load` |
| `runtime.py` | `Runtime`: loads a model, compiles and caches constraints, `generate` |
| `fields.py`, `lookup.py` | Two decoding modes built on the same controllers: block-parallel object fields and prompt-lookup drafts |

## From a schema or a regex to a token table

### 1. JSON schema to regex

`Runtime.compile` and `GrammarCompiler.compile_json_schema` first pass the schema through `json_safe_schema`. This step exists because outlines-core builds a string's regex as `"` + pattern + `"` with the pattern unchanged. As a result:

- a top-level `|` in a `pattern` splits the whole value;
- a pattern that can match a quote, a backslash or a control character produces text that is not JSON.

`json_safe_schema` fixes this as follows:

- **Patterns.** Each `pattern` is rewritten (`_json_pattern`) into a grouped regex over the string's contents as JSON writes them: a quote becomes `\"` and a newline becomes `\n`. The decoded string is then exactly what the pattern matches. A pattern is a search, as JSON Schema reads it (`_searched`): a top-level branch without `^` may have text before its match, one without `$` text after. zod's `startsWith("https://")` is `^https:\/\/`; read as a whole string, the only value it allowed was `"https://"`.
- **Formats.** A `format` outlines-core has no regex for (it knows `date-time`, `date`, `time`, `uuid`, `email` and `uri`) is dropped as an annotation, which is how JSON Schema reads formats by default; `ipv4` becomes a pattern. Pydantic's `Path`, `SecretStr` and `IPv4Address` were a 400 ("Format path is not supported by Outlines").
- **Unconstrained properties.** A property whose schema is `{}` or `true` becomes an explicit `anyOf` of the six JSON types (`_ANY`).
- **`allOf`.** Its parts are merged into one schema (`_merge_all_of`): properties and `required` joined, the tighter of two bounds, a `$ref` part resolved, a property in two parts as the `allOf` of both. outlines-core wrote the parts' JSON one after the other (`{"a":1}{"b":"x"}`), and ignored an `allOf` next to `properties`. Parts that cannot be joined (no type in common, two different patterns) raise `GrammarError`.
- **Refusals.** Property names that contain quotes, backslashes or control characters are refused with `GrammarError`. So are contradictory bounds (`minItems` > `maxItems`, and the same check for lengths and property counts).

`is_recursive` then picks the path:

- **Non-recursive schemas** (no `$ref` on a cycle) become one regex through `schema_regex(schema, whitespace)`: outlines-core's `build_regex_from_schema`, with the integer bounds it leaves out put back. `whitespace` defaults to `""`, which means compact output.
  - outlines-core ignores `minimum` and `maximum` (a rating from 1 to 5 could come out as 7). `schema_regex` writes each bounded integer as a placeholder string, which outlines-core copies as it is, and puts the range's own regex (`integer_regex`) in its place. Draft 4's boolean `exclusiveMinimum`/`exclusiveMaximum` are read too; a bound past JavaScript's safe integers (zod writes ±9007199254740991 for every int) counts as none, since its regex would cost about 50 automaton states a field. Bounds on `number` are not enforced.
- **Recursive schemas** go to the pushdown path (see below).

The server compiles `response_format: {"type": "json_object"}` with its own whitespace, `JSON_OBJECT_WS = r"( |\n *)?"` (a space, or a newline and an indent). In compact form, a model that habitually writes `{` and a newline was pushed into keys such as `"/a"`. Schemas stay compact because their keys are forced.

### 2. Regex to a byte DFA

`compile_dfa(pattern, max_states=200_000, n_calls=0)` runs `parse`, `compile_nfa`, `determinize` and `trim_and_minimize` in that order. The supported syntax is the subset that JSON-schema compilers emit:

| Construct | Notes |
|---|---|
| Literals; `\n \t \r \f \v \0`; `\xNN`, `\uNNNN`, `\UNNNNNNNN` | `\U` values above U+10FFFF are accepted only as nonterminal calls (`CALL_BASE` and up) |
| `[...]`, `[^...]`, ranges | `\b` inside a class means backspace |
| `.` | Any code point except newline |
| `\d \w \s`, `\D \W \S` | ASCII only |
| `( )`, `(?: )`, `\|` | Lookaround and named groups raise `RegexError` |
| `* + ? {m} {m,} {m,n} {,n}` | `{,n}` means `{0,n}`; a lazy `?` is accepted (same language); counts above `MAX_REPEAT = 10_000` are rejected |
| `^ $` | Ignored inside the regex, which is matched whole (a `response_format` regex, `guided_regex`); a JSON Schema `pattern`'s top-level `^` and `$` say where it may not be searched (`_searched`) |

How the automaton is built:

- **Bytes, not characters.** Code points are compiled to UTF-8 byte sequences (`utf8_sequences`, with surrogates excluded), so the automaton reads the same bytes as the vocabulary trie.
- **State cap.** Subset construction stops fail-closed at `max_states` and raises `BudgetExceeded`, which is a request error.
- **Trimming.** `trim_and_minimize` drops every state that cannot reach acceptance (`coreachable`). Every remaining state can therefore still finish. A pattern that matches nothing raises.
- **Minimization.** Moore refinement runs on deduplicated byte columns, with rows hashed to integers (`_row_classes`). The start state is renumbered to 0.

### 3. The vocabulary as bytes

`TokenVocab.from_hf(hf_tokenizer, n_model, eos_ids)` gives every token id its byte string, or `None` for special and padding ids.

- **Byte-level or SentencePiece.** `_is_byte_level` reads the tokenizer's decoder to choose between GPT-2 byte-level BPE (Llama 3, Qwen, Ministral 3) and SentencePiece pieces (`▁` for a space, `<0xNN>` byte fallback: Gemma, Mistral 7B). Without a decoder it counts `Ġ` against `▁` pieces.
- **Special tokens get no bytes, except those in FORMAT_TOKENS (below).** This includes Llama 3's control tokens, which are flagged special only in `added_tokens_decoder`. A grammar therefore can never write a role header.
- **Added tokens keep their text.** Added tokens that are not special, such as Qwen's `<tool_call>` and `</think>`, keep their surface text. The tokens in `FORMAT_TOKENS` keep their text even where the tokenizer marks them special, because a model emits these control tokens as one token, never letter by letter:

```
[TOOL_CALLS]  [ARGS]  [CALL_ID]  <|python_tag|>  <tool_call>  </tool_call>  <think>  </think>
<|tool_call>  <tool_call|>  <|"|>  <|channel>  <channel|>  <|think|>  [THINK]  [/THINK]
<function  </function>  <param  </param>
```

The last four are MiniCPM5's. It writes its calls with special tokens; without these entries a call could only be spelled letter by letter, and the model then answered in prose instead of calling.

`ByteTrie.build(vocab)` stores the trie level by level:

- `parent[d][j]` and `byte[d][j]` give a node's parent and its last byte;
- `tok_level` and `tok_index` locate each token's node.

### 4. DFA x trie: the token table

`compile_constraint(pattern, vocab, trie, max_dfa_states=200_000, work_budget=None)` returns a `CompiledConstraint`:

| Field | Shape | Meaning |
|---|---|---|
| `dfa` | | The minimized byte DFA |
| `next_state` | (n_model, Q) | The state after each token from each state; `DEAD` (-1) if the token is not allowed |
| `allowed` | (Q, n_model) bool | The mask per state; EOS ids are allowed exactly at accepting states |
| `forced_bytes` | Q entries | The maximal forced byte chain from each state |
| `only_eos` | (Q,) | Accepting states with no byte successors; the controller reports `finished()` there |
| `stats` | | `dfa_states`, `product_work`, `allowed_min`/`allowed_max`, `forced_states`, timings |

`token_table` handles all Q start states in one pass over the trie, which the comments call the universal-source lift:

- Level d of the trie carries an array of shape (nodes at depth d, Q) holding DFA states.
- Each level is computed from its parent level with one gather on the byte column.
- Rows whose states are all dead are pruned before descending.
- Tokens that end at level d copy their row out.
- Work is counted and checked against `work_budget` (`WorkBudgetExceeded`).
- During the pass, states are int16 while Q+1 < 2**15 (int32 otherwise). The returned `next_state` is int32, with DEAD = -1.

**Memory guard.** Before the table is built, its memory is estimated at about 7 bytes per (state, token), including its copies. Above `LOCALS_MAX_TABLE_GB` (default 3), compilation fails with a message that suggests a smaller schema or `--walk`.

- The comment's example: a string with `maxLength` 1000 (10k states) on a 248k-token vocabulary would need 17 GB and end the server with a Metal error.
- `runtime.py` notes that a 40-state regex already takes 60-100 MB on Qwen3.5's 248k vocabulary.

### 5. On the GPU: `LocalsController`

`LocalsController(constraint, planner, use_forced=True, canonical=False)`. Note that `Compiled.controller` and the server both pass `canonical=True`. The device holds three tables:

- `allowed`: the (Q, V) mask;
- `_gpu_T()`: `next_state`, with `DEAD` replaced by the start state (unreachable under the mask), in int16 when Q < 2**15;
- `_gpu_cover()`: the minimum token length per state (the boundary rule, see below).

The state itself is a device scalar (`gpu_state`):

- `gpu_mask(qg)` is `allowed[qg] & (tok_len >= cover[qg])`;
- `gpu_next(y, qg)` is `T[y, qg]`.

Because the next state is computed on the device from the sampled token, the next step's mask needs no host round trip. The engine can therefore build step n+1 while step n runs. The host mirrors the state in `accept`.

**Sparse head.** When at most `SPARSE_MAX = 4096` tokens are allowed at a state, `sparse_ids` returns their ids. The single-stream `Engine` (the pipelined and lookup loops, and only without a custom `sampler`) then computes the LM head for those rows only (`kernels.head_rows`), which skips most of the 4-bit head's weight traffic. The server's batch engine always computes the dense head.

**Caching.**

- Per-constraint device tables live in `_TABLES`, a `_TableCache` keyed by the `id` of their owners. Entries hold only weak references, so an entry goes when its constraint goes.
- Per-state entries (`sparse`, `budget_ids`, `pda_sparse`) are kept least recently used under `LOCALS_TABLE_CACHE_GB` (default 0.5).
- `Runtime` keeps compiled constraints least recently used: at most `LOCALS_MAX_COMPILED` (16) of them and `LOCALS_COMPILED_GB` (1.5) of tables.
- `Runtime` builds the device tables at compile time (`_prepare_device_tables`), because building them lazily costs about 0.25 s per schema inside the first request.

| Limit or variable | Default | Where |
|---|---|---|
| DFA states (table path) | 200_000 | `compile_dfa`, `compile_constraint`, `GrammarCompiler(max_dfa_states=...)` |
| DFA states (walk path) | 400_000 | `walk.compile_walk` |
| `MAX_REPEAT` | 10_000 | `regex.py` |
| `LOCALS_MAX_TABLE_GB` | 3 | `constraint.compile_constraint` |
| `LOCALS_TABLE_CACHE_GB` | 0.5 | `engine._TableCache` |
| `LOCALS_MAX_COMPILED` | 16 | `Runtime` |
| `LOCALS_COMPILED_GB` | 1.5 | `Runtime._cput` |
| `SPARSE_MAX` | 4096 | `LocalsController`, `WalkController`, `PDAController` |

## The walk kernel (`--walk`)

`locals serve --walk` (also a flag of the app's commands) replaces the token table with one Metal kernel per decode step, named `locals_dfa_walk`.

**What the kernel does.** It launches one thread per (vocabulary token, batch row). Each thread walks its token's bytes through the byte DFA from the row's current state and writes the state it reaches (int16, -1 for dead). That row is both the mask and the transition:

- the mask is `reached != -1`, or EOS at accepting states, combined with the cover rule;
- the next state is `row[y]`.

The state stays on the device, so decoding is as pipelined as with a table.

**Why.** The module docstring gives the reasoning: a configuration is reached by one path, so nothing about it needs storing; it can be recomputed. Memory is the automaton (256 x |Q| int16 entries) plus every token's bytes packed with offsets (`PackedVocab`), whatever the grammar's size. There is no compile pass beyond the automaton.

Details:

- **Compiling.** `compile_walk(rt, regex, max_dfa_states=400_000)` returns `(WalkConstraint, ForcedPlanner)`. `Runtime.compile_walk` caches it by regex. Only regular constraints are supported.
- **State limit.** States must fit int16 (< 2**15). In the batch engine this applies to all walk automata combined, which share one table with global state 0 as the free state.
- **Sparse head.** At a state with at most 4096 allowed tokens, `sparse_ids` computes the ids once with a CPU walk (`lazy.token_rows` with one root).
- **No budget masking, no healing by default.** `WalkController` inherits the no-op budget hooks, and its `heal` defaults to `False`. The batch engine's walk path applies no budget mask either.
- **No mixing.** A batch cannot mix table and walk controllers (`NotImplementedError`). With `--walk`, the server uses walk controllers for every schema and regex request.
- **Typed tools.** `--typed-tools` is meant to be used with it: typed tool calls produce automata with thousands of states.

## Recursive schemas: the pushdown automaton

### Grammar (`grammar.py`)

`compile_grammar(schema)` returns a `Grammar` with these parts:

- `rules`: one regex per nonterminal;
- `names`: the definition paths, where `"#"` is the root;
- `root`: the root nonterminal. If the root is a bare `$ref` to a recursive definition, that definition is used directly.

Definitions that lie on a cycle of the reference graph become nonterminals; `_cyclic_refs` finds them with Tarjan's algorithm. Every other `$ref` is inlined, and nesting deeper than 64 raises.

A call of nonterminal i is written in a regex as the escape `\U` of `CALL_BASE + i`, where `CALL_BASE = 0x200000` lies above Unicode. `regex.py` turns that escape into one edge on symbol `256 + i`, a DFA column past the 256 byte columns.

The compiler assembles objects, arrays and alternatives itself. Leaf types (strings, numbers, enums, consts, formats, length bounds) reuse outlines-core's regexes. Output is compact, with no whitespace.

- **Objects.** Properties come in schema order. Optional properties before the first required one are `(p,)?`, and later ones are `(,p)?`. If every property is optional, the object is any non-empty subset in schema order, or empty. `additionalProperties` other than absent or `false` raises.
- **Arrays** need `items`, and honour `minItems`/`maxItems`.
- **`anyOf` and `oneOf`** both compile to an alternation. `allOf` arrives merged by `json_safe_schema` (below).
- **A `type` list** is split per type, keeping only that type's keywords (`_for_type`).
- **Not supported:** `{}`/`true` values, and references inside a leaf type.

### Pushdown automaton (`build_pda`)

Each rule becomes a DFA with `n_calls` extra columns, and the DFAs are concatenated into one table with offsets. The machine works as follows:

- **Call.** Taking a call edge pushes the caller's return state and continues at the callee's start. Call edges are resolved into `eff_next`/`eff_push`, so the call is taken on the callee's first bytes.
- **Eager pop.** A callee that reaches a terminal state (accepting, with no successors, such as after `}` or `]`) pops at once.
- **Lazy pop.** An accepting state that still has successors (a number that could continue) pops only when the next byte is not its own.

`build_pda` raises `GrammarError` wherever the machine would not be deterministic (LL(1)):

- a called nonterminal starts with a call, or matches the empty string;
- a return state accepts (end of output would depend on the stack);
- a byte both continues a state and starts a callee ("ambiguous at state");
- an accepting, non-terminal state of a callee shares a byte with a follower of its return states ("ambiguous return").

### Token tables with a symbolic stack (`token_tables`)

This is the same trie pass as for a DFA, but rows carry a symbolic stack: pushes made inside a token are matched by pops inside the same token. Per token, at most `MAX_PUSH = 3` pushes and `MAX_POP = 2` real pops (pops below the stack the token started on) are resolved. Tokens that need more are masked. The module notes this loses nothing, because `}` exists as a token of its own.

Each (token, state) entry falls into one of three cases:

| Case | Entry |
|---|---|
| A: no real pop | `next_state[t, q] = q'`; `aux = pattern id \| pops << 8` |
| B: ends exactly at its single real pop | `next_state = RET` (-2): continue at the popped return state |
| C: continues past a real pop, or pops twice | Depends on the popped return states: stored in `ctx_next`/`ctx_aux`, indexed by token, state and a context made of the top `MAX_POP` stack entries ((C+1)**`MAX_POP` contexts) |

Limits and rules:

- Distinct push sequences (patterns) are capped at `MAX_PATTERNS = 255`, the 8 bits of `aux`.
- Tables are int16, so the state count must satisfy Q+2 < 2**15.
- Tables over 3 GB are refused.
- EOS is allowed at accepting states with an empty stack.

### `PDAController` (`pdactl.py`)

The device configuration is an int32 vector `[q, ptr, stack[0..63], S]` (`STACK_DEPTH = 64`). `S` is the sum of the shortest byte completions of the return states on the stack, used for the budget.

- **`gpu_mask`** reads the columns at `q`, with the context override for case C. A token is allowed when its pops fit (`pops <= ptr`, or a `RET` that pops exactly to the root's end) and its pushes stay within `STACK_DEPTH`. EOS is allowed where the state accepts and `ptr == 0`.
- **`gpu_next`** applies the pops, writes the pushed pattern and updates `S`, all on the device.
- **Depth limit.** Pushes past `STACK_DEPTH` are masked. If no token is allowed at all, `accept` raises `ConstraintExhausted`. The engines stop with `finish="exhausted"`, and the server reports `finish_reason` `"length"`.
- **`q`** is `(state, tuple(stack))`, hashable, used as the key for caches and statistics.

**Where recursive schemas run.** They work through `Runtime.compile` / `Runtime.generate` and through `GrammarCompiler.compile_json_schema`. Without `--walk`, the server's batch engine refuses them: the request ends with `"JSON schemas that refer to themselves ($ref on a cycle, as in a tree) are not supported by the server"`, before anything is compiled. With `--walk`, the schema goes to `Runtime.compile_walk`, which does not check for recursion and passes it to outlines-core's `build_regex_from_schema`. The batch engine itself also raises `NotImplementedError` for a `PDAController`.

## Forced spans

### What is forced

- **DFA.** `forced_chains(dfa)` follows, from each state, the states that are not accepting and have exactly one live byte. The bytes along the way are the forced chain. The DFA is trimmed, so chains are finite.
- **PDA.** `PDA.forced_bytes(q, stack)` follows across calls and pops on the real stack. It stops where EOS is an option and is capped at 4096 bytes.

### Canonical token split

Many token sequences spell the same bytes. `ForcedPlanner(hf_tokenizer, vocab, constraint, heal=True, commit_stable_tail=True, boundary_rule=True)` emits the one the tokenizer itself would produce:

1. It cuts the forced string at pre-tokenizer boundaries (`pre_tokenizer.pre_tokenize_str`); for GPT-style BPE, tokens never cross them.
2. It encodes the committed segment in one tokenizer call (`add_special_tokens=False`). The segment runs from the start of the span, or of the healed previous token, to the last committed block boundary. It then checks that the tokens' bytes reproduce the segment exactly. If they do not, there is no plan.
3. Added non-special tokens are treated as blocks of their own, because the tokenizer matches them before pre-tokenization. A block that is an added token is always stable.

Only the last block can be unstable, because bytes after it may merge into it. `_tail_is_stable` tries every byte the constraint allows next: ASCII bytes directly, and non-ASCII bytes through representatives (`é — 中 € ٣` and a non-ASCII space). The last block is committed only if none of them merges. A block ending in whitespace is never treated as stable. An unstable block is held back.

### Boundary cover rule (`canonical`)

`cover_rule(q)` applies when the remaining forced bytes at `q`:

- form one pre-token block,
- do not end in whitespace, and
- are exactly one vocabulary token whose bytes equal the block.

The next sampled token must then be at least that many bytes long. This rules out non-canonical prefixes such as `":` followed by `"To`. The device mask applies it as `tok_len >= cover[q]`.

The "bytes equal the block" condition matters for SentencePiece: such a tokenizer encodes "rue" as "▁rue", and a 3-byte cover would then allow no token at all.

### Healing

When a span is planned, the last sampled token has not been fed to the model yet. If the canonical split merges that token with the start of the span, the planner re-encodes the token's bytes together with the span and sets `replaces_prev`. The engine then feeds the re-encoded tokens instead of the sampled one, so no KV rollback is needed.

`ForcedPlan` has the fields `tokens`, `replaces_prev`, `q_end`, `committed_forced_bytes` and `cover`. Plans are cached per (previous token, configuration).

### How the engines feed them

- **`Engine._generate_pipelined`.** The sampled token and its span go into one forward pass; the module docstring notes that m known tokens cost one weight-streaming pass instead of m decode steps.
  - `may_force()` (backed by `_may_force`) says whether any token allowed now can lead to a forced span.
  - If none can, step n+1 is built and committed before waiting on step n.
  - If one can, the host snapshots the cache (`_snapshot`), waits for the token, and then either commits the prebuilt step or restores the snapshot and feeds the span.
  - States that rarely force (seen at least 3 times, forced in under 25% of visits) run the next body early anyway (`EARLY_BODY`).
- **`HybridBatchEngine` drip mode** (the default for the server). A span is queued and fed one token per step, so every step is a uniform (B, 1) batch. Spans are planned without healing, because the sampled token has already been fed.
- **`Matcher.jump_forward()`** returns the span, planned without healing, for use in your own decode loop.
- **Turning it off.** `forced=False` (`Runtime.generate`, `Matcher`) or `use_forced=False` (controllers) disables spans. The server reports `forced_tokens` in `usage.locals`.

## Budget-aware masking

**The problem.** The engines stop right after a sampled token once `len(out) >= max_tokens`, and forced tokens count toward `len(out)`. Without care, an output cut there stops in the middle of a value.

**The tables.** `budget_tables(c, planner, cover, tok_len)` computes two values per DFA state:

- `need(q)`, the smallest remaining budget from which `q` can still finish:
  - 1 at an accepting state (sample EOS);
  - 0 after a forced span that ends in a state with no successors;
  - otherwise the minimum cost over the tokens allowed at `q`, with the cover rule applied.
- `cost(q')`, the budget a sampled token must leave when it leads to `q'`:
  - 1 if `q'` accepts;
  - otherwise `1 + F(q') + need(E(q'))`, where `F` is the length of the forced span planned after `q'` (without healing) and `E` is the state after that span.

These are found with a Bellman-Ford fixpoint over the token-step graph.

**The mask.** With `r` tokens left, the mask keeps the tokens with `cost(q') <= r`, plus EOS. By induction, `need(q) <= r` holds at every sampling point. An output cut by `max_tokens` therefore always ends in an accepting state, and spans never push an output past `max_tokens`.

**When it is skipped.**

- Where `need(q) > r` already holds (`max_tokens` below the shortest valid output), the budget mask is dropped. That is the behaviour without it.
- `budget_active(r)` skips the work when `r >= max_cost` and no allowed token leads to a state that cannot finish (`has_inf`).

**Healing near the end.** Healing can change a span's length. Once `r <= heal_threshold`, the engines plan spans without healing (`heal_ok`). The threshold is `max_cost` plus the longest token (in bytes) plus the longest forced chain.

**PDA variant.** The bound is the shortest byte completion (`_byte_need`), since every token covers at least one byte. It decomposes over the stack: `need(q, stack) = bytes(q) + Σ bytes(r)` over the return states. A token costs `1 + need(after)`. The mask is active when `r < need + (MAX_PUSH + 1) * bn_max + MARGIN` (with `MARGIN = 16`), which allows for the device being one token ahead of the host.

**Where it applies.** `Engine` (it passes `r` at every step), `Matcher.next_token_mask(tokens_left=...)`, and `LocalsController` rows in the batch engine. It does not apply in `WalkController` or in `logits_processor`.

## Tool-call formats

### Detection

`formats.detect(tokenizer)` reads the chat template through `chat_template`. A dict of named templates resolves to `tool_use`, then `default`, then the first entry. `detect_from_template` then checks, in this order:

```
"<function=" and "<parameter="                 -> qwen_xml
'<function name="' and '<param name="'         -> minicpm_xml
"<|tool_call_start|>"                          -> lfm2_python
"[TOOL_CALLS]": with "[ARGS]"                  -> mistral_v11
                with "tool_results"            -> mistral_legacy
                otherwise                      -> mistral
"<|tool_call>call:"                            -> gemma4
"<arg_key>" and "<arg_value>"                  -> flint_xml
"<tool_call>"                                  -> hermes
'"parameters"' and ("ipython" or "<|python_tag|>") -> llama3_json
anything else                                  -> fallback
```

The server logs `tool calls: NAME; thinking: STYLE` when the engine loads. `/v1/models` reports `tool_call_format` and `parallel_tool_calls` (the format's `multi`).

### Formats

```
qwen_xml     <tool_call>\n<function=NAME>\n<parameter=P>\nV\n</parameter>\n</function>\n</tool_call>
hermes       <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>
llama3_json  [<|python_tag|>]{"name": "NAME", "parameters": {...}}          (the whole reply)
mistral      [TOOL_CALLS] [{"name": "NAME", "arguments": {...}}, ...]
mistral_v11  [TOOL_CALLS]NAME[ARGS]{...}  (repeated per call, after the turn's text)
minicpm_xml  <function name="NAME"><param name="P">V</param>...</function>   (V with <, & or \n in <![CDATA[...]]>)
lfm2_python  <|tool_call_start|>[NAME(P='text', N=3, O={"k": 1}), ...]<|tool_call_end|>
gemma4       <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>
flint_xml    <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>   (V as it is if a string, else JSON)
fallback     the hermes form; the tools are described in the system prompt
```

| `name` | Families (formats.py) | Several calls per reply | Arguments in the grammar | `replay_raw` |
|---|---|---|---|---|
| `qwen_xml` | Qwen3.5, Qwen3-Coder | yes, separated by `\n` | Parameter names `[A-Za-z0-9_.\-]+` or one of the offered tools' own (`$select`, `größe`), values any text without `</parameter>` (typed with `--typed-tools`) | yes |
| `hermes` | Qwen2.5, Qwen3, Hermes | yes, separated by `\n` | `json_args_regex()` | yes |
| `llama3_json` | Llama 3.1-3.3 | no | `json_args_regex()` under `"parameters"` | yes |
| `mistral` | Mistral 7B v0.3 | yes, in one JSON list | `json_args_regex()` | no |
| `mistral_legacy` | Older Mistral template (roles `tool_calls`/`tool_results`) | yes | as `mistral` | yes |
| `mistral_v11` | Ministral 3, Devstral, Mistral Small 3.2+ | yes | `json_args_regex()` | no (inherited from `mistral`) |
| `minicpm_xml` | MiniCPM5 | yes, separated by `\n` | Parameter names `[A-Za-z0-9_.\-]+` (`PARAM_RE_`); value a CDATA block or `[^<&\n]*` | yes |
| `lfm2_python` | LFM2.5 | yes, in one Python-style list | Single-quoted string, number, `True/False/None`, JSON object or JSON list | yes |
| `gemma4` | Gemma 4 | yes | Keys `[A-Za-z_][A-Za-z0-9_.\-]*`; values `<\|"\|>`-quoted strings, numbers, booleans, `null`, nested objects and lists three levels deep | no |
| `flint_xml` | Recursive's Flint | yes, back to back or separated by `\n` | Keys `[A-Za-z0-9_.\-]+`, values any text without `</arg_value>` (typed with `--typed-tools`) | yes |
| `fallback` | Templates without tools (Gemma 3, ...) | no: a model without tool training chains dependent calls blindly | `json_args_regex()` | yes |

`json_args_regex(space=r"[ ]?")` accepts any JSON object nested four levels deep (objects of lists of objects of lists, as Hermes Agent's clarify tool needs). It allows JSON string escapes including `\uXXXX`, JSON numbers, and an optional single space between JSON tokens, because templates write `", "` and `": "`. Its docstring puts it at about 700 automaton states. Arguments nested deeper cannot be written in these formats: Claude Code's `AskUserQuestion` puts its option objects at level five, so a model in one of them writes the options as strings. Five levels would take 1420 states, a 2.5 GB token table per tool set; Qwen's XML and MiniCPM's values have no such limit; an LFM2 value nests four levels, one more than these formats' arguments, which is enough for `AskUserQuestion`; and `--walk` needs no table.

In every format, tool names are restricted to the offered names by `_alts(names)`, with `[A-Za-z0-9_.\-]+` when no names are given.

### The reply grammar

`oai.reply_regex(tools, tool_choice="auto", thinking=False, typed=False, fmt=None, preamble=True, parallel=True)` handles Qwen's XML calls itself and delegates every other format to `fmt.reply_regex`. The grammar is free text that cannot start a call, followed by calls whose fixed parts are literals. Those literals are forced spans, so a reply always parses.

| `tool_choice` | Reply body |
|---|---|
| `"none"` | Free text only. The tools stay in the prompt, so the prompt cache still matches turns that offered them. |
| `"auto"` | Free text, then optionally the calls |
| `"required"` | (Free text if `preamble`), then one or more calls |
| `{"function": {"name": N}}` | (Free text if `preamble`), then exactly one call to N. If N is not offered, any offered name is allowed. |

The server always passes `preamble=False`, because a small model given room to talk first can talk until the token limit. It passes `parallel=False` when the request sets `parallel_tool_calls: false`, which allows at most one call. The server builds a reply grammar only for the `locals` engine, and only when the request has tools. With tools and a `response_format` (or `guided_regex`) at once, the grammar is the calls or the output: `(reply_regex(tools, "required", ...)|format regex)` for `auto`, the calls only for `required` or a named function, the output only for `none`; a reply that repeats the previous call is decoded again as the output. It also skips the grammar when `ignore_eos` is set, unless `tool_choice` is `"required"` or a named function.

How free text excludes a call. `not_containing(*lits)` is an exact regex for strings that contain none of the given literals, all of which must start with the same character and not contain it again. The string is split at every occurrence of that lead character, and each segment must not begin with the rest of any literal. The free text of each format:

- **hermes, fallback, mistral_v11, minicpm_xml, lfm2_python:** `not_containing(opener)` (the `ToolFormat.free_text` default, or `MistralV11`'s own copy of it)
- **qwen_xml:** `not_containing("<tool_call>", "<function=")` (no call with or without its wrapper)
- **gemma4:** `not_containing("<|tool_call>", "<|channel>", "<channel|>")`
- **llama3_json:** text, possibly empty, that starts with neither `{` nor `<|p`. The regex `([^{<](.|\n)*|<([^|]|\|[^p])(.|\n)*|<|<\|)?` rules out every text that starts with `<|p`, not only `<|python_tag|>`.
- **mistral:** text that does not start with `[` and does not contain `[TOOL_CALLS]`

`llama3_json` and `mistral` put their calls at the start of the reply. For them, `"auto"` is "calls or free text", and `"required"` (or a named function) is the calls alone.

**Thinking.** With `thinking`, the body is preceded by `not_containing("</think>") + "</think>" + think_sep` for qwen_xml (`oai.reply_regex`) and for every format that uses `ToolFormat.reply_regex` (hermes, mistral_v11, minicpm_xml, lfm2_python, fallback), so the reply starts inside an open think block. llama3_json, mistral and mistral_legacy override `reply_regex` and ignore `thinking`. `think_sep` is `\n\n`, or `""` for `lfm2_python`. Gemma 4 instead prefixes `thought_regex()`, an optional `<|channel>thought ... <channel|>` block, whatever the thinking setting; Flint likewise an optional `<think> ... </think>` block. When the server steers a repeated call with a text prefix, it removes that optional block from the regex.

**Typed Qwen calls.** `--typed-tools` switches Qwen's XML calls to `typed_call_regex`. Every parameter name is then fixed in schema order, required parameters are mandatory, and values are typed by `_value_regex`:

- `enum`: alternation of the values, a member that is not a string as JSON writes it (`null`, `true`);
- `integer`: `-?[0-9]+`;
- `number`: a decimal with optional exponent;
- `boolean`: `true|false`;
- an integer, number or boolean whose `type` list includes `"null"` also allows `null`;
- anything else: free text.

These automata have thousands of states, so the comment pairs the flag with the walk kernel.

**Typed Flint calls.** For Flint the flag gives `FlintXML.reply_regex` the tools' schemas (`specs`), and `typed_call` writes each call as Qwen's typed calls are written: parameter names in schema order, required ones mandatory, others optional. Values come from `_flint_value`, as Flint's template writes them: a string (or an untyped parameter) as it is, any text without `</arg_value>`; an `enum` as its alternatives; `integer`, `number`, `boolean` and `null` as JSON; arrays and objects as JSON nested three levels. The parameter names are then forced text: `<arg_key>filePath</arg_key><arg_value>` after a call's name is a span of five tokens, which one request alone feeds in one pass (docs/engine, `_solo`). Other formats ignore `typed`.

### Parsing

`oai.parse_reply(text, tools, fmt, thinking)` returns `(reasoning, content, tool_calls)` in OpenAI form:

1. `split_reasoning` separates the reasoning:
   - For formats that open thinking themselves (`opens_thinking`, Gemma 4), a reply that starts with `think_open` has a thought block.
   - Otherwise, reasoning is the `<think>` ... `</think>` block, if the prompt opened it or the reply opens it. A `</think>` later in a reply that did not open a block (in code, in an answer about it) is text.
2. `fmt.split(text)` gives `(content, calls part)`, and `fmt.parse_calls(part)` gives `[(name, args)]`.
3. Calls to tools the request did not offer are dropped.
4. String values are converted with `coerce` to the JSON type their parameter's schema declares.
5. For formats other than qwen_xml, a calls part that parsed to nothing stays as text. For qwen_xml, the content is always the text before the first `<tool_call>`, so a calls part that does not parse is dropped, not kept as text.
6. Each call gets an id from `fmt.new_id()`: `call_` plus 24 hex characters, or nine hex characters for Mistral, whose templates check ids.

`coerce(value, schema)` follows these rules:

- An enum member that is not a string (`null`, `true`, `3`) is returned when the text is its JSON form and not itself a member.
- If the schema allows a string, or declares no type, the text is kept as is ("1.0", or a file that happens to be JSON, stays text).
- `null` becomes `None` when null is allowed.
- Integers and numbers are parsed; NaN and infinity are rejected.
- Booleans accept `true/yes/1` and `false/no/0`.
- Arrays and objects are parsed with `json.loads` and type-checked.
- Anything else stays text.

Format-specific parsing details:

- **qwen_xml.** `qwen_calls` reads each value up to its own `</parameter>`, which the grammar keeps out of values. A value may therefore contain `</function>` or a whole example call. A call cut before `</function>` is not a call. A missing `</tool_call>` at the very end is tolerated.
- **llama3_json.** A preamble sentence before `{"name"` is accepted when the call parses, has a dict under `parameters` (or `arguments`) and ends the reply. A code-fence opener (for example ```json) at the end of the preamble is removed from the content, and backticks after the call are ignored.
- **gemma4.** A reply that starts with `call:` without its opener is accepted. Arguments are converted by `_gemma_to_json`.
- **lfm2_python.** Complete calls are found by bracket matching outside quoted strings; each call's name and keyword arguments are split at brackets and quotes (`_py_args`), so names the grammar allows but Python does not (`resolve-library-id`, `-n`) parse, and each value is read with `ast`.
- **fallback** (`loose_calls`). Two more shapes count as a call when they name an offered tool:
  - a fenced JSON block at the end of the reply;
  - a reply that is one JSON object.

  Such a call may be wrapped in `tool_call`, `function_call` or `call`. The name is read from `name`, `tool` or `tool_name`, and the arguments from `arguments`, `parameters`, `args` or `input`.

**Server guards.** `--max-calls` (default 6) stops a reply after that many complete calls (`count_complete`) and cuts it with `cut_after`. A duplicate call is cut to the unique ones.

**Streaming.** `ReplyStream(thinking, fmt).feed(text)` turns the growing text into `("reasoning", delta)` and `("content", delta)` events:

- It switches to tool mode at the format's opener and stops emitting content.
- It holds back any suffix that could be the start of a tag (`_safe`).
- For `llama3_json`, `mistral` and `mistral_legacy`, the first non-space character decides between text and a call. After text, `llama3_json` also watches for `{"name"`.
- `fallback` holds text back from a code fence or a leading `{` onward.
- `unsent(content)` releases text that was held back when the reply turned out to be text after all.

### Messages per format

**Replay.** When `replay_raw` is set, an assistant turn whose calls came from this server is rendered from the exact reply text it was parsed from (`convert_messages(raw=...)`). The prompt then re-tokenizes to the tokens the model produced, and the prompt cache is hit. `mistral`, `mistral_v11` (which inherits it from `MistralV3`) and `gemma4` turn this off (`mistral_legacy` turns it back on):

- Mistral templates count an assistant turn without `tool_calls` in their role alternation;
- Gemma 4's template writes tool results only after a turn's `tool_calls`.

`prepare(msgs, tools)` adapts the messages to each template:

| Format | Message handling |
|---|---|
| `llama3_json` | Splits an assistant turn with several calls into one turn per call (the template raises otherwise); passes JSON tool results as objects; drops an empty `tool_calls` key |
| `mistral`, `mistral_v11` | `_merge_users` merges consecutive user or assistant turns and drops empty assistant turns. `_mistral_ids` rewrites tool-call ids to nine letters and digits, and inserts an assistant `"Done."` turn between tool results and a following user turn where the template's alternation check needs one. |
| `mistral_legacy` | System text is folded into the first user turn; calls become `tool_calls` role turns and results become `tool_results` turns |
| `gemma4` | Text written before calls moves into `reasoning_content`; otherwise the template ends the turn mid-task |
| `fallback` | Tools are written into the system prompt (`FALLBACK_INSTRUCTIONS`); earlier calls are rewritten in the hermes form; tool results become user turns in `<tool_response>` tags; consecutive turns of one role are merged; the template gets no tools |
| Others | Unchanged |

## Thinking styles and `reasoning_effort`

`formats.thinking_style(tokenizer)` classifies the chat template:

| Style | Detected when | Example (from the code) | What the server does |
|---|---|---|---|
| `channel` | Template has `<\|channel>` and `enable_thinking` | Gemma 4 | Passes the switch; thoughts are in `<\|channel>thought ... <channel\|>`; the grammar allows an optional thought block at the start |
| `none` | Neither `enable_thinking` nor `<think>` | | Thinking off; no `enable_thinking` keyword is passed |
| `switch` | `enable_thinking` but no `<think>` | | Thinking kept off: the parser does not know how its thoughts are marked |
| `model` | With thinking on, the generation prompt does not end in `<think>` | Qwen3 | The model writes `<think>` itself; offered only when the template has `enable_thinking` |
| `open` | The prompt ends in `<think>` with thinking on, not with it off | Qwen3.5 | The reply starts inside the think block |
| `always` | The prompt ends in `<think>` even with thinking off | LFM2.5 | With thinking off, the server appends `</think>` to the prompt, so the model answers at once |

`offers_thinking` is true for `open`, `channel` and `always`, and for `model` when the template has `enable_thinking`. Once an engine is loaded, `/v1/models` lists a `-thinking` model id only when it is true. Before any engine is loaded it lists the `-thinking` ids anyway (`thinks = not loaded or offers_thinking(...)`).

`Server.prepare` decides `thinking` in this order:

1. `thinking_of(model, --thinking)`: on when the model id contains `think`, or when the server was started with `--thinking`.
2. `chat_template_kwargs.enable_thinking` (a bool) overrides it.
3. `reasoning_effort` (or the Responses API's `reasoning.effort`) equal to `"none"` turns it off.
4. Style `none` or `switch` turns it off.
5. `response_format` `json_object`, `json_schema` or `regex` (or `guided_regex`) turns it off, because the grammar writes the JSON from the first token.
6. A model for which `offers_thinking` is false keeps it off.

`reasoning_effort` is passed to the template only when thinking is on, and only if `chat_template_kwargs` did not already set it. `reasoning_efforts(template)` reads the values the template accepts from its `reasoning_effort ... not in (...)` check. If there is such a list, `effort_for(asked, allowed)` picks the nearest allowed value on the scale `none, minimal, low, medium, high, xhigh, max`:

- a tie goes up;
- an unknown value maps to the first allowed one.

For a template that accepts only `xhigh`, `medium` and `low` (and raises on anything else), `high` maps to `xhigh` (a tie between `medium` and `xhigh`), and `minimal` maps to `low`.

## Library API

The grammars work without the server.

```python
import locals

# 1. Our engine: forced spans in the same pass, GPU-resident masks, token budget
rt = locals.load("mlx-community/Qwen3.5-4B-MLX-4bit")      # Runtime.load; short names such as "qwen" work
res = rt.generate("Extract the person from: ...", schema=SCHEMA, max_tokens=512)
print(res.text, res.forced_tokens, res.finish)

# 2. A grammar in any decode loop
vocab = locals.Vocabulary.from_tokenizer(tokenizer)        # mlx-lm TokenizerWrapper or HF tokenizer
compiler = locals.GrammarCompiler(vocab)                    # max_dfa_states=200_000, work_budget=None
grammar = compiler.compile_json_schema(SCHEMA)              # or compile_regex(r"..."), compile_tool_calls(tools)
m = grammar.matcher()                                       # forced=True, canonical=True
while not m.is_terminated():
    logits = m.apply_token_mask(logits, tokens_left=left)   # -inf for forbidden tokens
    tok = sample(logits)
    m.accept_token(tok, check=False)                        # False (state unchanged) if not allowed
    feed(tok, *m.jump_forward())                            # forced tokens: feed, do not sample

# 3. A logits processor for mlx-lm
from mlx_lm import generate
generate(model, tokenizer, prompt, logits_processors=[locals.logits_processor(grammar)])
```

| Name | Signature and behaviour |
|---|---|
| `Vocabulary.from_tokenizer` | `(tokenizer, vocab_size=None, eos_ids=None)`. `vocab_size` is the logits width (defaults to the tokenizer's size). `from_model(model, tokenizer)` reads the width off the model. Properties: `n_tokens`, `eos_ids`. |
| `GrammarCompiler` | `compile_regex(regex)`; `compile_json_schema(schema, whitespace="")` (recursive schemas give a pushdown grammar); `compile_tool_calls(tools, tool_choice="auto", typed=False, thinking=False)` (Qwen's XML format only); `clear_cache()`. Grammars are cached by the regex they compile to (key `("regex", regex)`), and pushdown grammars by their normalized schema text. A second source that yields the same regex gets the first `CompiledGrammar` back, with that grammar's `kind` and `source`. With a slow tokenizer (no pre-tokenizer), the planner is `None` and forced spans are off. |
| `CompiledGrammar` | `kind` (`"regex"`, `"json_schema"`, `"tool_calls"`), `source`, `recursive`, `n_states`, `vocab_size`, `table_bytes`, `matcher(forced=True, canonical=True)`, `controller(...)` |
| `Matcher` | `next_token_mask(tokens_left=None)`; `apply_token_mask(logits, tokens_left=None)` (pads or cuts the mask to the logits width; a no-op once terminated); `accept_token(token, check=True) -> bool`; `jump_forward() -> List[int]` (accepted as a side effect, planned without healing); `is_completed()`; `is_terminated()`; `state`; `fork()`; `reset()` |
| `logits_processor` | `(grammar, canonical=False)`. Masks only: mlx-lm samples one token per step, so forced text is not skipped. A call whose context is not exactly one token longer than the previous call's starts a new sequence. A disallowed token raises `ValueError`. No token budget, so mlx-lm's `max_tokens` can cut the output mid-value. Not for mlx-lm's speculative `draft_model`. The matcher is `processor.matcher`. |
| `Runtime.generate` | `(prompt, schema=None, regex=None, max_tokens=512, temp=0.0, seed=0, forced=True, canonical=True, chat=True, reuse_prompt=True, grammar=None)`. `grammar` takes a `CompiledGrammar`. The prompt goes through the chat template unless `chat=False` or token ids are given. |
| `Runtime.compile` / `compile_walk` | `(schema=None, regex=None, whitespace="")`: a table constraint (or a pushdown one for recursive schemas), or a walk constraint (regular only) |

## Other decoders on the same controllers

- **`Engine.generate(..., lookup=True)`** (experimental). Greedy decoding with prompt-lookup drafts (`lookup.py`). Under a constraint, drafts proposed on the first pass and after a forced span are walked through the automaton on the host (`step_config`) and cut at the first disallowed token. A draft that rides on an assumed token (`guess_for`) is not walked: its first disallowed position fails the per-position masked verification instead. Every draft position is masked with the constraint's state there, so the accepted prefix equals greedy decoding up to floating-point ties.
- **`fields.parallel_fields(rt, schema, prompt_ids)`.** Decodes each object field as its own branch, all in one batch. It is an approximation (a value does not see earlier values), unlike forced spans, but the stitched object always matches the schema.
- **`LLGuidanceController`** (`engine.py`). An llguidance matcher used as a baseline controller.
