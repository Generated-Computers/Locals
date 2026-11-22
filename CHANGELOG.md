# Changelog

## 0.2.3 (unreleased)

- `python -m locals.convert_inkling --adapter FILE.gguf --out DIR` makes an Inkling 2 Mini build with a direction
  adapter applied, beside the build it is applied to (`--base`; its weight files are linked, the new build is 0.7 GB
  of its own). A direction adapter is a rank-1 LoRA on every layer's attention output and experts' output and on
  the embedding whose B is one unit direction a layer and whose A is minus that direction times the weights: the
  layer's output without its part along the direction. AtomicChat's Inkling-2-Mini-Uncensored-LoRA is one (its A is
  minus the direction times this build's own 8-bit attention weights to a cosine of 0.99997). Attention's output
  weights and the embedding are written again without the direction at their 8 bits (1% of it left, their
  rounding); the experts stay 4-bit as released and their summed output is cleared of it as the model runs, in the
  kernel that sums them (`inkling.combine`, `inkling2_loop.without`). On an M4 Pro, after the first token, against
  the build without it: a file written through a tool 70.9 tokens/s (73.7), a new file 60.6 (57.2), prose 34.7
  (35.1). An adapter of another shape is refused. What the adapter does to the model's answers was not measured
  here; its makers report MMLU 79.15% against 80.65%.
- Inkling 2 Mini keeps a long conversation's prefix on the SSD too (Inkling 2 only; the disk cache kept system
  prompts alone): the state at the start of the last message, every 2,048 tokens read past the last one saved
  (`disk_checkpoint_tokens`), back into the prompt cache as a conversation's checkpoint when a request starts with
  it and the memory holds less. The prompt cache has room for two or three such states beside the weights on a 24 GB
  Mac, and one that fell out of it, or a restarted server, read the conversation again at 300 tokens/s. A 7k-token
  conversation after a restart, on an M4 Pro: the first token after 0.5 to 0.8 s where it took 22; saving it took
  0.2 s of the first reading, and 0.48 GB of the disk cache's 4 (`LOCALS_DISK_CACHE_GB`). The idea is oMLX's cold
  tier (github.com/jundot/omlx), on the checkpoints Locals already makes. A 32k-token conversation's state is 1.4
  GB, more than the 1.25 GB prompt cache holds, so it is the request's own cache straight from the disk
  (`DiskPromptCache.load`): its requests answered after 142 to 154 s each, every one reading it all again (226
  tokens/s at that length), and answer after 1.0 to 1.2 s once it is saved. A 96k-token prompt is refused at once
  on this Mac, where 64,512 tokens fit beside the weights; the docs have the table by length.
- `python -m locals.convert_inkling` downloads only what the release's folder lacks: `--draft-only` gets the draft
  model's 0.64 GB alone (a folder from a build without it has no `mtp.safetensors`, and one deleted after the build
  had the whole 19.5 GB downloaded again), and a build made again gets the draft model's file.
- Inkling 2 Mini over long conversations, on a 24 GB Mac. A request that went on from a cached 6,666-token prefix
  with 40 tokens to read was given room to read 2,048, and the prompt cache was emptied for it, the prefix the next
  request went on from included: every other request of a 7k-token conversation read it again, 21 s. Three causes:
  - A prompt read up to a checkpoint went in chunks of 2,048 tokens whatever chunk the model sets (`_run`; Inkling
    sets 512 for what a pass holds: 1.0 GB against 2.9).
  - A read was recorded under its chunk's size, not under the tokens it read (`_note_read`), so a prompt's last 40
    tokens stood for a whole chunk.
  - Room was kept for the largest read seen. A model may say that its passes hold memory by their size
    (`read_room_by_size`: Inkling's are evaluated a visit at a time) and is then given the room of the nearest
    recorded size, for the tokens the prompt cache does not have (`_read_bytes`, `_make_room`).

  With them the cache keeps what it is for, and fills: 2.4 GB, 6% of the Mac's memory free and 3 GB in swap after a
  7k-token conversation. Where the weights leave the engine under 5 GB, Inkling asks for a prompt cache of 1.25 GB
  (`prompt_cache_bytes`; `--cache-gb` sets another): 12% free, and the third request of that conversation starts
  after 0.6 s where it took 21.
- Inkling 2 Mini's draft model against none, a request alone after 0.9k, 3k and 7k tokens of conversation (a file
  read in an earlier turn; tokens/s after the first, an M4 Pro): thinking about the code 37.5 to 55.8, 34.4 to 45.6
  and 37.3 to 64.8; the function written out again 50.9 to 62.2, 54.2 to 59.6 and 55.6 to 58.0; an edit through a
  tool 39.1 to 65.5 at 0.9k and 36.2 to 61.5 at 7k; a 900-token file written through a tool 44.0 to 59.9 at 0.9k and
  42.0 to 69.9 at 3k. Replies with and without it part where a pass rounds otherwise, so some cases are not the same
  text and are left out. Timed in the engine's loop, a pass with a draft is about 38 + 7.3 n ms and the draft
  model's draft 12 ms, where the rules that decide when to draft count 25 + 6 n and 7: set by the timed numbers
  they measured worse in the one run made, on a Mac short of memory at the time, so they are as they were.
- An agent on Inkling 2 Mini reads each prompt token once. Three things had tokens read again:
  - Typed tool calls wrote a call's arguments in the schema's order. The model's template writes an earlier turn's
    sorted by name (`dictsort`), and the model, left to itself, writes them sorted too (`content` before `path`): a
    finished reply was not what the next request held in its place, and every step of an agent read its step before
    again. `formats.Inkling.typed_call` sorts them.
  - Clients send a turn with calls back without its reasoning, and the template writes an earlier turn's. The
    server writes such a turn from the reply it kept (`formats.Inkling.restore`: the reasoning, the text to the
    character, the arguments in the reply's own nesting) as long as the client's calls are the reply's.
  - Requests asked at once with one prompt, as the versions `locals run` makes of a part, each read it. On Inkling 2
    (other models admit requests as before) the engine reads once to where a waiting request's prompt parts from this one's and keeps that for it (the two longest such
    prefixes, from 64 tokens on), and a "user" checkpoint within 32 tokens of the prefix a request was found is not
    copied again.

  `locals run`'s two-file task (4 slots, 2 versions, an M4 Pro): 4,755 prompt tokens read where the server's cache
  had none became 1,100 to 1,600 (21 s of reading became 6 to 8), and the task 34 to 40 s from 49 to 50.
- Inkling 2 Mini reads prompts faster. A prompt chunk's attention weights come from one kernel (`inkling.soft`: the
  bias by distance, the mask and the softmax with its sink in one pass over the scores, six arrays of the scores'
  size as MLX operations); a sliding layer scores a block of queries against the keys that block sees
  (`inkling2_loop.QUERY_BLOCK`) where it scored every query against the window and the whole chunk; a pass of 128
  tokens or more is evaluated a visit at a time (`EVAL_TOKENS`: a 256-token chunk held 1.14 GB of arrays at once and
  holds 0.50, one of 512 holds 0.99 where it held 1.97); and chunks are 512 tokens (`inkling.PREFILL_CHUNK`, 256
  before). Time to the first token with nothing cached, an M4 Pro: 1,471 tokens 5.35 to 4.66 s, 3,471 tokens 13.2
  to 11.2 s, 6,997 tokens 28.1 to 23.3 s (249 to 301 tokens/s). Tried and left out: the step's attention kernel for
  a chunk's sliding layers (a threadgroup of 1,024 threads a query: 1,096 ms a 256-token chunk against 959).
- Locals needs mlx-lm 0.32.2 or later: Inkling 2's build keeps its experts in NVFP4 with a scale per expert, which
  mlx-lm 0.32.0 could not load. The README says how to install it and make its build.
- Thinking Machines' Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, a looped mixture of experts: 24 layers of
  weights, 48 layer visits a token, 26.4B parameters) runs on Locals. `python -m locals.convert_inkling` makes its build
  for a 24 GB Mac from the release (15.6 GB: the experts of layers 0 to 19 as released, NVFP4, bit for bit; those of
  layers 20 to 23 from MXFP8 to NVFP4 with the scale of each block that loses least; attention's projections, the
  embedding and the output layer 8-bit; text only). `models/inkling2_loop.py` is the reference computation (what
  ml-explore/mlx-lm#2086 computes, under its parameter names); `inkling.py` takes a token's step in six kernels a
  visit where the reference makes about 125 MLX operations. An M4 Pro, one request: 18.5 tokens/s on the reference
  path, 34 with the kernels (29.5 ms a token, 21 of them weight reads at the memory's speed); 42 with two requests at
  once, 55 with four. `locals run` works on it (its tool calls, its reasoning and its text are messages of their own:
  `formats.Inkling`; thinking off is its least reasoning effort).
- Measured and left out for Inkling 2 Mini: attention's projections at 4 bits (the same next token as the 8-bit build
  79% of the time on 1024 tokens, the text's perplexity from 18.5 to 24.8, for under 2 ms a token); the output layer
  at 4 bits (96.4%, 1 ms).
- A grammar may write Inkling's message markers as the tokens they are (`vocab.FORMAT_TOKENS`). Spelled letter by
  letter, as a special token otherwise is, the model answered a tool's result by calling the tool again.
- A span the grammar forces is fed with Inkling's message markers as the tokens they are, though its tokenizer marks
  them special (`forced.ForcedPlanner` with `vocab.INKLING_TOKENS`; it split only the added tokens that are not
  special). A forced
  `<|message_model|>` went to Inkling as `<`, `|`, `message`, ..., after which the model no longer saw the start of
  a message: it wrote its tool call two or three times (the server cut the turn as a repeated call: 139 tokens for a
  call of 58) or lost the thread after a tool's result. Other models' special tokens are split as before.
- Inkling 2 Mini for agents, on an M4 Pro: prompts are read in chunks of 256 tokens (270 tokens/s; 150 in the
  engine's 1,024, 190 in 64); a request alone gets its forced spans in one pass and drafts of what followed the same
  three tokens before (its windows and convolutions go back to the last token kept: `inkling.rewind`), a pass over
  2 to 16 tokens in the same kernels as a step (45 ms for 2, 79 for 5, where a step is 30), drafts only as wide as
  their recent acceptance pays for (a drafted token costs 0.4 of a step here, kept or not); and the server keeps a
  prompt's prefixes from 192 tokens instead of 1,024 (`checkpoint_tokens`), so `locals run`'s agents, whose prompts
  are 900 to 1,100 tokens, start each step from the last one: a two-file task with four agents went from 63 s to 43.
  After the first token a request with tools runs at 38 to 46 tokens/s; a tool set's grammar takes 4 to 5 s the first
  time (its token table is 1.2 to 1.8 GB over 201,024 tokens, and the cache of compiled grammars holds one).
- Inkling 2 Mini's experts are read once for the rows that share them (`locals/inkling_experts.py`, for up to 32
  rows of a step or a pass). MLX's `gather_qmm` runs one matrix-vector product a (token, expert) pair, and at 4 bits
  that is bound by the arithmetic of unpacking a weight (8 operations a weight, 6 of them the unpacking), so an
  expert costs the same for every token routed to it. `slots` groups a pass's pairs by expert on the GPU and `xmm`
  unpacks each weight word once for two tokens' dot products, with the softcapped SwiGLU applied as the gate/up
  product is written (MLX's bits). Consecutive tokens of a text share experts (4 use 19 of their 32, 8 use 29 of
  64) and so do different requests' rows: the experts of an 8-token pass 59 to 41 ms, of a 4-token pass 33.5 to
  23.7, of one token 10.3 to 9.7; a pass of n tokens of one text about 25 + 6 n ms (it was 20 + 12 n), in kernels
  up to 32 tokens (`inkling.SEQ_MAX`, 16 before). Requests at once on an M4 Pro: four 71.6 to 79.4 tokens/s, eight
  79.5 to 94.6. `LOCALS_INKLING_XMM=0` is MLX's gather.
- Lookup drafts on a model whose drafted tokens cost (Inkling 2): as long as the recent drafts that got that far kept
  their i-th token more often than it costs, past that twice the longest one kept whole (2, 4, 8 ... 31 tokens over a
  long copy), under grammars too (they were off there). Where a request has lookup drafts and the draft model, it
  takes the one whose recent tries gained more steps (`hybrid_batch._draft_gain`) and tries the other one time in 16.
  After the first token, on an M4 Pro, with both: a file written through a tool 75.6 tokens/s (36.9 without drafts,
  17 on stock MLX), a new file through a tool 59.6 (37.4), an edit through a tool 50 to 53 (46), code 40.3 (37.6),
  prose 36.0 (37.4). (A "plain reply that copies from its prompt", 56.4 against 48.8, stood here: that request asked
  for a tool it was not given, and the reply was a call of no tool and then one argument over and over. The drafts
  were of that loop. Real copies are in the entry on long conversations.)
- Inkling 2 Mini's draft model (the release's `mtp.safetensors`, "DSpark": three layers that draft up to 7 tokens in
  one small pass) for a request alone: `locals/inkling_draft.py`, a part of the build (`python -m
  locals.convert_inkling --draft-only` adds its 0.64 GB to a build already made; `LOCALS_INKLING_DRAFT=0` runs without
  it). Its context is what the target's stream held entering three visits and at its output, for every token read:
  the target's passes put that into three caches after its own, so prompt caches, checkpoints and rewinds carry them
  (with the whole prompt in its context the target keeps 6.2 of its 7 tokens of a tool call that writes a file, 2.8
  with the reply alone). The engine tries its tokens as far as it is sure of them (probability over 0.8), under the
  request's grammar, in the pass that verifies lookup drafts, and sets it aside for a while when its tries lose
  (`hybrid_batch._model_draft_seen`). On an M4 Pro, after the first token: a file written through a tool 36.9 to
  63.5 tokens/s (479 of 498 drafted tokens kept), a new file through a tool 37.4 to 51.4, code in a plain reply 37.6
  to 39.1; prose 37.4 to 35.9 (its drafts seldom pay, and a step now also feeds the draft caches). Requests at once
  are as before (four: 72.8 and 73.4 tokens/s): drafts are for a request alone. A request's usage says what the draft
  model did (`locals.drafts.model`: kept, drafted, passes, tries, unsure, pauses).
- Inkling 2 Mini's grammars are walked (`walk_constraints`, a model's own choice of the server's `--walk`;
  `LOCALS_WALK=0` for the tables): the mask comes from the automaton each step, so a set of tools costs no token
  table (1.3 GB on the host and 0.4 on the GPU before), and its automaton is made in a thread of its own, not in the
  batch loop (other models' walked grammars are made in the batch loop, as before). A walked grammar's forced spans still go in one pass for a request alone (`hybrid_batch._solo_ok`,
  Inkling only), and its tool calls are typed without `--typed-tools` (`formats.Inkling.typed_call`;
  `LOCALS_TYPED_TOOLS=0` for any JSON object): the tool's parameters sorted by name, the required ones
  there, each value of its type, the JSON's name the header's. The first request with a set of tools answers after
  2.0 s where it took 6.6 (733-token prompt, M4 Pro), a call of 75 tokens has 25 forced where it had 18, and
  requests with tools run at 45 to 52 tokens/s after the first token (38 to 46). `locals run`'s two-file task with
  4 slots: 55 s and 15% of memory free at the end, where the tables took 61 s and left 11%.
- Inkling 2 Mini's attention kernel takes the three query heads of a key/value head in one threadgroup, so a
  window's keys and values are read once instead of three times: 5.1 to 3.7 ms of a step for one request, 26.8 to
  11.3 for eight. Requests at once on an M4 Pro, each writing 320 tokens: 35 tokens/s for one, 54 for two, 73 for
  four, 81 for eight (the experts are 8 ms a row and step: each row reads its own).
- On Inkling 2 Mini a request's admission counts the state its checkpoints copy besides the keys and values
  (`hybrid_batch._copies_state`, in `_fits` and `_make_room`): its sliding windows and convolution states, 160 MB a
  copy, which four agents with three checkpoints each held uncounted (an M4 Pro with 24 GB went to 6% free). Other
  models count their copies as before.
- On Inkling 2 Mini the server's line for a request says how many drafted tokens were kept (`drafts 97/105 in 90
  passes`), and its usage has them (`locals.drafts`).
- `Engine` (`locals chat --local`) reads prompts in a model's own chunk where it has one.

- On a busy Mac, a prompt chunk now and then went into a copy of the cache's key and value buffers instead of the
  buffers themselves: MLX writes into a buffer only when nothing else holds it, and the GPU's command buffers of the
  chunk before let go of theirs a moment after it was done (later while the CPU is busy). Each copy was a whole buffer,
  and a chunk the GPU then dropped could not be read again from where it began. Each chunk now waits for them first
  (about 30 µs; prompt reading is as fast).
- Qwenputer's first start, without its authors' package, gives the command that installs it where Locals is: with
  Locals installed by `uv tool install` (or the install script), `uv tool install ... --with "chipformer @ ..."`,
  since pip does not reach uv's environment; otherwise `pip install`. The Qwenputer docs give both.

## 0.2.2 (2026-11-04)

- **Qwenputer**, in two versions (`--model qwenputer` and `--model qwenputer-cpu`: divagr1925/Qwen3.5-4B-Qwenputer):
  Qwen3.5-4B with a RISC-V computer compiled into 11 layers before Qwen's own, which runs the model's `calc` tool
  exactly (integers and fractions of up to 62 digits). Qwen's 32 layers are Qwen3.5-4B's own weights and run as that
  model does. The computer is made once from the 18.8 GB checkpoint into a 123 MB pack (`~/.locals/qwenputer`);
  loading its firmware needs the authors' package (`docs/qwenputer`). A calc exchange the model runs before its
  answer comes back as `reasoning_content`. The computer reads and writes only the rows of its recurrent state a token
  touches (one written, at most two read, of 128) and makes every token of a run itself, nothing drafted: on a CPU
  performance core, in C built with the Mac's clang (3,300 trace tokens/s on an M2 Pro, where MLX operations one
  token at a time made 142), or else stepped on the GPU. A reply that calls calc stays in the batch, parked, while the
  run is made, and a finished reply's slot is taken in place by the next request. Every value equals transformers'
  layers, and every run the reference machine's.
  - `qwenputer` runs calc alone, on the authors' firmware: 60 arithmetic questions, 59 right at 155 tokens/s in total 8 at a time (Qwen3.5-4B: 5).
  - `qwenputer-cpu`'s computer also does work a client's tools would do on the Mac's CPU, slower: a toolbox of 60
    tools (counting, lengths, reversing, sorting, dates and weekdays, bases; and in C, compiled for the computer and
    checked against Python's own: JSON, tables, versions and lists, logic and regular expressions, statistics, money,
    primes, units, numerals, text, times, check digits, hashes, notes, and a small Python, `run_python`, within 1,250
    steps), offered to a request whose question asks for them; a call of up to about 1,000 tokens goes to it in parts.
    It checks each call of the client's tools against its schema before the client gets it (a wrong one goes back to
    the model), and each long number the answer copies from its results. 80 toolbox questions: 80 right in 59 s, against
    38 for `qwenputer` and for Qwen3.5-4B, and 197 s for Qwen3.5-4B with the same tools run by the client; 45 of the C
    tools' kinds: 40 right, Qwen3.5-4B 31; 60 arithmetic questions: 60 right at 107 tokens/s (151 without its copy
    checks).
- **A request's handler hears at once that its job is done**: the reply was written at the handler's next look at its
  queue, up to 0.25 s later.
- **A conversation's next turn no longer restarts the engine** when its first prompt chunk is marked: the turn read
  the previous reply's keys and values, strided views, and the engine dropped its prompt cache and read the turn
  from its start.
- **A downloaded model's context length is read from the cache**: the server asked Hugging Face for the model's
  `config.json` on every start, and when the model's repository had a newer revision, the cache's `refs/main` moved
  to it, without the weights: the model counted as not downloaded, the next start downloaded it again, and an
  offline start failed.
- `LOCALS_DISK_CACHE_GB` that is not a number is a warning again (4 GB), not a model that does not load.
- **Liquid's LFM3** (`--model lfm3`: LiquidAI/LFM3-1.7B-MLX-4bit, 1.0 GB). No mlx-lm release has LFM3 yet, so its code
  ships with Locals (`locals/models/lfm3.py`): LFM2's gated convolutions, 7 wide, and attention over a 256-token window
  or the whole context, with a sigmoid gate on each head, q/k norms, RoPE on half of each full-attention head's
  frequencies, and sandwich norms. It computes what Liquid's PyTorch code does (float32 logits within 0.001 on a
  759-token prompt; the same 40 greedy tokens through the cache). On an M2 Pro with 16 GB (`locals bench`), 620
  tokens/s for eight requests at once where mlx-lm, running the same code, gives 207; 153 for one where it gives 143.
  Only 6 of its 24 layers keep every token's keys and values (12 KB a token): on the 16 GB M2 Pro a 245,547-token
  request was read in 10.5 minutes, the one fact hidden in it found, with 4.9 GB of GPU memory at the peak.
  It thinks by default, as its template does (`"reasoning_effort": "none"` or `enable_thinking: false` turn it off):
  in the end-to-end suite (eight agent tools, 18 coding tasks, an M2 Pro) it finished 15 thinking and 10 not, and six
  OpenCode agents at once finished 6 and 3; with thinking off it reasoned in its answers instead.
- With thinking off, LFM3 still opened a think block of its own, or reasoned, closed a block it never opened and gave
  its answer twice (4 of 11 replies of the 4-bit build). For a chat template like LFM3's (`<think></think>` closed
  empty), a reply with thinking off is now kept from think tags by its grammar, for about 1% of the writing speed;
  other models' replies are as before. LFM3's calls are Qwen3.5's form with two newlines between calls, and the
  grammar now writes them as its template does.
- With thinking on, a chat template's `preserve_thinking` switch (LFM3's, LFM2.5's) is turned on unless the client
  sets it: earlier replies keep their reasoning as written, so the conversation stays a prefix of the cache. Claude
  Code on LFM3 (thinking on, its tool results followed by text, a user message to the template) read the conversation
  again from its first reply at every step: 55% of prompt tokens cached and a rename past 15 minutes, now 95% and 86 s.
- A prompt chunk the GPU drops is read again from where it began, instead of the request from its start. macOS ends a
  Metal command buffer that holds the GPU while the display needs it ("Internal Error (0000000e)", which MLX leaves to
  the OS): 10 minutes into a 245,547-token prompt, the request was read again from its start (19 minutes in all). The
  caches go back to where the chunk began, and it is read again after a pause, in half the tokens.
- Loading a model whose config.json names its own Python code for transformers (LFM3) no longer stops at
  transformers' question whether to run it (in a terminal, 15 seconds before it went on without).

- One request on a model with sliding-window layers (Flint) no longer waits for the GPU after every token: the
  check before a forced span's one-pass step read the layers' padding from the GPU each token, 35 reads that waited
  for the step in flight. It now reads it once, only where a span is about to go in one pass. On an M5 Pro, Flint
  wrote 89 tokens/s for one request with the check and 138 without it (mlx-lm: 101).
- OpenCode 2 works with Locals as OpenCode 1 does. `locals code` opened OpenCode 2's interface with `-m`, which it
  rejects ("Unrecognized flag: -m"); the model (and the agent `--agent` names) now goes in OPENCODE_CONFIG_CONTENT,
  which wins over a project's own opencode.json, and OpenCode's server runs for that launch (`--standalone`)
  instead of its shared background service. `locals swarm`, `locals solve` and `locals solve --plan` speak OpenCode
  2's server API (sessions, prompts and waits, interrupts, permission replies), and read its messages as version 1's.
  The agents are not given OpenCode 2's web search and code mode tools. Tested with OpenCode 1.18.32 and 2.1.2.
- **Recursive's Flint** (`--model flint`: mlx-community/flint-mixed-3-4bit, 17.8 GB, a 24 GB Mac and up). The model's
  code ships with Locals (mlx-lm has it only in an unreleased pull request), its tool calls have their own format
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`; it was taken for the hermes form,
  whose grammar held Flint to JSON calls), and its reasoning, which it opens itself, comes back as reasoning. Kernels
  for its router, experts and q/k norms: on an M4 Pro with 24 GB, 117 tokens/s for one request where mlx-lm gives 85,
  319 for eight at once where it gives 202; prompts are read 7 to 12% faster than mlx-lm reads them from 4,096 tokens
  (docs/benchmarks, section 11).
- On a Mac with the memory for a model but macOS's default GPU limit below what it needs (Flint on 24 GB), the memory
  check says how to raise the limit (`sudo sysctl iogpu.wired_limit_mb=...`) instead of only naming a smaller model.
- Flint's matrix products tuned to MLX's own kernel skip Locals's routing: 4 us of Python a call, 0.8 ms of a
  one-token step.
- Decode rows and a prompt chunk in one pass for Flint too: replies streaming while an 8,000-token prompt is read
  stall for at most 2.8 s instead of 9.7.
- The projections `locals.patch` fuses away from Flint no longer stay in MLX's buffer cache (2.5 GB, which a 24 GB
  Mac swapped for).
- `LOCALS_KV_BITS=8`: Flint's full-attention keys and values in 8 bits, as its config asks, read by a kernel of
  their own on the GPU's matrix units with Flint's attention sinks; a 24 GB Mac takes 67,584-token requests instead
  of about 35,800, for 4 to 13% of the writing speed.
- `--typed-tools` types Flint's tool calls too: the grammar writes each call's parameter names in the schema's order,
  required ones present, values of their types. On 12 OpenCode-like tasks Flint broke 4 of 12 greedy calls and 11 of
  39 sampled ones without it (`path` for `filePath`, `bash` without its `description`), none with it.
- One request alone gets its forced spans in one pass over Flint's sliding-window and 8-bit caches too: those
  caches cannot be rewound, which drafts need and forced spans do not. Flint's typed calls take 10%
  fewer passes (about the same time: a pass of several tokens reads each token's experts).
- Flint's expert kernels no longer end a command buffer each: MLX ends one once its inputs pass 50 MB, counted at
  their whole size, and the kernels were handed a layer's whole expert tensors (350 MB). Handed one-expert views of
  the same buffers, a one-token step takes 8.75 ms instead of 10.2: 115 tokens/s for one request instead of 102, and
  8 to 9% more after long prompts.
- Flint's sliding-window layers read a prompt chunk in bands of 512 queries against the keys their window reaches,
  instead of the whole chunk under a mask: prompts 11% faster from 4,096 tokens (844 tokens/s instead of 760 at 4k,
  560 instead of 504 at 32k), the same logits.
- Flint reads prompts in 2,048-token chunks to about 30k tokens (a model's `prefill_chunk_tokens` and
  `prefill_scores`, set where its attention holds no chunk's scores): a 30k prompt's first token after 54 s instead
  of 58, a 60k one with `LOCALS_KV_BITS=8` after 159 s instead of 177.
- Flint's full-attention layers attend without the batch caches' mask when no row is padded (all true, and 40% of
  MLX's attention at 30k keys): after a 30k-token prompt Flint writes 63.7 tokens/s instead of 55.9, after 13k 74.5
  instead of 70.1.
- Flint's expert kernels take experts of 2, 3, 4, 5, 6 and 8 bits (mlx-lm's mixed recipes put some layers at 6), and
  its router kernel names real experts even for a row of NaN logits. A Flint request alone gets its forced spans in
  one pass once it is no longer left-padded.
- Flint's passes of 5 to 8 tokens (a typed tool call's forced spans) are padded to 9 queries in its attention:
  MLX's attention takes a path twice as slow from 5 to 8 queries (2.1 ms a full layer at 8k keys for 8, 1.2 for 9).
- Flint's expert kernels read an expert once for every row of a batch that chose it, the same outputs bit for bit:
  284 tokens/s at 8 requests at once instead of 265, 313 at 16 instead of 288 (305 instead of 281 through the server).
- Flint with several requests at once, the same tokens: from 4 rows the expert kernels, bound by their arithmetic as
  much as by the weights they read, convert each input chunk once for several weight rows and turn 3-bit values into
  floats without an int-to-float conversion; the expert choice also orders the experts' rows (no sort kernel), and
  each layer's residual adds and next norm are inside its kernels (an add and RMS norm kernel laid out as MLX's,
  the same bits). 305 tokens/s at 8 at once instead of 284, 225 at 4 instead of 212, 340 at 16 instead of 313.
- A kernel family for Flint's matmuls of up to 16 rows, `fragq`: the weights go into the GPU's matrix units as
  1 + q/16 in the activations' 16-bit type (two by one shift, mask and or), the scale and bias per group on a float
  accumulator, a quarter of the matrix product's arithmetic spent on dequantizing instead of a half. The tuner times
  Flint's layers in a chain of calls that wait on each other, as a model runs them (back to back without waiting it
  chose kernels that made Flint 2% slower), and times again its shapes timed before the family. Flint's layers have
  tuning entries of their own (`…xbfloat16flint`): every other model's layers, of the same shapes too, keep their
  kernels, timings and choices. Flint: 317 tokens/s at 8 at once instead
  of 302, 352 at 16 instead of 339, 227 at 4 instead of 224; through the server, 305 at 8 clients.
- Recursive's own 4-bit build of Flint (RecursiveSI/flint-int4, 22.6 GB) runs on a 16 GB Mac: the GPU holds its
  weights without the experts (1.2 GB) and slots for recently used experts (40% of GPU memory), and the rest are
  read from the weight files as tokens need them (locals/models/flint_streamed.py, which loads that build on any
  Mac; mlx-community's builds load whole as before). On the M2 Pro: about 6 tokens per second, a 6,000-token prompt
  read in 19 s, and OpenCode fixed a bug in a small project with it.
- Flint's decode passes run its MoE blocks in Metal kernels of their own: the layer's router and the routers that
  guess the next layers' experts in one product and one selection (0.27 ms of a layer's GPU time became 0.04),
  and the routed and shared experts in two kernels instead of about fourteen; the experts already on the GPU are
  computed while the others are read. Without reads a token takes 31 ms instead of 40.
- Where Flint's layers mostly find their experts on the GPU, a decode step asks the host only when one is missing:
  each layer looks its experts up on the GPU, and the step runs windows of layers at once, resuming after a layer
  that missed one (its later layers' KV caches restored). With every expert on the GPU, a token takes 20 ms
  instead of 31 (50 tokens per second on the M2 Pro). Steps switch to it when 90% of the layers find their
  experts (LOCALS_FLINT_OPTIMISTIC: auto, always, 0).
- Flint reads a prompt's layers into two reused buffers, the next layer while the GPU computes this one, and copies
  the experts already on the GPU instead of reading them: a 1,024-token pass after some decoding, 5.4 s instead of
  6.9. Its GPU memory holds the files' bytes in every compute dtype, converted on the GPU when used.
- On M1 and M2 Flint computes in float16: its one value past float16's range (an expert's SwiGLU product in layer
  46, about 68,000) is taken in float32 and scaled by a power of 2. It is closer to a float32 run than bfloat16
  (logit error median 0.77 against 2.0) and a 4,096-token prompt pass takes 8.0 s instead of 10.1. A model that
  keeps its values in float16's range says so (``float16_safe``) and skips the runtime's dtype probe.
- A sliding layer of Flint attends over a long pass in blocks of 256 queries, each over the keys its window reaches
  (the same softmax): 11.6 ms a layer over 4,096 tokens instead of 57 with the window as a mask over every key. A
  4,096-token prompt pass takes 6.9 s (10.9 before these changes).
- Flint's results no longer depend on when expert reads end: the second call of a split MoE block sums every
  expert's contribution in the experts' order, taking those of the first call from it (the same bits as one call;
  a float32 sum in another order rounded differently now and then). The layer sets for prompt passes are let go
  after 2 s of decoding without one (0.84 GB) and made again for the next prompt (66 ms).
- `python -m locals.models.pack MODEL` writes an expert-major copy of a streamed model's experts (21.4 GB for
  Flint, in ~/.cache/locals/experts, 34 s on the M2 Pro). Flint's experts are stacked per tensor in its files, nine
  slices of one expert in nine places; from the copy each is one read, scattered into its slot: decoding 10.7
  tokens per second instead of 7.3, a 4,096-token prompt pass 6.3 s instead of 6.7. Used when present.
- LOCALS_EXPERT_RAM_GB: experts Flint evicts from the GPU are kept in memory outside the GPU's share, purgeable
  (macOS takes it back before it would swap), and a later miss copies them back instead of reading the SSD.
- Experts are read the most urgent first, the parts of one expert at once, bypassing macOS's file cache (which
  held the same experts a second time); the slot taken is the least used one (uses counted less as they age), and
  more experts are read ahead the more surely they are guessed. At most four guessed reads run at once
  (LOCALS_EXPERT_GUESS_READS), so a read a layer waits for finds the SSD free: 139 ms a token instead of 156.
  While a layer of Flint waits for its reads, the residual with the shared expert and the experts in slots goes
  through the next layer's norm and router: it names the next layer's experts far better than this layer's
  input (86% of them in its top 6 against 78%), and the next layer's reads ahead are revised (87 -> 83 ms a token).
- Mixture-of-experts models decode several requests at once with each expert read and decoded once a step: the
  step's (token, expert) pairs grouped by expert, one threadgroup per used expert and block of its rows (MLX's
  affine 4-bit and mxfp4). On the M2 Pro with Mellum2.1-12B-A2.5B (64 experts, 8 a token) 16 requests decode at
  187 tokens per second together against 142 with mlx-lm's batch_generate (1.32x), 4 at 131 against 115; one
  request as before (86 against 85.5). From about 5 tokens an expert MLX's sorted kernel fills its tiles and is
  used instead (64 requests: 367 against mlx-lm's 291). LOCALS_MOE_KERNEL=0: MLX's kernels.
- A model can set its own prompt passes (HybridBatchEngine.prefill_tokens).

## 0.2.1 (2026-10-21)

- The installer installs the release from PyPI; run again, it updates to the newest release. `LOCALS_SOURCE` set to
  GitHub's archive installs the main branch.
- The README on PyPI: `uv tool install locals` and `uv tool upgrade locals`.
- `locals menubar` asks for `uv tool install 'locals[menubar]'` when rumps is missing.
- Tested on Python 3.14 too, which a plain `uv tool install locals` picks on a Mac without a newer Python.

## 0.2.0 (2026-10-21)

First public release.

- **Up to 3x faster with many requests at once.** GPU kernels written
  and tuned for the few requests at a time that agents send, float16 on M1 and
  M2, fused layers, and a prompt cache that keeps each agent's conversation:
  271 tokens/s with eight requests at once where mlx-lm gives 87 (an M2 Pro with
  16 GB), and no slower for one request. On an M5 Pro, where mlx-lm is much
  faster, 1.5x with eight to sixteen requests at once. What each part
  contributes is measured in docs/why-faster.
- **Many agents on one task.** `locals run` puts a team of agents on a task, each
  in its own copy of the project, with your tests as the check; the project is
  changed only when they pass.
- **One command for coding agents.** `locals code` opens OpenCode on a local
  model; `locals launch` does the same for Claude Code, OpenHands, Cline, Kilo
  Code, OpenClaw, Hermes Agent, Aider, editor extensions and apps that work with
  Ollama. `locals models` says which model to use for agents.
- **Switching from Ollama, LM Studio, mlx-lm or llama.cpp** in a command or two
  (README, "Coming from"): `locals launch ollama` answers apps set up for
  Ollama and the `ollama` command itself, Ollama's model names work, LM
  Studio's MLX models run where they are, and a GGUF file or an Ollama name
  Locals does not know gets the MLX version of the same model to use.
- **An OpenAI-, Anthropic- and Ollama-compatible server** on MLX, running in the
  background (`locals start`, `locals status`, `locals stop`) or in the menu bar.
  Anthropic's Messages API (`/v1/messages`) is what Claude Code and the Anthropic
  SDK speak; each step of a Claude Code session comes from the prompt cache.
- **Files appear as the agent writes them.** Tool calls stream while the model
  writes them, so an agent writing a file shows it line by line, instead of
  nothing for 10 to 30 seconds and then the whole file at once (Qwen3.5).
- **Tool calls and JSON that always parse**, in each model's own format: Qwen,
  Gemma 4, Llama, Mistral, LFM2.5, MiniCPM5 and Hermes-style models.
- **Models for every Mac.** A default that fits your memory, short names for
  tested models (`qwen`, `gemma`, `lfm`, `minicpm`), any MLX model from Hugging
  Face, and a clear message for models that do not fit or come in the GGUF
  format.
- **Safe defaults.** The server answers only apps on your Mac, and the
  commands of `locals run` and `locals solve` agents can write only inside their
  own copy of the project (and the temporary and cache folders).
- **Long conversations.** A request can be as long as fits in your Mac's
  memory next to the model, instead of a fixed 64k tokens: about 178,000 tokens
  for Qwen3.5-4B on a 16 GB Mac (`--max-context` still sets it). The next turn
  continues from where the previous reply began, so a follow-up question about
  a 138,000-token document starts in 2.4 seconds; it used to read the whole
  conversation again (eight minutes), and agents re-read their last tool result
  on every turn. Four multi-turn conversations at once now take 83% of their
  prompts from memory instead of 28%, and finish 36 turns in four minutes,
  against 23 before.
- **No pause while a long prompt is read.** When a new request with a long
  prompt arrives, the requests already answering keep streaming: their next
  tokens are computed in the same passes over the model as the new prompt,
  instead of waiting for all of it (12 seconds for 8,000 tokens on a 4B model).
  Short prompts are read before long ones: a new agent's first token comes in
  7 seconds instead of 41 while another agent's 30,000-token prompt is read.
- **Works with benchmark tools.** `ignore_eos` and
  `stream_options.include_usage` as vLLM and OpenAI handle them, so InferenceX,
  vLLM's and aiperf's clients measure Locals as they measure other servers.
- **Measures itself on a new Mac.** How prompts are read (MLX's 4-bit kernel or
  a dense one) is timed on each GPU instead of fixed for M2-class chips, and
  `locals tune --report` shows where Locals gains on this Mac: MLX's matrix
  kernels against the chosen ones for each number of requests at once. On
  Qwen3.5, tuning also times the whole model reading prompts with MLX's own
  recurrent-layer op (which has a version for the M5's neural accelerators)
  and keeps it where it is faster. Batches of 8 to 16 requests are computed
  at a larger, faster size where tuning finds one: on an M5 Pro, 12 requests
  at once ran 10-13% faster on a 4B model and 19-20% on a 9B. On the M5, a
  4-bit matrix kernel for its neural accelerators (`nax`) is among the kernels
  tuning chooses from: timed alone, one decode step's matrix products take
  1.6-3.4x less time than with MLX's kernels at 6 to 32 requests. It is
  experimental (whole decode steps are not measured yet); `LOCALS_NAX=0` turns
  it off.
- **Smoother commands.** `locals chat` and `locals generate` use the background
  server instead of loading a second copy of the model, so `--n` works with every
  model; `locals status`, `stop` and `--version` answer in a fraction of a second;
  `locals code --run` shows each step as it happens; `locals` alone shows what it
  can do; the installer no longer needs git.
- **A steadier team of agents.** `locals run` tells you at once when the check
  command cannot run (for example pytest is not installed), instead of failing
  after minutes; a check that already passes before the run guards the result
  instead of ending it after the first file; "fix parser.py" finds
  `src/.../parser.py`; agents that are stopped stop generating at once.
- **Fixed:** sixteen long requests arriving at once no longer run the GPU out
  of memory; a stream that waits for a long prompt no longer sends a line that
  strict clients failed to read; a second command no longer starts a second
  server while the first is still loading the model; OpenCode always runs on the
  local model, even when a project's own settings name another one; a mistyped
  model name is no longer saved as the default; a follow-up message in an agent
  session no longer reads the previous turn again (Qwen3.5 and Gemma 4); pressing
  Esc or Ctrl-C in an agent stops its reply on the server too; with MLX 0.32.4, a
  new user's first task on an M2 no longer fails with "Maximum threads per
  threadgroup"; two OpenCode agents started in the same second no longer fail
  with "database is locked"; an agent that repeats its last call is led to a
  different call (to another tool if it insists), instead of a plan in words or
  a question that ended its task; Aider no
  longer writes a new file named after a sentence of the model's reply;
  several Gemma 4 requests at once with prompts longer than 512 tokens no
  longer get empty or wrong replies; eight or more requests at once no longer
  get "Connection reset by peer" on macOS 27; a model that nearly fills the
  Mac's memory (a 27B on 24 GB) runs several requests at once instead of one at
  a time, without running out of memory with sixteen: each request's own state
  is counted, the room kept for reading a prompt is what reading one took, and
  MLX keeps fewer freed buffers when the model leaves little room;
  several long prompts joining the requests already answering no longer hold
  their memory twice for a moment (three 27k-token prompts took a 4B model on
  a 16 GB Mac past the GPU's memory limit); while a request waits for memory,
  the running ones' growth is still made room for; a long prompt stops being
  read as soon as its client leaves; on a Mac where less than 64k tokens fit
  next to a large model, a longer request gets a context error instead of
  running the GPU out of memory.
- **Fixed, settings and commands:** `locals disconnect` no longer deletes an
  OpenCode provider of your own named "mlx", nor Hermes settings for another
  local server (LM Studio, llama.cpp); OpenClaw configs with commas or colons
  inside strings are read; `locals tune` next to a running server no longer
  erases tuning results (it asks to stop the server first); a model loading in
  `locals serve` is not loaded a second time by another command; a fine-tune
  with the same name as the running model is loaded instead of ignored; a
  server restarted after an upgrade keeps its `--ollama-port`, `--max-context`
  and other options; `locals --port N chat` uses that port; `locals bench` no
  longer leaves a second engine loaded; `locals start` while `locals serve` is
  still fetching the model on the same port waits for it instead of loading
  the model a second time; an agent left running when another command
  switched the server to another model gets a message saying so, where its
  requests were answered by the other model; a kernel shape whose tuning
  stops locals twice is left on MLX's kernel instead of stopping every start.
- **Fixed, APIs:** web pages using the OpenAI SDK can call the server
  (`Authorization` passes the CORS check); request bodies sent in chunks are
  read; Anthropic `stop_sequences` report `stop_sequence`; an Anthropic
  assistant prefill is continued; a file of many identical lines written
  through a tool call is no longer cut; replies stream as they are written
  when no tools are offered (Llama and Mistral formats); errors on the
  Anthropic and Ollama APIs come in their own shapes; the stock engine keeps
  serving after an error; a tool parameter that is a nested model (`$ref`, as
  FastMCP and Pydantic write it) or an optional number arrives as an object or
  a number instead of its JSON text, and a parameter schema written as `true`
  no longer fails the request; a request with a large JSON schema no longer
  pauses every other reply while its grammar is made (31 s for one with
  `maxItems: 50`, now 0.1 s), and one too large to serve is refused in seconds
  with a clear message (it took 214 s with `maxLength: 6000`); a JSON schema
  with `allOf` (a base and an extension, as OpenAPI and zod write them) gets
  one object with all the properties, where the reply was two objects back to
  back and no JSON; an integer stays within its schema's `minimum` and
  `maximum` (a rating from 1 to 5 could come out as 7); a request with tools
  and a `response_format` at once (an OpenAI Agents SDK agent with an output
  type) can call the tools, where the model made their answers up as JSON;
  Qwen can write tool parameters named like `$select`, `größe` or `user id`;
  a JSON schema `pattern` is a search, as JSON Schema says (zod's
  `startsWith("https://")` allowed only `"https://"` itself), and a `format`
  without a regex (Pydantic's `Path`, `SecretStr`) is no longer a 400; LFM2
  calls to tools named like `resolve-library-id` (Context7) are read as calls
  instead of coming back as text.
- **Fixed, safety and steadiness:** `locals run` writes the files it changed
  together or none of them, keeps their line endings, bytes and permissions,
  and a stop during that write lets it finish and says which files changed;
  each `locals solve` attempt works in a clone of its own (an agent's
  `git stash pop` no longer reached your stash), and OpenCode and the checks
  there can write only inside the attempt's copy, also through a link out of
  the project; copies of the project go as soon as they are read, and a
  stopped `solve --plan`, `swarm` or `/n` turn leaves none behind and keeps
  what finished; a system-prompt snapshot whose folder was deleted under a
  running server no longer fails every request in flight; `locals stop`
  during kernel tuning no longer counts as a crash of that kernel; every
  malformed request gets a 4xx answer, and a client that stops reading no
  longer holds a server thread; a reply cut at `--max-tokens` is said, and
  `locals generate --schema` then exits 1; OpenClaw's and OpenCode's configs
  are found where those tools look; the installer uses Python 3.12, the
  version Locals is tested on; a request's own `top_p`, `top_k` and `seed`
  apply to it (they were ignored), so a seeded request alone gives the same
  reply; a server tuning kernels while idle waits while another Locals
  server works, instead of keeping timings taken on a busy GPU; after an MLX
  upgrade, the kernels this Mac timed are timed again; `solve --plan` no
  longer stops its last version when the others gave nothing.
