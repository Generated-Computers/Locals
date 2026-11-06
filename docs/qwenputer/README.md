# Qwenputer: a computer inside the model

[Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) (Divyansh Agrawal) is Qwen3.5-4B with a RISC-V
computer compiled into its weights: 11 layers in front of Qwen's own 32 that are an RV32IM + Zba core with a bignum
coprocessor. Its RAM, ROM and registers are the Gated DeltaNet states of those layers. When Qwen writes a call of
its `calc` tool, the computer layers capture the call's text. At `</tool_call>` the computer runs the calc
firmware, its own trace tokens one after another, which Qwen's layers never see. It prints the exact result, which
comes back to Qwen as ordinary text, one byte a token.

```
locals start --model qwenputer        # calc on the computer: the faster one
locals start --model qwenputer-cpu    # its computer also doing work of the Mac's CPU: slower
```

Two versions of the same model:

| | `qwenputer` | `qwenputer-cpu` |
|---|---|---|
| The computer runs | `calc`, the authors' firmware | `calc` and a toolbox of 60 programs ([below](#the-toolbox-work-a-clients-tools-would-do-done-by-the-computer)) |
| It also | | checks the client's tool calls and the numbers an answer copies, keeps notes, runs a small Python |
| 60 arithmetic questions, 8 at a time | 59 right, 155 tokens/s in total | 60 right, 107 tokens/s |
| The same, one at a time | 47 tokens/s | 27 tokens/s |
| 80 toolbox questions | 38 right in 113 s (Qwen works them out itself) | 80 right in 59 s |
| In an editor | `locals/qwen3.5-4b-qwenputer` | `locals/qwen3.5-4b-qwenputer-cpu` |

`qwenputer` is Qwenputer as its authors made it, run fast. `qwenputer-cpu` is the idea taken further: work a client's
tools would do on the Mac's CPU (counting, sorting, dates, JSON, tables, statistics, regular expressions, a program),
done by the model's own computer instead, exactly, at the cost of speed.

The first start downloads the 18.8 GB checkpoint and makes a 123 MB pack of the computer from it, in about 30 seconds.
The pack goes in `~/.locals/qwenputer/` (`LOCALS_HOME`), and the checkpoint can be deleted afterwards. Loading the
firmware into the pack needs the authors' package, once:

```
pip install "chipformer @ git+https://github.com/divagr18/qwenputer@ca13310e2f9f788d1dd3feeea8213c1b0c23913c#subdirectory=experimental/riscv"
```

(`LOCALS_CHIPFORMER` may name a checkout's `experimental/riscv` folder instead.) Both versions read the same
checkpoint and pack. The `calc` tool is always offered to it, and the client never sees a call of it: a calc exchange the
model runs before its answer comes back as `reasoning_content`, the answer as `content`. Calls of the client's own
tools are the client's, as with any model.

```json
{"reasoning_content": "<tool_call>\n<function=calc>\n<parameter=expression>\n123456789 * 987654321\n</parameter>\n</function>\n</tool_call>\n<tool_response>\n121932631112635269\n</tool_response>",
 "content": "123456789 * 987654321 = 121,932,631,112,635,269"}
```

## What the checkpoint is, measured

`locals/models/qwenputer.py` reads the checkpoint one tensor at a time:

- **Qwen's part is Qwen3.5-4B.** Quantized to 4 bits, its 32 layers equal mlx-community's Qwen3.5-4B-MLX-4bit bit
  for bit. Locals runs them as that model, with its kernels.
- **The computer stays in its own dims.** Its layers read and write only the 1,408 dims after Qwen's 2,560, and of
  those 1,327 at most. They have 1.05 million nonzero weights, which take 796 distinct values.
- **The computer's norms never depend on Qwen's part.** Every embedding row carries 62 "ballast" dims of 2^40, so
  each RMSNorm of a computer layer divides by exactly 2^37. Qwen's part of the residual is below fp32's resolution
  of that sum. Outside the ballast the residual stays at 128 or less, so the sum is the same in any order.
- **Qwen's norms see only Qwen's part.** After the computer, a text token's residual is Qwen's embedding row plus
  the computer's output code (+-2^-20, set where the computer predicts the next token) and two marks (2^-40 and 0).
- **The computer's embedding rows are compact.** Every text row is one shared row plus the token's first 16 bytes as
  bits, except 262 special tokens. A trace token's row is its family's row plus one row for each set payload bit,
  except the QS and QD families and the controls.

`convert` makes the pack from these facts: the computer's layers cut to their nonzero rows and columns, and the
embedding as those few rows. `add_snapshot` adds the firmware: the parked calc program's state, as the authors'
`core.snapshot` lays it into the recurrent states. Decoding follows the authors' `rv/chat.py`. Every token goes
through the computer layers; a text token also goes through Qwen's. The next token is the computer's where its
output code is set, and Qwen's otherwise.

## Running the computer

Every token of a run is the computer's own prediction from the token before: Locals drafts nothing, certifies nothing
against a draft, and the batch engine's prompt-lookup drafts are off for Qwenputer's replies. Each way of running the
computer gives the same values as transformers' own Qwen3.5 layers on the checkpoint, at every position and in every
state (`tests/test_qwenputer.py`; 369 positions of prompt, tool call and trace tokens against transformers):

- **Rows a token touches, in place.** The computer's memory is the recurrent state: row k of a head is the cell a
  one-hot key k names. A token's key writes at most one row and its query reads at most two (every token measured).
  So a head updates those rows in place and leaves the other 125 alone, which is the full update's result, since it
  adds exactly 0 to them. The two counter heads of layer 0, whose state decays, update only the rows a mask has seen
  written.
- **Known tokens read in block passes.** The computer takes over only after `</tool_call>`, so it reads the prompt
  and Qwen's text there, in passes of up to 2,048 tokens, which also check it predicts nothing earlier. It keeps its
  state at the prompt's checkpoints: the calc tool's system turn is 369 of a question's 415 tokens.
- **A run on a CPU core.** A run is a chain: each token depends on the one before through every layer (Jacobi
  iteration, guessing a window and reading it again, settles 1.6 tokens a pass). It is made by the GPU stepper's
  operations in their order, in C (`locals/models/qwenputer_cpu.py`, built once with the Mac's clang), on a
  performance core. The computer's sparsity makes a core fast, and every shortcut gives the same bits:
  - a sparse product walks only the columns whose input is nonzero (7 to 23% of the normed residual, at most 16 of
    a layer's 4,096 head outputs) in ascending order. Each row stores its entries in that order, so its terms are
    added as the stepper adds them, and a zero input adds only zeros, which change no sum;
  - a head that no query or key entry touches, whose state does not decay, changes nothing and outputs 0;
  - the attention layers' convolution channels are stepped four at a time (NEON), each channel's sums in the
    stepper's order; one whose inputs are all +0 gives +0 and keeps its +0 state. SiLU is a table over the 65,536
    bf16 values its input takes, made by the same expression;
  - a norm's sum of squares is the ballast's alone when every other square is under half a unit in the last place of
    that sum (checked, else summed in order): each addition rounds back to it;
  - the 62 ballast dims never change and come first in every row: a product's sums over them are made once a run.

  A run alone shares its attention layers' key and value heads and its MLP layers' units out to a few cores (the
  same work on whichever core does it: each row's sum in its order).
- **Or on the GPU.** Without clang (or with `LOCALS_QWENPUTER_CPU=0`) runs are stepped on the GPU: two kernels a
  layer, the tokens decided by a kernel, blocks of 32 steps compiled once and queued ahead, the runs of several
  replies stepped as one batch that runs join and leave. Next to Qwen's batch its kernels wait on the memory Qwen's
  weights keep busy, which is why a core is the default.

In a server, a reply that calls calc stays in the batch at `</tool_call>`, parked: its row reads nothing (its
recurrent layers keep their state, and the attention slots of the steps it waits become left padding). The computer
reads the call and makes the run in a thread of its own, the run on a core (several replies' runs at once; a run
alone on up to 4). The core works on the computer's state where the GPU keeps it: on Apple silicon the GPU's memory is
the CPU's, so the run updates the device's own buffers in place, and nothing is copied out or back. The row then goes on, Qwen reading the computer's echoed result in one pass of its own. A finished
reply's slot waits a few steps, and the next request takes it in place (its cache written into the slot, not the
whole batch copied twice). `LOCALS_QWENPUTER_PARK=0` keeps the earlier way: the reply leaves the batch at the call and
comes back as a request that continues its kept cache.

## The toolbox: work a client's tools would do, done by the computer

calc is one program the computer runs. `qwenputer-cpu` adds more, so that work a client would otherwise run on the
Mac's CPU for the model (a tool's code) is done by the model's own computer, exactly:

| Tool | What it does |
|---|---|
| `count(text, find)` | how many times a letter or a piece of text occurs (non-overlapping, ASCII letters in either case) |
| `length(text)` | a text's characters (UTF-8) |
| `reverse(text)` | a text backwards, character by character |
| `sort(numbers)` | up to 256 numbers (up to 9 digits and 9 decimals: `12`, `-3`, `4.5`, `.5`) in ascending order, each as written; a fraction, a thousands separator, a unit or an exponent is an error, not a number read in two |
| `weekday(date)` | the day of the week of a date (`YYYY-MM-DD`, the Gregorian calendar) |
| `days_between(start, end)` | the days from one date to another |
| `add_days(date, days)` | the date a number of days after a date (before it, if negative), within the years 0000 to 9999 |
| `words(text)` | a text's words |
| `to_base(number, base)` | a number below 2^32 (decimal, or `0x`, `0b`, `0o` and its digits) in another base (2 to 36) |
| `count_letters(text, letters)` | how many characters of a text are any of the letters (a word's vowels: `aeiou`; ASCII letters in either case) |

They are RV32 assembly (`locals/models/qwenputer_tools.s`, built with the authors' assembler), linked after calc's
firmware: where calc's main loop resumes from its park, a jump goes to the toolbox. A call whose first rows are
calc's as Qwen writes them (`\n`, `<`, `function`, `=`, `calc`, `>`, checked a word at a time) goes on in calc's own
code at once, and gets the response calc's firmware gives (its run is 52 trace tokens longer); any other goes to the
dispatcher, which reads the call's function name from the text buffer (the call's tokens, as the bridge captures
them), and a calc call goes on in calc's code from there, every register as calc parked. Another tool's call is parsed, run, and answered as calc answers (`<tool_response>` ... `</think>`).
The firmware is one more snapshot in the chip pack (made once, in about 2 s), and a tool's run on the computer equals
the authors' reference machine's, token for token (`tests/test_qwenputer.py`). A call takes 2,000 to 3,700 trace
tokens: 0.5 to 1.1 s on a core.

The bridge captures a call's first 80 tokens, the first 16 bytes of each. The server gives the computer a toolbox
call with each longer token (a run of spaces or dashes, a Chinese word) as the vocabulary's shorter tokens it is made
of, and a call of more than 80 tokens in parts: calls of 80 rows whose first row marks them (the first part, one
between, the last), whose text the toolbox adds to its own (15 KB), parking without a word, a run each; at the last
part it runs the call, and only that response is shown, and read by Qwen. Qwen's own text stays as it wrote it. Through the server, one at a
time: the words of a 220-word text, or how many times "the" occurs in it, right in 11.4 s (3 parts, then a run of
15,000 to 17,000 trace tokens: 4.2 to 4.4 s on a core; the rest is Qwen writing the text into its call and
answering); 45 integers sorted, 13.2 s; 20 prices, 8.3 s.

A request is offered the tools its last user message asks for (letters, characters, words or backwards; sorting;
dates or days of the week; binary, octal, hexadecimal or another base): with any of them in the prompt, Qwen answered the arithmetic questions below differently (54 to 58
of 60 right, against 59 with calc alone: it went on to check calc's result by hand), so a question that asks for none
of them gets calc's prompt alone. A tool the client defines under the same name is the client's. The value a toolbox
tool prints goes to Qwen in the tokenizer's own tokens, as a client's tool result does (the computer prints a byte a
token, and Qwen copied a word written backwards a letter wrong from those). `LOCALS_QWENPUTER_TOOLS=1` gives
`qwenputer` the toolbox, and `LOCALS_QWENPUTER_TOOLS=0` takes it from `qwenputer-cpu`.

## The toolbox in C

Most of the toolbox is C (`locals/models/qwenputer_c/`), compiled for the computer (RV32IM + Zba) by its `build.py`
with a clang that targets RISC-V (zig's, from `pip install ziglang`, or LLVM's) and linked at fixed addresses after
calc's firmware and the assembly: code from 0x48000, read-only data from 0x22000, its tables in RAM's 0x800..0x1a00.
The dispatcher calls it for a function it has not got. The compiled code is in the repository
(`qwenputer_tools_c.s`, with the sources' digest: a test fails if they differ), so a Mac that serves Qwenputer needs
no RISC-V compiler. Built for the host (`host.c`), each tool is checked against Python's own (json, re, Decimal,
statistics, hashlib, zlib, base64, datetime, brute-force logic; `tests/test_qwenputer_c.py`), and on the computer
each run equals the authors' reference machine's, token for token.

| Kind | Tools |
|---|---|
| JSON | `json_check` (valid, or where and why not), `json_format` (pretty or compact), `json_get` (a path: `users[0].name`, `items[*].price`), `json_count` |
| Tables (CSV, tab-separated, Markdown) | `column_stats` (count, sum, average, median, min, max), `table_sort`, `table_filter` (`> 100`, `= Paris`, `contains York`), `table_totals` (a column added up by group) |
| Versions and lists | `compare_versions`, `sort_versions` (semver's order), `compare_lists` (in both, in one only) |
| Logic and patterns | `truth_table`, `logic_equivalent` (a counterexample if not), `regex_find`, `regex_replace` (Python's syntax and matches) |
| Numbers | `stats` (mean, median, mode, standard deviations), `percent_change`, `round_number` (half up), `compound_interest`, `loan_payment`, `is_prime`, `factor` (to 19 digits: Miller-Rabin, Pollard-Brent), `convert_units` (exact factors), `roman_numeral`, `number_to_words` |
| Text | `replace_text`, `change_case` (snake, camel, kebab...), `text_part` (the n-th word, line, character or sentence), `sort_lines`, `unique_items`, `word_frequency`, `is_palindrome`, `is_anagram`, `validate` (email, IPv4, URL, UUID, date, time, hex color), `encode` and `decode` (base64, hex, URL, ROT13, Caesar, Morse, binary), `edit_distance` |
| Times | `time_add`, `time_between`, `age`, `date_info` (ISO week, day of the year, leap year), `unix_time`, `convert_timezone` (fixed offsets) |
| Codes | `check_digit` (Luhn, ISBN, EAN, UPC, IBAN), `hash_text` (SHA-256, CRC-32, Adler-32) |
| Notes | `remember`, `recall`, `forget`: in the computer's RAM, and the conversation's earlier notes given to the next request's computer with its first notes call |
| A small Python | `run_python`: integers (to 2^62), strings, lists, tuples, dicts, loops, functions, comprehensions, f-strings, math's gcd, isqrt, factorial, comb; no floats (an exact `/`, else an error), no `try` or `class`; within 1,250 steps (a builtin's scanning, copying and converting counted too), about a loop of 250 rounds, which keeps a run inside the computer's limit; a loop's round gives back the memory it made unless something of it went into an older list, dict or variable |

Big numbers (statistics' sums of squares, money's powers, IBANs) are exact in 512 bits. A call takes 3,000 to 8,000
trace tokens (1 to 2 s on a core) for most of them; `stats` of 8 numbers 37,000, `compound_interest` 84,000,
`loan_payment` 110,000 and `factor(600851475143)` 124,000 (about 28 s). A `run_python` program walks its tree at
about 90 trace tokens a step: a loop of 100 rounds is 63,000 (15 s), and at its 1,250 steps a run is at most about
215,000, whatever the program does (measured on the reference machine).

## What made the tools cheaper

A tool's time on the computer is the length of its run: every trace token is one step through all 11 layers, about
0.3 ms on a core. These changes made the runs shorter, in trace tokens (measured on the reference machine when each
change was made):

| Tool | What changed | Before | After |
|---|---|---|---|
| `calc` under the toolbox | a call whose first rows are calc's goes on in calc's code at once, checked a word at a time, instead of the dispatcher reading its name a byte at a time and saving every register | 473 more than calc's own | 52 more |
| `count`, 1,100 characters | letters folded with a table, not a call a byte | 38,000 | 16,000 |
| `count_letters`, 1,100 characters | the same, and the letters as a 26-bit mask | 63,000 | 20,000 |
| `sort`, 201 numbers | a Shell sort | 206,000 | 88,000 |
| `stats`, 120 numbers | sorted, summed and squared in machine words where the numbers fit 62 bits; big division a word at a time (Knuth's algorithm D) | 369,000 | 141,000 |
| `run_python`, a 20-line program | names found through first-letter chains, not by going through each list; operators by precedence climbing, not a call for each of six levels | 139,000 | 92,000 |
| `run_python`, a 36-line program | the same | 240,000 | 151,000 |

Measured in the computer's instructions (a trace token is about 1.3 of them), on inputs of about 950 tokens: `json_get`
of `[*].name` 137,000 to 104,000, `column_stats` 224,000 to 95,000 (row numbers in the table's own memory, 24-byte rows,
the median by selection instead of a sort), `table_totals` 139,000 to 115,000, `table_filter` 108,000 to 89,000. On
such inputs `regex_find` of `\d+` (it ran out of steps) and `regex_replace` (it ran out of memory) now finish: a greedy
one-character repeat is one instruction of the regex machine, a class's first byte is skipped to, and a pattern that
starts with `\b` starts only at word boundaries. Finding where a parameter ends reads 4 bytes a step.

## The computer's own checks

`qwenputer-cpu`'s (they are programs of its toolbox):

- **The client's tool calls.** With the client's tools in the request, a call of one of them waits at its
  `</tool_call>` while the computer checks it against the tool's schema (`check_call`: required and unknown
  parameters, the JSON types `oai.coerce` reads, enums). A right call goes to the client as written, and the model
  goes on after it; a wrong one is the computer's exchange (reasoning before content), the model reads what is
  wrong and writes the call again (twice at most). The check is a run of 6,000 to 8,500 trace tokens: 1.7 to 2.3 s a
  call. A call the computer cannot check goes as written: one longer than it reads (about 1,000 tokens with the
  schema), one inside the model's reasoning, one without a function name, and one whose check fails on the
  computer's side (its run, or an error of its own such as running out of memory). `LOCALS_QWENPUTER_CHECK_CALLS=0`
  turns it off.
- **Numbers the answer copies.** A number of six digits or more in the answer, after a result of the computer's as
  long and not written in the question, waits while the computer compares it with the results (`copy_check`, on a
  computer of its own, one a core); a miscopied one is shown as the result, written as the model wrote numbers
  (its thousands separators, its minus sign). The reply's end waits for its last check. A number longer than 100
  tokens is shown as written, unchecked: held, a model writing ",000" on and on was hidden from the server's loop
  guard until the reply's token limit. This turned the arithmetic's
  one miss (10^38 with a group of zeros dropped) right, and costs throughput where every answer is a long number:
  60 of 60 right at 101 tokens/s in total against 59 of 60 at 143 (the 60 questions, 8 at a time, the same day).
  `LOCALS_QWENPUTER_COPY=0` turns it off.

- **Runs past the limit.** A run that goes past 250,000 trace tokens (a minute or two on a core) is cut: the model
  reads an error as the tool's response ("give it a smaller job"), and the reply's next call gets a computer made
  again from the reply so far. A run that ends with its own budget's error (a program over `run_python`'s steps, a
  pattern over `regex`'s) counts as cut too: it took a minute to fail, and the model wrote another program, and
  another. After two such runs in a reply, a toolbox call gets an error at once, without a run, and `calc` still
  runs.
- **Notes.** The server keeps the notes calls (remember, forget) a reply's computer ran by the conversation before
  the reply and its content, so another conversation never gets them; the next request's computer gets the notes
  they left (the newest that fit its 2,048 bytes, and the call), with its first notes call.

## Numbers

On an M2 Pro with 16 GB. The computer's speed is on a recorded calc run (`123456789 * 987654321`: 2,460 trace
tokens after the firmware), every run equal to the authors' reference machine's, token for token:

| The computer's run | Tokens/s |
|---|---|
| MLX operations, one token at a time | 142 |
| Stepped on the GPU | 1,690 (5,085 for 8 runs at once) |
| Stepped on the GPU while Qwen decodes 8 replies | 740 |
| On a performance core, alone or next to Qwen's batch | 3,300 (4,100 shared out to 3 or 4 cores) |
| Reading known tokens (a prompt) in block passes | 24,400 |

Through the server, 60 arithmetic questions with exact answers (`docs/qwenputer/arithmetic.py`, temperature 0):

| | Right | 8 at a time: tokens/s in total (the 60 in) | One at a time |
|---|---|---|---|
| `qwenputer` | 59 | 153 to 157 (43 s) | 47.3 tokens/s |
| `qwenputer-cpu` | 60 | 105 to 109 (61 s) | 27.1 tokens/s |
| `qwenputer-cpu`, copy checks off | 59 | 151 (44 s) | |
| Qwen3.5-4B | 5 | 237 (136 s: its replies are 5 times longer) | |

Without copy checks (`qwenputer`), the one missed is Qwen's copy: the computer's result for 10^38 is right, and the
answer writes it with a group of three zeros missing. `qwenputer-cpu` is slower on every question, as its copy checks
hold each long number the answer copies until the computer has read it (a run of its own), and with the toolbox's
firmware a calc call goes through its dispatch first. One request at a time, Qwenputer decodes plain text at 67.6 tokens/s (Qwen3.5-4B:
68.1) and answers `Use calc: what is 123456789 * 987654321?` in 2.5 s, with 121,932,631,112,635,269. Qwen3.5-4B,
asked the same, wrote 121,932,699,999,999,999 in 3.1 s.

80 questions for the toolbox (`docs/qwenputer/toolbox.py`: 8 each of letter counts, lengths, words backwards, sorting,
weekdays, days between dates, dates some days on, words in a sentence, numbers in binary, octal or hexadecimal, and a
word's vowels; temperature 0, 8 at a time):

| | Right | Time | Requests |
|---|---|---|---|
| `qwenputer-cpu` | 80 | 59 s (50 s on an earlier day) | 80 |
| Qwen3.5-4B, the client running the same tools (Python) | 80 | 197 s | 160 |
| `qwenputer` (calc alone) | 38 | 113 s | 80 |
| Qwen3.5-4B, no tools | 38 | 106 s | 80 |

Without the toolbox Qwen wrote no word backwards right (0 of 8), at most 2 of 8 weekdays, spans of days or dates some
days on, and 1 to 3 of 8 numbers in another base (it sorts and counts a sentence's words right), writing three and a
half to four times the tokens reasoning about them. With the same tools run by the client, it answers as the toolbox
does, but every question takes a second request (the tool's result sent back and read), and the 80 take four times
as long; the client's Python spent 2.3 ms on the tools themselves. Arithmetic stays as it was with the toolbox.

45 questions of the C tools' kinds (`docs/qwenputer/toolbox_more.py`: 3 each of JSON, tables, versions, lists,
logic, regular expressions, statistics, money, primes, units, roman numerals, text, times, check digits and a
computation; temperature 0, 8 at a time; the answer the last bold number, else the last number):

| | Right | Time | Reply tokens |
|---|---|---|---|
| `qwenputer-cpu` | 40 to 41 | 217 to 248 s | 13,974 to 14,284 |
| Qwen3.5-4B, no tools | 31 | 110 s | 24,579 |

Qwen alone got no prime factor, check digit or regular expression's count right, 1 of 3 roman numerals and truth
tables; with the toolbox all the primes, roman numerals, money and units are right. Of `qwenputer-cpu`'s misses, three
replies state the right answer and the grader reads another number (it takes the last one in bold: a row's label, the
card number with its check digit); in the others Qwen ran out of its 1,500 tokens checking a right result by hand.

## Limits

- Replies are not held to a schema or a regular expression. A request that asks for one gets an error, and calls of
  the client's tools are written without our grammar.
- A calc exchange after the model has begun its answer is left out of the reply (the model still read it).
- `calc` is the authors' firmware: integers and fractions of up to 62 digits, decimals for `+ - *`.
- The computer is slow next to the Mac's CPU (about 3,500 instructions a second on a core): a `run_python`
  program has 1,250 steps and is about 700 bytes at most (parsing it takes most of the computer's 10 KB for a
  program; a longer one gets an error that says so), `factor` finds factors up to about 20,000 (100,000 below 2^32) and
  says when a number left has larger ones, and heavy tools take 20 to 30 s.
- A toolbox call holds up to about 1,000 tokens and 7,000 characters (Qwen writes each digit as one: about 200
  numbers to sort, 700 words); a longer one gets an error, and Qwen answers on its own. A long call is slow on the computer: counting a
  900-token text's words takes a run of 100,000 trace tokens (about 30 s on a core), writing it backwards 200,000.
