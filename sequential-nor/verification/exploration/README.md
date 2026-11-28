# Explorations after Theorem 160 (not refereed, not part of any proof)

These are records of searches and exact computations in floating point that guided the work on
"all bases from noise 0.455". Nothing in the paper depends on them. Negative results are kept
so that they are not repeated.

## table_hunt/ — is there a base that defeats the table iteration above 0.455?

Conjecture tested: for every base and every noise level from 0.455 on, the table iteration with
reach C log2 n finishes in O(log n) rounds.

- No family with polynomial growth was found (fixed, mixed, random-window, multiplicative, block,
  barrier, hierarchical and hill-climbed periodic bases; noise 0.46, 0.50, 0.60; n up to 2^26).
- If no base pointer lies within the reach, the table does nothing: the rounds equal the pruned
  depth. So the question is the pruned iteration on bases whose pointers are all longer than
  about log n.
- The worst families at noise 0.46 are a base of period 4 with distances 160, 106, 318, 54 and
  the base with distance 49 * 2^v2(i). Their rounds fit (a log n)^2 to about 3%, a plain
  logarithm misses by 40 to 100%. The data cannot separate a logarithm with a large constant from
  a squared logarithm. So "O(log n) for every base from 0.455" may be false as stated; a
  polylogarithmic bound is the plausible target.
- There is no visible change of behaviour near 0.455: the growth slows smoothly from noise 0.36
  to 0.60.
- The sound set-valued variant can be much slower than the plain table iteration (distances 49
  and 50 alternating: 1375 against 153 rounds at n = 2^24, noise 0.46).
- The bundle layouts of the hardness proofs were not part of the hunt.

`hunt.c` is the simulator (usage in its header), `validate.py` compares it with the definitions on
small instances, `run_families.py`, `climb.py`, `climb2.py` drive the runs, `fits.py` fits the
growth; the `.txt` and `.tsv` files are the outputs.

## several_distances/ — two rows do not suffice for two distances

Base of period 2 with two distances that preserve the parity of the position: the upper row
reads two lower rows alternately. `per2.c` and `per2b.c` compute, with a fixed density d of ones
among resampled reads, the largest ratio

    E[open positions of the upper row] / E[open positions below]

over periodic input rows, for the exact chain with 64 states. At noise 0.46 and d = 0.4 the ratio
is 1.2 to 1.6 when only the positions that the upper row reads are counted, and 1.4 to 2.3 with
half weights on all positions of both lower rows; it is 1.12 at noise 0.60 and 0.73 at 0.70.
So the two-row argument of Theorem 160 cannot reach 0.455 for several distances. The reason: an
open run of a lower row is split between two upper rows, and seeds with gaps each start a run.

## more_rows/ — one distance, more rows: far below the hardness level

`per3.c` is the same computation for one distance with R rows above the input row (R = 2 is the
chain of Theorem 160). Largest ratio of the top row to the row below it, over all periodic inputs
of period 8 and hill climbing at period 24 to 30, with d = 0.338 (0.3345 from noise 0.30 down):

| noise | R = 2 | R = 3 | R = 4 |
|---|---|---|---|
| 0.455 | 0.948 | 0.801 | |
| 0.44 | 0.988 (one word) | 0.817 | |
| 0.42 | 1.029 (one word) | 0.838 | |
| 0.40 | | 0.857 | |
| 0.36 | | 0.896 | |
| 0.30 | | 0.946 | |
| 0.25 | | 0.984 | |
| 0.20 | | 1.019 | 0.966 |
| 0.10 | | 1.075 | 1.001 |

This led to Theorem 164 (three rows, proved from noise 0.30 on). Four rows should go lower still.

## Other closed routes (from the work before Theorem 160)

- A fixed distribution of hard instances does not follow from the embeddings (Remark 241).
- For all bases: no choice of norm or weights improves the chain count (the row sums of the
  powers of the kernel are exact); conditioning only on long pointers gives a relevance of about
  0.889 against 0.9 per gate; fractional moments give the same recursion.
- For the potentials of Lemma 159: potentials that are nonnegative at every state, and a density
  that may vary over the whole interval [l_4, mu_4] inside one window, are too pessimistic; narrow
  boxes of the density around its value at the start of the window work.
