# Sequential NOR on Random and Smoothed Instances: source and verification

The paper is [`../Sequential_NOR_on_Random_and_Smoothed_Instances.pdf`](../Sequential_NOR_on_Random_and_Smoothed_Instances.pdf)
(Generated Computers, 2026). This folder holds what stands behind it.

- [`paper.tex`](paper.tex): the LaTeX source.
- [`verification/scripts/`](verification/scripts/): the programs that the proofs and remarks
  cite. Section 22 of the paper says what each one does and how to run it. Programs that are part
  of a proof work in exact rational or interval arithmetic.
- [`verification/referee-scripts/`](verification/referee-scripts/): programs that automated
  referees wrote to check sections of the paper; kept as a record.
- [`verification/exploration/`](verification/exploration/): records of searches that are not part
  of any proof, with their negative results.
- The Lean files for the deterministic lemmas of the embeddings are in
  [`../sequential-nor-lean/`](../sequential-nor-lean/).

## Results added in this version

- **Fixed-distance bases, counting positions (Theorems 160 and 164).** On the base in which
  every gate reads the gate H positions back, with H large, the relevance-pruned iteration
  finishes in O(log n) rounds at every noise level from 0.30 on. The earlier bound for this
  family started at 0.69. The level 0.30 is below 0.4468, up to which the smoothing of the worst
  base is as hard as P in BPNC; so at those noise levels the hardness belongs to particular
  bases and not to long pointers as such. The proof follows the open positions of two, then
  three, consecutive rows as a Markov chain and uses potentials checked in interval arithmetic
  (`rows_check.c`, `rows3_check.c`).
- **Dependent reads (Theorem 231).** A first proved bound for designs whose reads depend on each
  other (`scan_check.c`).
- **Greedy independent sets (Corollary 239).** The problem is the lexicographically first maximal
  independent set on a path with one extra edge per vertex. With a fraction of those edges
  random it stays as hard as the worst case up to 0.4468, is in NC for every such graph from
  0.7288 on, and the standard parallel greedy algorithm is slow on some of these graphs below
  0.3819.
- **What did not work** is recorded in `verification/exploration/README.md`: no base was found
  that defeats the table iteration above noise 0.455, but the worst families look like a squared
  logarithm; and the two-row argument fails for bases with two distances.

Not refereed by a human. The proofs were developed with the assistance of an AI system; the
paper's status paragraph and Section 22 say which steps rest on computations and which were read
by automated referees. Nothing here bears on P versus NC in the worst case.
