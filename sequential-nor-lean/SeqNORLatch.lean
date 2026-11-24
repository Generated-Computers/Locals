/-!
# The latch chain of the latch embedding: the pair identity and monotonicity

A gate computes `nor p y`, where `p` is the value just before it and `y` the value it reads.
Two consecutive gates that read `y` and `y'` turn the state `x` into `(x || y) && !y'`
(`pair`): a one read by the first gate sets the state, a one read by the second resets it.
A chain runs over a list of such pairs (`lat`).  It is monotone: raising the state before the
chain, raising the reads of first gates and lowering the reads of second gates can only raise the
result (`lat_mono`).  This is part (a) of the lemma "latch" of the note.  `sandwich` is the
deterministic step in the proof of the latch embedding theorem: if every read lies between a
lower and an upper version, the last gate of the chain lies between the chain run from state 0 on
(lower first reads, upper second reads) and the chain run from state 1 on (upper first reads,
lower second reads), whatever the state before the chain.  Lean 4 core only.
-/

namespace SeqNORLatch

def nor (a b : Bool) : Bool := !(a || b)

/-- Two consecutive gates form a set/reset latch. -/
theorem pair (x y y' : Bool) : nor (nor x y) y' = ((x || y) && !y') := by
  cases x <;> cases y <;> cases y' <;> rfl

/-- A chain: the gates read the pairs of the list in order. -/
def lat (x : Bool) : List (Bool × Bool) → Bool
  | [] => x
  | (y, y') :: rest => lat (nor (nor x y) y') rest

/-- `a ≤ b` for bits. -/
def le (a b : Bool) : Prop := a = true → b = true

/-- One list of reads is below another: first entries are smaller, second entries larger. -/
inductive Below : List (Bool × Bool) → List (Bool × Bool) → Prop
  | nil : Below [] []
  | cons {y y' z z' : Bool} {ps qs : List (Bool × Bool)} :
      le y z → le z' y' → Below ps qs → Below ((y, y') :: ps) ((z, z') :: qs)

theorem step_mono {x x2 y z y' z' : Bool} (hx : le x x2) (hy : le y z) (hy' : le z' y') :
    le (nor (nor x y) y') (nor (nor x2 z) z') := by
  cases x <;> cases x2 <;> cases y <;> cases z <;> cases y' <;> cases z' <;>
    simp_all [le, nor]

theorem lat_mono {ps qs : List (Bool × Bool)} (h : Below ps qs) :
    ∀ {x x2 : Bool}, le x x2 → le (lat x ps) (lat x2 qs) := by
  induction h with
  | nil => intro x x2 hx; exact hx
  | cons hy hy' _ ih => intro x x2 hx; exact ih (step_mono hx hy hy')

theorem sandwich {lo mid hi : List (Bool × Bool)} (h1 : Below lo mid) (h2 : Below mid hi)
    (x : Bool) : le (lat false lo) (lat x mid) ∧ le (lat x mid) (lat true hi) :=
  ⟨lat_mono h1 (fun h => nomatch h), lat_mono h2 (fun _ => rfl)⟩

end SeqNORLatch
