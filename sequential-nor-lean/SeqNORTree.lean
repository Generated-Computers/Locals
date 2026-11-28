/-!
# The NOR-tree embedding: exactness and domination, in abstract form

Machine-checked core of Section 4 of the note
"Sequential NOR: average-case logspace, and hardness of smoothed instances up to noise 3/20".

A gate of Sequential NOR computes `nor p y`, where `p` is the value just before it and `y` is the
value it reads.  In a smoothed instance `y` is the value at the listed pointer, flipped when the
gate *misreads*.  The model below keeps exactly this and forgets positions:

* a leaf is a zero gate followed by a gate that reads a copy of an input;
* a node is one gate; it reads the last gate of its first block `L` and comes right after its
  second block `R` (post-order layout), so `R` starts right after `L`.

`domination` is Lemma 26 (with every flag replaced by the smallest event it is known to contain:
a misread for `B`, a misread of a `0` for `A`).  `exact` and `want_tree` are Lemma 25.
Lean 4 core only, no Mathlib; the only axioms used are `propext`, `Quot.sound` and
`Classical.choice` (see the `#print axioms` lines at the end).
-/

namespace SeqNORTree

def nor (a b : Bool) : Bool := !(a || b)

/-- A block of the compiled instance together with what happened to it under smoothing.

`leaf b cv mz ml`: the input has value `b`, the copy that is read holds `cv`, and `mz`, `ml` say
whether the zero gate and the reading gate misread.

`node L R mk`: `L` is the block read through the pointer, `R` the block right before the gate,
and `mk` says whether the gate misreads. -/
inductive Blk where
  | leaf (b cv mz ml : Bool)
  | node (L R : Blk) (mk : Bool)

open Blk

/-- The value of the last gate of a block, given the value `p` just before its first gate. -/
def out : Blk → Bool → Bool
  | leaf _ cv mz ml, p => nor (nor p (true != mz)) (cv != ml)
  | node L R mk, p => nor (out R (out L p)) (out L p != mk)

/-- The intended value: the value in the unsmoothed instance. -/
def want : Blk → Bool
  | leaf b _ _ _ => !b
  | node L R _ => nor (want R) (want L)

/-- The dominating event `D` of Section 4.3.  `B`-flags are replaced by "the gate misreads" and
`A`-flags by "the gate misreads a `0`". -/
def dom : Blk → Bool → Bool
  | leaf b cv mz ml, _ => if b then (cv != b) || ml else (cv != b) || mz || (ml && !cv)
  | node L R mk, p =>
    match want L, want R with
    | false, false => dom L p || dom R (out L p) || (mk && !(out L p))
    | true, true => dom R (out L p) && (dom L p || mk)
    | true, false => dom L p || mk
    | false, true => dom R (out L p)

/-- **Lemma 26 (domination).** A block whose last gate differs from its intended value has its
dominating event, whatever the value before its first gate. -/
theorem domination : ∀ (t : Blk) (p : Bool), out t p ≠ want t → dom t p = true := by
  intro t
  induction t with
  | leaf b cv mz ml =>
    intro p
    cases b <;> cases cv <;> cases mz <;> cases ml <;> cases p <;> decide
  | node L R mk ihL ihR =>
    intro p h
    have hL := ihL p
    have hR := ihR (out L p)
    simp only [out, want, dom] at h ⊢
    generalize out R (out L p) = oR at *
    generalize dom R (out L p) = dR at *
    generalize out L p = oL at *
    generalize dom L p = dL at *
    generalize want L = wL at *
    generalize want R = wR at *
    cases wL <;> cases wR <;> cases oL <;> cases oR <;> cases mk <;> cases dL <;> cases dR <;>
      simp_all [nor]

/-- No misread anywhere, and every copy holds the value of its input. -/
def clean : Blk → Bool
  | leaf b cv mz ml => (cv == b) && !mz && !ml
  | node L R mk => clean L && clean R && !mk

/-- **Lemma 25 (exactness).** A clean block ends with its intended value, whatever precedes it. -/
theorem exact : ∀ (t : Blk), clean t = true → ∀ p : Bool, out t p = want t := by
  intro t
  induction t with
  | leaf b cv mz ml =>
    intro h p
    cases b <;> cases cv <;> cases mz <;> cases ml <;> cases p <;> simp_all [clean, out, want, nor]
  | node L R mk ihL ihR =>
    intro h p
    simp only [clean, Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨⟨hL, hR⟩, hk⟩ := h
    simp [out, want, ihL hL, ihR hR, hk]

/-- The dominating event is not vacuous: it fails on every clean block. -/
theorem dom_clean : ∀ (t : Blk), clean t = true → ∀ p : Bool, dom t p = false := by
  intro t
  induction t with
  | leaf b cv mz ml =>
    intro h p
    cases b <;> cases cv <;> cases mz <;> cases ml <;> simp_all [clean, dom]
  | node L R mk ihL ihR =>
    intro h p
    simp only [clean, Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨⟨hL, hR⟩, hk⟩ := h
    simp only [dom, ihL hL, ihR hR, hk]
    cases want L <;> cases want R <;> simp

/-- The complete tree of depth `h` over an input of value `b`, with no noise. -/
def tree : Nat → Bool → Blk
  | 0, b => leaf b b false false
  | h + 1, b => node (tree h b) (tree h b) false

theorem clean_tree : ∀ (h : Nat) (b : Bool), clean (tree h b) = true := by
  intro h
  induction h with
  | zero => intro b; cases b <;> rfl
  | succ h ih => intro b; simp [tree, clean, ih]

/-- **Lemma 25, values.** A tree of even depth holds the complement of its input, a tree of odd
depth holds the input. -/
theorem want_tree : ∀ (h : Nat) (b : Bool), want (tree h b) = if h % 2 = 0 then !b else b := by
  intro h
  induction h with
  | zero => intro b; simp [tree, want]
  | succ h ih =>
    intro b
    have hmod : (h + 1) % 2 = 0 ↔ ¬ (h % 2 = 0) := by omega
    by_cases hh : h % 2 = 0
    · have : ¬ ((h + 1) % 2 = 0) := by omega
      cases b <;> simp [tree, want, ih, hh, this, nor]
    · have : (h + 1) % 2 = 0 := by omega
      cases b <;> simp [tree, want, ih, hh, this, nor]

/-- The cell of a gate `NOR(u, v)`: two trees of odd depth and one top gate.  Without noise its
last gate holds `nor v u`, whatever precedes the cell. -/
theorem cell_exact (d : Nat) (hd : d % 2 = 1) (u v p : Bool) :
    out (node (tree d u) (tree d v) false) p = nor v u := by
  have hc : clean (node (tree d u) (tree d v) false) = true := by
    simp [clean, clean_tree]
  rw [exact _ hc p]
  have hne : ¬ (d % 2 = 0) := by omega
  simp [want, want_tree, hne]

end SeqNORTree

#print axioms SeqNORTree.domination
#print axioms SeqNORTree.dom_clean
#print axioms SeqNORTree.exact
#print axioms SeqNORTree.want_tree
#print axioms SeqNORTree.cell_exact
