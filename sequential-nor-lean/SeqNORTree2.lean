/-!
# The refined NOR-tree embedding: exactness and domination, in abstract form

Machine-checked core of the refined embedding of Section 4 of the note (robust zero, pair leaves,
double top).  As in `SeqNORTree.lean`, a gate computes `nor p y`, where `p` is the value just
before it and `y` the value it reads, which is the value at its listed pointer flipped when the
gate *misreads*.  Positions are forgotten.

* A leaf over an input of value `b` has six gates:
  `z0` reads position 0 (which holds 1), `g` reads position 1 (which holds 0), `z` reads
  position 0, `l1` reads a copy `c2` of the input, `l2` reads position 1, `l3` reads a copy `c1`.
* A node is one gate that reads the last gate of its first block and comes right after its
  second block.
* A cell over blocks `U`, `V` has three top gates after them: `t1` reads the last gate of `U`,
  `t2` reads position 1, `t3` reads the last gate of `U`.

`dom` is the dominating event, with every flag replaced by the smallest event it is known to
contain: a misread for a `B`-flag, a misread of a `0` for an `A`-flag.  Lean 4 core only.
-/

namespace SeqNORTree2

def nor (a b : Bool) : Bool := !(a || b)

/-- The six gates of a leaf, given the value `p` before it.  Returns the last gate. -/
def leafOut (c1 c2 mz0 mg mz ml1 ml2 ml3 p : Bool) : Bool :=
  let z0 := nor p (true != mz0)
  let g  := nor z0 (false != mg)
  let z  := nor g (true != mz)
  let l1 := nor z (c2 != ml1)
  let l2 := nor l1 (false != ml2)
  nor l2 (c1 != ml3)

/-- The dominating event of a leaf over an input of value `b`.
For `b = 0`: a copy is bad, or the robust zero fails (`(B_{z0} ∪ A_g) ∩ B_z`), or `l1` or `l3`
misreads a `0`.  For `b = 1`: (`c1` bad or `B_{l3}`) and (`c2` bad or `B_{l1}` or `A_{l2}`). -/
def leafDom (b c1 c2 mz0 mg mz ml1 ml2 ml3 : Bool) : Bool :=
  if b then ((c1 != b) || ml3) && ((c2 != b) || ml1 || ml2)
  else (c1 != b) || (c2 != b) || ((mz0 || mg) && mz) || (ml1 && !c2) || (ml3 && !c1)

/-- **Leaf domination**: a leaf whose last gate is not `!b` has its event, whatever precedes it. -/
theorem leaf_domination : ∀ b c1 c2 mz0 mg mz ml1 ml2 ml3 p : Bool,
    leafOut c1 c2 mz0 mg mz ml1 ml2 ml3 p ≠ !b → leafDom b c1 c2 mz0 mg mz ml1 ml2 ml3 = true := by
  decide

/-- **Leaf exactness**: with both copies equal to `b` and no misread, the leaf holds `!b`. -/
theorem leaf_exact : ∀ b p : Bool,
    leafOut b b false false false false false false p = !b := by
  decide

/-- The event is not vacuous: it fails on a clean leaf. -/
theorem leaf_dom_clean : ∀ b : Bool,
    leafDom b b b false false false false false false = false := by
  decide

/-- Blocks: leaves and nodes. -/
inductive Blk where
  | leaf (b c1 c2 mz0 mg mz ml1 ml2 ml3 : Bool)
  | node (L R : Blk) (mk : Bool)

open Blk

def out : Blk → Bool → Bool
  | leaf _ c1 c2 mz0 mg mz ml1 ml2 ml3, p => leafOut c1 c2 mz0 mg mz ml1 ml2 ml3 p
  | node L R mk, p => nor (out R (out L p)) (out L p != mk)

def want : Blk → Bool
  | leaf b _ _ _ _ _ _ _ _ => !b
  | node L R _ => nor (want R) (want L)

def dom : Blk → Bool → Bool
  | leaf b c1 c2 mz0 mg mz ml1 ml2 ml3, _ => leafDom b c1 c2 mz0 mg mz ml1 ml2 ml3
  | node L R mk, p =>
    match want L, want R with
    | false, false => dom L p || dom R (out L p) || (mk && !(out L p))
    | true, true => dom R (out L p) && (dom L p || mk)
    | true, false => dom L p || mk
    | false, true => dom R (out L p)

/-- **Domination for blocks.** -/
theorem domination : ∀ (t : Blk) (p : Bool), out t p ≠ want t → dom t p = true := by
  intro t
  induction t with
  | leaf b c1 c2 mz0 mg mz ml1 ml2 ml3 =>
    intro p h
    exact leaf_domination b c1 c2 mz0 mg mz ml1 ml2 ml3 p h
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

/-- The three top gates of a cell, given the last gates `lu`, `lv` of its blocks. -/
def topOut (lu lv m1 m2 m3 : Bool) : Bool :=
  let t1 := nor lv (lu != m1)
  let t2 := nor t1 (false != m2)
  nor t2 (lu != m3)

/-- The dominating event of a cell, by the intended values `wu`, `wv` of its blocks, given the
events `du`, `dv` of the blocks.  `(0,0)`: `D_u ∪ D_v ∪ A_{t1} ∪ A_{t3}`; `(1,0)` and `(1,1)`:
`D_u ∪ (B_{t3} ∩ (B_{t1} ∪ A_{t2}))`; `(0,1)`: `D_v ∪ A_{t2}`. -/
def topDom (wu wv du dv lu m1 m2 m3 : Bool) : Bool :=
  match wu, wv with
  | false, false => du || dv || (m1 && !lu) || (m3 && !lu)
  | true, _ => du || (m3 && (m1 || m2))
  | false, true => dv || m2

/-- **Cell domination.**  If each block is wrong only on its event, then a cell whose last gate
differs from `nor wv wu` has its event. -/
theorem top_domination : ∀ wu wv du dv lu lv m1 m2 m3 : Bool,
    (lu ≠ wu → du = true) → (lv ≠ wv → dv = true) →
    topOut lu lv m1 m2 m3 ≠ nor wv wu → topDom wu wv du dv lu m1 m2 m3 = true := by
  decide

/-- **Cell exactness**: correct blocks and no misread give `nor wv wu`. -/
theorem top_exact : ∀ wu wv : Bool, topOut wu wv false false false = nor wv wu := by
  decide

/-- The complete tree of depth `h` over an input of value `b`, with no noise. -/
def tree : Nat → Bool → Blk
  | 0, b => leaf b b b false false false false false false
  | h + 1, b => node (tree h b) (tree h b) false

def clean : Blk → Bool
  | leaf b c1 c2 mz0 mg mz ml1 ml2 ml3 =>
      (c1 == b) && (c2 == b) && !mz0 && !mg && !mz && !ml1 && !ml2 && !ml3
  | node L R mk => clean L && clean R && !mk

theorem exact : ∀ (t : Blk), clean t = true → ∀ p : Bool, out t p = want t := by
  intro t
  induction t with
  | leaf b c1 c2 mz0 mg mz ml1 ml2 ml3 =>
    intro h p
    simp only [clean, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at h
    obtain ⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩ := h
    subst h1 h2 h3 h4 h5 h6 h7 h8
    simp only [out, want]
    exact leaf_exact _ p
  | node L R mk ihL ihR =>
    intro h p
    simp only [clean, Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨⟨hL, hR⟩, hk⟩ := h
    simp [out, want, ihL hL, ihR hR, hk]

theorem clean_tree : ∀ (h : Nat) (b : Bool), clean (tree h b) = true := by
  intro h
  induction h with
  | zero => intro b; cases b <;> rfl
  | succ h ih => intro b; simp [tree, clean, ih]

theorem want_tree : ∀ (h : Nat) (b : Bool), want (tree h b) = if h % 2 = 0 then !b else b := by
  intro h
  induction h with
  | zero => intro b; simp [tree, want]
  | succ h ih =>
    intro b
    by_cases hh : h % 2 = 0
    · have : ¬ ((h + 1) % 2 = 0) := by omega
      cases b <;> simp [tree, want, ih, hh, this, nor]
    · have : (h + 1) % 2 = 0 := by omega
      cases b <;> simp [tree, want, ih, hh, this, nor]

/-- **Cell exactness for trees of odd depth**: without noise the last top gate holds
`nor v u`, whatever precedes the cell. -/
theorem cell_exact (d : Nat) (hd : d % 2 = 1) (u v p : Bool) :
    topOut (out (tree d u) p) (out (tree d v) (out (tree d u) p)) false false false = nor v u := by
  have hu := exact (tree d u) (clean_tree d u) p
  have hv := exact (tree d v) (clean_tree d v) (out (tree d u) p)
  have hne : ¬ (d % 2 = 0) := by omega
  rw [hv, hu, want_tree, want_tree]
  simp only [hne, ite_false]
  exact top_exact u v

end SeqNORTree2

#print axioms SeqNORTree2.leaf_domination
#print axioms SeqNORTree2.domination
#print axioms SeqNORTree2.top_domination
#print axioms SeqNORTree2.exact
#print axioms SeqNORTree2.cell_exact
