#!/bin/sh
# search_bases.sh: the searches over repeating bases quoted in Section 9.  Build the tools first:
#   cc -O2 -o first_moment first_moment.c -lm ; cc -O2 -o canonical canonical.c -lm
# Each block prints the largest growth rates found, with the base (its distances) after the rate.
# usage: sh search_bases.sh        (about three minutes)
F=./first_moment; C=./canonical
echo "Kleene-prefix count at noise 0.74, period <= 4, pointers <= 7 back:"
( for a in 1 2 3 4 5 6 7; do echo "$($F 0 0.74 0 0 1 $a) $a"
  for b in 1 2 3 4 5 6 7; do echo "$($F 0 0.74 0 0 2 $a $b) $a $b"
   for c in 1 2 3 4 5 6 7; do echo "$($F 0 0.74 0 0 3 $a $b $c) $a $b $c"
    for d in 1 2 3 4 5 6 7; do echo "$($F 0 0.74 0 0 4 $a $b $c $d) $a $b $c $d"; done; done; done; done ) | sort -rn | head -4
echo "families (2,b) and (3,b), b <= 19, at noise 0.74:"
( for b in 2 3 5 7 9 11 13 15 17 19; do echo "$($F 0 0.74 0 0 2 2 $b) 2 $b"; echo "$($F 0 0.74 0 0 2 3 $b) 3 $b"; done ) | sort -rn | head -3
echo "locally canonical count at noise 0.74, period <= 3, pointers <= 5 back (period 4 takes 90 s more):"
( for a in 1 2 3 4 5; do for b in 1 2 3 4 5; do echo "$($C 0.74 0 1 2 $a $b) $a $b"
   for c in 1 2 3 4 5; do echo "$($C 0.74 0 1 3 $a $b $c) $a $b $c"; done; done; done ) | sort -rn | head -3
echo "Dlast count at noise 0.64, period 2 up to 12 back and period 3 up to 8 back:"
( for a in 1 2 3 4 5 6 7 8 9 10 11 12; do for b in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$($F 1 0.64 0 0 2 $a $b) $a $b"; done; done
  for a in 1 2 3 4 5 6 7 8; do for b in 1 2 3 4 5 6 7 8; do for c in 1 2 3 4 5 6 7 8; do echo "$($F 1 0.64 0 0 3 $a $b $c) $a $b $c"; done; done; done ) | sort -rn | head -4
