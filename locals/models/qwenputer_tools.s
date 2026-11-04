; Qwenputer's toolbox: tools the model's own RV32 computer runs (chipformer's assembler, chipformer.rv.asm).
;
; Linked after the calc firmware (calc-bridge.elf): its main loop resumes from its park at RESUME, whose first
; instruction (lui a1, 0x8) is replaced by a jump to from_calc. A run then reads the call's text from the text buffer
; (row i of the call's tokens at 0x8000 + 16 i) and dispatches on <function=NAME>: calc goes on in calc's own code
; with every register as it parked; the other tools run here, print the tool response as calc does
;   \n<tool_response>\nVALUE\n</tool_response>\n<think>\n\n</think>\n
; and park at their own exit ecall (the next GO resumes at dispatch).
;
; Memory (calc never allocates: its data ends at 0x390): calc's registers at 0x700 + 4 x, the function's name at
; NAME (0x6c0), the call's text from BUF (0x3000, up to 0x6c00) and its end at 0x6f0, the parameters' table at
; PTAB (0x1a00: name, name end, value, value end; 4 at most) and its count at 0x6f4, sort's numbers from NUMS
; (0x2000, 256 of 16 bytes: whole part, fraction in billionths, text start, text end), the stack below calc's own. A
; calc call reads only the function's name before calc goes on.
;
; The host gives the computer a call's text so that each row holds a whole token (a token of more than 16 bytes as
; shorter ones), and a call of more than the 80 rows in parts: calls whose first row is the byte 2 (the first
; part), 1 (one between) or 3 (the last), whose other rows the toolbox adds to BUF's text, parking without a word
; after each but the last (0x6f8: 1 while a call's parts come, 2 if they did not fit; an ordinary call sets it
; back to 0). A call of too many parts comes as one of too_long (an error the computer prints).

.text
from_calc:                           ; a call that starts as Qwen writes calc's (rows "\n" "<" "function" "=" "calc" ">")
    sw x5, 0x714(zero)               ; goes on in calc at once: t0 and t1 kept, a1 as calc's replaced instruction sets it
    sw x6, 0x718(zero)               ; (its text buffer); any other call is read by the dispatcher
    lui a1, 0x8
    lw t0, 0x50(a1)
    li t1, 0x636c6163                ; ("calc")
    bne t0, t1, fc_slow
    lw t0, 0x54(a1)
    bnez t0, fc_slow
    lw t0, 0x40(a1)
    li t1, 0x3d                      ; ("=")
    bne t0, t1, fc_slow
    lw t0, 0x30(a1)
    li t1, 0x636e7566                ; ("func")
    bne t0, t1, fc_slow
    lw t0, 0x34(a1)
    li t1, 0x6e6f6974                ; ("tion")
    bne t0, t1, fc_slow
    lw t0, 0x38(a1)
    bnez t0, fc_slow
    lw t0, 0x20(a1)
    li t1, 0x3c                      ; ("<")
    bne t0, t1, fc_slow
    lw t0, 0x10(a1)
    li t1, 0xa                       ; ("\n")
    bne t0, t1, fc_slow
    lw t0, 0x60(a1)
    li t1, 0x3e                      ; (">")
    bne t0, t1, fc_slow
    lw x5, 0x714(zero)
    lw x6, 0x718(zero)
    j RESUME_NEXT
fc_slow:
    lw x5, 0x714(zero)
    lw x6, 0x718(zero)
    sw x1, 0x704(zero)
    sw x2, 0x708(zero)
    sw x3, 0x70c(zero)
    sw x4, 0x710(zero)
    sw x5, 0x714(zero)
    sw x6, 0x718(zero)
    sw x7, 0x71c(zero)
    sw x8, 0x720(zero)
    sw x9, 0x724(zero)
    sw x10, 0x728(zero)
    sw x11, 0x72c(zero)
    sw x12, 0x730(zero)
    sw x13, 0x734(zero)
    sw x14, 0x738(zero)
    sw x15, 0x73c(zero)
    sw x16, 0x740(zero)
    sw x17, 0x744(zero)
    sw x18, 0x748(zero)
    sw x19, 0x74c(zero)
    sw x20, 0x750(zero)
    sw x21, 0x754(zero)
    sw x22, 0x758(zero)
    sw x23, 0x75c(zero)
    sw x24, 0x760(zero)
    sw x25, 0x764(zero)
    sw x26, 0x768(zero)
    sw x27, 0x76c(zero)
    sw x28, 0x770(zero)
    sw x29, 0x774(zero)
    sw x30, 0x778(zero)
    sw x31, 0x77c(zero)
    j dispatch

park:
    li a0, 0
    li a1, 0
    li a2, 0
    li a7, 93
    ecall
dispatch:
    lw sp, 0x708(zero)
    addi sp, sp, -256                ; (a stack of our own below calc's frame)
    li t0, 0x8010
    lbu t0, 0(t0)                    ; row 1: 2, 1 or 3, a part of a long call
    addi t1, t0, -1
    li t2, 3
    bltu t1, t2, part
    sw zero, 0x6f8(zero)             ; (an ordinary call: no long call's parts before it)
    call fname
    la a1, n_calc
    call is_name
    beqz a0, nx_1
    j to_calc
nx_1:
    li a0, 0x8010
    li a1, 0x3000
    call gather
    j tools
part:                                ; its rows from row 2 after the text so far (the first part's: at BUF)
    li t1, 3
    beq t0, t1, last_part
    li a1, 0x3000
    li t1, 2
    beq t0, t1, pt_first
    lw t1, 0x6f8(zero)
    li t2, 1
    bne t1, t2, pt_full              ; (no first part before it, or the text did not fit)
    lw a1, 0x6f0(zero)
pt_first:
    li a0, 0x8020
    call gather
    li t0, 1
    bnez a0, pt_set
pt_full:
    li t0, 2
pt_set:
    sw t0, 0x6f8(zero)
    j park
last_part:
    lw t0, 0x6f8(zero)
    sw zero, 0x6f8(zero)
    li t1, 1
    bne t0, t1, too_long             ; (no parts before it, or they did not fit)
    li a0, 0x8020
    lw a1, 0x6f0(zero)
    call gather
    beqz a0, too_long
    call bname
tools:                               ; (an assembly tool's parameters are read before it runs)
    la a1, n_count
    call is_name
    beqz a0, nx_2
    la s11, do_count
    j with_params
nx_2:
    la a1, n_length
    call is_name
    beqz a0, nx_3
    la s11, do_length
    j with_params
nx_3:
    la a1, n_reverse
    call is_name
    beqz a0, nx_4
    la s11, do_reverse
    j with_params
nx_4:
    la a1, n_sort
    call is_name
    beqz a0, nx_5
    la s11, do_sort
    j with_params
nx_5:
    la a1, n_weekday
    call is_name
    beqz a0, nx_6
    la s11, do_weekday
    j with_params
nx_6:
    la a1, n_between
    call is_name
    beqz a0, nx_7
    la s11, do_between
    j with_params
nx_7:
    la a1, n_add
    call is_name
    beqz a0, nx_8
    la s11, do_add
    j with_params
nx_8:
    la a1, n_words
    call is_name
    beqz a0, nx_9
    la s11, do_words
    j with_params
nx_9:
    la a1, n_base
    call is_name
    beqz a0, nx_10
    la s11, do_base
    j with_params
nx_10:
    la a1, n_letters
    call is_name
    beqz a0, nx_11
    la s11, do_letters
    j with_params
nx_11:
    la a1, n_long
    call is_name
    beqz a0, nx_12
too_long:
    call open_resp
    la a0, m_long
    call puts
    j close_park
nx_12:                               ; the tools in C (qwenputer_c/): c_main(close_park) runs one and ends at close_park,
    call open_resp                   ; or returns for a name it has not either
    la a0, close_park
    li t0, 0x48000
    jalr t0
    la a0, m_unknown
    call puts
    j close_park
with_params:
    call params
    jr s11

to_calc:                             ; calc's registers as it parked, its replaced instruction, and on in its code
    lw x1, 0x704(zero)
    lw x2, 0x708(zero)
    lw x3, 0x70c(zero)
    lw x4, 0x710(zero)
    lw x5, 0x714(zero)
    lw x6, 0x718(zero)
    lw x7, 0x71c(zero)
    lw x8, 0x720(zero)
    lw x9, 0x724(zero)
    lw x10, 0x728(zero)
    lw x11, 0x72c(zero)
    lw x12, 0x730(zero)
    lw x13, 0x734(zero)
    lw x14, 0x738(zero)
    lw x15, 0x73c(zero)
    lw x16, 0x740(zero)
    lw x17, 0x744(zero)
    lw x18, 0x748(zero)
    lw x19, 0x74c(zero)
    lw x20, 0x750(zero)
    lw x21, 0x754(zero)
    lw x22, 0x758(zero)
    lw x23, 0x75c(zero)
    lw x24, 0x760(zero)
    lw x25, 0x764(zero)
    lw x26, 0x768(zero)
    lw x27, 0x76c(zero)
    lw x28, 0x770(zero)
    lw x29, 0x774(zero)
    lw x30, 0x778(zero)
    lw x31, 0x77c(zero)
    lui a1, 0x8
    j RESUME_NEXT

; ---- the call's text: at a1, the text buffer's rows from a0 (up to row 80), each up to its first NUL (a row's 16
; bytes read in straight-line code); its end at 0x6f0. a0: 1, or 0 if BUF has no room for 80 rows more (nothing read)
gather:
    mv t0, a0
    li t1, 80
    mv t2, a1
    li t3, 0x66ff                    ; (0x6c00 - 80 x 16 - 1)
    bgtu t2, t3, g_full
g_row:
    beqz t1, g_done
    lbu t3, 0(t0)
    beqz t3, g_done
    sb t3, 0(t2)
    lbu t3, 1(t0)
    beqz t3, g_1
    sb t3, 1(t2)
    lbu t3, 2(t0)
    beqz t3, g_2
    sb t3, 2(t2)
    lbu t3, 3(t0)
    beqz t3, g_3
    sb t3, 3(t2)
    lbu t3, 4(t0)
    beqz t3, g_4
    sb t3, 4(t2)
    lbu t3, 5(t0)
    beqz t3, g_5
    sb t3, 5(t2)
    lbu t3, 6(t0)
    beqz t3, g_6
    sb t3, 6(t2)
    lbu t3, 7(t0)
    beqz t3, g_7
    sb t3, 7(t2)
    lbu t3, 8(t0)
    beqz t3, g_8
    sb t3, 8(t2)
    lbu t3, 9(t0)
    beqz t3, g_9
    sb t3, 9(t2)
    lbu t3, 10(t0)
    beqz t3, g_10
    sb t3, 10(t2)
    lbu t3, 11(t0)
    beqz t3, g_11
    sb t3, 11(t2)
    lbu t3, 12(t0)
    beqz t3, g_12
    sb t3, 12(t2)
    lbu t3, 13(t0)
    beqz t3, g_13
    sb t3, 13(t2)
    lbu t3, 14(t0)
    beqz t3, g_14
    sb t3, 14(t2)
    lbu t3, 15(t0)
    beqz t3, g_15
    sb t3, 15(t2)
    addi t2, t2, 16
    j g_next
g_1:
    addi t2, t2, 1
    j g_next
g_2:
    addi t2, t2, 2
    j g_next
g_3:
    addi t2, t2, 3
    j g_next
g_4:
    addi t2, t2, 4
    j g_next
g_5:
    addi t2, t2, 5
    j g_next
g_6:
    addi t2, t2, 6
    j g_next
g_7:
    addi t2, t2, 7
    j g_next
g_8:
    addi t2, t2, 8
    j g_next
g_9:
    addi t2, t2, 9
    j g_next
g_10:
    addi t2, t2, 10
    j g_next
g_11:
    addi t2, t2, 11
    j g_next
g_12:
    addi t2, t2, 12
    j g_next
g_13:
    addi t2, t2, 13
    j g_next
g_14:
    addi t2, t2, 14
    j g_next
g_15:
    addi t2, t2, 15
    j g_next
g_next:
    addi t0, t0, 16
    addi t1, t1, -1
    j g_row
g_done:
    sb zero, 0(t2)
    sw t2, 0x6f0(zero)
    li a0, 1
    ret
g_full:
    li a0, 0
    ret

; bname: NAME <- the function's name read from BUF (after <function= up to >, at most 31 bytes); empty if none
bname:
    addi sp, sp, -16
    sw ra, 12(sp)
    li a0, 0x3000
    lw a1, 0x6f0(zero)
    la a2, s_function
    call find
    li t0, 0x6c0
    beqz a0, bn_end
    addi a0, a0, 10
    lw a1, 0x6f0(zero)
    li t2, 0x6df
bn_byte:
    bgeu a0, a1, bn_end
    bgeu t0, t2, bn_end
    lbu t1, 0(a0)
    li t3, 62
    beq t1, t3, bn_end
    sb t1, 0(t0)
    addi t0, t0, 1
    addi a0, a0, 1
    j bn_byte
bn_end:
    sb zero, 0(t0)
    lw ra, 12(sp)
    addi sp, sp, 16
    ret

; fname: NAME <- the function's name (after <function= up to >, at most 31 bytes, NUL-terminated), read from the
; text buffer's rows (the bytes of each row up to its first NUL; an empty row ends the call); empty if none
fname:
    li t0, 0x8010                    ; the row
    li t1, 0                         ; the byte in it
    li t2, 0                         ; how much of "<function=" matched
    li a3, 0                         ; the name's length (-1: not reading it yet)
    li a4, -1
    la a5, s_function
    li t6, 0x8510
fn_byte:
    bgeu t0, t6, fn_end
    li t5, 16
    beq t1, t5, fn_row
    add t5, t0, t1
    lbu t4, 0(t5)
    beqz t4, fn_row
    addi t1, t1, 1
    bgez a4, fn_name
    add t5, a5, t2                   ; matching "<function=" (a mismatch starts again, at 1 if it is a '<')
    lbu t5, 0(t5)
    beq t4, t5, fn_match
    li t2, 0
    li t5, 60
    bne t4, t5, fn_byte
    li t2, 1
    j fn_byte
fn_match:
    addi t2, t2, 1
    li t5, 10
    bne t2, t5, fn_byte
    li a4, 0
    j fn_byte
fn_name:
    li t5, 62
    beq t4, t5, fn_end
    li t5, 31
    bge a3, t5, fn_end
    addi t5, a3, 0x6c0
    sb t4, 0(t5)
    addi a3, a3, 1
    j fn_byte
fn_row:
    addi t0, t0, 16
    li t1, 0
    lbu t5, 0(t0)
    bnez t5, fn_byte
fn_end:
    addi t5, a3, 0x6c0
    sb zero, 0(t5)
    ret

; is_name: whether NAME equals the NUL-terminated a1 (a0: 1 or 0)
is_name:
    li t0, 0x6c0
in_loop:
    lbu t1, 0(t0)
    lbu t2, 0(a1)
    bne t1, t2, in_no
    beqz t1, in_yes
    addi t0, t0, 1
    addi a1, a1, 1
    j in_loop
in_yes:
    li a0, 1
    ret
in_no:
    li a0, 0
    ret

; find: the first place in [a0, a1) where the NUL-terminated a2 starts (a0), or 0
find:
    lbu t4, 0(a2)
    bgeu a0, a1, f_none
f_outer:                             ; (4 bytes a step while 4 are left)
    addi t5, a0, 4
    bgtu t5, a1, f_one
    lbu t3, 0(a0)
    beq t3, t4, f_try
    lbu t3, 1(a0)
    beq t3, t4, f_try1
    lbu t3, 2(a0)
    beq t3, t4, f_try2
    lbu t3, 3(a0)
    beq t3, t4, f_try3
    mv a0, t5
    bltu a0, a1, f_outer
    j f_none
f_one:
    lbu t3, 0(a0)
    beq t3, t4, f_try
    addi a0, a0, 1
    bltu a0, a1, f_one
    j f_none
f_try3:
    addi a0, a0, 1
f_try2:
    addi a0, a0, 1
f_try1:
    addi a0, a0, 1
f_try:
    mv t0, a0
    mv t1, a2
f_inner:
    lbu t2, 0(t1)
    beqz t2, f_found
    bgeu t0, a1, f_none
    lbu t3, 0(t0)
    bne t2, t3, f_next
    addi t0, t0, 1
    addi t1, t1, 1
    j f_inner
f_next:
    addi a0, a0, 1
    bltu a0, a1, f_outer
    j f_none
f_found:
    ret
f_none:
    li a0, 0
    ret

; params: PTAB <- the call's parameters in one pass over BUF: each one's name and its value (one newline trimmed at
; each end), 4 at most; their count at 0x6f4
params:
    addi sp, sp, -32
    sw ra, 28(sp)
    sw s0, 24(sp)
    sw s1, 20(sp)
    sw s2, 16(sp)
    li s0, 0x3000                    ; where the search goes on
    li s1, 0                         ; how many
pr_loop:
    li t0, 4
    bge s1, t0, pr_done
    mv a0, s0
    lw a1, 0x6f0(zero)
    la a2, s_param
    call find
    beqz a0, pr_done
    addi a0, a0, 11                  ; the name
    slli s2, s1, 4
    li t0, 0x1a00
    add s2, s2, t0
    sw a0, 0(s2)
    lw a1, 0x6f0(zero)
pr_name:
    bgeu a0, a1, pr_done
    lbu t1, 0(a0)
    li t2, 62
    beq t1, t2, pr_value
    addi a0, a0, 1
    j pr_name
pr_value:
    sw a0, 4(s2)
    addi a0, a0, 1
    bgeu a0, a1, pr_done
    lbu t1, 0(a0)
    li t2, 10
    bne t1, t2, pr_start
    addi a0, a0, 1
pr_start:
    sw a0, 8(s2)
    la a2, s_endparam
    call find
    beqz a0, pr_done
    addi s0, a0, 12
    lw t0, 8(s2)
    bleu a0, t0, pr_end
    lbu t1, -1(a0)
    li t2, 10
    bne t1, t2, pr_end
    addi a0, a0, -1
pr_end:
    sw a0, 12(s2)
    addi s1, s1, 1
    j pr_loop
pr_done:
    sw s1, 0x6f4(zero)
    lw ra, 28(sp)
    lw s0, 24(sp)
    lw s1, 20(sp)
    lw s2, 16(sp)
    addi sp, sp, 32
    ret

; param: the value [a0, a1) of the parameter named by the NUL-terminated a0, or a0 = 0
param:
    li t0, 0x1a00
    lw t1, 0x6f4(zero)
pm_entry:
    beqz t1, pm_none
    lw t2, 0(t0)                     ; the entry's name [t2, t3)
    lw t3, 4(t0)
    mv t4, a0
pm_cmp:
    lbu t5, 0(t4)
    beq t2, t3, pm_last
    beqz t5, pm_next
    lbu t6, 0(t2)
    bne t5, t6, pm_next
    addi t2, t2, 1
    addi t4, t4, 1
    j pm_cmp
pm_last:
    bnez t5, pm_next
    lw a0, 8(t0)
    lw a1, 12(t0)
    ret
pm_next:
    addi t0, t0, 16
    addi t1, t1, -1
    j pm_entry
pm_none:
    li a0, 0
    li a1, 0
    ret

; ---- output: put a0 (a byte, or 256.. a special token), puts a NUL-terminated string, put_int a signed integer
put:
    li a1, 0
    li a2, 0
    li a7, 1
    ecall
    ret

puts:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    mv s0, a0
ps_loop:
    lbu a0, 0(s0)
    beqz a0, ps_done
    call put
    addi s0, s0, 1
    j ps_loop
ps_done:
    lw ra, 12(sp)
    lw s0, 8(sp)
    addi sp, sp, 16
    ret

put_int:
    addi sp, sp, -32
    sw ra, 28(sp)
    sw s0, 24(sp)
    sw s1, 20(sp)
    mv s0, a0
    bgez s0, pi_pos
    li a0, 45
    call put
    neg s0, s0
pi_pos:
    li s1, 0                         ; digits on the stack, least significant first
pi_digit:
    li t0, 10
    remu t1, s0, t0
    divu s0, s0, t0
    add t2, sp, s1
    sb t1, 0(t2)
    addi s1, s1, 1
    bnez s0, pi_digit
pi_out:
    addi s1, s1, -1
    add t2, sp, s1
    lbu a0, 0(t2)
    addi a0, a0, 48
    call put
    bnez s1, pi_out
    lw ra, 28(sp)
    lw s0, 24(sp)
    lw s1, 20(sp)
    addi sp, sp, 32
    ret

open_resp:                           ; \n<tool_response>\n
    addi sp, sp, -16
    sw ra, 12(sp)
    li a0, 10
    call put
    li a0, 256
    call put
    li a0, 10
    call put
    lw ra, 12(sp)
    addi sp, sp, 16
    ret

close_park:                          ; \n</tool_response>\n<think>\n\n</think>\n, then park
    li a0, 10
    call put
    li a0, 257
    call put
    li a0, 10
    call put
    li a0, 258
    call put
    li a0, 10
    call put
    li a0, 10
    call put
    li a0, 259
    call put
    li a0, 10
    call put
    j park

bad_args:
    call open_resp
    la a0, m_args
    call puts
    j close_park

; space: a0 = whether the byte a0 is a space, tab, newline or carriage return (uses t6 alone)
space:
    li t6, 32
    beq a0, t6, is_sp
    li t6, 9
    beq a0, t6, is_sp
    li t6, 10
    beq a0, t6, is_sp
    li t6, 13
    beq a0, t6, is_sp
    li a0, 0
    ret
is_sp:
    li a0, 1
    ret

; lower: a0 to lower case (ASCII letters)
lower:
    li t0, 65
    blt a0, t0, lo_done
    li t0, 90
    bgt a0, t0, lo_done
    addi a0, a0, 32
lo_done:
    ret

; ---- count: how many times find occurs in text (non-overlapping, ASCII letters in either case): find folded to
; lower case at 0x1b00 (256 bytes at most), each place in text compared with its first byte, folded (the fold table)
do_count:
    la a0, p_text
    call param
    beqz a0, bad_args
    mv s2, a0
    mv s3, a1
    la a0, p_find
    call param
    beqz a0, bad_args
    mv s4, a0
    sub s6, a1, a0
    blez s6, bad_args
    li t0, 256
    bgt s6, t0, nc_big
    la s5, fold
    li s9, 0x1b00
    mv t0, s4
    mv t1, s9
c_fold:
    bgeu t0, a1, c_ready
    lbu t3, 0(t0)
    add t3, t3, s5
    lbu t3, 0(t3)
    sb t3, 0(t1)
    addi t0, t0, 1
    addi t1, t1, 1
    j c_fold
c_ready:
    lbu s10, 0(s9)                   ; find's first byte
    li s7, 0
    sub s11, s3, s6                  ; the last place find can start
    mv s8, s2
c_scan:
    bgtu s8, s11, c_done
    lbu t0, 0(s8)
    add t0, t0, s5
    lbu t0, 0(t0)
    beq t0, s10, c_try
    addi s8, s8, 1
    j c_scan
c_try:
    li t1, 1
c_cmp:
    beq t1, s6, c_hit
    add t0, s8, t1
    lbu t0, 0(t0)
    add t0, t0, s5
    lbu t0, 0(t0)
    add t2, s9, t1
    lbu t2, 0(t2)
    bne t0, t2, c_miss
    addi t1, t1, 1
    j c_cmp
c_hit:
    addi s7, s7, 1
    add s8, s8, s6
    j c_scan
c_miss:
    addi s8, s8, 1
    j c_scan
c_done:
    call open_resp
    mv a0, s7
    call put_int
    j close_park
nc_big:
    j s_big

; ---- length: the characters of text (UTF-8: the bytes that do not continue a character)
do_length:
    la a0, p_text
    call param
    beqz a0, bad_args
    li s7, 0
l_loop:
    bgeu a0, a1, l_done
    lbu t0, 0(a0)
    andi t0, t0, 192
    li t1, 128
    beq t0, t1, l_next
    addi s7, s7, 1
l_next:
    addi a0, a0, 1
    j l_loop
l_done:
    call open_resp
    mv a0, s7
    call put_int
    j close_park

; ---- reverse: text's characters in reverse order (each UTF-8 character's bytes kept in order)
do_reverse:
    la a0, p_text
    call param
    beqz a0, bad_args
    mv s2, a0
    mv s3, a1
    call open_resp
r_char:
    bleu s3, s2, r_done
    addi s4, s3, -1                  ; the character's first byte: back over continuation bytes
r_back:
    bleu s4, s2, r_out
    lbu t0, 0(s4)
    andi t0, t0, 192
    li t1, 128
    bne t0, t1, r_out
    addi s4, s4, -1
    j r_back
r_out:
    mv s5, s4
r_byte:
    bgeu s5, s3, r_next
    lbu a0, 0(s5)
    call put
    addi s5, s5, 1
    j r_byte
r_next:
    mv s3, s4
    j r_char
r_done:
    j close_park

; ---- sort: the numbers in numbers (a minus or U+2212, up to 9 digits, a point and up to 9 digits: 12, -3, 4.5, .5;
; anything else separates them), ascending, each printed as written, with ", " between them. A number with a letter
; right after it (1e3, 10k, 0x1f), a slash between digits (3/4), or a comma between digits where white space
; separates the numbers too (1,000, 250), is an error, not a number cut in two
do_sort:
    la a0, p_numbers
    call param
    beqz a0, bad_args
    mv s2, a0
    mv s3, a1
    li s4, 0                         ; how many
    li s11, 0                        ; whether white space is among them
    mv s6, s2
s_sp:
    bgeu s6, s3, s_scan
    lbu a0, 0(s6)
    addi s6, s6, 1
    call space
    beqz a0, s_sp
    li s11, 1
s_scan:
    bgeu s2, s3, s_sort
    lbu t0, 0(s2)
    mv s6, s2                        ; the number's text from here (its minus)
    li s7, 0                         ; negative
    addi t2, s2, 1                   ; (after a minus)
    li t1, 45
    beq t0, t1, s_minus
    li t1, 0xe2                      ; (U+2212: e2 88 92)
    bne t0, t1, s_digits
    addi t2, s2, 3
    bgtu t2, s3, s_skip
    lbu t3, 1(s2)
    li t1, 0x88
    bne t3, t1, s_skip
    lbu t3, 2(s2)
    li t1, 0x92
    bne t3, t1, s_skip
s_minus:                             ; a minus before a number (a digit, or a point and a digit)
    bgeu t2, s3, s_skip
    lbu t3, 0(t2)
    addi t3, t3, -48
    li t4, 10
    bltu t3, t4, s_neg
    li t4, -2                        ; ('.' - '0')
    bne t3, t4, s_skip
    addi t3, t2, 1
    bgeu t3, s3, s_skip
    lbu t3, 0(t3)
    addi t3, t3, -48
    li t4, 10
    bgeu t3, t4, s_skip
s_neg:
    li s7, 1
    mv s2, t2
s_digits:
    lbu t0, 0(s2)
    addi t1, t0, -48
    li t2, 10
    bltu t1, t2, s_whole
    li t2, 46                        ; a point before a digit: a number with no whole part
    bne t0, t2, s_skip
    addi t3, s2, 1
    bgeu t3, s3, s_skip
    lbu t3, 0(t3)
    addi t3, t3, -48
    li t2, 10
    bgeu t3, t2, s_skip
    li s8, 0
    li s9, 0
    j s_point
s_whole:
    mv a0, s2
    li a1, 10
    call num
    li t0, 10
    beq a2, t0, s_big                ; (a 10th digit)
    mv s8, a0                        ; whole part
    mv s2, a1
    li s9, 0                         ; fraction, in billionths
    bgeu s2, s3, s_store
    lbu t0, 0(s2)
    li t1, 46
    bne t0, t1, s_after
s_point:
    addi t2, s2, 1
    bgeu t2, s3, s_store
    lbu t3, 0(t2)
    addi t3, t3, -48
    li t4, 10
    bgeu t3, t4, s_store             ; (a point not before a digit: not the number's)
    mv a0, t2
    li a1, 10
    call num
    li t0, 10
    beq a2, t0, s_big
    mv s9, a0
    mv s2, a1
s_scale:
    li t0, 9
    bge a2, t0, s_frac
    li t0, 10
    mul s9, s9, t0
    addi a2, a2, 1
    j s_scale
s_frac:
    bgeu s2, s3, s_store
    lbu t0, 0(s2)
s_after:                             ; t0: the byte right after the number
    ori t1, t0, 32
    addi t1, t1, -97
    li t2, 26
    bltu t1, t2, s_nums
    li t2, 47                        ; a slash before a digit (3/4): a fraction, not two numbers
    bne t0, t2, s_comma
    addi t3, s2, 1
    bgeu t3, s3, s_store
    lbu t4, 0(t3)
    addi t4, t4, -48
    li t2, 10
    bltu t4, t2, s_nums
    j s_store
s_comma:
    li t2, 44
    bne t0, t2, s_store
    beqz s11, s_store
    addi t3, s2, 1
    bgeu t3, s3, s_store
    lbu t4, 0(t3)
    addi t4, t4, -48
    li t2, 10
    bltu t4, t2, s_nums
s_store:
    beqz s7, s_pos
    neg s8, s8
    neg s9, s9
s_pos:
    li t1, 256
    bge s4, t1, s_many
    slli t0, s4, 4
    li t1, 0x2000
    add t0, t0, t1
    sw s8, 0(t0)
    sw s9, 4(t0)
    sw s6, 8(t0)
    sw s2, 12(t0)
    addi s4, s4, 1
    j s_scan
s_skip:
    addi s2, s2, 1
    j s_scan
s_big:
    call open_resp
    la a0, m_big
    call puts
    j close_park
s_nums:
    call open_resp
    la a0, m_nums
    call puts
    j close_park
s_many:
    call open_resp
    la a0, m_many
    call puts
    j close_park
s_sort:                              ; Shell sort on (whole part, fraction), gaps from gaps
    beqz s4, bad_args
    la s11, gaps
s_gap:
    lw s5, 0(s11)
    beqz s5, s_print
    addi s11, s11, 4
    bge s5, s4, s_gap
    slli s3, s5, 4                   ; (the gap in bytes)
    mv s6, s5
s_outer:
    bge s6, s4, s_gap
    slli t0, s6, 4
    li t1, 0x2000
    add t0, t0, t1
    lw s7, 0(t0)                     ; the key
    lw s8, 4(t0)
    lw s9, 8(t0)
    lw s10, 12(t0)
    sub t2, s6, s5
s_inner:
    bltz t2, s_place
    slli t3, t2, 4
    li t1, 0x2000
    add t3, t3, t1
    lw t4, 0(t3)
    blt t4, s7, s_place
    bne t4, s7, s_move
    lw t5, 4(t3)
    ble t5, s8, s_place
s_move:
    lw t4, 0(t3)
    lw t5, 4(t3)
    lw t6, 8(t3)
    lw a0, 12(t3)
    add t3, t3, s3
    sw t4, 0(t3)
    sw t5, 4(t3)
    sw t6, 8(t3)
    sw a0, 12(t3)
    sub t2, t2, s5
    j s_inner
s_place:
    add t2, t2, s5
    slli t3, t2, 4
    li t1, 0x2000
    add t3, t3, t1
    sw s7, 0(t3)
    sw s8, 4(t3)
    sw s9, 8(t3)
    sw s10, 12(t3)
    addi s6, s6, 1
    j s_outer
s_print:
    call open_resp
    li s6, 0
s_each:
    bge s6, s4, s_end
    beqz s6, s_first
    li a0, 44
    call put
    li a0, 32
    call put
s_first:
    slli t0, s6, 4
    li t1, 0x2000
    add t0, t0, t1
    lw s7, 8(t0)
    lw s8, 12(t0)
s_byte:
    bgeu s7, s8, s_next
    lbu a0, 0(s7)
    call put
    addi s7, s7, 1
    j s_byte
s_next:
    addi s6, s6, 1
    j s_each
s_end:
    j close_park

; ---- dates (the proleptic Gregorian calendar)
; num: the unsigned number of up to a1 digits at a0 (a0: its value, a1: the next byte's address, a2: digits read)
num:
    li t0, 0
    li a2, 0
n_loop:
    bge a2, a1, n_done
    lbu t1, 0(a0)
    addi t1, t1, -48
    li t2, 10
    bgeu t1, t2, n_done
    li t2, 10
    mul t0, t0, t2
    add t0, t0, t1
    addi a0, a0, 1
    addi a2, a2, 1
    j n_loop
n_done:
    mv a1, a0
    mv a0, t0
    ret

; date: the day number (days since 1970-01-01) of the date YYYY-MM-DD in [a0, a1): a0 = 1 and a1 = the day
; number, or a0 = 0
date:
    addi sp, sp, -48
    sw ra, 44(sp)
    sw s0, 40(sp)
    sw s1, 36(sp)
    sw s2, 32(sp)
    sw s3, 28(sp)
    sw s4, 24(sp)
    mv s0, a0
    mv s1, a1
d_space:
    bgeu s0, s1, d_bad
    lbu a0, 0(s0)
    call space
    bnez a0, d_skip
    j d_year
d_skip:
    addi s0, s0, 1
    j d_space
d_year:
    mv a0, s0
    li a1, 4
    call num
    li t0, 4
    bne a2, t0, d_bad
    mv s2, a0                        ; year
    mv s0, a1
    bgeu s0, s1, d_bad
    lbu t0, 0(s0)
    li t1, 45
    bne t0, t1, d_bad
    addi a0, s0, 1
    li a1, 2
    call num
    beqz a2, d_bad
    mv s3, a0                        ; month
    mv s0, a1
    bgeu s0, s1, d_bad
    lbu t0, 0(s0)
    li t1, 45
    bne t0, t1, d_bad
    addi a0, s0, 1
    li a1, 2
    call num
    beqz a2, d_bad
    mv s4, a0                        ; day
    mv s0, a1
d_tail:                              ; only spaces after the date
    bgeu s0, s1, d_check
    lbu a0, 0(s0)
    call space
    bnez a0, d_tail_next
    j d_bad
d_tail_next:
    addi s0, s0, 1
    j d_tail
d_check:
    li t0, 1
    blt s3, t0, d_bad
    li t0, 12
    bgt s3, t0, d_bad
    li t0, 1
    blt s4, t0, d_bad
    la t0, month_days                ; the month's length (February: 29 in a leap year)
    addi t1, s3, -1
    add t0, t0, t1
    lbu t2, 0(t0)
    li t1, 2
    bne s3, t1, d_len
    li t1, 4
    rem t3, s2, t1
    bnez t3, d_len
    li t1, 100
    rem t3, s2, t1
    bnez t3, d_leap
    li t1, 400
    rem t3, s2, t1
    bnez t3, d_len
d_leap:
    li t2, 29
d_len:
    bgt s4, t2, d_bad
    ; days from civil (H. Hinnant): y -= m <= 2; era = floor(y / 400); yoe = y - 400 era;
    ; doy = (153 (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; doe = 365 yoe + yoe / 4 - yoe / 100 + doy;
    ; days = 146097 era + doe - 719468
    li t0, 2
    bgt s3, t0, d_y
    addi s2, s2, -1
d_y:
    mv t0, s2
    bgez t0, d_era
    addi t0, t0, -399
d_era:
    li t1, 400
    div t0, t0, t1                   ; era
    mul t2, t0, t1
    sub t2, s2, t2                   ; yoe
    li t3, 2
    bgt s3, t3, d_m3
    addi t3, s3, 9
    j d_doy
d_m3:
    addi t3, s3, -3
d_doy:
    li t4, 153
    mul t3, t3, t4
    addi t3, t3, 2
    li t4, 5
    div t3, t3, t4
    add t3, t3, s4
    addi t3, t3, -1                  ; doy
    li t4, 365
    mul t5, t2, t4
    li t4, 4
    div t6, t2, t4
    add t5, t5, t6
    li t4, 100
    div t6, t2, t4
    sub t5, t5, t6
    add t5, t5, t3                   ; doe
    li t4, 146097
    mul t6, t0, t4
    add t6, t6, t5
    li t4, 719468
    sub a1, t6, t4
    li a0, 1
    j d_ret
d_bad:
    li a0, 0
    li a1, 0
d_ret:
    lw ra, 44(sp)
    lw s0, 40(sp)
    lw s1, 36(sp)
    lw s2, 32(sp)
    lw s3, 28(sp)
    lw s4, 24(sp)
    addi sp, sp, 48
    ret

bad_date:
    call open_resp
    la a0, m_date
    call puts
    j close_park

; ---- weekday: the day of the week of date
do_weekday:
    la a0, p_date
    call param
    beqz a0, bad_args
    call date
    beqz a0, bad_date
    addi t0, a1, 4                   ; 1970-01-01 was a Thursday (4, from Sunday 0)
    li t1, 7
    rem t0, t0, t1
    bgez t0, w_pos
    addi t0, t0, 7
w_pos:
    li t1, 10
    mul t0, t0, t1
    la a0, day_names
    add s2, a0, t0
    call open_resp
    mv a0, s2
    call puts
    j close_park

; ---- days_between: the days from start to end (negative if end comes first)
do_between:
    la a0, p_start
    call param
    beqz a0, bad_args
    call date
    beqz a0, bad_date
    mv s2, a1
    la a0, p_end
    call param
    beqz a0, bad_args
    call date
    beqz a0, bad_date
    sub s2, a1, s2
    call open_resp
    mv a0, s2
    call put_int
    j close_park

; (jumps to the error handlers above: out of a branch's reach from the tools below)
nf_args:
    j bad_args
nf_date:
    j bad_date
nf_big:
    j s_big

; ---- printing numbers: put_pad a0 (unsigned) in decimal, zero-padded to a1 digits; put_base a0 (unsigned) in base a1
put_pad:
    addi sp, sp, -48
    sw ra, 44(sp)
    sw s0, 40(sp)
    sw s1, 36(sp)
    mv s0, a0
    mv s1, a1
    li t0, 0
pp_digit:
    li t1, 10
    remu t2, s0, t1
    divu s0, s0, t1
    add t3, sp, t0
    sb t2, 0(t3)
    addi t0, t0, 1
    bnez s0, pp_digit
pp_pad:
    bge t0, s1, pp_out
    add t3, sp, t0
    sb zero, 0(t3)
    addi t0, t0, 1
    j pp_pad
pp_out:
    mv s1, t0
pp_loop:
    addi s1, s1, -1
    add t3, sp, s1
    lbu a0, 0(t3)
    addi a0, a0, 48
    call put
    bnez s1, pp_loop
    lw ra, 44(sp)
    lw s0, 40(sp)
    lw s1, 36(sp)
    addi sp, sp, 48
    ret

put_base:
    addi sp, sp, -48
    sw ra, 44(sp)
    sw s0, 40(sp)
    sw s1, 36(sp)
    sw s2, 32(sp)
    mv s0, a0
    mv s2, a1
    li s1, 0
pb_digit:
    remu t2, s0, s2
    divu s0, s0, s2
    add t3, sp, s1
    sb t2, 0(t3)
    addi s1, s1, 1
    bnez s0, pb_digit
pb_loop:
    addi s1, s1, -1
    add t3, sp, s1
    lbu a0, 0(t3)
    li t0, 10
    blt a0, t0, pb_dec
    addi a0, a0, 55                  ; (A..Z)
    j pb_put
pb_dec:
    addi a0, a0, 48
pb_put:
    call put
    bnez s1, pb_loop
    lw ra, 44(sp)
    lw s0, 40(sp)
    lw s1, 36(sp)
    lw s2, 32(sp)
    addi sp, sp, 48
    ret

; sint: the signed integer in [a0, a1) (spaces around it, a sign, up to 9 digits): a0 = 1 and a1 its
; value, or a0 = 0
sint:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    sw s2, 0(sp)
    mv s0, a0
    mv s1, a1
    li s2, 0
si_lead:
    bgeu s0, s1, si_bad
    lbu t0, 0(s0)
    mv a0, t0
    call space
    bnez a0, si_skip
    li t1, 45
    beq t0, t1, si_minus
    li t1, 43
    beq t0, t1, si_plus
    j si_num
si_skip:
    addi s0, s0, 1
    j si_lead
si_minus:
    li s2, 1
si_plus:
    addi s0, s0, 1
si_num:
    mv a0, s0
    li a1, 9
    call num
    beqz a2, si_bad
    mv s0, a1
    mv t2, a0
si_tail:
    bgeu s0, s1, si_ok
    lbu a0, 0(s0)
    call space
    bnez a0, si_tail_next
    j si_bad
si_tail_next:
    addi s0, s0, 1
    j si_tail
si_ok:
    beqz s2, si_pos
    neg t2, t2
si_pos:
    li a0, 1
    mv a1, t2
    j si_ret
si_bad:
    li a0, 0
    li a1, 0
si_ret:
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    lw s2, 0(sp)
    addi sp, sp, 16
    ret

; ---- add_days: the date a number of days after date (before it, if negative); the civil date of a day number by
; H. Hinnant's algorithm (z += 719468; era = floor(z / 146097); doe = z - 146097 era; yoe = (doe - doe/1460 + doe/36524 -
; doe/146096) / 365; doy = doe - (365 yoe + yoe/4 - yoe/100); mp = (5 doy + 2) / 153; d = doy - (153 mp + 2)/5 + 1;
; m = mp < 10 ? mp + 3 : mp - 9; y = yoe + 400 era + (m <= 2)), within the years 0000..9999
do_add:
    la a0, p_date
    call param
    beqz a0, nf_args
    call date
    beqz a0, nf_date
    mv s2, a1
    la a0, p_days
    call param
    beqz a0, nf_args
    call sint
    beqz a0, nf_args
    add s2, s2, a1
    li t0, 719468
    add t0, s2, t0
    li t1, 146097
    mv t2, t0
    bgez t2, a_era
    li t3, 146096
    sub t2, t2, t3
a_era:
    div t2, t2, t1                   ; era (rounded down)
    mul t3, t2, t1
    sub t3, t0, t3                   ; doe
    li t1, 1460
    div t4, t3, t1
    sub t4, t3, t4
    li t1, 36524
    div t5, t3, t1
    add t4, t4, t5
    li t1, 146096
    div t5, t3, t1
    sub t4, t4, t5
    li t1, 365
    div t4, t4, t1                   ; yoe
    li t1, 400
    mul s3, t2, t1
    add s3, s3, t4                   ; y
    li t1, 365
    mul t6, t4, t1
    li t1, 4
    div t5, t4, t1
    add t6, t6, t5
    li t1, 100
    div t5, t4, t1
    sub t6, t6, t5
    sub t6, t3, t6                   ; doy
    li t1, 5
    mul t5, t6, t1
    addi t5, t5, 2
    li t1, 153
    div t5, t5, t1                   ; mp
    li t1, 153
    mul s5, t5, t1
    addi s5, s5, 2
    li t1, 5
    div s5, s5, t1
    sub s5, t6, s5
    addi s5, s5, 1                   ; d
    li t1, 10
    blt t5, t1, a_m3
    addi s4, t5, -9
    j a_y
a_m3:
    addi s4, t5, 3                   ; m
a_y:
    li t1, 2
    bgt s4, t1, a_check
    addi s3, s3, 1
a_check:
    bltz s3, bad_range
    li t1, 9999
    bgt s3, t1, bad_range
    call open_resp
    mv a0, s3
    li a1, 4
    call put_pad
    li a0, 45
    call put
    mv a0, s4
    li a1, 2
    call put_pad
    li a0, 45
    call put
    mv a0, s5
    li a1, 2
    call put_pad
    j close_park

bad_range:
    call open_resp
    la a0, m_range
    call puts
    j close_park

; ---- words: the words of text (runs of characters between spaces, tabs, newlines and carriage returns)
do_words:
    la a0, p_text
    call param
    beqz a0, nf_args
    li s7, 0
    li t2, 1                         ; after a space
wd_loop:
    bgeu a0, a1, wd_done
    lbu t0, 0(a0)
    li t1, 32
    beq t0, t1, wd_space
    addi t1, t0, -9                  ; (tab, newline, vertical tab, form feed, carriage return)
    li t3, 5
    bltu t1, t3, wd_space
    beqz t2, wd_next
    addi s7, s7, 1
    li t2, 0
    j wd_next
wd_space:
    li t2, 1
wd_next:
    addi a0, a0, 1
    j wd_loop
wd_done:
    call open_resp
    mv a0, s7
    call put_int
    j close_park

; ---- to_base: number (decimal, or 0x / 0b / 0o and its digits; below 2^32) written in base (2..36)
do_base:
    la a0, p_number
    call param
    beqz a0, nf_args
    mv s2, a0
    mv s3, a1
    la a0, p_base
    call param
    beqz a0, nf_args
    call sint
    beqz a0, nf_args
    mv s4, a1
    li t0, 2
    blt s4, t0, bad_base
    li t0, 36
    bgt s4, t0, bad_base
b_lead:
    bgeu s2, s3, nf_args
    lbu t0, 0(s2)
    li t1, 32
    beq t0, t1, b_skip
    li t1, 10
    beq t0, t1, b_skip
    j b_prefix
b_skip:
    addi s2, s2, 1
    j b_lead
b_prefix:
    li s5, 10
    addi t2, s2, 1
    bgeu t2, s3, b_start
    li t1, 48
    bne t0, t1, b_start
    lbu t3, 0(t2)
    ori t3, t3, 32                   ; (lower case)
    li t1, 120
    li t4, 16
    beq t3, t1, b_pre
    li t1, 98
    li t4, 2
    beq t3, t1, b_pre
    li t1, 111
    li t4, 8
    beq t3, t1, b_pre
    j b_start
b_pre:
    mv s5, t4
    addi s2, s2, 2
b_start:
    li s6, 0
    li s8, 0
b_digit:
    bgeu s2, s3, b_end
    lbu t0, 0(s2)
    li t1, 48
    blt t0, t1, b_end
    li t1, 57
    ble t0, t1, b_dec
    ori t0, t0, 32
    li t1, 97
    blt t0, t1, b_end
    li t1, 122
    bgt t0, t1, b_end
    addi t0, t0, -87
    j b_val
b_dec:
    addi t0, t0, -48
b_val:
    bge t0, s5, b_end
    mulhu t1, s6, s5
    bnez t1, nf_big
    mul t1, s6, s5
    add t2, t1, t0
    bltu t2, t1, nf_big
    mv s6, t2
    addi s8, s8, 1
    addi s2, s2, 1
    j b_digit
b_end:
    beqz s8, nf_args
b_tail:
    bgeu s2, s3, b_out
    lbu t0, 0(s2)
    li t1, 32
    beq t0, t1, b_tail_next
    li t1, 10
    beq t0, t1, b_tail_next
    j nf_args
b_tail_next:
    addi s2, s2, 1
    j b_tail
b_out:
    call open_resp
    mv a0, s6
    mv a1, s4
    call put_base
    j close_park

bad_base:
    call open_resp
    la a0, m_base
    call puts
    j close_park

; ---- count_letters: how many characters of text are any of the letters (ASCII letters in either case; what else
; letters holds, such as the commas and spaces of "a, e, i, o, u", is not a letter of the set): the set a bit a letter
do_letters:
    la a0, p_text
    call param
    beqz a0, nl_args
    mv s2, a0
    mv s3, a1
    la a0, p_letters
    call param
    beqz a0, nl_args
    la s6, fold
    li s9, 0                         ; the set (a: bit 0)
    li t3, 26
cl_set:
    bgeu a0, a1, cl_any
    lbu t1, 0(a0)
    addi a0, a0, 1
    add t1, t1, s6
    lbu t1, 0(t1)
    addi t1, t1, -97
    bgeu t1, t3, cl_set
    li t2, 1
    sll t2, t2, t1
    or s9, s9, t2
    j cl_set
cl_any:
    beqz s9, nl_args
    li s7, 0
cl_char:
    bgeu s2, s3, cl_done
    lbu t1, 0(s2)
    addi s2, s2, 1
    add t1, t1, s6
    lbu t1, 0(t1)
    addi t1, t1, -97
    bgeu t1, t3, cl_char
    srl t2, s9, t1
    andi t2, t2, 1
    add s7, s7, t2
    j cl_char
cl_done:
    call open_resp
    mv a0, s7
    call put_int
    j close_park
nl_args:
    j bad_args

.rodata
s_function: .byte 60, 102, 117, 110, 99, 116, 105, 111, 110, 61, 0
s_param: .byte 60, 112, 97, 114, 97, 109, 101, 116, 101, 114, 61, 0
s_endparam: .byte 60, 47, 112, 97, 114, 97, 109, 101, 116, 101, 114, 62, 0
n_calc: .byte 99, 97, 108, 99, 0
n_count: .byte 99, 111, 117, 110, 116, 0
n_length: .byte 108, 101, 110, 103, 116, 104, 0
n_reverse: .byte 114, 101, 118, 101, 114, 115, 101, 0
n_sort: .byte 115, 111, 114, 116, 0
n_weekday: .byte 119, 101, 101, 107, 100, 97, 121, 0
n_between: .byte 100, 97, 121, 115, 95, 98, 101, 116, 119, 101, 101, 110, 0
p_text: .byte 116, 101, 120, 116, 0
p_find: .byte 102, 105, 110, 100, 0
p_numbers: .byte 110, 117, 109, 98, 101, 114, 115, 0
p_date: .byte 100, 97, 116, 101, 0
p_start: .byte 115, 116, 97, 114, 116, 0
p_end: .byte 101, 110, 100, 0
m_unknown: .byte 101, 114, 114, 111, 114, 32, 40, 110, 111, 32, 115, 117, 99, 104, 32, 116, 111, 111, 108, 41, 0
m_args: .byte 101, 114, 114, 111, 114, 32, 40, 109, 105, 115, 115, 105, 110, 103, 32, 97, 114, 103, 117, 109, 101, 110, 116, 115, 41, 0
m_big: .byte 101, 114, 114, 111, 114, 32, 40, 116, 111, 111, 32, 98, 105, 103, 41, 0
m_date: .byte 101, 114, 114, 111, 114, 32, 40, 110, 111, 116, 32, 97, 32, 100, 97, 116, 101, 58, 32, 89, 89, 89, 89, 45, 77, 77, 45, 68, 68, 41, 0
n_add: .byte 97, 100, 100, 95, 100, 97, 121, 115, 0
n_words: .byte 119, 111, 114, 100, 115, 0
n_base: .byte 116, 111, 95, 98, 97, 115, 101, 0
p_days: .byte 100, 97, 121, 115, 0
p_number: .byte 110, 117, 109, 98, 101, 114, 0
p_base: .byte 98, 97, 115, 101, 0
n_letters: .byte 99, 111, 117, 110, 116, 95, 108, 101, 116, 116, 101, 114, 115, 0
p_letters: .byte 108, 101, 116, 116, 101, 114, 115, 0
n_long: .byte 116, 111, 111, 95, 108, 111, 110, 103, 0
m_long: .byte 101, 114, 114, 111, 114, 32, 40, 116, 104, 101, 32, 99, 97, 108, 108, 32, 105, 115, 32, 116, 111, 111, 32, 108, 111, 110, 103, 32, 102, 111, 114, 32, 116, 104, 101, 32, 99, 111, 109, 112, 117, 116, 101, 114, 58, 32, 97, 116, 32, 109, 111, 115, 116, 32, 97, 98, 111, 117, 116, 32, 49, 44, 48, 48, 48, 32, 116, 111, 107, 101, 110, 115, 32, 111, 114, 32, 55, 44, 48, 48, 48, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 115, 41, 0
m_nums: .byte 101, 114, 114, 111, 114, 32, 40, 110, 117, 109, 98, 101, 114, 115, 32, 115, 117, 99, 104, 32, 97, 115, 32, 49, 50, 44, 32, 45, 51, 44, 32, 52, 46, 53, 58, 32, 110, 111, 32, 102, 114, 97, 99, 116, 105, 111, 110, 115, 44, 32, 116, 104, 111, 117, 115, 97, 110, 100, 115, 32, 115, 101, 112, 97, 114, 97, 116, 111, 114, 115, 44, 32, 117, 110, 105, 116, 115, 32, 111, 114, 32, 101, 120, 112, 111, 110, 101, 110, 116, 115, 41, 0
m_many: .byte 101, 114, 114, 111, 114, 32, 40, 109, 111, 114, 101, 32, 116, 104, 97, 110, 32, 50, 53, 54, 32, 110, 117, 109, 98, 101, 114, 115, 41, 0
m_range: .byte 101, 114, 114, 111, 114, 32, 40, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101, 58, 32, 121, 101, 97, 114, 115, 32, 48, 48, 48, 48, 32, 116, 111, 32, 57, 57, 57, 57, 41, 0
m_base: .byte 101, 114, 114, 111, 114, 32, 40, 98, 97, 115, 101, 58, 32, 50, 32, 116, 111, 32, 51, 54, 41, 0
fold:                                ; each byte, ASCII letters in lower case
    .byte 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    .byte 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    .byte 64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91, 92, 93, 94, 95
    .byte 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
    .byte 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159
    .byte 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191
    .byte 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223
    .byte 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
.align 2
gaps: .word 701, 301, 132, 57, 23, 10, 4, 1, 0  ; (Ciura's)
month_days: .byte 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
day_names:
    .byte 83, 117, 110, 100, 97, 121, 0, 0, 0, 0
    .byte 77, 111, 110, 100, 97, 121, 0, 0, 0, 0
    .byte 84, 117, 101, 115, 100, 97, 121, 0, 0, 0
    .byte 87, 101, 100, 110, 101, 115, 100, 97, 121, 0
    .byte 84, 104, 117, 114, 115, 100, 97, 121, 0, 0
    .byte 70, 114, 105, 100, 97, 121, 0, 0, 0, 0
    .byte 83, 97, 116, 117, 114, 100, 97, 121, 0, 0
