/* Where the toolbox's C goes in the computer's memory (chipformer's map: RAM at 0, DROM at 0x20000, ROM at
   0x40000): code from 0x48000 (c_main first), after calc's and qwenputer_tools.s's; read-only data from 0x22000;
   zeroed data in RAM's 0x800..0x1a00 (calc's data ends at 0x390, the dispatcher's at 0x780). No initialized data. */
ENTRY(c_main)
MEMORY
{
  RAM (rw) : ORIGIN = 0x800, LENGTH = 0x1200
  DROM (r) : ORIGIN = 0x22000, LENGTH = 0x1e000
  ROM (rx) : ORIGIN = 0x48000, LENGTH = 0x38000
}
SECTIONS
{
  .text : { KEEP(*(.text.entry)) *(.text .text.*) } > ROM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } > RAM
  /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.eh_frame*) *(.note*) }
}
