/* The toolbox's C built for this Mac, to test it (tests/test_qwenputer_c.py): `host < calls` prints the responses
   to the calls' texts (as the bridge gives them: \n<function=NAME>\n<parameter=...>...</function>\n), the calls
   separated by a byte 0x1e and run one after another on one memory (as the computer runs them: the notes stay), the
   responses separated by the same byte. */
#undef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0 /* (string.h's memcpy and the others as functions: base.c has its own) */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tools.h"

jmp_buf host_end;
char host_heap[16384], *host_heap_end = host_heap + sizeof host_heap, host_low[5120];
int c_main(void);

int main(void) {
    static char buf[1 << 17], name[64];
    size_t n = fread(buf, 1, sizeof buf, stdin);
    char *p = buf, *end = buf + n;
    for (int i = 0; p <= end; i++) {
        char *e = memchr(p, 0x1e, end - p);
        if (!e) e = end;
        char *f = strstr(p, "<function="), *g = f ? strchr(f, '>') : 0;
        name[0] = 0;
        if (f && g && g < e && g - f - 10 < (long)sizeof name) memcpy(name, f + 10, g - f - 10), name[g - f - 10] = 0;
        call_name = name;
        text = p;
        text_end = e;
        if (i) putchar(0x1e);
        if (!setjmp(host_end) && !c_main()) printf("error (no such tool)");
        p = e + 1;
    }
    return 0;
}
