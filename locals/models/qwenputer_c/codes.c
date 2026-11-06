/* Check digits and hashes: check_digit (Luhn: cards, IMEI; ISBN-10, ISBN-13, EAN-13, UPC-A; IBAN) and hash_text
   (CRC-32, Adler-32, SHA-256 of the text's UTF-8 bytes). */
#include "tools.h"

void check_digit(void) {
    str t = trim(param("number")), k = trim(param("kind"));
    if (!have(t) || t.p == t.e) fail("missing argument: number");
    char d[64];
    int n = 0, letters = 0;
    for (const char *p = t.p; p < t.e; p++) {
        if (digit(*p) || ((*p | 32) == 'x') || (unsigned)((*p | 32) - 'a') < 26) {
            if (n == 60) fail("a number of more than 60 characters");
            d[n++] = (char)((unsigned)(*p - 'a') < 26 ? *p - 32 : *p);
            letters |= !digit(*p);
        } else if (*p != ' ' && *p != '-')
            fail("a number of digits (spaces and dashes are left out)");
    }
    static const char LUHN[] = "luhn", ISBN10[] = "isbn10", ISBN13[] = "isbn13", EAN[] = "ean13", UPC[] = "upc", IBAN[] = "iban";
#define K(z) (str_ieq(k, (str){z, &z[sizeof z - 1]}))
    int kind = !have(k) || k.p == k.e ? -1 : K(LUHN) || K("card") || K("credit card") || K("imei") ? 0 : K(ISBN10) || K("isbn-10") ? 1
               : K(ISBN13) || K("isbn-13") ? 2 : K(EAN) || K("ean") || K("ean-13") ? 3 : K(UPC) || K("upc-a") ? 4 : K(IBAN) ? 5
               : K("isbn") ? (n == 10 ? 1 : 2) : -2;
#undef K
    if (kind == -2) fail("kind: luhn (a card or IMEI), isbn10, isbn13, ean13, upc or iban");
    if (kind == -1) kind = letters ? (n == 10 ? 1 : 5) : n == 13 ? 2 : n == 12 ? 4 : 0;
    int want, got;
    static const char *const NAMES[] = {"Luhn", "ISBN-10", "ISBN-13", "EAN-13", "UPC-A", "IBAN"};
    if (kind == 5) { /* the first 4 to the end, letters as 10..35, the whole mod 97 is 1 */
        if (n < 15 || n > 34 || !((unsigned)(d[0] - 'A') < 26) || !((unsigned)(d[1] - 'A') < 26) || !digit(d[2]) || !digit(d[3]))
            fail("an IBAN: 2 letters, 2 check digits, then up to 30 letters and digits");
        u32 r = 0;
        for (int i = 0; i < n; i++) {
            char c = d[(i + 4) % n];
            if (digit(c)) r = (r * 10 + (c - '0')) % 97;
            else r = (r * 100 + (c - 'A' + 10)) % 97;
        }
        /* the right check digits: 98 - (the number with 00 in their place) mod 97 */
        u32 r0 = 0;
        for (int i = 0; i < n; i++) {
            char c = (i == n - 2 || i == n - 1) ? '0' : d[(i + 4) % n]; /* (its check digits as 00) */
            if (digit(c)) r0 = (r0 * 10 + (c - '0')) % 97;
            else r0 = (r0 * 100 + (c - 'A' + 10)) % 97;
        }
        want = (int)(98 - r0), got = (d[2] - '0') * 10 + (d[3] - '0');
        say(r == 1 ? "valid IBAN" : "not a valid IBAN: its check digits should be ");
        if (r != 1) put('0' + want / 10), put('0' + want % 10);
        return;
    }
    for (int i = 0; i < n; i++)
        if (!digit(d[i]) && !(kind == 1 && i == n - 1 && d[i] == 'X')) fail("digits only (an ISBN-10 may end in X)");
    /* the check digit of the number as a whole (its last digit) and the one to append to it (as a body without one) */
    int len = kind == 1 ? 10 : kind == 2 || kind == 3 ? 13 : kind == 4 ? 12 : 0;
    if (len && n != len && n != len - 1) {
        say("error (");
        say(NAMES[kind]);
        say(" has ");
        say_int(len);
        say(" digits, or ");
        say_int(len - 1);
        say(" without its check digit)");
        done();
    }
    int wants[2] = {-1, -1}; /* [whole]: the check digit of the body without it, and of the whole number */
    for (int whole = 1; whole >= 0; whole--) {
        int m = whole ? n - 1 : n; /* (the body's digits) */
        if (len && m != len - 1) continue;
        if (whole && n < 2) continue;
        if (kind == 0) { /* every second digit from the right of the body doubled */
            int s = 0;
            for (int i = m - 1, dbl = 1; i >= 0; i--, dbl ^= 1) {
                int v = (d[i] - '0') * (dbl ? 2 : 1);
                s += v > 9 ? v - 9 : v;
            }
            want = (10 - s % 10) % 10;
        } else if (kind == 1) {
            int s = 0;
            for (int i = 0; i < 9; i++) s += (d[i] - '0') * (10 - i);
            want = (11 - s % 11) % 11;
        } else {
            int s = 0;
            for (int i = 0; i < m; i++) s += (d[i] - '0') * ((kind == 4 ? i % 2 == 0 : i % 2 == 1) ? 3 : 1);
            want = (10 - s % 10) % 10;
        }
        wants[whole] = want;
    }
    /* a Luhn number either way: valid as written first, else the digit to append first (a number that is not valid as
       written is most likely one without its check digit, and Qwen took the first digit it read) */
    got = n ? (d[n - 1] == 'X' ? 10 : d[n - 1] - '0') : -1;
    int ok = wants[1] >= 0 && got == wants[1];
    if (wants[0] >= 0 && !ok && kind == 0) {
        say("the digit to append to it as a number without its check digit: ");
        put('0' + wants[0]);
        if (wants[1] >= 0) say("; ");
    }
    if (wants[1] >= 0) {
        if (ok) say("valid "), say(NAMES[kind]), say(" (its check digit ");
        else say(kind == 0 && wants[0] >= 0 ? "as a whole number, not valid " : "not valid as "), say(NAMES[kind]),
             say(kind == 0 && wants[0] >= 0 ? " (its last digit should be " : " (its check digit should be ");
        put(wants[1] == 10 ? 'X' : '0' + wants[1]);
        if (!ok) say(", not "), put(d[n - 1]);
        put(')');
    }
    if (wants[0] >= 0 && (ok || kind != 0)) {
        if (wants[1] >= 0) say("; ");
        say(kind == 0 ? "as a number without its check digit, the digit to append: " : "its check digit: ");
        put(wants[0] == 10 ? 'X' : '0' + wants[0]);
    }
}

/* ---- hashes */
static u32 rotr(u32 x, int n) { return x >> n | x << (32 - n); }

static void sha256(const u8 *m, size_t len, u8 out[32]) {
    static const u32 K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    u32 h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t blocks = (len + 9 + 63) / 64;
    for (size_t b = 0; b < blocks; b++) {
        u32 w[64];
        size_t base = b * 64;
        if (base + 64 <= len) /* (a block of the message alone) */
            for (int i = 0; i < 16; i++) {
                const u8 *q = m + base + i * 4;
                w[i] = (u32)q[0] << 24 | (u32)q[1] << 16 | (u32)q[2] << 8 | q[3];
            }
        else
            for (int i = 0; i < 16; i++) {
                u32 v = 0;
                for (int k = 0; k < 4; k++) {
                    size_t at = base + i * 4 + k;
                    u8 c = at < len ? m[at] : at == len ? 0x80 : 0;
                    if (at >= blocks * 64 - 8) c = (u8)((u64)len * 8 >> (8 * (blocks * 64 - 1 - at)));
                    v = v << 8 | c;
                }
                w[i] = v;
            }
        for (int i = 16; i < 64; i++) {
            u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        u32 a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            u32 t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            u32 t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = bb, bb = a, a = t1 + t2;
        }
        h[0] += a, h[1] += bb, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 4; k++) out[i * 4 + k] = (u8)(h[i] >> (24 - 8 * k));
}

static void say_hex(u32 v, int digits) {
    for (int i = digits - 1; i >= 0; i--) put("0123456789abcdef"[v >> (4 * i) & 15]);
}

void hash_text(void) {
    str t = param("text"), a = trim(param("algorithm"));
    if (!have(t)) fail("missing argument: text");
    const u8 *p = (const u8 *)t.p;
    size_t n = t.e - t.p;
    static const char CRC[] = "crc32", ADLER[] = "adler32", SHA[] = "sha256";
#define K(z) str_ieq(a, (str){z, &z[sizeof z - 1]})
    if (!have(a) || a.p == a.e || K(SHA) || K("sha-256")) {
        if (n > 640) fail("SHA-256 of at most 640 bytes on the computer: a longer text's would take past its run");
        u8 h[32];
        sha256(p, n, h);
        for (int i = 0; i < 32; i++) say_hex(h[i], 2);
    } else if (K(CRC) || K("crc-32") || K("crc")) {
        u32 *table = alloc(sizeof(u32) * 256), c = 0xffffffffu; /* (a byte at a time from a table) */
        for (u32 i = 0; i < 256; i++) {
            u32 x = i;
            for (int k = 0; k < 8; k++) x = x >> 1 ^ (0xedb88320u & -(x & 1));
            table[i] = x;
        }
        for (size_t i = 0; i < n; i++) c = c >> 8 ^ table[(c ^ p[i]) & 255];
        say_hex(~c, 8);
    } else if (K(ADLER) || K("adler-32")) {
        u32 s1 = 1, s2 = 0;
        for (size_t i = 0; i < n;) { /* (the modulus every 5,552 bytes, as zlib takes it) */
            size_t e = n - i > 5552 ? i + 5552 : n;
            for (; i < e; i++) s1 += p[i], s2 += s1;
            s1 %= 65521, s2 %= 65521;
        }
        say_hex(s2 << 16 | s1, 8);
    } else
        fail("algorithm: sha256, crc32 or adler32 (md5 and sha1 are not on the computer)");
#undef K
}
