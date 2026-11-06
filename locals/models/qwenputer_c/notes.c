/* Notes the computer keeps: remember, recall, forget. They stay in its RAM from one call to the next (in a reply,
   and in a conversation whose computer the server keeps or rebuilds from the conversation's remember calls). */
#include "tools.h"

#define STORE 2048
static struct {
    u32 used;          /* bytes of `data` in use: notes one after another, key length, value length, key, value */
    u8 data[STORE];
} notes; /* (zeroed data: no notes in a new computer) */

typedef struct {
    u8 *at;
    str key, value;
} note;

static int note_at(u8 *p, note *n) {
    if (p >= notes.data + notes.used) return 0;
    int kl = p[0], vl = p[1] | p[2] << 8;
    n->at = p;
    n->key = (str){(const char *)p + 3, (const char *)p + 3 + kl};
    n->value = (str){n->key.e, n->key.e + vl};
    return 1;
}

static u8 *after(note *n) { return (u8 *)n->value.e; }

static int find_note(str key, note *out) {
    note n;
    for (u8 *p = notes.data; note_at(p, &n); p = after(&n))
        if (str_ieq(n.key, key)) return *out = n, 1;
    return 0;
}

static void drop(note *n) {
    u8 *end = notes.data + notes.used, *next = after(n);
    memmove(n->at, next, end - next);
    notes.used -= (u32)(next - n->at);
}

static void put_note(str k, str v) {
    u8 *p = notes.data + notes.used;
    p[0] = (u8)(k.e - k.p);
    p[1] = (u8)(v.e - v.p), p[2] = (u8)((v.e - v.p) >> 8);
    memcpy(p + 3, k.p, k.e - k.p);
    memcpy(p + 3 + (k.e - k.p), v.p, v.e - v.p);
    notes.used += 3 + (u32)(k.e - k.p) + (u32)(v.e - v.p);
}

/* a conversation's notes from its earlier calls, in a computer that has not got them (the server adds them to the
   first notes call of a request as _replay: a line each, "remember", key, value or "forget", key, tab-separated) */
static void replay(void) {
    str r = param("_replay");
    if (!have(r)) return;
    notes.used = 0;
    for (str s = r; s.p < s.e;) {
        const char *nl = find_byte(s.p, s.e, '\n'), *q;
        str line = {s.p, nl ? nl : s.e}, op, k, v = {0, 0};
        s.p = nl ? nl + 1 : s.e;
        for (q = line.p; q < line.e && *q != '\t'; q++) {
        }
        op = (str){line.p, q};
        k.p = q < line.e ? q + 1 : q;
        for (q = k.p; q < line.e && *q != '\t'; q++) {
        }
        k.e = q;
        if (q < line.e) v = (str){q + 1, line.e};
        k = trim(k), v = trim(v);
        if (k.p == k.e || k.e - k.p > 60) continue;
        note old;
        if (find_note(k, &old)) drop(&old);
        if (str_is(op, "remember") && have(v) && notes.used + 3 + (k.e - k.p) + (v.e - v.p) <= STORE) put_note(k, v);
    }
}

static str key_of(void) {
    str k = trim(param("key"));
    if (!have(k)) k = trim(param("name"));
    if (have(k) && k.e - k.p >= 2 && (*k.p == '"' || *k.p == '\'') && k.e[-1] == *k.p) k.p++, k.e--;
    return k;
}

/* the notes as "key: value, ...", within about `room` bytes (a long value cut at 60, then "...") */
static void say_notes(int room) {
    note n;
    int i = 0;
    for (u8 *p = notes.data; note_at(p, &n); p = after(&n)) {
        int len = (int)(n.key.e - n.key.p) + 2 + (int)(n.value.e - n.value.p > 63 ? 63 : n.value.e - n.value.p);
        if (i && len > room) {
            say(", ...");
            return;
        }
        if (i++) say(", ");
        say_str(n.key);
        say(": ");
        if (n.value.e - n.value.p > 63) say_str((str){n.value.p, n.value.p + 60}), say("...");
        else say_str(n.value);
        room -= len + 2;
    }
}

/* a key that asks for every note ("all", "every", "notes"...), letter case aside */
static int all_key(str k) {
    static const char *const ALL[] = {"*", "all", "every", "everything", "notes", "all notes", "everything you remember",
                                      "list", 0};
    for (int i = 0; ALL[i]; i++)
        if (str_ieq(k, (str){ALL[i], ALL[i] + strlen(ALL[i])})) return 1;
    return 0;
}

void remember(void) {
    replay();
    str k = key_of(), v = trim(param("value"));
    if (!have(k) || k.p == k.e) fail("missing argument: key, the note's name");
    if (!have(v)) fail("missing argument: value, what to remember");
    if (k.e - k.p > 60) fail("a key of more than 60 characters");
    note old;
    int had = find_note(k, &old);
    u32 need = 3 + (u32)(k.e - k.p) + (u32)(v.e - v.p), freed = had ? (u32)(after(&old) - old.at) : 0;
    if (notes.used - freed + need > STORE) fail("the notes are full (about 2,000 characters): forget one first");
    if (had) {
        say("updated \"");
        say_str(k);
        say("\" (it was: ");
        say_str(old.value);
        say(")");
        drop(&old);
    } else {
        say("remembered \"");
        say_str(k);
        put('"');
    }
    put_note(k, v);
}

void recall(void) {
    replay();
    str k = key_of();
    note n;
    if (have(k) && k.p < k.e && find_note(k, &n)) { /* (a note of that name first, even one named "list") */
        say_str(n.value);
        return;
    }
    if (!have(k) || k.p == k.e || all_key(k)) {
        if (!notes.used) {
            say("no notes yet");
            return;
        }
        int i = 0;
        for (u8 *p = notes.data; note_at(p, &n); p = after(&n)) {
            if (i++) put('\n');
            say_str(n.key);
            say(": ");
            say_str(n.value);
        }
        return;
    }
    say("error (no note \"");
    say_str(k);
    if (notes.used) say("\"; the notes: "), say_notes(400);
    else say("\"; no notes yet");
    put(')');
    done();
}

void forget(void) {
    replay();
    str k = key_of();
    note n;
    if (!have(k) || k.p == k.e) fail("missing argument: key");
    if (!find_note(k, &n)) {
        say("error (no note \"");
        say_str(k);
        say("\")");
        done();
    }
    drop(&n);
    say("forgot \"");
    say_str(k);
    put('"');
}
