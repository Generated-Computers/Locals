/* Dates and times (the Gregorian calendar, no leap seconds): time_add, time_between, age, date_info, unix_time,
   convert_timezone. A time is HH:MM[:SS] (and am / pm), a moment YYYY-MM-DD HH:MM[:SS] (or with T); a time zone a
   fixed offset (UTC+5:30, -08:00) or a name that is one (EST, CEST, IST, JST...): a place's own (New York's), whose
   offset changes with daylight saving, needs the offset the date has. */
#include "tools.h"

static i64 days_from_civil(i64 y, int m, int d) { /* (Howard Hinnant's) days since 1970-01-01 */
    y -= m <= 2;
    i64 era = (y >= 0 ? y : y - 399) / 400;
    i64 yoe = y - era * 400;
    i64 doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static void civil_from_days(i64 z, i64 *y, int *m, int *d) {
    z += 719468;
    i64 era = (z >= 0 ? z : z - 146096) / 146097;
    i64 doe = z - era * 146097;
    i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    i64 mp = (5 * doy + 2) / 153;
    *d = (int)(doy - (153 * mp + 2) / 5 + 1);
    *m = (int)(mp < 10 ? mp + 3 : mp - 9);
    *y = yoe + era * 400 + (*m <= 2);
}

static int leap(i64 y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static int month_days(i64 y, int m) {
    static const int D[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return D[m - 1] + (m == 2 && leap(y));
}

static const char *const WEEKDAYS[] = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
static const char *const MONTHS[] = {"January", "February", "March", "April", "May", "June", "July", "August",
                                     "September", "October", "November", "December"};

static int weekday(i64 days) { return (int)(((days % 7) + 7 + 3) % 7); } /* (1970-01-01 a Thursday: 0 Monday) */

/* a moment: seconds since 1970-01-01 00:00 (of its own zone), and whether it had a date */
typedef struct {
    i64 s;
    int dated, z; /* z: an offset was given (in tz, minutes) */
    int tz;
} moment;

static int num_at(const char **p, const char *e, int digits, int *v) {
    int n = 0, k = 0;
    while (*p < e && digit(**p) && k < digits) n = n * 10 + (**p - '0'), (*p)++, k++;
    *v = n;
    return k;
}

static int zone(str z, int *minutes); /* below */

static int read_moment(str t, moment *m) {
    t = trim(t);
    const char *p = t.p, *e = t.e;
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0;
    m->dated = m->z = 0, m->tz = 0;
    if (e - p >= 10 && p[4] == '-' && p[7] == '-') {
        const char *q = p;
        if (num_at(&q, e, 4, &y) != 4 || *q++ != '-' || num_at(&q, e, 2, &mo) != 2 || *q++ != '-' || num_at(&q, e, 2, &d) != 2)
            return 0;
        if (mo < 1 || mo > 12 || d < 1 || d > month_days(y, mo)) return 0;
        m->dated = 1;
        p = q;
        if (p < e && (*p == 'T' || *p == ' ')) p++;
        while (p < e && *p == ' ') p++;
    }
    if (p < e && digit(*p)) {
        if (!num_at(&p, e, 2, &h)) return 0;
        if (p < e && *p == ':') {
            p++;
            if (num_at(&p, e, 2, &mi) != 2) return 0;
            if (p < e && *p == ':') {
                p++;
                if (num_at(&p, e, 2, &se) != 2) return 0;
            }
        } else if (!m->dated && !(p < e && (*p == ' ' || lower((u8)*p) == 'a' || lower((u8)*p) == 'p')))
            return 0;
        while (p < e && *p == ' ') p++;
        if (e - p >= 2 && (lower((u8)p[1]) == 'm') && (lower((u8)*p) == 'a' || lower((u8)*p) == 'p')) { /* am, pm */
            if (h < 1 || h > 12) return 0;
            h = h % 12 + (lower((u8)*p) == 'p') * 12;
            p += 2;
            if (p < e && *p == '.') p++;
        }
        if (h > 23 || mi > 59 || se > 59) return 0;
    } else if (!m->dated)
        return 0;
    while (p < e && *p == ' ') p++;
    if (p < e) { /* a zone after it */
        if (*p == 'Z' && p + 1 == e) m->z = 1, m->tz = 0;
        else if (zone((str){p, e}, &m->tz)) m->z = 1;
        else return 0;
    }
    m->s = (m->dated ? days_from_civil(y, mo, d) * 86400 : 0) + h * 3600 + mi * 60 + se;
    return 1;
}

static void say2(int v) { put('0' + v / 10 % 10), put('0' + v % 10); }

static void say_moment(i64 s, int dated, int secs) {
    i64 days = s >= 0 ? s / 86400 : -((-s + 86399) / 86400), r = s - days * 86400;
    if (dated) {
        i64 y;
        int m, d;
        civil_from_days(days, &y, &m, &d);
        if (y < 1 || y > 9999) fail("a date outside the years 1 to 9999");
        put('0' + y / 1000 % 10), put('0' + y / 100 % 10), put('0' + y / 10 % 10), put('0' + y % 10);
        put('-'), say2(m), put('-'), say2(d), put(' ');
    }
    say2((int)(r / 3600)), put(':'), say2((int)(r / 60 % 60));
    if (secs || r % 60) put(':'), say2((int)(r % 60));
}

/* an amount of time: 2h 45m, 90 minutes, 1 day 3 hours, -30 min, 1:30 (h:mm), 1.5 hours */
static int read_amount(str t, i64 *secs) {
    t = trim(t);
    const char *p = t.p, *e = t.e;
    int neg = 0, any = 0;
    *secs = 0;
    if (p < e && (*p == '-' || *p == '+')) neg = *p == '-', p++;
    while (p < e && *p == ' ') p++;
    { /* h:mm[:ss] */
        const char *q = p;
        int h, mi, se = 0;
        if (num_at(&q, e, 4, &h) && q < e && *q == ':') {
            q++;
            if (num_at(&q, e, 2, &mi) == 2) {
                if (q < e && *q == ':') q++, num_at(&q, e, 2, &se);
                if (q == e) {
                    *secs = (i64)h * 3600 + mi * 60 + se;
                    if (neg) *secs = -*secs;
                    return 1;
                }
            }
        }
    }
    while (p < e) {
        while (p < e && (*p == ' ' || *p == ',')) p++;
        if (e - p >= 4 && !memcmp(p, "and ", 4)) p += 4;
        if (p >= e) break;
        i64 v = 0;
        int frac = 0, fd = 0, k = 0;
        while (p < e && digit(*p)) {
            if (++k > 12) return 0; /* (no amount of time the calendar's 10,000 years have) */
            v = v * 10 + (*p++ - '0');
        }
        if (p < e && *p == '.')
            for (p++; p < e && digit(*p) && fd < 3; p++) frac = frac * 10 + (*p - '0'), fd++;
        if (!k && !fd) return 0;
        while (p < e && *p == ' ') p++;
        const char *u = p;
        while (p < e && (unsigned)((*p | 32) - 'a') < 26) p++;
        str unit = {u, p};
        static const struct {
            const char *z;
            i64 s;
        } U[] = {{"s", 1}, {"sec", 1}, {"secs", 1}, {"second", 1}, {"seconds", 1}, {"m", 60}, {"min", 60}, {"mins", 60},
                 {"minute", 60}, {"minutes", 60}, {"h", 3600}, {"hr", 3600}, {"hrs", 3600}, {"hour", 3600}, {"hours", 3600},
                 {"d", 86400}, {"day", 86400}, {"days", 86400}, {"w", 604800}, {"wk", 604800}, {"week", 604800}, {"weeks", 604800}};
        i64 mult = -1;
        for (size_t i = 0; i < sizeof U / sizeof U[0] && mult < 0; i++)
            if (str_ieq(unit, (str){U[i].z, U[i].z + strlen(U[i].z)})) mult = U[i].s;
        if (mult < 0) return 0;
        static const int P10[] = {1, 10, 100, 1000};
        *secs += v * mult + frac * mult / P10[fd];
        if (*secs > 400000000000000LL) return 0;
        any = 1;
    }
    if (neg) *secs = -*secs;
    return any;
}

static void say_duration(i64 s) {
    if (s < 0) say("minus "), s = -s;
    i64 d = s / 86400, h = s / 3600 % 24, m = s / 60 % 60, se = s % 60;
    int first = 1;
#define PART(v, one, many)                                                                                             \
    if (v) {                                                                                                           \
        if (!first) say(", ");                                                                                         \
        say_int(v);                                                                                                    \
        say(v == 1 ? one : many);                                                                                      \
        first = 0;                                                                                                     \
    }
    PART(d, " day", " days")
    PART(h, " hour", " hours")
    PART(m, " minute", " minutes")
    PART(se, " second", " seconds")
#undef PART
    if (first) say("0 minutes");
}

static moment need_moment(const char *name) {
    moment m;
    str t = param(name);
    if (!have(t) || !read_moment(t, &m)) {
        say("error (");
        say(name);
        say(": a time (14:30, 2:30 pm) or a date and time (2024-03-10 14:30), with a zone if it has one)");
        done();
    }
    return m;
}

void time_add(void) {
    moment m = need_moment("time");
    i64 a;
    if (!read_amount(param("amount"), &a)) fail("amount: e.g. 2h 45m, 90 minutes, -1 day, 1:30");
    i64 r = m.s + a;
    say_moment(r, m.dated, 0);
    if (!m.dated) {
        i64 days = r >= 0 ? r / 86400 : -((-r + 86399) / 86400);
        if (days == 1) say(" (the next day)");
        else if (days == -1) say(" (the day before)");
        else if (days) say(" ("), say_int(days > 0 ? days : -days), say(days > 0 ? " days later)" : " days earlier)");
    }
}

void time_between(void) {
    moment a = need_moment("start"), b = need_moment("end");
    i64 s1 = a.s - (a.z ? a.tz * 60 : 0), s2 = b.s - (b.z ? b.tz * 60 : 0), d = s2 - s1;
    int next = 0;
    if (!a.dated && !b.dated && d < 0) d += 86400, next = 1; /* (an end before the start: the next day's) */
    say_duration(d);
    if (d % 60 == 0 && d >= 3600) say(" ("), say_int(d / 60), say(" minutes)");
    if (next) say(" (the end the next day)");
}

static int read_date(str t, i64 *days) {
    moment m;
    if (!read_moment(t, &m) || !m.dated) return 0;
    *days = m.s >= 0 ? m.s / 86400 : -((-m.s + 86399) / 86400);
    return 1;
}

void age(void) {
    i64 a, b;
    if (!read_date(param("birth_date"), &a)) fail("birth_date: YYYY-MM-DD");
    str on = param("on_date");
    if (!have(on) || !read_date(on, &b)) fail("on_date: YYYY-MM-DD (the computer has no clock: give today's date)");
    if (b < a) fail("on_date is before birth_date");
    i64 y1, y2;
    int m1, d1, m2, d2;
    civil_from_days(a, &y1, &m1, &d1);
    civil_from_days(b, &y2, &m2, &d2);
    /* whole years and months (as Python's dateutil counts them: a month on from the 31st is the month's last day),
       then the days from there */
    i64 y = y2 - y1;
    int mo = m2 - m1 - (d2 < d1);
    if (mo < 0) y--, mo += 12;
    i64 yy = y1 + y + (m1 - 1 + mo) / 12;
    int mm = (m1 - 1 + mo) % 12 + 1, dd = d1 < month_days(yy, mm) ? d1 : month_days(yy, mm);
    int d = (int)(b - days_from_civil(yy, mm, dd));
    say_int(y);
    say(y == 1 ? " year, " : " years, ");
    say_int(mo);
    say(mo == 1 ? " month and " : " months and ");
    say_int(d);
    say(d == 1 ? " day (" : " days (");
    say_int(b - a);
    say(" days)");
}

void date_info(void) {
    i64 z;
    if (!read_date(param("date"), &z)) fail("date: YYYY-MM-DD");
    i64 y;
    int m, d;
    civil_from_days(z, &y, &m, &d);
    int wd = weekday(z), doy = (int)(z - days_from_civil(y, 1, 1)) + 1;
    /* the ISO week: the week of the Thursday of this one */
    i64 thu = z - wd + 3, ty;
    int tm, td;
    civil_from_days(thu, &ty, &tm, &td);
    int week = (int)((thu - days_from_civil(ty, 1, 1)) / 7) + 1;
    say(WEEKDAYS[wd]);
    say(", ");
    say(MONTHS[m - 1]);
    put(' ');
    say_int(d);
    say(", ");
    say_int(y);
    say(": day ");
    say_int(doy);
    say(" of ");
    say_int(365 + leap(y));
    say(leap(y) ? " (a leap year)" : " (not a leap year)");
    say(", ISO week ");
    say_int(week);
    say(" of ");
    say_int(ty);
    say(", quarter ");
    say_int((m - 1) / 3 + 1);
    say(", ");
    say(MONTHS[m - 1]);
    say(" has ");
    say_int(month_days(y, m));
    say(" days");
}

void unix_time(void) {
    str v = trim(param("value"));
    if (!have(v) || v.p == v.e) fail("value: a Unix timestamp (seconds) or a date and time (UTC unless a zone is given)");
    num n;
    if (read_num(v, &n, 0) && !n.frac && (v.e - v.p != 10 || v.p[4] != '-')) {
        int ms = n.v > 100000000000LL || n.v < -100000000000LL;
        i64 s = ms ? n.v / 1000 : n.v;
        say_moment(s, 1, 1);
        say(" UTC");
        if (ms) say(" (the number read as milliseconds)");
        return;
    }
    moment m;
    if (!read_moment(v, &m) || !m.dated) fail("value: a Unix timestamp or YYYY-MM-DD HH:MM[:SS]");
    say_int(m.s - (m.z ? m.tz * 60 : 0));
    if (!m.z) say(" (the time read as UTC)");
}

static const struct {
    const char *z;
    int min;
} ZONES[] = {{"UTC", 0}, {"GMT", 0}, {"WET", 0}, {"BST", 60}, {"IST", 330}, {"CET", 60}, {"CEST", 120}, {"EET", 120},
             {"EEST", 180}, {"WEST", 60}, {"MSK", 180}, {"EST", -300}, {"EDT", -240}, {"CST", -360}, {"CDT", -300},
             {"MST", -420}, {"MDT", -360}, {"PST", -480}, {"PDT", -420}, {"AKST", -540}, {"AKDT", -480}, {"HST", -600},
             {"JST", 540}, {"KST", 540}, {"SGT", 480}, {"HKT", 480}, {"AEST", 600}, {"AEDT", 660}, {"ACST", 570},
             {"AWST", 480}, {"NZST", 720}, {"NZDT", 780}, {"PKT", 300}, {"GST", 240}, {"SAST", 120}, {"BRT", -180},
             {"ART", -180}, {"WIB", 420}, {"ICT", 420}, {"PHT", 480}};

static int zone(str z, int *minutes) {
    z = trim(z);
    for (size_t i = 0; i < sizeof ZONES / sizeof ZONES[0]; i++)
        if (str_ieq(z, (str){ZONES[i].z, ZONES[i].z + strlen(ZONES[i].z)})) return *minutes = ZONES[i].min, 1;
    const char *p = z.p, *e = z.e;
    static const char UTC[] = "UTC", GMT[] = "GMT";
    if (e - p >= 3 && (str_ieq((str){p, p + 3}, (str){UTC, UTC + 3}) || str_ieq((str){p, p + 3}, (str){GMT, GMT + 3}))) p += 3;
    if (p >= e || (*p != '+' && *p != '-')) return 0;
    int neg = *p++ == '-', h = 0, m = 0;
    if (!num_at(&p, e, 2, &h)) return 0;
    if (p < e && *p == ':') p++;
    if (p < e) num_at(&p, e, 2, &m);
    if (p != e || h > 14 || m > 59) return 0;
    *minutes = (neg ? -1 : 1) * (h * 60 + m);
    return 1;
}

static void say_zone(int m) {
    say("UTC");
    if (!m) return;
    put(m < 0 ? '-' : '+');
    if (m < 0) m = -m;
    say_int(m / 60);
    if (m % 60) put(':'), say2(m % 60);
}

void convert_timezone(void) {
    moment m = need_moment("time");
    int fz = 0, tz = 0;
    str from = param("from"), to = param("to");
    if (m.z) fz = m.tz;
    else if (!have(from) || !zone(from, &fz)) {
        say("error (from: a time zone as an offset (UTC+5:30, -08:00) or a name of one (EST, CEST, JST); a place's own "
            "changes with daylight saving: give the offset its date has)");
        done();
    }
    if (!have(to) || !zone(to, &tz)) fail("to: a time zone as an offset (UTC+9) or a name of one (JST)");
    i64 r = m.s - (i64)fz * 60 + (i64)tz * 60;
    say_moment(r, m.dated, 0);
    put(' ');
    say_zone(tz);
    if (!m.dated) {
        i64 days = r >= 0 ? r / 86400 : -((-r + 86399) / 86400);
        if (days > 0) say(" (the next day)");
        if (days < 0) say(" (the day before)");
    }
}
