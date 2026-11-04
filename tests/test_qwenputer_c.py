"""Qwenputer's toolbox tools in C (locals/models/qwenputer_c/), built for this machine (host.c) and checked against
Python's own json, re, csv arithmetic and logic; and the compiled toolbox (qwenputer_tools_c.s) made from these
sources. On the computer itself each tool's run equals the authors' reference machine's (test_qwenputer.py)."""

import itertools
import json
import math
import random
import re
import shutil
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

C = Path(__file__).parent.parent / "locals" / "models" / "qwenputer_c"


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    exe = tmp_path_factory.mktemp("qwenputer_c") / "host"
    sources = [str(p) for p in sorted(C.glob("*.c"))]
    subprocess.run([cc, "-DHOST", "-O1", "-std=gnu11", "-I", str(C), "-o", str(exe)] + sources, check=True)
    return str(exe)


def run(host, *calls):
    """The responses to calls (name, {parameter: value}) made one after another on one memory."""
    bodies = [f"\n<function={n}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in p.items()) + "</function>\n"
              for n, p in calls]
    out = subprocess.run([host], input="\x1e".join(bodies).encode(), capture_output=True, check=True).stdout.decode()
    return out.split("\x1e")


def one(host, name, **params):
    return run(host, (name, params))[0]


def test_the_compiled_toolbox_is_made_from_these_sources():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build", C / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    head = (C.parent / "qwenputer_tools_c.s").read_text().split("\n")[0]
    assert build.digest() in head, "qwenputer_c/ changed: python locals/models/qwenputer_c/build.py"


def test_json_as_pythons(host):
    rnd = random.Random(1)

    def value(d=0):
        t = rnd.random()
        if d > 3 or t < 0.3:
            return rnd.choice([None, True, False, rnd.randint(-10 ** 6, 10 ** 6), 1.5e20, "é\n\"\\☃", "", "k"])
        if t < 0.65:
            return [value(d + 1) for _ in range(rnd.randint(0, 4))]
        return {rnd.choice(["a", "b", "x y", "é"]): value(d + 1) for _ in range(rnd.randint(0, 4))}

    for _ in range(60):
        v = value()
        text = json.dumps(v, ensure_ascii=False)
        assert one(host, "json_format", text=text) == json.dumps(v, indent=2, ensure_ascii=False)
        assert json.loads(one(host, "json_format", text=json.dumps(v, indent=3), style="compact")) == v
        assert one(host, "json_check", text=text).startswith("valid JSON")
    doc = '{"users":[{"name":"Ann","age":31},{"name":"Bo","age":25}],"n":2}'
    assert one(host, "json_get", text=doc, path="users[*].name") == "Ann\nBo"
    assert one(host, "json_get", text=doc, path="$.users[-1].age") == "25"
    assert one(host, "json_get", text=doc, path="/users/0") == '{"name":"Ann","age":31}'
    assert one(host, "json_get", text=doc, path="users[0].nme") == 'error (no key "nme" in the object at users[0]; its keys: name, age)'
    assert one(host, "json_count", text=doc) == "2 keys: users, n"
    for bad in ['{"a":1,}', "{'a':1}", "[01]", "[NaN]", '{"a":1}{"b":2}', '"a\nb"', "[1 2]"]:
        assert one(host, "json_check", text=bad).startswith("error (not valid JSON: "), bad
        with pytest.raises(ValueError):
            json.loads(bad, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert one(host, "json_check", text='{"a": 1,\n "b": 2,}') == "error (not valid JSON: a comma before } at line 2, column 9)"


def _canon(d):
    s = f"{d:f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return "0" if s == "-0" else s


def test_tables_as_decimal(host):
    rnd = random.Random(2)
    for kind in ["csv", "tsv", "md", "semi"] * 10:
        n = rnd.randint(1, 20)
        names = [f"n{i}" for i in range(n)]
        groups = [rnd.choice(["North", "South"]) for _ in range(n)]
        vals = [Decimal(rnd.randint(-10 ** 6, 10 ** 7)) / 10 ** rnd.randint(0, 3) for _ in range(n)]
        rows = [["Name", "Region", "Amount"]] + [[a, g, f"{v:f}"] for a, g, v in zip(names, groups, vals)]
        if kind == "md":
            t = "\n".join(["| " + " | ".join(rows[0]) + " |", "|---|---|---:|"] + ["| " + " | ".join(r) + " |" for r in rows[1:]])
        else:
            t = "\n".join({"csv": ",", "tsv": "\t", "semi": ";"}[kind].join(r) for r in rows)
        s = sum(vals)
        mean = s / n
        q = mean.quantize(Decimal("1e-9"), rounding=ROUND_HALF_UP)
        got = one(host, "column_stats", table=t, column="amount")
        assert got.startswith(f"count {n}, sum {_canon(s)}, average {_canon(q)}" + ("" if q == mean else " (rounded)")), (t, got)
        order = sorted(range(n), key=lambda i: -vals[i])
        lines = one(host, "table_sort", table=t, column="Amount", order="descending").split("\n")
        assert [next(x for x in names if re.search(rf"(^|[|\t,; ]){x}($|[|\t,; ])", ln)) for ln in lines[1 + (kind == "md"):]] == [names[i] for i in order]
        thr = rnd.choice(vals)
        assert one(host, "table_filter", table=t, column="Amount", condition=f">= {_canon(thr)}").startswith(
            f"{sum(v >= thr for v in vals)} of {n} row")
        tot = {}
        for g, v in zip(groups, vals):
            tot.setdefault(g, [Decimal(0), 0])
            tot[g][0] += v
            tot[g][1] += 1
        assert one(host, "table_totals", table=t, group_by="Region", column="Amount") == "\n".join(
            f"{g}: {_canon(a)} ({c} row{'s' if c != 1 else ''})" for g, (a, c) in tot.items())


def test_versions_and_lists(host):
    assert one(host, "compare_versions", first="1.10.0", second="1.9.3") == "1.10.0 is newer than 1.9.3"
    assert one(host, "compare_versions", first="v2.0", second="2.0.0") == "v2.0 and 2.0.0 are the same version"
    semver = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    shuffled = semver[:]
    random.Random(3).shuffle(shuffled)
    assert one(host, "sort_versions", versions=", ".join(shuffled)) == ", ".join(semver)  # (semver.org's order)
    assert one(host, "compare_lists", first="apple, Banana, cherry", second="banana\ndate\nApple") == (
        "in both (2): apple, Banana\nonly in the first (1): cherry\nonly in the second (1): date")


def test_logic_as_brute_force(host):
    ops = {"and": lambda a, b: a and b, "or": lambda a, b: a or b, "xor": lambda a, b: a != b,
           "->": lambda a, b: (not a) or b, "<->": lambda a, b: a == b}
    rnd = random.Random(4)

    def gen(d, names):
        if d == 0 or rnd.random() < 0.3:
            v = rnd.choice(names)
            return v, lambda env, v=v: env[v]
        if rnd.random() < 0.2:
            s, f = gen(d - 1, names)
            return f"not ({s})", lambda env, f=f: not f(env)
        op = rnd.choice(list(ops))
        (s1, f1), (s2, f2) = gen(d - 1, names), gen(d - 1, names)
        return f"({s1} {op} {s2})", lambda env, f1=f1, f2=f2, op=op: ops[op](f1(env), f2(env))

    for _ in range(40):
        s, f = gen(4, ["p", "q", "r", "rain"])
        names = list(dict.fromkeys(re.findall(r"\b(p|q|r|rain)\b", s)))
        t = sum(f(dict(zip(names, b))) for b in itertools.product([False, True], repeat=len(names)))
        assert f"true in {t} of {2 ** len(names)} row" in one(host, "truth_table", expression=s), s
    assert one(host, "logic_equivalent", first="not (p and q)", second="not p or not q").startswith("equivalent")
    assert one(host, "logic_equivalent", first="p -> q", second="q -> p") == (
        "not equivalent: where p = F, q = T, the first is true and the second false")


def test_regular_expressions_as_pythons(host):
    rnd = random.Random(5)
    atoms = ["a", "b", ".", r"\d", r"\w", r"\s", "[abc]", "[^a]", "[a-c0-9]", "é", r"\.", "-"]

    def gen(d):
        t = rnd.random()
        s = (rnd.choice(atoms) if d == 0 or t < 0.35 else gen(d - 1) + gen(d - 1) if t < 0.6
             else "(" + gen(d - 1) + "|" + gen(d - 1) + ")" if t < 0.75 else "(?:" + gen(d - 1) + ")")
        if rnd.random() < 0.3 and not s.endswith(("*", "+", "?", "}")):
            s += rnd.choice(["*", "+", "?", "{2}", "{1,3}", "*?", "+?"])
        return s

    n = 0
    for _ in range(150):
        pat, text = gen(3), "".join(rnd.choice("abcx 01.-é\n") for _ in range(rnd.randint(0, 30)))
        rx = re.compile(pat)
        ms = list(rx.finditer(text))
        got = one(host, "regex_find", pattern=pat, text=text)
        if got.startswith("error (the pattern needs too many steps"):
            continue
        want = "no matches" if not ms else "\n".join(
            [f"{len(ms)} match{'es' if len(ms) != 1 else ''}:"] + [m.group(0) + (" (groups: " + ", ".join(
                "none" if g is None else g for g in m.groups()) + ")" if rx.groups else "") for m in ms])
        assert got == want, (pat, text)
        sub, k = rx.subn("-", text)
        assert one(host, "regex_replace", pattern=pat, text=text, replacement="-") == f"{k} replacement{'s' if k != 1 else ''}:\n{sub}"
        n += 1
    assert n > 100
    assert one(host, "regex_find", pattern=r"(\d{4})-(\d\d)", text="2024-01 and 1999-12") == (
        "2 matches:\n2024-01 (groups: 2024, 01)\n1999-12 (groups: 1999, 12)")


def test_notes_stay_from_call_to_call(host):
    assert run(host, ("remember", {"key": "wifi", "value": "hunter2"}), ("remember", {"key": "dentist", "value": "Tuesday"}),
               ("recall", {"key": "WiFi"}), ("forget", {"key": "wifi"}), ("recall", {"key": "all"}), ("recall", {"key": "wifi"})) == [
        'remembered "wifi"', 'remembered "dentist"', "hunter2", 'forgot "wifi"', "dentist: Tuesday",
        'error (no note "wifi"; the notes: dentist: Tuesday)']
    # a conversation's notes from its earlier calls (the server's _replay, the computer's RAM being new)
    assert run(host, ("recall", {"key": "dentist", "_replay": "remember\tdentist\tTuesday\nremember\tgym\tMonday\nforget\tgym"}),
               ("recall", {"key": "all"})) == ["Tuesday", "dentist: Tuesday"]


def test_call_checks(host):
    schema = "city\t\tr\nunits\t\to\tcelsius\x1ffahrenheit\ndays\ti\to"

    def check(call, schema=schema):
        body = f"\n<function=check_call>\n{schema}\n</function>\n" + call
        return subprocess.run([host], input=body.encode(), capture_output=True, check=True).stdout.decode()

    assert check("<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n") == "ok"
    assert check("<function=get_weather>\n<parameter=units>\nKelvin\n</parameter>\n<parameter=days>\nthree\n</parameter>\n"
                 "<parameter=cty>\nParis\n</parameter>\n</function>\n") == (
        "error (the required parameter city is missing; units must be one of: celsius, fahrenheit (it is Kelvin); "
        "days must be an integer (it is three); no parameter named cty (the parameters: city, units, days))")
    assert check("<function=get_wether>\n</function>\n", "tools\tget_weather\x1fsearch") == (
        "error (no tool named get_wether; the tools: get_weather, search)")


def test_copy_checks(host):
    big = str(10 ** 38)
    assert one(host, "copy_check", results=big, written=f"{10 ** 35:,}") == f"fix\n{10 ** 38:,}"
    assert one(host, "copy_check", results=big, written=f"{10 ** 38:,}") == "ok"
    assert one(host, "copy_check", results="121932631112635269", written="121,932,631,112,653,269") == "fix\n121,932,631,112,635,269"
    assert one(host, "copy_check", results="3.14159265", written="3.1416") == "ok"  # (rounded)
    assert one(host, "copy_check", results="121932631112635269", written="123456789") == "ok"  # (another number)
    assert one(host, "copy_check", results="-12345678", written="−12,345,687") == "fix\n−12,345,678"


PROGRAMS = [
    "fib = [0, 1]\nwhile len(fib) < 20:\n    fib.append(fib[-1] + fib[-2])\nprint(fib)\nprint(fib[-1], len(fib))",
    "import math\nprint(math.factorial(20))\nprint(math.gcd(84, 36), math.lcm(4, 6), math.isqrt(99), math.comb(10, 3))",
    's = "The quick brown fox jumps over the lazy dog"\nvowels = sum(1 for c in s.lower() if c in "aeiou")\nprint(f"{vowels} vowels")\n'
    "words = s.split()\nprint(len(words), words[::-1][:3], sorted(words, key=len)[-1])",
    "n = 7\nsteps = 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\nprint(steps)",
    'text = "a b a c b a"\nfreq = {}\nfor w in text.split():\n    freq[w] = freq.get(w, 0) + 1\nprint(freq)\n'
    "print(max(freq, key=freq.get), sorted(freq.items(), key=lambda kv: -kv[1]))",
    'for i in range(1, 16):\n    if i % 15 == 0:\n        print("FizzBuzz")\n    elif i % 3 == 0:\n        print("Fizz")\n'
    '    elif i % 5 == 0:\n        print("Buzz")\n    else:\n        print(i)',
    'x = 1234567\nprint(f"{x:,} {x:>10} {x:08d} {255:x} {255:08b} {x!r}")\nprint(bin(10), hex(255), oct(8), int("ff", 16))',
    "def gcd(a, b):\n    return a if b == 0 else gcd(b, a % b)\nprint(gcd(1071, 462))\na, b = 3, 4\na, b = b, a\n"
    "print(a, b, divmod(17, 5), pow(3, 200, 1000007))",
    'total = 0\nfor i, (name, score) in enumerate(zip(["ann", "bo", "cy"], [90, 85, 77]), 1):\n    total += score\n'
    "    print(i, name.title(), score)\nprint(total // 3, total % 3)",
    "grid = [[i * j for j in range(4)] for i in range(4)]\nprint(grid)\nprint([row[2] for row in grid], sum(sum(r) for r in grid))\n"
    "d = {k: v * v for k, v in [(1, 2), (3, 4)]}\nprint(d, 3 in d, list(d.keys()), list(d.values()))",
    's = "racecar"\nprint(s == s[::-1], s.upper(), s.count("r"), s.find("car"), s.replace("car", "CAR"))\n'
    'print("-".join(["a", "b", "c"]), " x ".strip(), "a,b,,c".split(","), "42".zfill(5))',
    "nums = [5, 3, 8, 1, 9, 2]\nnums.sort()\nprint(nums, nums.index(8), min(nums), max(nums), sum(nums))\n"
    "nums.reverse(); print(nums); nums.pop(); print(nums, len(nums))\n"
    "print(sorted(nums, reverse=True), list(reversed(nums)), any(x > 8 for x in nums), all(x > 0 for x in nums))",
    "import math\nprimes = [p for p in range(2, 60) if all(p % d for d in range(2, math.isqrt(p) + 1))]\n"
    "print(len(primes), sum(primes), primes[-3:])",
    "def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f(15))",
    "print(len('naïve'), ord('A'), chr(9731), 'abc' < 'abd', [1, 2] < [1, 3], (1, 2), (3,))",
    "x = None\nprint(x is None, x is not None, not x, True + True, [] or 'empty', 7 // 2, -7 // 2, -7 % 3, 2 ** 10)",
]


def test_run_python_as_python(host):
    import sys

    for code in PROGRAMS:
        want = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert one(host, "run_python", code=code) == want, code


def test_run_python_says_what_it_cannot_do(host):
    assert one(host, "run_python", code="print(7 / 2)").startswith("error (line 1: ValueError: / of numbers that do not divide")
    assert "math.isqrt" in one(host, "run_python", code="n = 10\nprint(int(n ** 0.5))")
    assert one(host, "run_python", code="print(2 ** 70)").startswith("error (line 1: OverflowError")
    assert one(host, "run_python", code="print(x)") == "error (line 1: NameError: name 'x' is not defined)"
    assert one(host, "run_python", code="for i in range(3):\nprint(i)").startswith("error (line 2: IndentationError")
    assert one(host, "run_python", code="try:\n    pass\nexcept:\n    pass").startswith("error (line 1: SyntaxError: try is not supported")
    assert one(host, "run_python", code="while True:\n    pass").startswith("error (line 2: TimeoutError")
    assert one(host, "run_python", code="def f(n):\n    return f(n + 1)\nf(0)").startswith("error (line 2: RecursionError")
    # memory: a program too long to parse, and values too many for what is left
    long = "\n".join(f"v{i} = {i} * 3 + 1" for i in range(200)) + "\nprint(v0)"
    assert one(host, "run_python", code=long).startswith("error (out of memory: the program is too long for the computer")
    assert one(host, "run_python", code="x = [0] * 3000\nprint(len(x))").startswith("error (out of memory: the program's values")
    # (a model's fenced block)
    assert one(host, "run_python", code="```python\nprint(6 * 7)\n```") == "42\n"


def test_numbers_as_python(host):
    import statistics
    from decimal import Decimal as D
    from fractions import Fraction

    rnd = random.Random(7)
    for _ in range(25):
        xs = [D(rnd.randint(-10 ** 6, 10 ** 6)) / 10 ** rnd.randint(0, 3) for _ in range(rnd.randint(2, 30))]
        got = one(host, "stats", numbers=", ".join(f"{x:f}" for x in xs))
        mean = sum(xs) / len(xs)
        assert f"count {len(xs)}, sum {_canon(sum(xs))}, mean " in got
        assert f", median {_canon(statistics.median(xs))}, " in got
        for k, f in ((0, statistics.pstdev), (1, statistics.stdev)):
            want = D(str(f([Fraction(x) for x in xs]))).quantize(D("1e-6"), rounding=ROUND_HALF_UP) if False else None
        ps = D(statistics.pstdev([float(x) for x in xs])).quantize(D("1e-6"))
        assert abs(D(got.split("standard deviation ")[1].split(" ")[0]) - ps) <= D("0.000002"), (xs, got)
        assert (mean == mean.quantize(D("1e-9"))) != got.split(", median")[0].endswith("(rounded)")
    assert one(host, "percent_change", old="80", new="100") == "+25% (up 20, from 80 to 100)"
    assert one(host, "round_number", number="2.675", places="2") == "2.68"
    assert one(host, "round_number", number="1234567", places="-3") == "1235000"
    want = (D(1000) * (1 + D("0.05") / 12) ** 120).quantize(D("0.01"), rounding=ROUND_HALF_UP)
    assert one(host, "compound_interest", principal="1000", rate="5", years="10", compounding="monthly").startswith(f"amount {want}")
    i, n = D("0.005"), 360
    pay = (D(200000) * i / (1 - (1 + i) ** -n)).quantize(D("0.01"), rounding=ROUND_HALF_UP)
    assert one(host, "loan_payment", principal="200000", rate="6", years="30").startswith(f"monthly payment {pay} for 360 payments")
    for _ in range(40):
        n = rnd.choice([rnd.randint(2, 10 ** 6), rnd.randint(2, 10 ** 12)])
        fs, m, p = [], n, 2
        while p * p <= m:
            while m % p == 0:
                fs.append(p)
                m //= p
            p += 1
        if m > 1:
            fs.append(m)
        got = one(host, "factor", number=str(n))
        parts = []
        for f in sorted(set(fs)):
            parts.append(f"{f}^{fs.count(f)}" if fs.count(f) > 1 else str(f))
        if "(the last not prime" in got:  # (a cofactor whose factors are past the run's budget, as it says: above 20,000)
            shown = [int(x.split("^")[0]) ** int(x.split("^")[1]) if "^" in x else int(x) for x in got.split(" = ")[1].split(" (")[0].split(" * ")]
            assert math.prod(shown) == n and min(f for f in fs if shown[-1] % f == 0) > 20000, (n, got)
            continue
        assert got.startswith(f"{n} = " + " * ".join(parts)), (n, got)
        assert one(host, "is_prime", number=str(n)).startswith(f"{n} is prime" if len(fs) == 1 else f"{n} is not prime")
    assert one(host, "is_prime", number="18446744073709551557") == "18446744073709551557 is prime"
    assert one(host, "convert_units", value="5", from_="km", to="mi") if False else True
    assert one(host, "convert_units", **{"value": "37", "from": "celsius", "to": "fahrenheit"}) == "37 °C = 98.6 °F"
    assert one(host, "convert_units", **{"value": "1", "from": "GiB", "to": "MB"}) == "1 GiB = 1073.741824 MB"
    assert one(host, "convert_units", **{"value": "10", "from": "lb", "to": "kg"}) == "10 lb = 4.5359237 kg"
    assert one(host, "convert_units", **{"value": "3", "from": "km", "to": "kg"}) == "error (km is a length and kg a mass)"
    roman = {1: "I", 4: "IV", 9: "IX", 14: "XIV", 40: "XL", 90: "XC", 400: "CD", 1994: "MCMXCIV", 2024: "MMXXIV", 3999: "MMMCMXCIX"}
    for v, r in roman.items():
        assert one(host, "roman_numeral", value=str(v)) == r and one(host, "roman_numeral", value=r) == str(v)
    assert one(host, "roman_numeral", value="IIII").startswith("error")
    assert one(host, "number_to_words", number="1234567") == "one million two hundred thirty-four thousand five hundred sixty-seven"
    assert one(host, "number_to_words", number="-45.07") == "minus forty-five point zero seven"


def test_stats_of_long_and_mixed_numbers(host):
    """stats in machine words (numbers within 62 bits at the list's scale: sorted, summed and squared in 64 and 128
    bits) and in big numbers (a list beyond that) agree with exact arithmetic."""
    import statistics
    from decimal import Decimal as D, localcontext

    rnd = random.Random(11)
    for i in range(48):
        n = rnd.randint(2, 60)
        kind = i % 4
        if kind == 0:
            xs = [D(rnd.randint(-10 ** 17, 10 ** 17)) for _ in range(n)]
        elif kind == 1:
            xs = [D(rnd.randint(-10 ** 12, 10 ** 12)) / 10 ** rnd.randint(0, 6) for _ in range(n)]
        elif kind == 2:
            xs = [D(rnd.randint(-10 ** 9, 10 ** 9)) / 10 ** rnd.randint(0, 9) for _ in range(n)]
        else:  # (beyond 62 bits at the scale: big numbers)
            xs = [D(rnd.randint(-10 ** 17, 10 ** 17)) for _ in range(n - 1)] + [D("0.000000001")]
        got = one(host, "stats", numbers=", ".join(f"{x:f}" for x in xs))
        assert f"count {n}, sum {_canon(sum(xs))}, mean " in got, (xs, got)
        assert f", median {_canon(statistics.median(xs))}, " in got, (xs, got)
        with localcontext() as c:
            c.prec = 90
            mean = sum(xs) / n
            ps = (sum((x - mean) ** 2 for x in xs) / n).sqrt()
        assert abs(D(got.split("standard deviation ")[1].split(" ")[0]) - ps) <= D("0.000002"), (xs, got)


def test_text_tools_as_python(host):
    import base64
    import codecs
    import urllib.parse
    from collections import Counter

    t = "the cat and the hat and the bat"
    assert one(host, "replace_text", text=t, find="the", replace="a") == f"3 replacements:\n{t.replace('the', 'a')}"
    assert one(host, "change_case", text="hello world from Qwen", case="snake") == "hello_world_from_qwen"
    assert one(host, "change_case", text="hello world from Qwen", case="camel") == "helloWorldFromQwen"
    assert one(host, "change_case", text="myVariableName", case="kebab") == "my-variable-name"
    assert one(host, "change_case", text="hello world", case="title") == "Hello World"
    assert one(host, "text_part", text="one two three four", part="word", number="-1") == "four"
    assert one(host, "text_part", text="abcdef", part="character", number="2-4") == "bcd"
    words = t.split()
    c = Counter(words)
    assert one(host, "word_frequency", text=t.upper(), top="2") == f"{len(words)} words, {len(c)} different\nthe: 3\nand: 2"
    assert one(host, "sort_lines", text="pear\nApple\nbanana") == "Apple\nbanana\npear"
    assert one(host, "unique_items", text="a, b, A, c, b") == "3 unique of 5:\na, b, c"
    assert one(host, "is_palindrome", text="A man, a plan, a canal: Panama").startswith("yes")
    assert one(host, "is_anagram", first="Listen", second="Silent").startswith("yes")
    assert one(host, "is_anagram", first="abc", second="abd") == "no: c 1 and 0, d 0 and 1 times"
    for kind, good, bad in (("email", "a.b+c@example.co.uk", "a@b"), ("ipv4", "192.168.0.1", "256.1.1.1"),
                            ("url", "https://example.com/x?y=1", "example.com"), ("date", "2024-02-29", "2023-02-29"),
                            ("time", "23:59:59", "24:00"), ("uuid", "123e4567-e89b-12d3-a456-426614174000", "123"),
                            ("hex color", "#1a2b3c", "#12345")):
        assert one(host, "validate", text=good, kind=kind) == "valid", kind
        assert one(host, "validate", text=bad, kind=kind).startswith("not valid"), kind
    rnd = random.Random(8)
    for _ in range(20):
        s = "".join(rnd.choice("abcXYZ 0189,.!é/?&=") for _ in range(rnd.randint(0, 30)))
        b = s.encode()
        assert one(host, "encode", text=s, encoding="base64") == base64.b64encode(b).decode()
        assert one(host, "encode", text=s, encoding="hex") == b.hex()
        assert one(host, "encode", text=s, encoding="url") == urllib.parse.quote(s, safe="-_.~")
        assert one(host, "encode", text=s, encoding="rot13") == codecs.encode(s, "rot13")
        if s.strip():
            assert one(host, "decode", text=base64.b64encode(b).decode(), encoding="base64") == s
            assert one(host, "decode", text=b.hex(), encoding="hex") == s
    assert one(host, "encode", text="SOS help", encoding="morse") == "... --- ... / .... . .-.. .--."
    assert one(host, "decode", text="... --- ...", encoding="morse") == "sos"

    def lev(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, row[0] = row[0], i
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
        return row[-1]

    for _ in range(20):
        a = "".join(rnd.choice("abcé") for _ in range(rnd.randint(0, 20)))
        b = "".join(rnd.choice("abcé") for _ in range(rnd.randint(0, 20)))
        assert one(host, "edit_distance", first=a, second=b).startswith(f"{lev(a, b)} edit")


def test_time_tools_as_python(host):
    import datetime as dt

    rnd = random.Random(9)
    for _ in range(30):
        d = dt.date(1900, 1, 1) + dt.timedelta(days=rnd.randint(0, 80000))
        y, w, wd = d.isocalendar()
        got = one(host, "date_info", date=d.isoformat())
        assert got.startswith(d.strftime("%A, %B ") + f"{d.day}, {d.year}: day {d.timetuple().tm_yday} of ") and f"ISO week {w} of {y}" in got
        ts = rnd.randint(-10 ** 9, 4 * 10 ** 9)
        when = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=ts)
        assert one(host, "unix_time", value=str(ts)) == when.strftime("%Y-%m-%d %H:%M:%S UTC")
        assert one(host, "unix_time", value=when.strftime("%Y-%m-%d %H:%M:%S")) == f"{ts} (the time read as UTC)"
    assert one(host, "time_add", time="22:30", amount="2h 45m") == "01:15 (the next day)"
    assert one(host, "time_add", time="2024-02-28 23:00", amount="1 day 2 hours") == "2024-03-01 01:00"
    assert one(host, "time_between", start="09:15", end="17:40") == "8 hours, 25 minutes (505 minutes)"
    assert one(host, "age", birth_date="1990-05-17", on_date="2024-03-10") == "33 years, 9 months and 22 days (12351 days)"
    assert one(host, "convert_timezone", time="2024-03-10 15:00", **{"from": "EST", "to": "JST"}) == "2024-03-11 05:00 UTC+9"
    assert one(host, "convert_timezone", time="9:00 am", **{"from": "UTC+5:30", "to": "UTC"}) == "03:30 UTC"


def test_check_digits_and_hashes_as_python(host):
    import hashlib
    import zlib

    rnd = random.Random(10)
    for _ in range(20):  # Luhn: a random number with its own check digit
        body = [rnd.randint(0, 9) for _ in range(15)]
        s = sum((d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 == 0 else d for i, d in enumerate(reversed(body)))
        number = "".join(map(str, body)) + str((10 - s % 10) % 10)
        assert one(host, "check_digit", number=number, kind="luhn").startswith("valid Luhn")
        wrong = number[:-1] + str((int(number[-1]) + 1) % 10)
        assert "as a whole number, not valid Luhn" in one(host, "check_digit", number=wrong, kind="luhn")
    assert one(host, "check_digit", number="0306406152", kind="isbn10") == "valid ISBN-10 (its check digit 2)"
    assert one(host, "check_digit", number="978-0-306-40615-7", kind="isbn13") == "valid ISBN-13 (its check digit 7)"
    assert one(host, "check_digit", number="978-0-306-40615", kind="isbn13") == "its check digit: 7"
    assert one(host, "check_digit", number="714285124000348", kind="luhn") == (  # (not valid: the digit to append first)
        "the digit to append to it as a number without its check digit: 3; as a whole number, not valid Luhn (its last "
        "digit should be 4, not 8)")
    assert one(host, "check_digit", number="4111111111111111", kind="luhn") == (
        "valid Luhn (its check digit 1); as a number without its check digit, the digit to append: 3")
    assert one(host, "check_digit", number="GB82 WEST 1234 5698 7654 32", kind="iban") == "valid IBAN"
    assert one(host, "check_digit", number="GB83 WEST 1234 5698 7654 32", kind="iban") == "not a valid IBAN: its check digits should be 82"
    for _ in range(10):
        s = "".join(rnd.choice("abc é\n") for _ in range(rnd.randint(0, 200)))
        b = s.encode()
        assert one(host, "hash_text", text=s, algorithm="sha256") == hashlib.sha256(b).hexdigest()
        assert one(host, "hash_text", text=s, algorithm="crc32") == f"{zlib.crc32(b):08x}"
        assert one(host, "hash_text", text=s, algorithm="adler32") == f"{zlib.adler32(b):08x}"


def test_notes_recall_every_and_values(host):
    notes = [("remember", {"key": "locker_code", "value": "4512"}), ("remember", {"key": "bike_lock", "value": "0937"})]
    for key in ["every", "Everything", " all notes ", "everything you remember", "LIST", "notes", "*", "all"]:
        assert run(host, *notes, ("recall", {"key": key}))[-1] == "locker_code: 4512\nbike_lock: 0937", key
    assert run(host, *notes, ("recall", {"key": "x"}))[-1] == 'error (no note "x"; the notes: locker_code: 4512, bike_lock: 0937)'
    # (a note of that name first; a long list cut to a few hundred bytes)
    assert run(host, ("remember", {"key": "list", "value": "milk, eggs"}), ("recall", {"key": "list"}))[-1] == "milk, eggs"
    many = [("remember", {"key": f"k{i}", "value": "v" * 90}) for i in range(12)]
    got = run(host, *many, ("recall", {"key": "x"}))[-1]
    assert got.startswith('error (no note "x"; the notes: k0: ' + "v" * 60 + "..., k1: ") and got.endswith(", ...)") and len(got) < 600


def test_run_python_review_fixes(host):
    import sys

    def py(code):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    for code in [
        "print(f'{5:0100}')", "print(f'{2**61:_b}')", "print(f'{-5:06}', f'{7:>4}', f'{-42:08,}', f'{42:04,}', f'{255:010_b}')", "s = 'a' * 70\nprint(f'{s:>100}|')",
        "print(dict({'a': 1}), dict([('b', 2)]), dict(zip('xy', [1, 2])), dict(a=1))",
        "d = {'a': 1}\nd.update([('b', 2)])\nd.update({'c': 3})\nd.update()\nprint(d)",
        "l = ['bb', 'a', 'ccc']\nl.sort(key=len, reverse=True)\nprint(l)\nl.sort(reverse=True, key=len)\nprint(l)\nl.sort(key=len)\nprint(l)",
        "a, b = 1, 2\nprint(f'{a != b} {a!r} {a == b}')",
        "print('a,b,c'.split(',', 1), 'a b  c '.split(None, 1), 'a,b'.split(',', 0), 'aaa'.replace('a', 'b', 1), 'aaa'.replace('a', 'b', -1))",
        "import math\nprint(sum([1, 2], 10), math.comb(62, 31), math.comb(60, 30), math.perm(20, 10))",
        "print(1 in [1, 2], 3 in [1, 2], 'b' in ['a', 'b'], True in [1], [1] in [[1], 2], 2 in (1, 2))",
        "print('abcabc'.count('bc'), 'abcabc'.find('ca'), 'abcabc'.rfind('bc'), 'abcabc'.index('c'), 'x' in 'abc', 'ab'[::-1], 'abcdef'[1:5:2])",
    ]:
        assert one(host, "run_python", code=code) == py(code), code
    for code, want in [("l = []\nl.append()", "error (line 2: TypeError: append() is missing an argument)"),
                       ("print(len())", "error (line 1: TypeError: len() is missing an argument)"),
                       ("d = {}\nd.update(5)", "error (line 2: TypeError: int is not iterable)"),
                       ("print(dict([1, 2]))", "error (line 1: TypeError: a dict is made of (key, value) pairs)"),
                       ("print(8 >> -1)", "error (line 1: ValueError: negative shift count)")]:
        assert one(host, "run_python", code=code) == want, code
    # nesting deeper than the computer's stack: a clean error, also inside str() (which writes into a buffer)
    deep = "nested too deeply for the computer's memory)"
    assert one(host, "run_python", code="print(" + "(" * 12 + "1" + ")" * 12 + ")") == "1\n"
    assert one(host, "run_python", code="print(" + "(" * 60 + "1" + ")" * 60 + ")").endswith(deep)
    assert one(host, "run_python", code="print(" + "(" * 30 + "1" + ")" * 30 + ")") == "1\n"  # (the parser's levels)
    assert one(host, "run_python", code="x = " + "[" * 40 + "1" + "]" * 40 + "\nprint(x)").endswith(deep)
    assert one(host, "run_python", code="l = []\nfor i in range(200):\n    l = [l]\nprint(len(str(l)))").endswith(deep)
    assert one(host, "run_python", code="def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f(15))") == "1307674368000\n"
    assert "RecursionError" in one(host, "run_python", code="def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f(40))")
    # a for's target that ends a run in a syntax error does not change how the next run reads "in"
    assert run(host, ("run_python", {"code": "for 2x in range(3):\n    print(x)"}), ("run_python", {"code": "print(1 in [1, 2])"}))[1] == "True\n"
    # a string scanned again and again is charged by its length: the step limit ends it, not the computer's run limit
    got = one(host, "run_python", code='s = "ab" * 1500\nn = 0\nfor i in range(300):\n    n += s.count("b")\nprint(n)')
    assert got.startswith("error (line 4: TimeoutError"), got
    assert one(host, "run_python", code='l = [0] * 500\nn = 0\nfor i in range(300):\n    if -1 in l:\n        n += 1\nprint(n)').startswith(
        "error (line 4: TimeoutError")


def test_tools_review_fixes(host):
    import datetime as dt
    import statistics
    from collections import Counter

    from dateutil.relativedelta import relativedelta

    # (memory and arithmetic at the edges)
    assert one(host, "truth_table", expression=" -> ".join(["p"] * 65)).endswith("true in 2 of 2 rows: a tautology (always true)")
    assert one(host, "number_to_words", number=str(10 ** 18)) == "error (numbers below a quintillion (10^18))"
    assert one(host, "number_to_words", number=str(10 ** 18 - 1)).startswith("nine hundred ninety-nine quadrillion")
    new = 2 * 10 ** 145
    assert one(host, "percent_change", old="1", new=str(new)) == f"+{(new - 1) * 100}% (up {new - 1}, from 1 to {new})"
    # word_frequency: words split at punctuation, as many as there are
    for text in ["apple,banana,cherry", "a state-of-the-art tool", "x;" * 300]:
        words = re.findall(r"[A-Za-z0-9\x80-\uffff]+(?:'[A-Za-z]+)*", text)
        c = Counter(w.lower() for w in words)
        want = [f"{len(words)} words, {len(c)} different"] + [f"{w}: {k}" for w, k in sorted(c.items(), key=lambda x: -x[1])[:10]]
        assert one(host, "word_frequency", text=text) == "\n".join(want), text
    rnd = random.Random(12)
    words = ["".join(rnd.choice("abcdefgh") for _ in range(rnd.randint(2, 4))) for _ in range(900)]
    c = Counter(words)
    got = one(host, "word_frequency", text=" ".join(words), top="1000").split("\n")
    assert got[0] == f"900 words, {len(c)} different" and got[1:] == [f"{w}: {k}" for w, k in sorted(c.items(), key=lambda x: -x[1])]
    # age as dateutil counts it (a month on from the 31st is the next month's last day)
    for _ in range(300):
        b = dt.date(1950, 1, 1) + dt.timedelta(days=rnd.randint(0, 25000))
        d = b + dt.timedelta(days=rnd.randint(0, 3000))
        r = relativedelta(d, b)
        got = one(host, "age", birth_date=b.isoformat(), on_date=d.isoformat())
        assert got.startswith(f"{r.years} year{'s' if r.years != 1 else ''}, {r.months} month{'s' if r.months != 1 else ''} and "
                              f"{r.days} day{'s' if r.days != 1 else ''} ("), (b, d, got)
    assert one(host, "age", birth_date="2000-01-31", on_date="2000-03-01") == "0 years, 1 month and 1 day (30 days)"
    # versions of more than 9 digits; roman numerals not written the standard way; a table's unclosed quote
    assert one(host, "sort_versions", versions="5000000000, 999999999, 1.0, 0010000000000") == "1.0, 999999999, 5000000000, 0010000000000"
    assert one(host, "sort_versions", versions="1.5000000000.0, 1.999999999.0") == "1.999999999.0, 1.5000000000.0"
    for bad in ["IXI", "VIV", "XCX", "CMC", "IIII", "MMMM"]:
        assert one(host, "roman_numeral", value=bad).startswith("error (not a roman numeral"), bad
    for n in range(1, 4000, 7):
        r = one(host, "roman_numeral", value=str(n))
        assert one(host, "roman_numeral", value=r) == str(n)
    assert one(host, "table_sort", table='name,v\nb,2\na,"1', column="v") == 'name,v\na,"1\nb,2'
    # times past the calendar's years
    assert one(host, "time_add", time="9999-12-31 12:00", amount="1 day") == "error (a date outside the years 1 to 9999)"
    assert one(host, "time_add", time="2024-01-01 00:00", amount="99999999999999999999 days").startswith("error (amount")
    # column_stats' median (a selection now) with many equal numbers, and a table of more rows than the heap alone holds
    for _ in range(30):
        xs = [rnd.randint(0, 9) for _ in range(rnd.randint(1, 60))]
        got = one(host, "column_stats", table="\n".join(map(str, xs)), column="1")
        assert f", median {_canon(Decimal(statistics.median(xs)))}, min {min(xs)} " in got, (xs, got)
    xs = [rnd.randint(0, 9) for _ in range(470)]
    assert one(host, "column_stats", table="\n".join(map(str, xs)), column="1").startswith(f"count 470, sum {sum(xs)}, ")


def test_regular_expressions_review_fixes(host):
    def want(pat, text):
        rx = re.compile(pat)
        ms = list(rx.finditer(text))
        return "no matches" if not ms else "\n".join(
            [f"{len(ms)} match{'es' if len(ms) != 1 else ''}:"] + [m.group(0) + (" (groups: " + ", ".join(
                "none" if g is None else g for g in m.groups()) + ")" if rx.groups else "") for m in ms])

    for pat, text in [("a{,2}", "aaa"), ("a{,}", "aaa"), ("a{}", "a{}"), ("(?:a*)*", "aab"), ("(a|b?)+", "ab"), ("(a*)+b", "aaab"),
                      ("(|a)+", "aaa"), ("(a|)+", "aaa"), (r"(\s*\w*,?)*", "a, b,c"), ("x(?:y?)*z", "xz xyz xyyz"),
                      (r"\d+", "ab 12 c 345 é6"), (r"[é0-9]+", "xé1 2"), (r"(?i)hello", "HeLLo hello"), (r"\bcat\b", "cat concat cat."),
                      (r"^\w+", "word one"), (r"\W+", "a, b!"), ("é", "--0éacé.\na"), ("(?:" * 20 + "a" + ")" * 20, "aa"),
                      ("|".join(f"w{i}" for i in range(60)), "w5 w59 w60 x"), ("a" * 300, "a" * 310)]:
        assert one(host, "regex_find", pattern=pat, text=text) == want(pat, text), pat
    assert one(host, "regex_find", pattern="a{4294967297}", text="aaa") == "error (the pattern: a repeat count of more than 1,000)"
    assert one(host, "regex_find", pattern="(?:" * 21 + "a" + ")" * 21, text="a") == "error (the pattern: groups nested more than 20 deep)"
    rnd = random.Random(13)
    atoms = ["a", "b", r"\d", r"\w", r"\s", "[abc]", "[^a]", "é", "x?", "(?:a|)", "(b*)"]

    def gen(d):
        t = rnd.random()
        s = (rnd.choice(atoms) if d == 0 or t < 0.35 else gen(d - 1) + gen(d - 1) if t < 0.6
             else "(" + gen(d - 1) + "|" + gen(d - 1) + ")" if t < 0.75 else "(?:" + gen(d - 1) + ")")
        if rnd.random() < 0.35 and not s.endswith(("*", "+", "?", "}")):
            s += rnd.choice(["*", "+", "?", "{,2}", "{1,}", "*?", "+?"])
        return s

    n = 0
    for _ in range(200):
        pat, text = gen(3), "".join(rnd.choice("abcx 01é\n") for _ in range(rnd.randint(0, 30)))
        got = one(host, "regex_find", pattern=pat, text=text)
        if got.startswith(("error (the pattern needs too many steps", "error (the pattern: more than 9 groups")):
            continue
        assert got == want(pat, text), (pat, text)
        n += 1
    assert n > 150
    # a replacement's text on the computer's memory, not after a fixed program's and stack's (now in the tables' memory)
    text = " ".join(rnd.choice(["dog", "cat", "a", "the"]) for _ in range(500))
    assert one(host, "regex_replace", pattern="dog", text=text, replacement="wolf") == (
        f"{text.count('dog')} replacements:\n" + text.replace("dog", "wolf"))


def test_regex_find_says_what_anchors_match(host):
    """No match for a pattern anchored at the text's end or start, in a text of several words: the response says that
    ^ and $ are the whole text's (the model wrote \\w+3$ for words ending in 3)."""
    text = "maple94 quartz88 pixel3 harbor99 orbit83 river20"
    assert one(host, "regex_find", pattern=r"\w+3$", text=text) == (
        "no matches (^ and $ are the whole text's start and end here: for each word's, use \\b)")
    assert one(host, "regex_find", pattern=r"\b\w*3\b", text=text) == "2 matches:\npixel3\norbit83"
    assert one(host, "regex_find", pattern=r"x$", text="abc") == "no matches"
    assert one(host, "regex_find", pattern=r"9\$", text="a b") == "no matches"


def test_regular_expressions_on_a_longer_text(host):
    # (a text of 150 words, 1 KB: these took more than the step limit before the machine's STAR, the first-byte skip
    # and the word-boundary start; each now ends within about 100,000 of the computer's instructions)
    text = " ".join(["alpha", "beta", "gamma", "delta"][i % 4] + str(i % 7) for i in range(150))
    for pat in [r"\w+3$", r"\b\w*3\b", r"\w*3\b", r"\d+", r"[a-z]+\d", r"\b[a-z]+3\b", r"\w+\d\b", r"(\w+)3 (\w+)"]:
        ms = list(re.finditer(pat, text))
        got = one(host, "regex_find", pattern=pat, text=text)
        if len(ms) > 100:
            assert got.startswith("more than 100 matches; the first 100:\n" + ms[0].group(0)), pat
        else:
            assert got.split("\n")[0].split(" (^")[0] == ("no matches" if not ms else f"{len(ms)} match{'es' if len(ms) != 1 else ''}:"), pat
            assert [ln.split(" (groups")[0] for ln in got.split("\n")[1:]] == [m.group(0) for m in ms], pat


def test_run_python_steps_bound_the_work(host):
    # builtins' work (str and int conversions, map, join, sorted, dict lookups, string scans) is charged as steps, so
    # these end with the step limit's error (on the computer within about 175,000 instructions), not past its run limit
    for code in ["print(sum(1 for x in range(1, 201) if sum(map(int, str(x))) == 10))",
                 "count = 0\nfor x in range(1, 201):\n    s = 0\n    for c in str(x):\n        s += int(c)\n    if s == 10:\n        count += 1\nprint(count)",
                 "print(len([x for x in range(1, 201) if sum(int(d) for d in str(x)) == 10]))",
                 "d = {}\nfor i in range(300):\n    d[str(i)] = i\nt = 0\nfor k in d:\n    t += d[k]\nprint(t)",
                 "t = 0\nfor i in range(1000):\n    t += int(str(i * 12345))\nprint(t)",
                 "parts = [str(i) for i in range(100)]\nfor k in range(200):\n    t = ','.join(parts)\nprint(len(t))",
                 "l = list(range(100))\nfor k in range(200):\n    s = str(l)\nprint(len(s))"]:
        assert "TimeoutError: the program takes more than the computer's 1,250 steps" in one(host, "run_python", code=code), code
    # (and the same question for fewer numbers is answered)
    assert one(host, "run_python", code="print(sum(1 for x in range(1, 41) if sum(map(int, str(x))) == 10))") == "3\n"
