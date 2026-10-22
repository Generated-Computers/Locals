"""The README's star history: the repository's stars from its first day to now, one light and one dark SVG.

Run: python make.py OUT_DIR [--repo OWNER/NAME] [--data stars.json]
Reads every stargazer's date from GitHub's API (GITHUB_TOKEN or GH_TOKEN; listing stargazers needs one), or
from --data ({"created": ISO date, "stars": [ISO dates]}). Drawn as a GitHub box in GitHub's colors (Primer), its
text in the font GitHub's pages use on the reader's system.
"""
import argparse
import bisect
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

FONTS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
STAR = ("M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1"
        "-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611"
        "L7.327.668A.75.75 0 0 1 8 .25Z")  # (Octicons' star-fill-16, MIT)

THEMES = {  # Primer's light and dark
    "light": dict(text="#1f2328", muted="#59636e", border="#d1d9e0", grid="#d1d9e0b3", canvas="#ffffff",
                  subtle="#f6f8fa", line="#218bff"),
    "dark": dict(text="#f0f6fc", muted="#9198a1", border="#3d444d", grid="#3d444db3", canvas="#0d1117",
                 subtle="#151b23", line="#4493f8"),
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, theme, title, desc):
        self.w, self.h, self.t, self.parts = w, h, THEMES[theme], []
        self.title, self.desc = title, desc

    def text(self, x, y, s, size=12, color="muted", anchor="start", bold=False):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.t[color]}"'
                          + (f' text-anchor="{anchor}"' if anchor != "start" else "")
                          + (' font-weight="600"' if bold else "") + f">{esc(s)}</text>")

    def line(self, x1, y1, x2, y2, color="grid", width=1):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                          f' stroke="{self.t[color]}" stroke-width="{width}"/>')

    def save(self, path):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"'
            f' role="img" aria-labelledby="t d">\n<title id="t">{esc(self.title)}</title><desc id="d">{esc(self.desc)}</desc>\n'
            f'<style>text {{ font-family: {FONTS}; }}</style>\n' + "\n".join(self.parts) + "\n</svg>\n")


def nice_ticks(vmax, n=4):
    vmax = max(vmax, 1)
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw)) if raw >= 1 else 1
    step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    step = max(1, step)
    top = math.ceil(vmax / step) * step
    return [i * step for i in range(int(round(top / step)) + 1)]


def date_ticks(start, end, n=6):
    """About n dates between start and end, on day boundaries (whole weeks or months when the span is long)."""
    days = max(1, (end - start).days)
    if days > 120:  # month starts
        out, d = [], datetime(start.year, start.month, 1, tzinfo=timezone.utc)
        months = []
        while d <= end:
            if d >= start:
                months.append(d)
            d = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1, tzinfo=timezone.utc)
        every = max(1, math.ceil(len(months) / n))
        return months[::every], "%b %Y"
    step = max(1, math.ceil(days / n))
    if step > 7:
        step = 7 * math.ceil(step / 7)
    first = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    return [first + timedelta(days=i) for i in range(0, days + 1, step)], None


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch(repo, token):
    """(creation date, [star dates]) of ``repo`` from GitHub's API."""
    def get(url):
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.star+json",
                                                   "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    created = parse(get(f"https://api.github.com/repos/{repo}")["created_at"])
    stars, page = [], 1
    while True:
        got = get(f"https://api.github.com/repos/{repo}/stargazers?per_page=100&page={page}")
        stars += [parse(s["starred_at"]) for s in got]
        if len(got) < 100:
            return created, stars
        page += 1


BIN_HOURS = 10


def day(d, year=False):
    return f"{d:%b} {d.day}" + (f", {d.year}" if year else "")  # (as GitHub writes dates)


def render(theme, repo, created, stars, now):
    W = H = 480
    x0, y0, pw, ph = 56, 128, W - 72, 300
    stars = sorted(min(t, now) for t in stars)  # (a clock behind GitHub's would put the last ones past the end)
    start = min([created] + stars[:1])
    span = max((now - start).total_seconds(), 1)
    total = len(stars)
    s = Svg(W, H, theme, f"{repo}: star history",
            f"GitHub stars of {repo} from {day(start, True)} to {day(now, True)}: {total:,} stars.")
    t = s.t
    # a box with a header, as on GitHub's pages
    s.parts.append(f'<clipPath id="box"><rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6"/></clipPath>')
    s.parts.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="{t["canvas"]}"/>')
    s.parts.append(f'<rect width="{W}" height="48" fill="{t["subtle"]}" clip-path="url(#box)"/>')
    s.line(0, 48.5, W, 48.5, "border")
    s.parts.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="none" stroke="{t["border"]}"/>')
    s.text(16, 29, "Star history", 14, "text", bold=True)
    s.text(W - 16, 29, repo, 12, "muted", "end")
    s.parts.append(f'<path d="{STAR}" transform="translate(16 71)" fill="{t["muted"]}"/>')
    s.parts.append(f'<text x="40" y="87"><tspan font-size="24" font-weight="600" fill="{t["text"]}">{total:,}</tspan>'
                   f'<tspan dx="6" font-size="14" fill="{t["muted"]}">{"star" if total == 1 else "stars"}</tspan></text>')
    s.text(W - 16, 87, f"Updated {day(now, True)}", 12, "muted", "end")
    ticks = nice_ticks(total, 4)
    top = ticks[-1]
    ybase = y0 + ph
    for v in ticks:
        y = ybase - ph * v / top
        if v:
            s.line(x0, y, x0 + pw, y, "grid")
        s.text(x0 - 8, y + 4, f"{v:,}", 12, "muted", "end")
    s.line(x0, ybase, x0 + pw, ybase, "border")
    dates, fmt = date_ticks(start, now, 4)
    for d in dates:
        x = x0 + pw * (d - start).total_seconds() / span
        if x0 + 14 <= x <= x0 + pw - 14:
            s.line(x, ybase, x, ybase + 4, "border")
            s.text(x, ybase + 20, d.strftime(fmt) if fmt else day(d), 12, "muted", "middle")
    # the count every few hours (10, or more over a long history: about 90 points), joined by straight lines
    step = max(BIN_HOURS, math.ceil(span / 3600 / 90)) * 3600
    pts = []
    for k in range(math.ceil(span / step) + 1):
        sec = min(k * step, span)
        n = bisect.bisect_right(stars, start + timedelta(seconds=sec))
        pts.append((x0 + pw * sec / span, ybase - ph * n / top))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    s.parts.append(f'<polyline points="{line}" fill="none" stroke="{t["line"]}" stroke-width="2.5"'
                   f' stroke-linejoin="round" stroke-linecap="round"/>')
    ex, ey = pts[-1]
    s.parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{t["line"]}" stroke="{t["canvas"]}" stroke-width="2"/>')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "Generated-Computers/Locals"))
    ap.add_argument("--data", help='a JSON file {"created": ISO date, "stars": [ISO dates]} instead of the API')
    args = ap.parse_args()
    if args.data:
        d = json.loads(Path(args.data).read_text())
        created, stars = parse(d["created"]), [parse(t) for t in d["stars"]]
    else:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise SystemExit("listing stargazers needs a token: GITHUB_TOKEN or GH_TOKEN")
        created, stars = fetch(args.repo, token)
    now = datetime.now(timezone.utc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme, name in (("light", "stars.svg"), ("dark", "stars-dark.svg")):
        render(theme, args.repo, created, stars, now).save(out / name)
    print(f"{len(stars)} stars since {created:%Y-%m-%d}: {out / 'stars.svg'}, {out / 'stars-dark.svg'}")


if __name__ == "__main__":
    main()
