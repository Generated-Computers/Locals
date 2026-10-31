"""Six test pictures and one question each (pics/questions.json). Usage: make_pictures.py OUTDIR"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
F = lambda n: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", n)
qs = {}

def cat42(w, h, size, name):
    im = Image.new("RGB", (w, h), "white"); g = ImageDraw.Draw(im)
    box = g.textbbox((0, 0), "CAT 42", font=F(size))
    g.text(((w - box[2]) // 2, (h - box[3]) // 2), "CAT 42", fill="black", font=F(size))
    im.save(out / name); qs[name] = "What text is in the picture? Answer with the text only."

cat42(640, 400, 64, "cat42_640x400.png")
cat42(1280, 800, 128, "cat42_1280x800.png")

im = Image.new("RGB", (640, 480), "white"); g = ImageDraw.Draw(im)
g.ellipse((40, 160, 200, 320), fill="red"); g.rectangle((250, 160, 410, 320), fill="blue")
g.polygon([(530, 160), (450, 320), (610, 320)], fill="green")
im.save(out / "shapes_640x480.png"); qs["shapes_640x480.png"] = "List the shapes from left to right with their colours."

im = Image.new("RGB", (900, 500), "white"); g = ImageDraw.Draw(im)
for i, line in enumerate(["Invoice 7319", "Date: 2026-10-31", "Total: 1,284.50 PLN", "Customer: Stefan Nowak"]):
    g.text((40, 30 + i * 85), line, fill="black", font=F(40))
im.save(out / "invoice_900x500.png"); qs["invoice_900x500.png"] = "What is the total, and who is the customer?"

im = Image.new("RGB", (512, 256), "white"); g = ImageDraw.Draw(im)
g.text((40, 110), "HELLO WORLD from a small font", fill="black", font=F(20))
im.save(out / "smallfont_512x256.png"); qs["smallfont_512x256.png"] = "What does it say? Answer with the text only."

im = Image.new("RGB", (768, 512), "white"); g = ImageDraw.Draw(im)
for i, (label, v) in enumerate([("A", 120), ("B", 340), ("C", 220)]):
    x = 120 + i * 220; g.rectangle((x, 440 - v, x + 120, 440), fill="gray"); g.text((x + 45, 450), label, fill="black", font=F(36))
g.line((80, 440, 720, 440), fill="black", width=3)
im.save(out / "bars_768x512.png"); qs["bars_768x512.png"] = "Which bar is the tallest, and which is the shortest?"

(out / "questions.json").write_text(json.dumps(qs, indent=1))
print(len(qs), "pictures")
