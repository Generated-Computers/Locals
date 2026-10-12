"""Workloads the model tests decode: extraction tasks with their JSON schemas (WORKLOADS), the
prompts for them, and recursive schemas (JOBS) with a validator."""

import json

WORKLOADS = [
    {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "status": {"enum": ["pending", "approved", "rejected"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age", "email", "status", "tags"],
        },
        "text": "Maria Gonzalez, 34, signed up with maria.g@example.com last week. Her application was approved. She is interested in hiking, photography and machine learning.",
    },
    {
        "name": "tool_call",
        "schema": {
            "type": "object",
            "properties": {
                "function": {"enum": ["get_weather", "search_web", "send_email", "create_calendar_event"]},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"enum": ["celsius", "fahrenheit"]},
                        "include_forecast": {"type": "boolean"},
                        "days": {"type": "integer"},
                    },
                    "required": ["location", "unit", "include_forecast", "days"],
                },
            },
            "required": ["function", "arguments"],
        },
        "text": "What's the weather going to be like in Lisbon over the next 5 days? Use celsius please and include the forecast.",
    },
    {
        "name": "product",
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"enum": ["USD", "EUR", "GBP", "JPY"]},
                "in_stock": {"type": "boolean"},
                "categories": {"type": "array", "items": {"type": "string"}},
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "width_cm": {"type": "number"},
                        "height_cm": {"type": "number"},
                        "depth_cm": {"type": "number"},
                    },
                    "required": ["width_cm", "height_cm", "depth_cm"],
                },
            },
            "required": ["product_name", "price", "currency", "in_stock", "categories", "dimensions"],
        },
        "text": "The Nordic Oak Desk costs 449.99 euros and is currently available. It fits in the furniture, office and home categories. It measures 120 cm wide, 75 cm tall and 60 cm deep.",
    },
    {
        "name": "event",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "location": {
                    "type": "object",
                    "properties": {"venue": {"type": "string"}, "city": {"type": "string"}, "country": {"type": "string"}},
                    "required": ["venue", "city", "country"],
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "role": {"enum": ["speaker", "organizer", "guest"]}},
                        "required": ["name", "role"],
                    },
                },
            },
            "required": ["title", "date", "location", "attendees"],
        },
        "text": "The Applied AI Summit takes place on 2026-11-12 at the Congress Center in Berlin, Germany. Dr. Aiko Tanaka and Lars Petersen will speak, and Nina Weber is organizing it.",
    },
    {
        "name": "classification",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"enum": ["very_negative", "negative", "neutral", "positive", "very_positive"]},
                "confidence": {"type": "number"},
                "topics": {"type": "array", "items": {"enum": ["shipping", "quality", "price", "customer_service", "packaging"]}},
                "needs_follow_up": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["sentiment", "confidence", "topics", "needs_follow_up", "summary"],
        },
        "text": "Review: The blender arrived two weeks late and the box was crushed, but it works great and was cheaper than elsewhere. Support never answered my emails though.",
    },
    {
        "name": "contact_card",
        "schema": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "street_address": {"type": "string"},
                "postal_code": {"type": "string"},
                "city": {"type": "string"},
                "country_code": {"enum": ["US", "GB", "DE", "FR", "JP", "BR"]},
                "phone_number": {"type": "string"},
                "is_primary_contact": {"type": "boolean"},
                "preferred_contact_method": {"enum": ["email", "phone", "sms", "mail"]},
            },
            "required": [
                "first_name", "last_name", "street_address", "postal_code", "city",
                "country_code", "phone_number", "is_primary_contact", "preferred_contact_method",
            ],
        },
        "text": "Please add Jean-Luc Moreau, 14 Rue des Lilas, 75011 Paris, France, phone +33 6 12 34 56 78, as our main contact. He prefers to be reached by text message.",
    },
]


def make_prompt(tokenizer, w):
    import json

    msg = (
        "Extract the information from the text below as JSON that follows this schema:\n"
        + json.dumps(w["schema"])
        + "\n\nText: "
        + w["text"]
        + "\n\nRespond with the JSON only."
    )
    return tokenizer.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False)


def compact_schema(s) -> str:
    """Keys, enum options and list/object shape only: the constraint supplies
    the syntax, so the prompt needs just the semantics."""
    if "enum" in s:
        return "|".join(s["enum"])
    t = s.get("type")
    if t == "object":
        return "{" + ", ".join(f"{k}: {compact_schema(v)}" for k, v in s["properties"].items()) + "}"
    if t == "array":
        return "[" + compact_schema(s["items"]) + ", ...]"
    return t


def prompt_text(w, schema_in_prompt, compact=False):
    if compact:
        head = "Extract the information from the text below as JSON: " + compact_schema(w["schema"])
    elif schema_in_prompt:
        head = "Extract the information from the text below as JSON that follows this schema:\n" + json.dumps(w["schema"])
    else:
        head = "Extract the information from the text below as JSON."
    return head + "\n\nText: " + w["text"] + "\n\nRespond with the JSON only."


TREE = {"type": "object", "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["name", "children"], "additionalProperties": False}
EXPR = {"$defs": {"expr": {"anyOf": [{"type": "number"}, {"type": "object", "properties": {
    "op": {"enum": ["+", "*", "-", "/"]}, "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 2}},
    "required": ["op", "args"], "additionalProperties": False}]}}, "$ref": "#/$defs/expr"}
THREAD = {"type": "object", "properties": {"author": {"type": "string"}, "text": {"type": "string"},
                                           "replies": {"type": "array", "items": {"$ref": "#"}}},
          "required": ["author", "text", "replies"], "additionalProperties": False}
FS = {"$defs": {"entry": {"type": "object", "properties": {
    "name": {"type": "string"}, "kind": {"enum": ["file", "dir"]}, "size": {"type": "integer"},
    "entries": {"type": "array", "items": {"$ref": "#/$defs/entry"}}},
    "required": ["name", "kind", "size", "entries"], "additionalProperties": False}}, "$ref": "#/$defs/entry"}
CATS = {"type": "object", "properties": {"category": {"type": "string"}, "count": {"type": "integer"},
                                         "subcategories": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["category", "count", "subcategories"], "additionalProperties": False}

JOBS = [
    ("solar", TREE, "Build a tree of the Solar System as nested JSON: the root is 'Solar System'; its children are the "
                    "planets Mercury, Venus, Earth, Mars and Jupiter; Earth has the child 'Moon'; Mars has 'Phobos' and "
                    "'Deimos'; Jupiter has 'Io' and 'Europa'."),
    ("expr", EXPR, "Write the arithmetic expression (2 + 3) * (4 - 1) * (10 / 5) as a JSON expression tree. A node is "
                   "either a number or an object with 'op' and 'args'."),
    ("thread", THREAD, "Convert this discussion into nested JSON. Alice wrote 'Is the build green?'. Bob replied to Alice "
                       "'Yes, since this morning'. Carol replied to Bob 'Great, merging now'. Dave replied to Alice "
                       "'The docs job is still failing'. Erin replied to Dave 'On it'."),
    ("fs", FS, "Describe this directory tree as nested JSON. 'project' (dir) contains 'README.md' (file, 1200 bytes), "
               "'src' (dir) and 'tests' (dir). 'src' contains 'main.py' (file, 4300 bytes) and 'util' (dir) with "
               "'io.py' (file, 800 bytes). 'tests' contains 'test_main.py' (file, 2100 bytes). Directory sizes are 0."),
    ("cats", CATS, "Turn this taxonomy into nested JSON with item counts. 'Electronics' (120 items) has 'Phones' (50) "
                   "and 'Laptops' (70); 'Phones' has 'Android' (30) and 'iPhone' (20); 'Laptops' has 'Gaming' (25) "
                   "and 'Ultrabooks' (45). 'Books' (200) has 'Fiction' (120) and 'Non-fiction' (80)."),
]


PHRASINGS = [
    "{task}\n\nRespond with JSON only, following this schema exactly:\n{schema}",
    "Task: {task}\nOutput format: a single JSON document valid against this schema (no prose):\n{schema}",
    "{task}\n\nJSON schema:\n{schema}\n\nAnswer with the JSON document and nothing else.",
]


def recursive_prompt(schema, task, variant: int = 0):
    return PHRASINGS[variant % len(PHRASINGS)].format(task=task, schema=json.dumps(schema))


def _resolve(root, ref):
    node = root
    if ref != "#":
        for part in ref[2:].split("/"):
            node = node[part]
    return node


def valid(schema, x, root=None):
    root = schema if root is None else root
    if "$ref" in schema:
        return valid(_resolve(root, schema["$ref"]), x, root)
    if "anyOf" in schema:
        return any(valid(s, x, root) for s in schema["anyOf"])
    if "enum" in schema:
        return x in schema["enum"]
    t = schema.get("type")
    if t == "object":
        if not isinstance(x, dict) or set(x) - set(schema.get("properties", {})):
            return False
        if any(k not in x for k in schema.get("required", [])):
            return False
        return all(valid(schema["properties"][k], v, root) for k, v in x.items())
    if t == "array":
        if not isinstance(x, list) or len(x) < schema.get("minItems", 0):
            return False
        return all(valid(schema["items"], v, root) for v in x)
    if t == "string":
        return isinstance(x, str)
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "boolean":
        return isinstance(x, bool)
    return False


def depth(x):
    if isinstance(x, dict):
        return 1 + max([depth(v) for v in x.values()] + [0])
    if isinstance(x, list):
        return max([depth(v) for v in x] + [0])
    return 0
