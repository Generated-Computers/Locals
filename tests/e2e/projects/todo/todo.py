"""A to-do list on the command line, kept in todos.json next to the current directory.

    python3 todo.py add "buy milk"
    python3 todo.py list
"""

import argparse
import json
import os

FILE = "todos.json"


def load(path=FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save(todos, path=FILE):
    with open(path, "w") as f:
        json.dump(todos, f, indent=1)


def add(todos, text):
    todos.append({"text": text, "done": False})
    return todos


def render(todos):
    """One line per to-do: '1. [ ] buy milk', with [x] for the ones done."""
    return [f"{i}. [{'x' if t['done'] else ' '}] {t['text']}" for i, t in enumerate(todos, 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="todo")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add")
    p.add_argument("text")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    todos = load()
    if args.command == "add":
        save(add(todos, args.text))
    elif args.command == "list":
        for line in render(todos):
            print(line)


if __name__ == "__main__":
    main()
