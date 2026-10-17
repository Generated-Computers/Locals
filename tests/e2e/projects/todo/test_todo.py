import os
import tempfile
import unittest

import todo


class TodoTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "todos.json")

    def test_add_and_render(self):
        todos = todo.add([], "buy milk")
        todo.add(todos, "call mom")
        self.assertEqual(todo.render(todos), ["1. [ ] buy milk", "2. [ ] call mom"])

    def test_save_and_load(self):
        todo.save(todo.add([], "x"), self.path)
        self.assertEqual(todo.load(self.path), [{"text": "x", "done": False}])

    def test_load_missing_file(self):
        self.assertEqual(todo.load(os.path.join(self.dir, "none.json")), [])


if __name__ == "__main__":
    unittest.main()
