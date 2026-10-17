from typing import Dict, List

from .models import Item


class Store:
    """Items by name."""

    def __init__(self):
        self.items: Dict[str, Item] = {}

    def add(self, item: Item) -> None:
        if item.name in self.items:
            self.items[item.name].quantity += item.quantity
        else:
            self.items[item.name] = item

    def remove(self, name: str, quantity: int = 1) -> None:
        item = self.items[name]
        if quantity > item.quantity:
            raise ValueError(f"only {item.quantity} {name} in stock")
        item.quantity -= quantity
        if item.quantity == 0:
            del self.items[name]

    def find_by_tag(self, tag: str) -> List[Item]:
        return sorted((i for i in self.items.values() if tag in i.tags), key=lambda i: i.name)

    def total_value(self) -> float:
        return sum(i.value for i in self.items.values())
