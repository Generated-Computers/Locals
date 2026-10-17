from dataclasses import dataclass, field
from typing import List


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.price * self.quantity
