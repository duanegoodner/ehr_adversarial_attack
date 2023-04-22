from dataclasses import dataclass


@dataclass
class Thing:
    item_a: int


thing = Thing(item_a=1)
thing.item_b = 2

print(thing.item_b)
