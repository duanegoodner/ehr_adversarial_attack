from typing import NamedTuple


class Thing(NamedTuple):
    val_a: int
    val_b: int


my_tuples = [Thing(val_a=1, val_b=2), Thing(val_a=3, val_b=4)]

test_item = Thing(val_a=1, val_b=2)

print(test_item in my_tuples)