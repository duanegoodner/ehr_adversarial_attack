from dataclasses import dataclass


@dataclass
class TestClass:
    item_a: int
    item_b: int


my_dict = {"item_a": 1, "item_b": 2}


my_object = TestClass(**my_dict)

