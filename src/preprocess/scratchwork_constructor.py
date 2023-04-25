from dataclasses import dataclass


@dataclass
class IncomingClass:
    item_a: int
    item_b: int


@dataclass
class ContainerClass:
    item_a: float
    item_b: float


my_incoming = IncomingClass(item_a=1, item_b=2)


my_container = ContainerClass(
    **{key: val + 0.5 for key, val in my_incoming.__dict__.items()}
)


# my_object = TestClass(**my_dict)

