from dataclasses import dataclass
from typing import Any


@dataclass
class TestClass:
    ref: str
    process_object: Any = None

    def transform(self):
        self.process_object = f"imported {self.ref}"


test_object = TestClass(ref="something")
print(test_object.process_object)
test_object.transform()
print(test_object.process_object)




