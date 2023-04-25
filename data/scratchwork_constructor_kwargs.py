from pathlib import Path


class TestClass:
    _expected_items = ["item_a", "item_b", "item_c"]

    def __init__(
            self,
            default_dir: Path,
            item_a: Path = None,
            item_b: Path = None,
            item_c: Path = None

    ):
        self.default_dir = default_dir
        self.item_a =




my_class = TestClass(default_dir=Path.cwd())


