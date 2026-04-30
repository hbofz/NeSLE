import unittest

from nesle.actions import (
    Button,
    COMPLEX_MOVEMENT_MASKS,
    RIGHT_ONLY_MASKS,
    SIMPLE_MOVEMENT_MASKS,
    encode_action,
)


class ActionTests(unittest.TestCase):
    def test_encode_noop(self):
        self.assertEqual(encode_action(["NOOP"]), 0)

    def test_encode_buttons(self):
        self.assertEqual(encode_action(["right", "A", "B"]), 0b10000011)
        self.assertEqual(encode_action([Button.START]), 0b00001000)

    def test_default_spaces(self):
        self.assertEqual(len(RIGHT_ONLY_MASKS), 5)
        self.assertEqual(len(SIMPLE_MOVEMENT_MASKS), 7)
        self.assertEqual(len(COMPLEX_MOVEMENT_MASKS), 12)
        self.assertEqual(SIMPLE_MOVEMENT_MASKS[1], encode_action(["right"]))

    def test_unknown_button_fails(self):
        with self.assertRaises(ValueError):
            encode_action(["jump"])


if __name__ == "__main__":
    unittest.main()
