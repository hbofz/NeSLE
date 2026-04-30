import unittest

from nesle.smb import compute_reward, read_ram


def ram_with_defaults():
    ram = bytearray(2048)
    ram[0x006D] = 1
    ram[0x0086] = 2
    ram[0x00B5] = 1
    ram[0x03B8] = 100
    ram[0x075A] = 2
    ram[0x075C] = 0
    ram[0x075F] = 0
    ram[0x0760] = 0
    ram[0x0756] = 1
    ram[0x07F8] = 4
    ram[0x07F9] = 0
    ram[0x07FA] = 0
    ram[0x07ED] = 1
    ram[0x07EE] = 7
    ram[0x07DE] = 0
    ram[0x07DF] = 1
    ram[0x07E0] = 2
    ram[0x07E1] = 3
    ram[0x07E2] = 4
    ram[0x07E3] = 5
    return ram


class SmbTests(unittest.TestCase):
    def test_read_ram(self):
        state = read_ram(ram_with_defaults())
        self.assertEqual(state.x_pos, 258)
        self.assertEqual(state.y_pos, 155)
        self.assertEqual(state.time, 400)
        self.assertEqual(state.coins, 17)
        self.assertEqual(state.score, 12345)
        self.assertEqual(state.life, 2)
        self.assertEqual(state.world, 1)
        self.assertEqual(state.stage, 1)
        self.assertEqual(state.area, 1)
        self.assertEqual(state.status, "tall")

    def test_reward_filters_large_x_reset(self):
        prev_ram = ram_with_defaults()
        next_ram = ram_with_defaults()
        next_ram[0x006D] = 0
        next_ram[0x0086] = 0
        prev = read_ram(prev_ram)
        curr = read_ram(next_ram)
        reward = compute_reward(prev, curr)
        self.assertEqual(reward.x, 0)

    def test_reward_components(self):
        prev_ram = ram_with_defaults()
        next_ram = ram_with_defaults()
        next_ram[0x0086] = 5
        next_ram[0x07FA] = 9
        next_ram[0x07F9] = 9
        next_ram[0x07F8] = 3
        prev = read_ram(prev_ram)
        curr = read_ram(next_ram)
        reward = compute_reward(prev, curr)
        self.assertEqual(reward.x, 3)
        self.assertEqual(reward.time, -1)
        self.assertEqual(reward.total, 2)

    def test_death_penalty(self):
        prev_ram = ram_with_defaults()
        next_ram = ram_with_defaults()
        next_ram[0x000E] = 0x0B
        reward = compute_reward(read_ram(prev_ram), read_ram(next_ram))
        self.assertEqual(reward.death, -25)

    def test_flag_get(self):
        ram = ram_with_defaults()
        ram[0x0016] = 0x31
        ram[0x001D] = 3
        self.assertTrue(read_ram(ram).flag_get)

    def test_short_ram_fails(self):
        with self.assertRaises(ValueError):
            read_ram(b"\x00")


if __name__ == "__main__":
    unittest.main()
