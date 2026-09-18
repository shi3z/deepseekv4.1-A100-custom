import unittest
import torch
from dsv41.engine import GenParams, apply_penalties

class TestPenalties(unittest.TestCase):
    def test_repetition_penalty(self):
        logits = torch.tensor([5.0, 5.0, 5.0, 5.0])
        # Token 1 appeared in generated tokens
        tokens = [1, 1]
        penalized = apply_penalties(logits, tokens, repetition_penalty=1.2)
        self.assertLess(penalized[1].item(), logits[1].item())
        self.assertEqual(penalized[0].item(), logits[0].item())
        self.assertEqual(penalized[2].item(), logits[2].item())

    def test_frequency_presence_penalty(self):
        logits = torch.tensor([10.0, 10.0, 10.0])
        # Token 0 appeared twice, Token 1 appeared once
        tokens = [0, 0, 1]
        penalized = apply_penalties(
            logits, tokens,
            repetition_penalty=1.0,
            presence_penalty=0.5,
            frequency_penalty=0.2,
            progressive_penalty=0.0,
        )
        # Token 0: 10 - (0.5 + 0.2*2) = 9.1
        self.assertAlmostEqual(penalized[0].item(), 9.1, places=4)
        # Token 1: 10 - (0.5 + 0.2*1) = 9.3
        self.assertAlmostEqual(penalized[1].item(), 9.3, places=4)
        # Token 2: 10.0
        self.assertEqual(penalized[2].item(), 10.0)

    def test_cycle_suppression(self):
        phrase = [10, 20, 30, 40]
        # phrase repeated once, and now at 10, 20, 30 -> next token 40 would repeat cycle
        tokens = phrase + [10, 20, 30]
        logits = torch.tensor([10.0] * 50)
        penalized = apply_penalties(logits, tokens, ban_cycles=True)
        # Token 40 should be suppressed heavily (-50)
        self.assertLess(penalized[40].item(), -30.0)
        # Other tokens not completing the cycle are not suppressed by cycle ban
        self.assertGreater(penalized[45].item(), 5.0)

    def test_windowing(self):
        logits = torch.tensor([5.0, 5.0, 5.0])
        # Window size is 2, so token 0 (which was at index 0) should fall out of window
        tokens = [0, 1, 2]
        penalized = apply_penalties(logits, tokens, repetition_penalty=1.5, window=2)
        # Token 0 should be untouched because it fell outside window
        self.assertEqual(penalized[0].item(), 5.0)
        # Tokens 1 and 2 should be penalized
        self.assertLess(penalized[1].item(), 5.0)
        self.assertLess(penalized[2].item(), 5.0)

if __name__ == "__main__":
    unittest.main()
