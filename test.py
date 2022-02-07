import unittest
from eval.score import Official
import logging
import numpy as np

logger = logging.getLogger()
logger.level = logging.DEBUG


class LoaderTestCase(unittest.TestCase):
    def test_loader(self):
        data_loader = Official()
        self.assertEqual(len(data_loader.all_data.columns), 7)
        self.assertEqual(len(data_loader.all_data), 10469)

    def test_submit(self):
        data_loader = Official()
        # Generate completely random sequences
        data_loader.final([np.random.choice([0, 1]) for _ in range(len(data_loader.final_data))])
        # Zip the task1.txt and submit - 0.0842931937


if __name__ == '__main__':
    unittest.main()
