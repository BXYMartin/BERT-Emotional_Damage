import unittest
from loader.official import OfficialLoader
from loader.custom import CustomLoader
import logging
import numpy as np

logger = logging.getLogger()
logger.level = logging.DEBUG


class LoaderTestCase(unittest.TestCase):
    def test_loader(self):
        data_loader = CustomLoader()
        self.assertEqual(len(data_loader.all_data.columns), 7)
        self.assertEqual(len(data_loader.all_data), 10469)

    def test_submit(self):
        data_loader = CustomLoader()
        # Generate completely random sequences
        data_loader.final([np.random.choice([0, 1]) for _ in range(len(data_loader.final_data))])
        # Zip the task1.txt and submit - 0.0842931937

    def test_fold(self):
        data_loader = CustomLoader()
        for k in [1, 5, 10, 20]:
            for train_data, test_data in data_loader.fold(k):
                self.assertEqual(len(train_data), len(data_loader.all_data) - int(len(data_loader.all_data) / k))
                self.assertEqual(len(test_data), int(len(data_loader.all_data) / k))
                break

    def test_split(self):
        data_loader = OfficialLoader()
        data_loader.split()
        self.assertEqual(len(data_loader.test_data), 2094)
        self.assertEqual(len(data_loader.train_data), 8375)



if __name__ == '__main__':
    unittest.main()
