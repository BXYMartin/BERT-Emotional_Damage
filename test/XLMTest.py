import unittest
import os
from loader.official import OfficialLoader
from model.XLM import XLM

class XLMTestCase(unittest.TestCase):
    def test_train(self):
        data_loader = OfficialLoader("Official-XLM-Clean-Test-Case")
        data_loader.split_upsample_cleaned()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = XLM(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-XLM-Clean-Test-Case")
        data_loader.split_upsample_cleaned()
        # data_loader.balance()
        model = XLM(data_loader, load_existing=True)
        #model = XLM(data_loader)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-XLM-Clean-Test-Case")
        model = XLM(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
