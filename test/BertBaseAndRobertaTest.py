import unittest
import os
from loader.official import OfficialLoader
from model.RoBERTa import RoBERTa
from model.BackTranslate import BackTranslate
from model.RoBERTaBase import RoBERTaBase
from model.BertBaseUncased import BertBaseUncased
from loader.custom import CustomLoader
import logging
import numpy as np

logger = logging.getLogger()
logger.level = logging.DEBUG


class RoBERTaTestCase(unittest.TestCase):
    def test_train(self):
        data_loader = OfficialLoader("Official-RoBERTa-Test-Case-Augmented")
        data_loader.split()
        data_loader.augmentation()
        model = RoBERTa(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-RoBERTa-Test-Case-Augmented")
        data_loader.split()
        #data_loader.balance()
        model = RoBERTa(data_loader, load_existing=True)
        model.predict()
        print(model.prediction)
        data_loader.eval(data_loader.test_data.label.tolist(), model.prediction)
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-RoBERTa-Test-Case-Augmented")
        model = RoBERTa(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)


class BERTBaseUncasedCase(unittest.TestCase):
    def test_train(self):
        data_loader = OfficialLoader("Official-BERTBaseUncased-Test-Case")
        data_loader.split()
        data_loader.augmentation()
        model = BertBaseUncased(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-BERTBaseUncased-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = BertBaseUncased(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-BERTBaseUncased-Test-Case")
        model = BertBaseUncased(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)


class RoBERTaBaseCase(unittest.TestCase):
    def test_train(self):
        data_loader = OfficialLoader("Official-RoBERTaBase-Test-Case")
        data_loader.split()
        data_loader.balance()
        model = RoBERTaBase(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-RoBERTaBase-Test-Case")
        data_loader.split()
        data_loader.balance()
        model = RoBERTaBase(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-RoBERTaBase-Test-Case")
        model = RoBERTaBase(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
