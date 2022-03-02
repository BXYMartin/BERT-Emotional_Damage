import unittest
import os
from loader.official import OfficialLoader
from model.TalkDownBert import TalkDownBert
from model.XLNet import XLNet

class TalkDownBertTestCase(unittest.TestCase):
    def test_train(self):
        data_loader = OfficialLoader("Official-TalkDownBert-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = TalkDownBert(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-TalkDownBert-Test-Case")
        data_loader.split_upsample()
        # data_loader.balance()
        #model = TalkDownBert(data_loader, load_existing=True)
        model = TalkDownBert(data_loader)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-TalkDownBert-Test-Case")
        model = TalkDownBert(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
