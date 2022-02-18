import unittest
import os

os.environ['PIP_CACHE_DIR'] = "/vol/bitbucket/ya321/.cache"
os.environ['TRANSFORMERS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ['HF_DATASETS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ["WANDB_DISABLED"] = "true"
from loader.official import OfficialLoader
from model.DebertaBase import DebertaBase
from model.RobertaBaseFrenkHate import RobertaBaseFrenkHate


class DebertaBaseTestCase(unittest.TestCase):
    def test_splitAAA(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        self.assertEqual(True, True)

    def test_train(self):
        data_loader = OfficialLoader("Official-DebertaBaseUncased-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        model = DebertaBase(data_loader)
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-DebertaBaseUncased-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = DebertaBase(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-DebertaBaseUncased-Test-Case")
        model = DebertaBase(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)


class RobertaBaseFrenkHateTestCase(unittest.TestCase):
    def test_process(self):
        data_loader = OfficialLoader()
        data_loader.process()
        self.assertEqual(True, True)

    def test_split_upsample(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        self.assertEqual(True, True)

    def test_augment_all(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        data_loader.augmentation_all()
        self.assertEqual(True, True)

    def test_train(self):
        data_loader = OfficialLoader("Official-RobertaBaseFrenkHate-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        data_loader.augmentation_all()
        model = RobertaBaseFrenkHate(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-RobertaBaseFrenkHate-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = RobertaBaseFrenkHate(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-RobertaBaseFrenkHate-Test-Case")
        model = RobertaBaseFrenkHate(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
