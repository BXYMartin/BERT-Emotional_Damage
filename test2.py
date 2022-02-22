import unittest
import os

os.environ['PIP_CACHE_DIR'] = "/vol/bitbucket/ya321/.cache"
os.environ['TRANSFORMERS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ['HF_DATASETS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ["WANDB_DISABLED"] = "true"
from loader.official import OfficialLoader
from model.DebertaBase import DebertaBase
from model.RobertaBaseFrenkHate import RobertaBaseFrenkHate
from model.DebertaLarge import DebertaLarge
from model.DebertaV2XLarge import DebertaV2XLarge
from model.DebertaV3Large import DebertaV3Large


class DebertaBaseTestCase(unittest.TestCase):
    def test_splitAAA(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        self.assertEqual(True, True)

    def test_split_upsample_all(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample_all()
        self.assertEqual(True, True)

    def test_train(self):
        data_loader = OfficialLoader("Official-DebertaBaseUncased-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
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

    def test_process(self):
        data_loader = OfficialLoader()
        data_loader.process('official_split_train_dataset_AAA.csv', 'official_split_train_dataset_AAA_truncation.csv')
        data_loader.process('official_split_test_dataset.csv', 'official_split_test_dataset_truncation.csv')
        data_loader.process('official_split_train_dataset_AAA_all.csv',
                            'official_split_train_dataset_AAA_all_truncation.csv')
        data_loader.process('dontpatronizeme_test.tsv',
                            data_loader.official_final_data_truncation_filename)
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

    def test_final_path(self):
        data_loader = OfficialLoader("Official-RobertaBaseFrenkHate-Test-Case")
        model = RobertaBaseFrenkHate(data_loader, load_existing=True)
        model.final('0')


class DebertaLargeTestCase(unittest.TestCase):
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
        data_loader = OfficialLoader("Official-DebertaLarge-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaLarge(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_train_512(self):
        data_loader = OfficialLoader("Official-DebertaLarge512-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaLarge(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-DebertaLarge-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = DebertaLarge(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-DebertaLarge-Test-Case")
        model = DebertaLarge(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

    def test_final_path(self):
        data_loader = OfficialLoader("Official-DebertaLarge-Test-Case")
        model = DebertaLarge(data_loader, load_existing=True)
        model.final('0')


class DebertaV2XLargeTestCase(unittest.TestCase):
    def test_process(self):
        data_loader = OfficialLoader()
        # data_loader.process()
        self.assertEqual(True, True)

    def test_split_upsample(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        self.assertEqual(True, True)



    def test_train_all_cleaned(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Test-Case")
        data_loader.split_upsample()
        data_loader.split_upsample_all_cleaned()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaV2XLarge(data_loader, skip_eval=True)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass




    def test_train_all(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Test-Case")
        data_loader.split_upsample()
        data_loader.split_upsample_all()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaV2XLarge(data_loader, skip_eval=True)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass



    def test_augment_all(self):
        data_loader = OfficialLoader()
        data_loader.split_upsample()
        data_loader.augmentation_all()
        self.assertEqual(True, True)

    def test_train(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaV2XLarge(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = DebertaV2XLarge(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Test-Case")
        model = DebertaV2XLarge(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

    def test_final_path(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Test-Case")
        model = DebertaV2XLarge(data_loader, load_existing=True)
        model.final('0')


class DebertaV3LargeTestCase(unittest.TestCase):
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
        data_loader = OfficialLoader("Official-DebertaV3Large-Test-Case")
        data_loader.split_upsample()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaV3Large(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_train_truncation(self):
        data_loader = OfficialLoader("Official-DebertaV3LargeTruncation-Test-Case")
        data_loader.split_upsample_truncation()
        # data_loader.augmentation()
        # data_loader.augmentation_all()
        model = DebertaV3Large(data_loader)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass

    def test_predict(self):
        data_loader = OfficialLoader("Official-DebertaV3Large-Test-Case")
        data_loader.split()
        # data_loader.balance()
        model = DebertaV3Large(data_loader, load_existing=True)
        model.predict()
        self.assertEqual(True, True)

    def test_final(self):
        data_loader = OfficialLoader("Official-DebertaV3Large-Test-Case")
        model = DebertaV3Large(data_loader, load_existing=True)
        model.final()
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

    def test_final_path(self):
        data_loader = OfficialLoader("Official-DebertaV3Large-Test-Case")
        model = DebertaV3Large(data_loader, load_existing=True)
        model.final('0')


if __name__ == '__main__':
    unittest.main()
