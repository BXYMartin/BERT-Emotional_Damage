import unittest
import os

os.environ['PIP_CACHE_DIR'] = "/vol/bitbucket/mb220/.cache"
os.environ['TRANSFORMERS_CACHE'] = "/vol/bitbucket/mb220/.cache"
os.environ['HF_DATASETS_CACHE'] = "/vol/bitbucket/mb220/.cache"
os.environ["WANDB_DISABLED"] = "true"
from loader.official import OfficialLoader
from util.opt import ThresholdOptimizer
from model.DebertaV2XLarge import DebertaV2XLarge

class DebertaV2XLargeTestCase(unittest.TestCase):
    def test_optimize_all_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Synonym-Test-Case")
        optimizer = ThresholdOptimizer(data_loader)
        optimizer.run()

    def test_final_all_with_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Synonym-Test-Case")
        model = DebertaV2XLarge(data_loader, save_prob=True, load_existing=True)
        model.final_with_threshold(threshold=0.95619658)
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

    def test_final_convert_all_with_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Synonym-Test-Case")
        for threshold in [0.95, 0.96, 0.97, 0.98, 0.99, 0.999]:
            data_loader.final_convert(threshold=threshold)
        self.assertEqual(True, True)


    def test_train_all(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Synonym-Test-Case")
        data_loader.split_upsample_cleaned_synonym()
        model = DebertaV2XLarge(data_loader, skip_eval=True)
        model.eval_step_size = 2000
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass
 
    def test_predict_all_on_test_set(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-All-Cleaned-Synonym-Test-Case")
        data_loader.split_upsample_cleaned_synonym()
        model = DebertaV2XLarge(data_loader, load_existing=True, skip_eval=False, save_prob=True)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.predict()
        self.assertEqual(True, True)
        pass



    def test_optimize_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Cleaned-Synonym-Test-Case")
        optimizer = ThresholdOptimizer(data_loader)
        optimizer.run()

    def test_final_with_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Cleaned-Synonym-Test-Case")
        model = DebertaV2XLarge(data_loader, save_prob=True, load_existing=True)
        model.final_with_threshold(threshold=0.95619658)
        data_loader.final(model.prediction)
        self.assertEqual(True, True)

    def test_final_convert_with_threshold(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Cleaned-Synonym-Test-Case")
        for threshold in [0.95, 0.96, 0.97, 0.98, 0.99, 0.999]:
            data_loader.final_convert(threshold=threshold)
        self.assertEqual(True, True)


    def test_train(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Cleaned-Synonym-Test-Case")
        data_loader.split_upsample_cleaned_synonym(use_all=False)
        model = DebertaV2XLarge(data_loader, skip_eval=False)
        model.eval_step_size = 2000
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.train()
        self.assertEqual(True, True)
        pass
 
    def test_predict_on_test_set(self):
        data_loader = OfficialLoader("Official-DebertaV2XLarge-Cleaned-Synonym-Test-Case")
        data_loader.split_upsample_cleaned_synonym()
        model = DebertaV2XLarge(data_loader, load_existing=True, skip_eval=False, save_prob=True)
        print(f'train_start, model.train_epochs = {model.train_epochs}, model.batch_size = {model.batch_size}')
        model.predict()
        self.assertEqual(True, True)
        pass



if __name__ == '__main__':
    unittest.main()
