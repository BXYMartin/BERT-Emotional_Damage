#!/bin/bash
#SBATCH --gres=gpu:1
source /homes/mb220/.bashrc 
cd /vol/bitbucket/mb220/course-work-for-natural-language-processing
#/vol/bitbucket/mb220/python-env/bin/python -m unittest test2.DebertaV2XLargeTestCase.test_predict_all_on_test_set
#/vol/bitbucket/mb220/python-env/bin/python -m unittest test2.DebertaV2XLargeTestCase.test_optimize_all_threshold
/vol/bitbucket/mb220/python-env/bin/python -m unittest test2.DebertaV2XLargeTestCase.test_final_all_with_threshold
