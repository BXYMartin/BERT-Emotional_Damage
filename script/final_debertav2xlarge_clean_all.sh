#!/bin/bash
#SBATCH --gres=gpu:1
source /homes/mb220/.bashrc 
cd /vol/bitbucket/mb220/course-work-for-natural-language-processing
/vol/bitbucket/mb220/python-env/bin/python -m unittest test2.DebertaV2XLargeTestCase.test_final_all_cleaned_with_threshold
