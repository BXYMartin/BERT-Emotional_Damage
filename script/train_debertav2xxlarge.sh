#!/bin/bash
#SBATCH --gres=gpu:2
source /homes/mb220/.bashrc 
cd /vol/bitbucket/mb220/course-work-for-natural-language-processing
/vol/bitbucket/mb220/python-env/bin/python -m unittest test3.DebertaV2XXLargeTestCase.test_train
