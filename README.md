# SemEval 2022 Task 4 SubTask 1: Patronizing and Condescending Language Detection
This is the natural language processing coursework repository for our team.. 

## EmotionalDamage Team
- Boyu Han
- Xinyu Bai
- Yuze An

## Directory

```bash
.
├── data `Preprocessed csv data for training and evaluation`
├── loader `Data loader`
├── main.py `Main python script to invoke functions`
├── model `All of our implemented models`
├── resource `Figures and data that is used to generate report`
├── runtime `Runtime cache and model checkpoints`
├── script `Scripts used to train on Slurm`
├── spec `Specifications of the task`
├── test `Python unittest directory`
└── util `Data analysis and performance optimization`
```

## Usage
### Run `main.py`
```bash

```

### Run unittest
For example, we can run dataloader unittest test with:
```bash
python -m unittest test.DataLoaderTest.LoaderTestCase.test_loader
```
Run LongformerLarge model training with:
```bash
python -m unittest test.LongformerLargeTest.LongformerLargeTestCase.test_train
```

## Experiment Result
All the results are stored in `resource` folder, including all figures, prediction files and labels.
