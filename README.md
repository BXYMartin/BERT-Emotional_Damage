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
python -u [--train int] [--model_name str] [--data_type type]
```
* [--train ] the default value is `0`
  * `1`: run training then testing 
  * `0`: return cached testing results of our final model: DeBERTaV2XLarge
* [--model_name ] determines which model to use and the default value is `DeBERTaV2XLarge`
  * The value can be [`DeBERTaV3Large`, `DeBERTaV2XLarge`, `DeBERTaBase`, `DeBERTaLarge`, `XLNet`, `Longformer`]
* [--data_type ] determines which type of data to use and the default value is `clean_upsample`
  * `clean_upsample`: Upsampled data without extra quotation marks
  * `synonym_clean_upsample`: Upsampled data without extra quotation marks uses synonym data augmentation technique
  * `plain_upsample`: Upsampled data

For example, you can train a `DeBERTaV2XLarge` model using `clean_upsample` data using:
```bash
python -u main.py --train 1 --model_name DeBERTaV2XLarge --data_type clean_upsample
```
**Note:** If you use `DeBERTaV2XLarge` which is our final model, an extra Bayesian 
Optimization process will be executed to maximum the model performance.
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
