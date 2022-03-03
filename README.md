# SemEval 2022 Task 4 SubTask 1: Patronizing and Condescending Language Detection
This is the natural language processing coursework repository for our team.. 

## Emotional Damage Team
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
* [--train ] the default value is `1`
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
**Note:** 
- If you use `DeBERTaV2XLarge` which is our final model, an extra Bayesian Optimization step will be executed to maximize the model performance. 
- Due to the randomness in the initialization of the model and the randomized batch sampler, the f1-score may be slightly lower than what we stated on the paper. Also, we performed early-stopping per iteration which is not used in consideration of time in this training process. If you run the command directly, the model will be trained on 1 epoch and the results are collected afterwards. 
- To reproduce our result, use early-stopping at about 4900 training iterations for batch size 3 (slightly less than 1 epoch) and train the model on full labelled dataset before invoking Bayesian Optimization methods.



### Run unittest
For example, we can run dataloader unittest test with:
```bash
python -m unittest test.DataLoaderTest.LoaderTestCase.test_loader
```
Run LongformerLarge model training with:
```bash
python -m unittest test.LongformerLargeTest.LongformerLargeTestCase.test_train
```

## Experiment result
All the results are stored in `resource` folder, including all figures, prediction files and labels.
