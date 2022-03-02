import argparse
import sys
import datetime
from model.DebertaV3Large import DebertaV3Large
from model.DebertaV2XLarge import DebertaV2XLarge
from model.DebertaBase import DebertaBase
from model.DebertaLarge import DebertaLarge
from model.XLNet import XLNet
from model.Longformer import Longformer
from loader.official import OfficialLoader
from util.analyze import DataAnalyseTestCase
from util.opt import ThresholdOptimizer

parser = argparse.ArgumentParser(description='Run')
parser.add_argument('--train', type=int, default=0,
                    help='1 run training then testing; 0 return cached testing results')
parser.add_argument('--model_name', type=str, default='DebertaV2XLarge', help='model name')
parser.add_argument('--data_type', type=str, default='clean_upsample', help='precessed data type')
args = parser.parse_args()
model_names = ['DeBERTaV3Large', 'DeBERTaV2XLarge', 'DeBERTaBase', 'DeBERTaLarge', 'XLNet', 'Longformer']
loader_types = ['clean_upsample', 'synonym_clean_upsample', 'plain_upsample']


def get_loader(loader_type, model_name):
    if loader_type not in loader_types:
        print(f'Please use a valid loader type, valid types are:\n{loader_types}')
        sys.exit(1)
    data_loader = OfficialLoader(model_name)
    if loader_type == 'clean_upsample':
        data_loader.split_upsample_cleaned()
    elif loader_type == 'synonym_clean_upsample':
        data_loader.split_upsample_cleaned_synonym()
    else:
        data_loader.split_upsample()
    return data_loader


def get_model(model_name, data_loader):
    if model_name not in model_names:
        print(f'Please use a valid model name, valid names are:\n{model_names}')
        sys.exit(1)
    if model_name == 'DebertaV3Large':
        nlp_model = DebertaV3Large(data_loader)
    elif model_name == 'DeBERTaV2XLarge':
        nlp_model = DebertaV2XLarge(data_loader)
    elif model_name == 'DeBERTaBase':
        nlp_model = DebertaBase(data_loader)
    elif model_name == 'DeBERTaLarge':
        nlp_model = DebertaLarge(data_loader)
    elif model_name == 'XLNet':
        nlp_model = XLNet(data_loader)
    else:
        nlp_model = Longformer(data_loader)
    return nlp_model


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    loader = get_loader(args.data_type, args.model_name)
    nlp_model = get_model(args.model_name, loader)
    if args.train:
        nlp_model = get_model(args.model_name, loader)
        nlp_model.train()
    else:
        DataAnalyseTestCase.test_all_label()
    endtime = datetime.datetime.now()
    print(f"[Total Time]:{(endtime - starttime).seconds / 60:.4f} minutes")
