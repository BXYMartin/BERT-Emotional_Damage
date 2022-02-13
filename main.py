from loader.official import OfficialLoader
from model.RoBERTa import RoBERTa

if __name__ == '__main__':
    data_loader = OfficialLoader("Official-RoBERTa-Test-Case")
    data_loader.split()
    data_loader.balance()
    model = RoBERTa(data_loader)
    model.train()
