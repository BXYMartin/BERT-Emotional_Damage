import numpy as np
import os
import logging
import warnings
import GPyOpt
from sklearn.metrics import precision_score, recall_score, f1_score


class ThresholdOptimizer:
    def __init__(self, loader):
        warnings.filterwarnings('ignore')
        self.data_loader = loader
        self.prob_path = os.path.join(self.data_loader.storage_folder, self.data_loader.prob_dir, self.data_loader.prob_filename)
        self.label_path = os.path.join(self.data_loader.storage_folder, self.data_loader.prob_dir, self.data_loader.label_filename)
        self.probs = np.loadtxt(self.prob_path)
        self.labels = np.loadtxt(self.label_path)
        # format digits to probabilities
        self.probs = 1 / (1 + np.exp(-self.probs))
        self.bounds = [{'name': 'threshold', 'type': 'continuous', 'domain': (0, 1)}]
        self.optimizer = GPyOpt.methods.BayesianOptimization(f=self.optimizer_step,
                                                             domain=self.bounds,
                                                             model_type='GP',
                                                             acquisition_type='EI',
                                                             maximize=True)

    def run(self, iteration=2000, time=6000, step=1e-5):
        self.optimizer.run_optimization(iteration, time, step)
        self.optimizer.plot_convergence(filename=os.path.join(self.data_loader.storage_folder, self.data_loader.prob_dir, "convergence.png"))
        self.optimal = self.optimizer.x_opt
        logging.info(f"Obtained optimal threshold {self.optimal}: F1-Score {self.optimizer_step(np.array([self.optimal]))}.")


    def optimizer_step(self, params):
        threshold = params[0]
        prediction = np.zeros(self.labels.shape)
        for index in range(self.labels.shape[0]):
            if self.probs[index, 1] > threshold:
                prediction[index] = 1
        precision = precision_score(self.labels, prediction)
        recall = recall_score(self.labels, prediction)
        f1 = f1_score(self.labels, prediction)
        logging.info(f"Optimizer step: Threshold {threshold}, Precision {precision}, Recall {recall}, F1-Score {f1}")
        return f1
        
