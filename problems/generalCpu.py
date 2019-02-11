# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import sys
sys.path.append('./../utils')
import solutionmanager as sm
from gridsearch import GridSearch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import random
import time


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        self.hidden_size_2 = self.hidden_size
        self.lr = solution.lr
        self.activation_hidden = solution.activation_hidden
        self.activation_output = solution.activation_output

        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear1_2 = nn.Linear(self.hidden_size, self.hidden_size_2)
        self.linear1_3 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.linear1_4 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.linear2 = nn.Linear(self.hidden_size_2, output_size)
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_2 = nn.BatchNorm1d(self.hidden_size_2, track_running_stats=False)
        self.batch_norm1_3 = nn.BatchNorm1d(self.hidden_size_2, track_running_stats=False)
        self.batch_norm1_4 = nn.BatchNorm1d(self.hidden_size_2, track_running_stats=False)
        self.batch_norm2 = nn.BatchNorm1d(output_size, track_running_stats=False)

    def forward(self, x):
        activations = self.solution.activations
        
        x = self.linear1(x)
        x = activations.get(self.activation_hidden)(x)
        x = self.batch_norm1(x)
        
        x = self.linear1_2(x)
        x = activations.get(self.activation_hidden)(x)
        x = self.batch_norm1_2(x)
        
        x = self.linear1_3(x)
        x = activations.get(self.activation_hidden)(x)
        x = self.batch_norm1_3(x)

        x = self.linear1_4(x)
        x = activations.get(self.activation_hidden)(x)
        x = self.batch_norm1_4(x)

        x = self.linear2(x)
        x = activations.get(self.activation_output)(x)

        return x


class Solution():
    def __init__(self):
        self.best_step = sys.maxsize
        self.sols = {}
        self.solsSum = {}
        self.hidden_size = 50
        self.lr = 0.01
        self.activation_hidden = 'relu6'
        self.activation_output = 'sigmoid'
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'rrelu0205': nn.RReLU(0.2, 0.5),
            'htang1': nn.Hardtanh(-1, 1),
            'htang2': nn.Hardtanh(-2, 2),
            'htang3': nn.Hardtanh(-3, 3),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'hardshrink': nn.Hardshrink(),
            'leakyrelu01': nn.LeakyReLU(0.1),
            'leakyrelu001': nn.LeakyReLU(0.01),
            'logsigmoid': nn.LogSigmoid(),
            'prelu': nn.PReLU(),
        }
        self.hidden_size_grid = [16, 20, 26, 32, 36, 40, 45, 50, 54]
        self.lr_grid = [0.0001, 0.001, 0.005, 0.01, 0.1, 1]

        # self.lr_grid = [0.1, .5, 1, 1.5, 2, 3, 5, 10]

        # self.activation_hidden_grid = list(self.activations.keys())
        # self.activation_output_grid = list(self.activations.keys())
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        criterion = F.binary_cross_entropy
        # optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        while True:
            time_left = context.get_timer().get_time_left()
            key = "{}_{}_{}_{}".format(self.lr, self.hidden_size, self.activation_hidden, self.activation_output)
            # No more time left, stop training
            if time_left < 0.1 or (model.solution.grid_search.enabled and step > 100):
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                self.sols[key] = -1
                break
            if key in self.sols and self.sols[key] == -1:
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            if total == correct:
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                if step < 21:
                    self.best_step = step
                    loss = criterion(output, target)
                    self.print_stats(step, loss, correct, total, model)
                    print("{:.4f}".format(float(self.solsSum[key])/self.sols[key]))
                    return step
            # calculate loss
            loss = criterion(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, step, loss, correct, total, model):
        print("LR={}, HS={}, ActivHidden={}, ActivOut={}, Step = {} Prediction = {}/{} Error = {}".format(model.lr,
                                                                                                              model.hidden_size, model.activation_hidden, model.activation_output, step, correct, total, loss.item()))

###
###
# Don't change code after this line
###
###


class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0


class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i >> j) & 1
                data[i, j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()


# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
