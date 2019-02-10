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
        self.lr = solution.lr
        self.activation_hidden = solution.activation_hidden
        self.activation_output = solution.activation_output

        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        activations = self.solution.activations
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = activations.get(self.activation_hidden)(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = activations.get(self.activation_output)(x)
        return x


class Solution():
    def __init__(self):
        self.hidden_size = 11
        self.lr = .1
        self.activation_hidden = 'leakyrelu001'
        self.activation_output = 'tanh'
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            # 'rrelu0103': nn.RReLU(0.1, 0.3),
            # 'rrelu0205': nn.RReLU(0.2, 0.5),
            'htang1': nn.Hardtanh(-1, 1),
            # 'htang2': nn.Hardtanh(-2, 2),
            # 'htang3': nn.Hardtanh(-3, 3),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            # 'selu': nn.SELU(),
            # 'hardshrink': nn.Hardshrink(),
            # 'leakyrelu01': nn.LeakyReLU(0.1),
            'leakyrelu001': nn.LeakyReLU(0.01),
            'logsigmoid': nn.LogSigmoid(),
            # 'prelu': nn.PReLU(),
        }
        # self.hidden_size_grid = [3, 7, 11, 13,  15, 19]
        # self.lr_grid = [0.01, 0.05, 0.1]
        self.hidden_size_grid = [3, 7, 11, 15, 20, 25]
        self.lr_grid = [0.001, 0.01, 0.1, 1, 10, 100]
        self.activation_hidden_grid = list(self.activations.keys())
        # self.activation_output_grid = list(self.activations.keys())
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(True)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1 or (model.solution.grid_search.enabled and step > 1000):
                break
            optimizer = optim.SGD(model.parameters(), lr=model.lr)
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
                return step
            # calculate loss
            loss = ((output-target)**2).sum()
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total, model)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, step, loss, correct, total, model):
        if step % 200 == 0: #and loss.item() < 0.1:
        # if step % 1000 == 0:
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
