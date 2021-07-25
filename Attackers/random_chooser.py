import numpy as np
import torch

class random_chooser:
    def __init__(self, list):
        self.name = 'random choice ' + str(list)
        self.choice_range = list
        self.last_choice = None

    def __str__(self):
        return self.name

    def choice(self, size=1, dtype=torch.float):
        self.last_choice = torch.tensor(np.random.choice(self.choice_range, size=size), dtype=dtype)
        return self.last_choice

    def __eq__(self, other):
        if not isinstance(other, random_chooser):
            return False

        if self.choice_range == other.choice_range:
            return True

        return False