#!/usr/bin/env python
# Created by "Thieu" at 10:09, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from itertools import permutations
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BF(Optimizer):
    """
    The Custom version of: Brute Force (BF)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.custom_based.CustomBF import BF
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1
    >>> model = BF(epoch)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_POS = 0  # position
    ID_TAR = 1  # fitness

    def __init__(self, epoch=1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        
    def initialize_variables(self):
        tmp_pop = []
        for pos in permutations(range(len(self.problem.ub))):
            pos = np.asarray(list(pos))
            target = self.get_target_wrapper(pos)
            tmp_pop.append([pos, target])
            
        self.pop = tmp_pop
        self.pop_size = len(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        self.pop = self.update_target_wrapper_population(self.pop)
