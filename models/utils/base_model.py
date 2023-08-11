import os
import time as tm
import numpy as np
import pandas as pd
from copy import deepcopy

from .node import Node
from .my_connection import MyConnection

class BaseModel(object):
    
    ID_WX1 = 0 # List ID - Probability for Popular
    ID_WX2 = 1 # List ID - Probability to Cost
    ID_WX3 = 2 # List ID - Probability for Time
    
    def __init__(self, base_params=None, model_params=None):
        self.model = base_params['model']
        self.minmax = "min" if base_params['minmax'] == None else base_params['minmax']
        self.maut_weights = {
            "popular" : 1,
            "cost" : 1,
            "time" : 1
        } if base_params['maut_weights'] == None else base_params['maut_weights']
        self.time_start = 8 if base_params['time_start'] == None else int(base_params['time_start'])
        self.time_end = 20 if base_params['time_end'] == None else int(base_params['time_end'])
        
        self.conn = MyConnection()
        self.time_start, self.time_end = self._convert_time2sec_(self.time_start), self._convert_time2sec_(self.time_end)
        
        self.hotel_node = self.conn.get_hotel_node_by_id(base_params["hotel_id"])
        self._tour_initialisation_(base_params["tour_ids"])
        self.n_days = 3 if base_params["n_days"] == None else int(base_params["n_days"])
        # self.print_train = base_params['print_train']
        self._model_params = model_params
        self._base_params = base_params
        
        self.best_solution = []
        self.outlier_solution = []
        self.params_logger = []
        
        self.log_to = base_params["log_to"]
        
    def _tour_initialisation_(self, tour_ids=None):
        self.tour_nodes = self.conn.get_tour_nodes_by_ids(tour_ids)
        self.tmp_tour_nodes = deepcopy(self.tour_nodes)
        self.n_cities = len(self.tour_nodes)

    def _check_time_start2end_(self, hotel_node:Node, tour_nodes:'list(Node)', position:'list(int)'):
        solution_nodes = np.asarray([tour_nodes[i] for i in position])
        accepted = []

        total_time = self.time_start

        start_node_accepted = False
        for i in range(len(solution_nodes)):
            temp_total_time = self.conn.get_hotel_dist_matrix(origin_id=hotel_node.id, dest_id=solution_nodes[i].id, hotel2tour=True)
            start_node_accepted = self._check_time_open_((total_time+temp_total_time), solution_nodes[i])
            if start_node_accepted:
                accepted.append(True)
                total_time += temp_total_time
                break
            else:
                accepted.append(False)

        for i in range(len(accepted), len(solution_nodes)):
            temp_total_time = self.conn.get_tour_dist_matrix(origin_id=solution_nodes[(i-1)].id, dest_id=solution_nodes[i].id)+solution_nodes[(i-1)].spend_time
            if not self._check_time_open_((total_time+temp_total_time), solution_nodes[i]):
                accepted.append(False)
            else:
                accepted.append(True)
                total_time += temp_total_time

            if total_time > self.time_end:
                break
        
        solution_nodes = solution_nodes[:len(accepted)]
        solution_nodes = solution_nodes[:len(accepted)]
        solution_nodes = list(solution_nodes[accepted])
        solution_position = position[:len(accepted)]
        solution_position = list(solution_position[accepted])
        
        # if len(solution_position) > 1 and total_time > self.time_end:
        #     total_time -= self.conn.get_tour_dist_matrix(origin_id=solution_nodes[-2].id, dest_id=solution_nodes[-1].id)+solution_nodes[-2].spend_time
        #     solution_nodes.pop(-1)
        #     solution_position.pop(-1)
        
        if len(solution_position) > 2 and total_time > self.time_end:
            for _ in range(0, len(solution_nodes)):
                total_time -= self.conn.get_tour_dist_matrix(origin_id=solution_nodes[-2].id, dest_id=solution_nodes[-1].id)+solution_nodes[-2].spend_time
                solution_nodes.pop(-1)
                solution_position.pop(-1)
                tmp_total_time = self.conn.get_hotel_dist_matrix(origin_id=solution_nodes[-1].id, dest_id=hotel_node.id, hotel2tour=False)+solution_nodes[-1].spend_time
                total_time += tmp_total_time
                
                if total_time <= self.time_end:
                    break
                
                total_time -= tmp_total_time
            
        return solution_position
    
    def check_time(self, solution):
        return self._check_time_start2end_(self.hotel_node, self.tour_nodes, solution)
    
    def _fitness_func_(self, x:float=0.0, low_best=True):
        """Fitness Function
        
        Parameters
        ----------
        
        x : float, default=0.0
            Input function.
        
        high_best: bool, default=True
            If True, return x. If Else, return 1.0-x.
            
        Returns
        -------
        
        fitness : float 
        """
        return x if low_best else 1.0-x
    
    def _multi_attribute_utility_theory_(self, weights:'list(float)', arr:'list(float)'):
        """Multi Attribute Utility Theory (MAUT)
        
        This method has custom with :func:`_fitness_func_`. 
        
        f(w,x) = w1*x1+w2*x2+w3*x3/3
        
        Parameters
        ----------
        
        weights : list of float, default=None
            weight for MAUT function, each element is representation for each element in arr
        
        arr: list of float, default=None
            arr is value
            
        Returns
        -------
        
        MAUT calculation : float
            return fitness calculation from MAUT
        """
        assert len(weights) == len(arr), "weights and arr must be same length"
        fx1 = weights[self.ID_WX1]*arr[self.ID_WX1]
        fx2 = weights[self.ID_WX2]*arr[self.ID_WX2]
        fx3 = weights[self.ID_WX3]*arr[self.ID_WX3]
        return sum([fx1,fx2,fx3])/3.
    
    def _normalize_(self, x, min_value:float=None, max_value:float=None):
        if min_value == max_value : return 1
        try:
            if isinstance(x, list):
                return [round((i - min_value) / (max_value - min_value), 15) for i in x]

            return round((x - min_value) / (max_value - min_value), 15)
        except ZeroDivisionError:
            return 0.0
    
    def _calculate_popular_fitness_(self, solution_nodes:'list(Node)', min_value:float, max_value:float):
        return self._normalize_(sum([x.rating for x in solution_nodes])/len(solution_nodes), min_value, max_value)
    
    def _calculate_cost_fitness_(self, solution_nodes:'list(Node)', min_value:float, max_value:float):
        return self._normalize_(sum([x.tarif for x in solution_nodes])/len(solution_nodes), min_value, max_value)
    
    def _calculate_time_fitness_(self, hotel_node:Node, solution_nodes:'list(Node)', minmax_dict:dict):
        tour2tour_normalize = lambda x:self._normalize_(x, minmax_dict["tour"]["min"], minmax_dict["tour"]["max"])
        tour2tour_time = [tour2tour_normalize(self.conn.get_tour_dist_matrix(origin_id=solution_nodes[(i-1)].id, dest_id=solution_nodes[i].id)+solution_nodes[(i-1)].spend_time) for i in range(1, len(solution_nodes))]
        
        try:
            tour2tour_time = sum(tour2tour_time)/(len(solution_nodes)-1)
        except ZeroDivisionError:
            tour2tour_time = 0.0
            
        hotel2tour_time = self.conn.get_hotel_dist_matrix(origin_id=hotel_node.id, dest_id=solution_nodes[0].id, hotel2tour=True)
        hotel2tour_time = self._normalize_(hotel2tour_time, minmax_dict["hotel2tour"]["min"], minmax_dict["hotel2tour"]["max"])
        
        tour2hotel_time = self.conn.get_hotel_dist_matrix(origin_id=solution_nodes[-1].id, dest_id=hotel_node.id, hotel2tour=False)+solution_nodes[-1].spend_time
        tour2hotel_time = self._normalize_(tour2hotel_time, minmax_dict["tour2hotel"]["min"], minmax_dict["tour2hotel"]["max"])
        
        
        return sum([tour2tour_time, hotel2tour_time, tour2hotel_time])/3.0
    
    def _calculate_fitness_(self, hotel_node:Node, tour_nodes:'list(Node)', position:'list(int)'):
        accepted_posittion = self.check_time(position)
        solution_nodes = [tour_nodes[i] for i in accepted_posittion]
        
        popular_fit = self._calculate_popular_fitness_(solution_nodes=solution_nodes, min_value=self.conn.attribute_minmax["popular_minmax"]["min"], max_value=self.conn.attribute_minmax["popular_minmax"]["max"])
        popular_fit = self._fitness_func_(popular_fit, low_best=False if self.minmax == "min" else True)
        
        cost_fit = self._calculate_cost_fitness_(solution_nodes=solution_nodes, min_value=self.conn.attribute_minmax["cost_minmax"]["min"], max_value=self.conn.attribute_minmax["cost_minmax"]["max"])
        cost_fit = self._fitness_func_(cost_fit, low_best=True if self.minmax == "min" else False)
        
        time_fit = self._calculate_time_fitness_(hotel_node=hotel_node, solution_nodes=solution_nodes, minmax_dict=self.conn.attribute_minmax["time_minmax"])
        time_fit = self._fitness_func_(time_fit, low_best=True if self.minmax == "min" else False)
        
        return self._multi_attribute_utility_theory_(weights=[self.maut_weights["popular"],self.maut_weights["cost"],self.maut_weights["time"]], arr=[popular_fit,cost_fit,time_fit])
    
    def _convert_time2sec_(self, time):
        if isinstance(time, list):
            return time[0]*3600 + time[1]*60
        return time*3600
    
    def _check_time_open_(self, total_time, tour_node:Node):
        return total_time >= self._convert_time2sec_(tour_node.jam_buka) and total_time < self._convert_time2sec_(tour_node.jam_tutup)
        
    
    # def _check_time_start2end_(self, hotel_node:Node, tour_nodes:'list(Node)', position:'list(int)'):
    #     solution_nodes = [tour_nodes[i] for i in position]
    #     accepted = True

    #     total_time = self.time_start
    #     total_time += self.conn.get_hotel_dist_matrix(origin_id=hotel_node.id, dest_id=solution_nodes[0].id, hotel2tour=True)

    #     sliced = 1
    #     for i in range(1, len(solution_nodes)):
    #         if not self._check_time_open_(total_time, solution_nodes[(i-1)]):
    #             accepted = False
    #             break
            
    #         total_time += self.conn.get_tour_dist_matrix(origin_id=solution_nodes[(i-1)].id, dest_id=solution_nodes[i].id)+solution_nodes[(i-1)].spend_time
    #         sliced = i+1

    #         if total_time > self.time_end:
    #             break

    #     solution_nodes = solution_nodes[:sliced]
    #     if accepted:
    #         for _ in range(0, len(solution_nodes)):
    #             total_time -= self.conn.get_tour_dist_matrix(origin_id=solution_nodes[-2].id, dest_id=solution_nodes[-1].id)+solution_nodes[-2].spend_time
    #             solution_nodes.pop(-1)
    #             tmp_total_time = self.conn.get_hotel_dist_matrix(origin_id=solution_nodes[-2].id, dest_id=hotel_node.id, hotel2tour=False)+solution_nodes[-2].spend_time
    #             total_time += tmp_total_time
                
    #             if total_time <= self.time_end:
    #                 break
                
    #             total_time -= tmp_total_time
            
    #     return position[:len(solution_nodes)], accepted
        
    def generate_stable_solution(self, solution, lb=None, ub=None):
        # print(f"Raw: {solution}")
        ## Bring them back to boundary
        solution = np.clip(solution, lb, ub)

        solution_set = set(list(range(0, len(solution))))
        solution_done = np.array([-1, ] * len(solution))
        solution_int = solution.astype(int)
        city_unique, city_counts = np.unique(solution_int, return_counts=True)

        ### Way 1: Stable, not random
        for idx, city in enumerate(solution_int):
            if solution_done[idx] != -1:
                continue
            if city in city_unique:
                solution_done[idx] = city
                city_unique = np.where(city_unique == city, -1, city_unique)
            else:
                list_cities_left = list(solution_set - set(city_unique) - set(solution_done))
                # print(list_cities_left)
                solution_done[idx] = list_cities_left[0]
        # print(f"What: {solution_done}")
        return solution_done
    
    def fitness_function(self, solution):
        try:
            return self._calculate_fitness_(self.hotel_node, self.tour_nodes, solution)
        except ZeroDivisionError:
            return 1.0 if self.minmax == "min" else 0.0
                    
    
    
    def params_tuner(self, model_params_tuning:dict=None, num_test:int=5, log_directory:str="logs/params", file_name:str=None):
        IDXNODE, IDXFITNESS, IDXDAY, IDXTIME, IDXSCORE = 2, 3, 4, 5, 6
        tmp_log_to = self.log_to
        self.log_to = None
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            
        if isinstance(model_params_tuning, dict):
            tmp_model_params = deepcopy(self._model_params)
            keys = model_params_tuning.keys()
            val = list(model_params_tuning.values())
            combination = np.array(np.meshgrid(*val)).T.reshape(-1, len(keys))
            
            best_model_params = None
            
            if file_name == None:
                file_name = f"tuning"
            
            file_name = f"{file_name}_{tm.strftime('%Y%m%d-%H%M%S')}"
            
            for idx, c in enumerate(combination):
                print("-"*50,f"{idx+1}/{len(combination)}","-"*50)
                tmp_params = {k:v.item() for k, v in zip(keys, c)}
                tmp_model_params.update(tmp_params)
                tmp_params_logger = [deepcopy(tmp_model_params), tmp_params, None, None, None, None, None]
                
                print(f"Parameter Training - {tmp_model_params}")
                tpl_node = []
                tpl_fitness = []
                tpl_day = []
                tpl_time = []
                for i in range(num_test):
                    # print(f"Iteration {i} Start")
                    time_start = tm.time()
                    self.train(tmp_model_params)
                    time_end = tm.time()
                    tpl_node.append(len(np.hstack([x[0] for x in self.best_solution])))
                    tmp_fits = np.hstack([x[1] for x in self.best_solution])
                    tpl_fitness.append(sum(tmp_fits)/len(tmp_fits))
                    tpl_day.append(len(self.best_solution))
                    tpl_time.append(time_end-time_start)
                    
                    print(f"Iteration {i} End - Node Count: {tpl_node[-1]}, AVG Fitness: {tpl_fitness[-1]}, Day Count: {tpl_day[-1]}, Time Elapse: {tpl_time[-1]}")
                
                tmp_params_logger[IDXNODE] = sum([x for x in tpl_node if x != 0])/num_test
                tmp_params_logger[IDXFITNESS] = sum([x for x in tpl_fitness if x != 0])/num_test
                tmp_params_logger[IDXDAY] = sum([x for x in tpl_day if x != 0])/num_test
                tmp_params_logger[IDXTIME] = sum([x for x in tpl_time if x != 0])/num_test
                print(f"Result - Node Count: {tmp_params_logger[IDXNODE]}, AVG Fitness: {tmp_params_logger[IDXFITNESS]}, Day Count: {tmp_params_logger[IDXDAY]}, Time Elapse: {tmp_params_logger[IDXTIME]}\n")
                self.params_logger.append(tmp_params_logger)
                
                tmp_avg_node = [x[IDXNODE] for x in self.params_logger]
                tmp_avg_fitness = [x[IDXFITNESS] for x in self.params_logger]
                tmp_avg_day = [x[IDXDAY] for x in self.params_logger]
                tmp_avg_time_process = [x[IDXTIME] for x in self.params_logger]
                for x in self.params_logger:
                    ww = self._fitness_func_(self._normalize_(x[IDXNODE], min(tmp_avg_node), max(tmp_avg_node)), True)
                    xx = self._fitness_func_(self._normalize_(x[IDXFITNESS], min(tmp_avg_fitness), max(tmp_avg_fitness)), True)
                    yy = self._fitness_func_(self._normalize_(x[IDXDAY], min(tmp_avg_day), max(tmp_avg_day)), False)
                    zz = self._fitness_func_(self._normalize_(x[IDXTIME], min(tmp_avg_time_process), max(tmp_avg_time_process)), False)
                    param_score = sum([ww,xx,yy,zz])/4
                    x[IDXSCORE] = param_score
                    if best_model_params == None or best_model_params[IDXSCORE] < x[IDXSCORE]:
                        best_model_params = x
                        
                file = pd.DataFrame(self.params_logger, columns=["full_params_tuned", "params_tuned", "avg_node", "avg_fitness", "avg_day", "avg_time_process", "param_score"])
                with open(f"{log_directory}/{file_name}.csv", "w") as f:
                    file.to_csv(f)
        
        self.log_to = tmp_log_to


    def experiment(self, 
                   model_params:dict=None, 
                   maut_weights:dict=None,
                   max_tour_ids:list=None,
                   num_test_epoch_list:list=[10, 15, 25, 50, 75, 100, 150, 250, 500, 750], 
                   num_test:int=5,
                   log_directory:str="logs/experiment",
                   file_name:str=None):
        tmp_log_to = self.log_to
        self.log_to = None
        self.best_solution = []
        IDXNODE, IDXFITNESS, IDXDAY, IDXTIME, IDXSCORE = 0, 1, 2, 3, 4
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            
        if file_name == None:
            file_name = f"experiment"
            
        file_name = f"{file_name}_{tm.strftime('%Y%m%d-%H%M%S')}"
            
        if not isinstance(model_params, dict):
            model_params = self._model_params
            
        tmp_maut_weights = self.maut_weights
        if isinstance(maut_weights, dict):
            self.maut_weights = maut_weights
            
        prob_experiment_name = "epoch"
        prob_experiment = num_test_epoch_list
        if max_tour_ids != None:
            prob_experiment_name = "max_tour"
            prob_experiment = max_tour_ids
            
        log_experiment = {}
        for idx, te in enumerate(prob_experiment):
            if max_tour_ids == None:
                model_params["epoch"] = te
            else:
                self._tour_initialisation_(list(range(0, te)))
            local_log_experiment = [None, None, None, None, None] # Node, Fitness, Day, Time, Score
            print("-"*50,f"{idx+1}/{len(prob_experiment)}","-"*50)
            print(f"Experiment Start {te} {prob_experiment_name}")

            tpl_node = []
            tpl_fitness = []
            tpl_day = []
            tpl_time = []
            for i in range(num_test):
                time_start = tm.time()
                self.train(model_params)
                time_end = tm.time()
                tpl_node.append(len(np.hstack([x[0] for x in self.best_solution])))
                tmp_fits = np.hstack([x[1] for x in self.best_solution])
                tpl_fitness.append(sum(tmp_fits)/len(tmp_fits))
                tpl_day.append(len(self.best_solution))
                tpl_time.append(time_end-time_start)
                
                print(f"Iteration {i} End - Node Count: {tpl_node[-1]}, AVG Fitness: {tpl_fitness[-1]}, Day Count: {tpl_day[-1]}, Time Elapse: {tpl_time[-1]}")

            local_log_experiment[IDXNODE] = sum([x for x in tpl_node if x != 0])/num_test
            local_log_experiment[IDXFITNESS] = sum([x for x in tpl_fitness if x != 0])/num_test
            local_log_experiment[IDXDAY] = sum([x for x in tpl_day if x != 0])/num_test
            local_log_experiment[IDXTIME] = sum([x for x in tpl_time if x != 0])/num_test
            print(f"Result - Node Count: {local_log_experiment[IDXNODE]}, AVG Fitness: {local_log_experiment[IDXFITNESS]}, Day Count: {local_log_experiment[IDXDAY]}, Time Elapse: {local_log_experiment[IDXTIME]}\n")
            log_experiment[te] = local_log_experiment
            
            tmp_avg_node = [x[IDXNODE] for x in list(log_experiment.values())]
            tmp_avg_fitness = [x[IDXFITNESS] for x in list(log_experiment.values())]
            tmp_avg_day = [x[IDXDAY] for x in list(log_experiment.values())]
            tmp_avg_time_process = [x[IDXTIME] for x in list(log_experiment.values())]
            for key, val in log_experiment.items():
                ww = self._fitness_func_(self._normalize_(val[IDXNODE], min(tmp_avg_node), max(tmp_avg_node)), True)
                xx = self._fitness_func_(self._normalize_(val[IDXFITNESS], min(tmp_avg_fitness), max(tmp_avg_fitness)), True)
                yy = self._fitness_func_(self._normalize_(val[IDXDAY], min(tmp_avg_day), max(tmp_avg_day)), False)
                zz = self._fitness_func_(self._normalize_(val[IDXTIME], min(tmp_avg_time_process), max(tmp_avg_time_process)), False)
                score = sum([ww,xx,yy,zz])/4
                log_experiment[key][IDXSCORE] = score
                    
            file = pd.DataFrame(list(log_experiment.values()), columns=["avg_node", "avg_fitness", "avg_day", "avg_time_process", "param_score"], index=list(log_experiment.keys()))
            file.index.name = prob_experiment_name
            with open(f"{log_directory}/{file_name}.csv", "w") as f:
                file.to_csv(f)
            
        self.log_to = tmp_log_to
        self.maut_weights = tmp_maut_weights
        if max_tour_ids != None:
            self._tour_initialisation_(self._base_params["tour_ids"])
    
    
    def train(self, model_params=None):
        if not isinstance(model_params, dict):
            model_params = self._model_params
        self.tour_nodes = deepcopy(self.tmp_tour_nodes)
        tmp_n_cities = self.n_cities
        day = 1
        self.best_solution = []
        self.outlier_solution = []
    
        while day <= self.n_days and tmp_n_cities > 1:
            LB = [0, ] * tmp_n_cities
            UB = [(tmp_n_cities - 0.01), ] * tmp_n_cities
            problem = {
                "fit_func": self.fitness_function,
                "lb": LB,
                "ub": UB,
                "minmax": self.minmax,        # Trying to find the minimum distance
                "log_to": self.log_to,        # console, None, file -> "log_file": "result.log"
                # "verbose":True,
                # "mode": "ES",
                # "quantity": 15,
                "amend_position": self.generate_stable_solution,
            }
            model = self.model(**model_params)
            current_best_position, current_best_fitness = model.solve(problem, mode="single") #mode="thread", n_workers=2
            # if current_best_fitness == 0.0: print(current_best_position, current_best_fitness, accepted_pos)
            if current_best_fitness != 0.0:
                accepted_pos = self.check_time(current_best_position)
                if len(accepted_pos) > 1: 
                    self.best_solution.append([[self.tour_nodes[i] for i in accepted_pos], current_best_fitness])
                    if self.log_to != None:
                        print(f"\nDay-{day} Current Best solution: {accepted_pos} - {[i.id for i in self.best_solution[-1][0]]}, Obj = Fitness MAUT ({self.minmax}): {current_best_fitness}")
                    day += 1
                else: self.outlier_solution.append([[self.tour_nodes[i] for i in accepted_pos], current_best_fitness])
                    
                tmp_pop_solution = deepcopy(list(accepted_pos))
                tmp_pop_solution.sort(reverse=True)
                
            else:
                self.outlier_solution.append([[self.tour_nodes[i] for i in current_best_position], current_best_fitness])
                    
                tmp_pop_solution = deepcopy(list(current_best_position))
                tmp_pop_solution.sort(reverse=True)

            for pos in tmp_pop_solution:
                self.tour_nodes.pop(pos)
            
            tmp_n_cities = len(self.tour_nodes)
            
        # if tmp_n_cities > 2 and self.n_days-day > 0:
        #     self.best_solution.append([list(self.tour_nodes), current_best_fitness])
        #     print(f"\nDay-{day} Current Best solution: {self.best_solution[-1]}")
        
        if tmp_n_cities != 0:
            self.outlier_solution.append([self.tour_nodes, 0.0])
            
        if self.log_to != None:
            print(f"\nFinal Best solution: {list(self.best_solution)}")
        
