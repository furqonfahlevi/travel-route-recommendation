from copy import deepcopy
from utils.base_model import BaseModel
from mealpy.swarm_based.FFA import OriginalFFA
            
if __name__ == '__main__':
    ## Setting parameters
    base_params = {
        "model": OriginalFFA,
        "maut_weights" : {
            "popular" : 0,
            "cost" : 1,
            "time" : 0
        },
        "time_start": 8,
        "time_end": 20,
        
        "n_days": 50,
        "hotel_id": 5,
        # "tour_ids": [0, 2, 4, 6, 7, 8, 9, 20, 23, 25, 29, 30, 31, 32, 33, 35, 40, 55, 42, 56],
        "tour_ids": list(range(0, 100)),
        
        "minmax": "max",
        "log_to":"console",
    }
    model_params = {
        "epoch": 50,           # iteration
        "pop_size": 25,        # population size
        "c1": 2.05,
        "c2": 2.05,
        "w_min": 0.4,
        "w_max": 0.9,
    }
    model = BaseModel(base_params=base_params, model_params=model_params)# ----Experiment
    max_tour_ids = [5, 15, 30, 45, 60, 75, 100]
    num_test_epoch_list = [10, 25, 50, 75, 100, 150, 250]
    # num_test_epoch_list = [10, 15, 25, 50, 75, 100, 150, 250, 500, 750]
    num_test = 5
    log_directory = "logs/experiment/pso_ffa/ffa"
    file_name = "ffa_weight_p0_c1_t0_Ndays_30epoch_tour"
    model.experiment(
        max_tour_ids=max_tour_ids,
        num_test_epoch_list=num_test_epoch_list,
        num_test=num_test,
        log_directory=log_directory,
        file_name=file_name
    )



