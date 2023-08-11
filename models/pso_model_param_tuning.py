from copy import deepcopy
from utils.base_model import BaseModel
from mealpy.swarm_based.PSO import OriginalPSO

if __name__ == '__main__':
    ## Setting parameters
    base_params = {
        "model": OriginalPSO,
        "maut_weights" : {
            "popular" : 1,
            "cost" : 1,
            "time" : 1
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
        "epoch": 30,           # iteration
        "pop_size": 50,        # population size
        "c1": 2.05,
        "c2": 2.05,
        "w_min": 0.4,
        "w_max": 0.9,
    }
    model = BaseModel(base_params=base_params, model_params=model_params)# ----Tuning
    model_params_tune = {
        "epoch": [15, 30, 50],           # iteration
        "pop_size": [25, 50, 100],        # population size
        "c1": 2.05,
        "c2": 2.05,
        "w_min": 0.4,
        "w_max": 0.9,   # Uncomment jika mau tuning parameternya
    }
    num_test = 5
    log_directory = "logs/params/pso"
    file_name = "pso"
    model.params_tuner(
        model_params_tuning=model_params_tune,
        num_test=num_test,
        log_directory=log_directory,
        file_name=file_name
    )



