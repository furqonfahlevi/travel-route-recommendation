from copy import deepcopy
from utils.base_model import BaseModel
from mealpy.custom_based.CustomBF import BF
            
if __name__ == '__main__':
    ## Setting parameters
    base_params = {
        "model": BF,
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
        "epoch": 1,           # iteration
    }
    model = BaseModel(base_params=base_params, model_params=model_params)# ----Experiment
    max_tour_ids = [5,8,11]
    num_test_epoch_list = [10, 25, 50, 75, 100, 150, 250]
    # num_test_epoch_list = [10, 15, 25, 50, 75, 100, 150, 250, 500, 750]
    num_test = 1
    log_directory = "logs/experiment/bf"
    file_name = "bf_weight_p0_c1_t0_Ndays_30epoch_tour"
    model.experiment(
        max_tour_ids=max_tour_ids,
        # num_test_epoch_list=num_test_epoch_list,
        num_test=num_test,
        log_directory=log_directory,
        file_name=file_name
    )



