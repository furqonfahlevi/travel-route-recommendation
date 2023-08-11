import sys, getopt
import yaml
import numpy as np
from prettytable import PrettyTable
from utils.base_model import BaseModel
from mealpy.swarm_based.CSO import OriginalCSO
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.evolutionary_based.CustomGA import GA
from mealpy.physics_based.SA import OriginalSA
from mealpy.custom_based.CustomBF import BF

def get_result(model):
    res = []
    for idx, day in enumerate(model.best_solution):
        res_perday = []
        res_perday.append(
            {
                "id":model.hotel_node.id,
                "name":model.hotel_node.name,
                "time_start":None,
                "time_end":(8,0),
                "cost":None,
                "dest_open":None,
                "dest_close":None,
                "dest_lat":model.hotel_node.lat,
                "dest_long":model.hotel_node.long,
            }
        )
        total_time = 8*3600
        tour_nodes = day[0]
        time = model.conn.get_hotel_dist_matrix(origin_id=model.hotel_node.id, dest_id=tour_nodes[0].id, hotel2tour=True)
        spend_time = 0
        total_time += time
        for i in range(len(tour_nodes)-1):
            jam_sampai_jam = int(np.floor(total_time/3600))
            jam_sampai_menit = int(np.round((total_time%3600)/60))
            time = 0
            spend_time = 0
            spend_time = tour_nodes[i].spend_time
            time = model.conn.get_tour_dist_matrix(origin_id=tour_nodes[i].id, dest_id=tour_nodes[i+1].id)+spend_time
            jam_berangkat_jam = int(np.floor((total_time+spend_time)/3600))
            jam_berangkat_menit = int(np.round(((total_time+spend_time)%3600)/60))
            total_time += time
            
            res_perday.append(
                {
                    "id":tour_nodes[i].id,
                    "name":tour_nodes[i].name,
                    "time_start":(jam_sampai_jam, jam_sampai_menit),
                    "time_end":(jam_berangkat_jam, jam_berangkat_menit),
                    "cost":tour_nodes[i].tarif,
                    "dest_open":tuple(tour_nodes[i].jam_buka),
                    "dest_close":tuple(tour_nodes[i].jam_tutup),
                    "dest_lat":tour_nodes[i].lat,
                    "dest_long":tour_nodes[i].long,
                }
            )
        
        jam_sampai_jam = int(np.floor(total_time/3600))
        jam_sampai_menit = int(np.round((total_time%3600)/60))
        spend_time = tour_nodes[-1].spend_time
        time = model.conn.get_hotel_dist_matrix(origin_id=model.hotel_node.id, dest_id=tour_nodes[-1].id, hotel2tour=False)+spend_time
        total_time+=time
        res_perday.append(
            {
                "id":model.hotel_node.id,
                "name":model.hotel_node.name,
                "time_start":(jam_sampai_jam, jam_sampai_menit),
                "time_end":None,
                "cost":None,
                "dest_open":None,
                "dest_close":None,
                "dest_lat":model.hotel_node.lat,
                "dest_long":model.hotel_node.long,
            }
        )
        res.append(res_perday)
        
    return res, np.hstack([x[0] for x in model.outlier_solution]) if len(model.outlier_solution) != 0 else None

def print_tour(tour):
    for day, tourperday  in enumerate(tour, start=1):
        table = PrettyTable()
        table.title = f"Day {day}"
        table.field_names = ["ID", "Name", "Waktu Tiba", "Waktu Keberangkatan", "Biaya", "Dest Open", "Dest Closed", "Dest Lat", "Dest Long"]
        for dest in tourperday:
            table.add_row(list(dest.values()))
          
        print(table)

def main(model:int, hotel_id:int, tour_ids:list, maut_weights:tuple, time_constraint:tuple=(8,20), n_days:int=3):
    """_summary_

    Args:
        model (int): 
        Model used for generating tour based on models_param.yaml
            0 -> Cat Swarm Optimization (CSO)
            1 -> Artificial Bee Colony (ABC)
            2 -> Simulated Annealing (SA)
            3 -> Genetic Algorithm (GA)
            4 -> Brute Force (BF)
        
        hotel_id (int): ID Hotel for start and finish the trip
        tour_ids (list): List of Destination
        maut_weights (tuple): Based on Multi Attribute Utility-Theory (MAUT) for used preferences (Popularity, Cost, Time)
        
        time_constraint (tuple, optional): Time Constraint. Defaults to (8,20).
        n_days (int, optional): Max Day to generate. Defaults to 3.

    Returns:
        List of Tour for N-Day, Outlier
    """
    with open("utils/models_param.yaml", "r") as f:
        model_loaded = yaml.load(f, Loader=yaml.FullLoader)
    
    base_params = {
        "model": eval(model_loaded[model]["model_package"]),
        "maut_weights" : {
            "popular" : maut_weights[0],
            "cost" : maut_weights[1],
            "time" : maut_weights[2]
        },
        "time_start": time_constraint[0],
        "time_end": time_constraint[1],
        
        "n_days": n_days,
        "hotel_id": hotel_id,
        "tour_ids": tour_ids,
        
        "minmax": "max",
        "log_to": None # None, "console"
    }
    
    print("-"*10, model_loaded[model]["model_name"], "-"*10)
    model = BaseModel(base_params=base_params, model_params=model_loaded[model]["params"])
    model.train()
    return get_result(model)
    
if __name__ == '__main__':
    model = None
    hotel_id = None
    tour_ids = None
    maut_weights = None
    time_constraints = (8,20)
    n_days = 3
    argumentList = sys.argv[1:]
 
    # Options
    options = "h:m:o:r:w:c:n:"
     
    # Long options
    # long_options = ["Help", "m", "hi", "ti", "mw", "tc", "nd"]
    long_options = ["Help", "model", "hotel_id", "tour_ids", "maut_weights", "time_constraints", "n_days"]
    
    try:
      # Parsing argument
      arguments, values = getopt.getopt(argumentList, options, long_options)
      for currentArgument, currentValue in arguments:
  
          if currentArgument in ("-h", "--Help"):
              print ("python running_models.py -m <model> -o <hotel_id> -r <tour_ids>, -w <maut_weight> -c <time_constraints> -n <n_days>")
          elif currentArgument in ("-m", "--model"):
              model = int(currentValue)
          elif currentArgument in ("-o", "--hotel_id"):
              hotel_id = int(currentValue)
          elif currentArgument in ("-r", "--tour_ids"):
              tour_ids = list([int(x) for x in currentValue.split(",")])
          elif currentArgument in ("-w", "--maut_weights"):
              maut_weights = tuple([int(x) for x in currentValue.split(",")])
          elif currentArgument in ("-c", "--time_constraints"):
              time_constraints = tuple([int(x) for x in currentValue.split(",")])
          elif currentArgument in ("-n", "--n_days"):
              n_days = int(currentValue)
        
    except getopt.error as err:
        print (str(err))
        
    tour, outlier = main(model, hotel_id, tour_ids, maut_weights, time_constraints, n_days)
    print_tour(tour)