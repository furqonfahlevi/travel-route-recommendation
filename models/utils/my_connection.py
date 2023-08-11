import json
import os

from .node import Node

class MyConnection(object):
    def __init__(self):
        self._load_data_()
        self.attribute_minmax = self._get_minmax_attribute()
        
    def _load_data_(self):
        tour_path = "../../data/tour.json"
        tour_dist_path = "../../data/distance_matrix_tour.json"
        hotel_path = "../../data/hotels.json"
        hotel2tour_dist_path = "../../data/distance_matrix_hotels2tour.json"
        tour2hotel_dist_path = "../../data/distance_matrix_tour2hotels.json"
        
        with open(tour_path, 'r') as f:
            self.data_tour = json.load(f)
        with open(tour_dist_path, 'r') as f:
            self.data_tour_dist = json.load(f)
        with open(hotel_path, 'r') as f:
            self.data_hotel = json.load(f)
        with open(hotel2tour_dist_path, 'r') as f:
            self.data_hotel2tour_dist = json.load(f)
        with open(tour2hotel_dist_path, 'r') as f:
            self.data_tour2hotel_dist = json.load(f)
            
    def _get_minmax_attribute(self):
        tmp_rating = [x.rating for x in self.get_tour_nodes_by_ids(range(len(self.data_tour)))]
        tmp_cost = [x.tarif for x in self.get_tour_nodes_by_ids(range(len(self.data_tour)))]
        tmp_data_tour_dist = []
        for i, data in enumerate(self.data_tour_dist):
            spend_time = self.data_tour[i]["time_spent"]
            for j in data:
                if j["distance_matrix"] != None:
                    tmp_data_tour_dist.append(j['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"]+spend_time)

        tmp_data_hotel2tour_dist = []            
        for i in self.data_hotel2tour_dist:
            for j in i:
                if j["distance_matrix"] != None:
                    tmp_data_hotel2tour_dist.append(j['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"])
            
        tmp_data_tour2hotel_dist = []         
        for i, data in enumerate(self.data_tour2hotel_dist):
            spend_time = self.data_tour[i]["time_spent"]
            for j in data:
                if j["distance_matrix"] != None:
                    tmp_data_tour2hotel_dist.append(j['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"]+spend_time)
        
        return {
            "popular_minmax":{
                "min":0,
                # "min":min(tmp_rating),
                "max":max(tmp_rating),
            },
            "cost_minmax":{
                "min":0,
                # "min":min(tmp_cost),
                "max":max(tmp_cost),
            },
            "time_minmax":{
                "tour":{
                    "min":min(tmp_data_tour_dist),
                    "max":max(tmp_data_tour_dist),
                },
                "hotel2tour":{
                    "min":min(tmp_data_hotel2tour_dist),
                    "max":max(tmp_data_hotel2tour_dist),
                },
                "tour2hotel":{
                    "min":min(tmp_data_tour2hotel_dist),
                    "max":max(tmp_data_tour2hotel_dist),
                },
            }
        }
            
    def get_tour_dist_matrix(self, origin_id, dest_id):
        try:
            return float(self.data_tour_dist[origin_id][dest_id]['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"])
        except:
            raise Exception(f"get_tour_dist_matrix {origin_id}, {dest_id}")
    
    def get_hotel_dist_matrix(self, origin_id, dest_id, hotel2tour=True):
        try:
            return float(self.data_hotel2tour_dist[origin_id][dest_id]['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"]) \
                if hotel2tour else float(self.data_tour2hotel_dist[origin_id][dest_id]['distance_matrix']["rows"][0]["elements"][0]["duration"]["value"])
        except:
            raise Exception(f"get_hotel_dist_matrix hotel2tour:{hotel2tour}-{origin_id}, {dest_id}")
            
    def get_tour_nodes_by_ids(self, ids:'list(int)'):
        nodes = list()
        for idx in ids:
            # if idx == 43:
            #     continue
            try:
                id = idx
                name = self.data_tour[idx]["place_name"]
                dest_type = "Tourist Attraction"
                lat = self.data_tour[idx]["gps_coordinates"]["latitude"]
                long = self.data_tour[idx]["gps_coordinates"]["longitude"]
                spend_time = self.data_tour[idx]["time_spent"]
                rating = self.data_tour[idx]["rating"]
                tarif = self.data_tour[idx]["tarif_weekday"]
                # print(self.data_tour[idx]["operating_hours"]["sunday"][0])
                jam_tmp = self.data_tour[idx]["operating_hours"]["sunday"] if self.data_tour[idx]["operating_hours"]["sunday"][0] != "Closed" \
                    else self.data_tour[idx]["operating_hours"]["tuesday"]
                jam_buka = jam_tmp[0]
                jam_tutup = jam_tmp[1]
                node = Node(id=id, name=name, dest_type=dest_type, lat=lat, long=long, spend_time=spend_time, rating=rating, tarif=tarif, jam_buka=jam_buka, jam_tutup=jam_tutup)
                nodes.append(node)
            except:
                print(idx)
            
        return nodes
    
    def get_hotel_node_by_id(self, id:'int'):
        id = id
        name = self.data_hotel[id]["place_name"]
        dest_type = "Hotel"
        lat = self.data_hotel[id]["gps_coordinates"]["latitude"]
        long = self.data_hotel[id]["gps_coordinates"]["longitude"]
        rating = self.data_hotel[id]["rating"]
        node = Node(id=id, name=name, dest_type=dest_type, lat=lat, long=long, rating=rating)
        
        return node