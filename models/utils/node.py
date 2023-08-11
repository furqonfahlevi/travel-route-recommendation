class Node(object):
    def __init__(self, id=None, name=None, dest_type=None, lat=None, long=None, spend_time=None, rating=None, tarif=None, jam_buka=None, jam_tutup=None):
        self.id = id
        self.name = name
        self.dest_type = dest_type
        self.lat = lat
        self.long = long
        self.spend_time = spend_time
        self.rating = rating
        self.tarif = tarif
        self.jam_buka = jam_buka
        self.jam_tutup = jam_tutup
        
    def __repr__(self):
       return f"Place id: {self.id} -> Name: {self.name}, Type: {self.dest_type}, Rating: {self.rating}, Cost: {self.tarif}, Spend Time (s): {self.spend_time}, Spend Time (s): {self.jam_buka}-{self.jam_tutup}"