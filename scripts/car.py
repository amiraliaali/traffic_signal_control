import random
class Car:
    def __init__(self, coordinate, initial_position_in_grid, num_available_cars=4) -> None:
        self.coordinate = coordinate
        self.position = initial_position_in_grid
        self.waited_time = 0
        self.car_num = random.choice([i+1 for i in range(num_available_cars)])

    def next_step(self):
        if self.coordinate == "x":
            return (self.position[0], self.position[1]-1)
        else:
            return (self.position[0]-1, self.position[1])
    
    def go_to_next_step(self):
        self.position = self.next_step()
    
    def get_position(self):
        return self.position
    
    def increment_waiting_time(self):
        self.waited_time += 1

    def get_waited_time(self):
        return self.waited_time
    
    def get_num(self):
        return self.car_num
    
    def get_coordinate(self):
        return self.coordinate