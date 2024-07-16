import random
class Car:
    def __init__(self, coordinate, initial_position_in_grid, traffic_light_position, junction_size, num_available_cars=4) -> None:
        self.coordinate = coordinate
        self.position = initial_position_in_grid
        self.initial_position = initial_position_in_grid
        self.waited_time = 0
        self.is_in_junction = False
        self.in_junction_counter = 0
        self.junction_size = junction_size
        self.car_num = random.choice([i+1 for i in range(num_available_cars)])
        self.traffic_light_position = traffic_light_position

    def next_step(self):
        if self.coordinate == "x":
            return (self.position[0], self.position[1]-1)
        else:
            return (self.position[0]-1, self.position[1])
    
    def go_to_next_step(self):
        self.position = self.next_step()
        if self.is_in_junction:
            if self.in_junction_counter > self.junction_size:
                self.is_in_junction = False
                self.in_junction_counter = 0
            self.in_junction_counter += 1

    def set_is_in_junction(self):
        self.is_in_junction = True

    def get_is_in_junction(self):
        return self.is_in_junction
    
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

    def get_initial_position(self):
        return self.initial_position
    
    def move_to_initial_position(self):
        self.position = self.initial_position

    def reset_waiting_time(self):
        self.waited_time = 0
    
    def reset(self):
        self.move_to_initial_position()
        self.reset_waiting_time()

    def has_passed_traffic_light(self):
        if self.coordinate == "x":
            return self.position[1] < self.traffic_light_position
        else:
            return self.position[0] < self.traffic_light_position