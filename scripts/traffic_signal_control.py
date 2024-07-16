import numpy as np
import cv2 as cv
import random
from car import Car

OBJECT_MAPPING = {
    "abandoned_area": 0,
    "traffic_light_red": -1,
    "traffic_light_green": 1,
    "streets": 2,
    "cars": 3
}

class TrafficSignalControl:
    def __init__(self, grid_size=15, num_traffic_lights=4) -> None:
        assert num_traffic_lights % 2 == 0, "number of traffic lights should be an even number"
        self.grid_size = grid_size
        self.num_traffic_lights = num_traffic_lights
        self.cars = list()
        self.traffic_lights_status = (0, 0)
        self.grid_without_cars = None
        self.create_grid(grid_size)

    def place_traffic_lights(self, grid):
        grid_centre = int(self.grid_size / 2)
        traffic_lights_range = int(self.num_traffic_lights / 2)

        grid[grid_centre-traffic_lights_range:grid_centre+traffic_lights_range, grid_centre+traffic_lights_range] = OBJECT_MAPPING["traffic_light_green"]
        grid[grid_centre+traffic_lights_range, grid_centre-traffic_lights_range:grid_centre+traffic_lights_range] = OBJECT_MAPPING["traffic_light_red"]

    def place_streets(self, grid):
        grid_centre = int(self.grid_size / 2)
        streets_range = int(self.num_traffic_lights / 2)

        grid[:self.grid_size, grid_centre-streets_range:grid_centre+streets_range] = OBJECT_MAPPING["streets"]
        grid[grid_centre-streets_range:grid_centre+streets_range, :self.grid_size] = OBJECT_MAPPING["streets"]
    
    def place_cars_in_grid(self, grid):
        grid_copy = grid.copy()
        for car in self.cars:
            grid_copy[car.get_position()] = OBJECT_MAPPING["cars"]
        
        return grid_copy

    def generate_car(self):
        grid_centre = int(self.grid_size / 2)
        streets_range = int(self.num_traffic_lights / 2)
        car_coordinate = random.choice(["x", "y"])
        car_position = (0,0)

        if car_coordinate == "x":
            random_initial_y_position = random.choice(list(range(grid_centre-streets_range, grid_centre+streets_range)))
            car_position = (random_initial_y_position, self.grid_size-1)

        elif car_coordinate == "y":
            random_initial_x_position = random.choice(list(range(grid_centre-streets_range, grid_centre+streets_range)))
            car_position = (self.grid_size-1, random_initial_x_position)

        car_already_in_this_position = False
        for car in self.cars:
            if car.get_position() == car_position:
                car_already_in_this_position = True

        if not car_already_in_this_position:
            self.cars.append(Car(car_coordinate, car_position))
        else:
            self.generate_car()

    def move_cars_by_one_step(self):
        # Create a new list of cars that are still within bounds
        self.cars = [car for car in self.cars if not (car.next_step()[0] < -1 or car.next_step()[1] < -1)]

        for car in self.cars:
            car_next_position = car.next_step()
            next_position_value_in_grid = self.grid_without_cars[car_next_position]

            if next_position_value_in_grid == OBJECT_MAPPING["traffic_light_red"] or next_position_value_in_grid == OBJECT_MAPPING["cars"]:
                car.increment_waiting_time()
            else:
                if next_position_value_in_grid == OBJECT_MAPPING["traffic_light_green"]:
                    car.go_to_next_step()
                car.go_to_next_step()
            
    def create_grid(self, grid_size):
        assert grid_size > 10, "grid size must be bigger than 10"

        if grid_size % 2 == 0:
            grid_size += 1
            print("Adding one to grid size so that it has odd dimensions.")

        grid = np.full((grid_size, grid_size), fill_value=OBJECT_MAPPING["abandoned_area"])
        self.place_streets(grid)
        self.place_traffic_lights(grid)
        self.grid_without_cars = grid

        # print(self.grid_without_cars)

if __name__ == "__main__":
    TSC = TrafficSignalControl()

