import numpy as np
import random
from car import Car
from torch import nn as nn
import torch
from replay_memory import ReplayMemory
from torch.optim import AdamW
from tqdm import tqdm
import copy
import torch.nn.functional as F
import cv2 as cv

OBJECT_MAPPING = {
    "abandoned_area": 0,
    "traffic_light_red": -1,
    "traffic_light_green": 1,
    "streets": 2,
    "cars": 3
}

ACTIONS_MAPPING = {
    "0": (-1, -1),
    "1": (-1, 1),
    "2": (1, -1),
}

class TrafficSignalControl:
    def __init__(self, grid_size=15, num_traffic_lights=4, total_running_time=150, states_dim=4, cars_num=3) -> None:
        assert num_traffic_lights % 2 == 0, "number of traffic lights should be an even number"
        self.grid_size = grid_size
        self.num_traffic_lights = num_traffic_lights
        self.cars_num = cars_num
        self.traffic_lights_position = None
        self.cars = list()
        self.traffic_lights_status = (-1, -1)
        self.grid_without_cars = None
        self.total_running_time = total_running_time
        self.current_running_time = 0
        self.all_frames = list()
        self.states_dim = states_dim
        self.actions_num = len(ACTIONS_MAPPING)
        self.q_network = self.generate_network()
        self.target_q_network = copy.deepcopy(self.q_network).eval()

        self.create_grid(grid_size)

    def place_traffic_lights(self, grid, horizontal_status, vertical_status):
        grid_centre = int(self.grid_size / 2)
        traffic_lights_range = int(self.num_traffic_lights / 2)

        grid[grid_centre-traffic_lights_range:grid_centre+traffic_lights_range, grid_centre+traffic_lights_range] = horizontal_status
        grid[grid_centre+traffic_lights_range, grid_centre-traffic_lights_range:grid_centre+traffic_lights_range] = vertical_status
        self.traffic_lights_position = grid_centre+traffic_lights_range

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
            self.cars.append(Car(car_coordinate, car_position, self.traffic_lights_position, self.num_traffic_lights))
        else:
            if len(self.cars) < self.num_traffic_lights * (self.grid_size-1-grid_centre):
                self.generate_car()

    def policy(self, state, epsilon=0.):
        if torch.rand(1) < epsilon:
            return torch.randint(self.actions_num, (1, 1))
        else:
            av = self.q_network(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)
    
    def train_deep_sarsa(self, episodes, alpha=0.0005, batch_size=64, gamma=0.99, epsilon=0.2, update_every=10):
        optim = AdamW(self.q_network.parameters(), lr=alpha)
        memory = ReplayMemory()
        stats = {"MSE Loss": [], "Returns": []}

        progress_bar = tqdm(range(1, episodes+1), desc="Training", leave=True)

        for episode in progress_bar:
            state = self.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done = self.step(state, action)
                memory.insert([state, action, reward, done, next_state])

                if memory.can_sample(batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)

                    qsa_b = self.q_network(state_b).gather(1, action_b)

                    next_action_b = self.policy(next_state_b, epsilon)
                    next_qsa_b = self.target_q_network(next_state_b).gather(1, next_action_b)

                    target_b = reward_b + ~done_b * gamma * next_qsa_b

                    loss = F.mse_loss(qsa_b, target_b)

                    self.q_network.zero_grad()
                    loss.backward()
                    optim.step()

                    stats["MSE Loss"].append(loss.item())
                
                state = next_state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

             # Update the progress bar description with the average reward
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                avg_loss = np.mean(stats["MSE Loss"][-update_every:]) if stats["MSE Loss"] else 0
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f}, Avg Loss: {avg_loss:.4f})")

            if episode % 50 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        return stats

    def move_cars_by_one_step(self):
        if random.choice([True, False]):
            self.generate_car()
            self.generate_car()
        # Create a new list of cars that are still within bounds
        self.cars = [car for car in self.cars if not (car.next_step()[0] < -1 or car.next_step()[1] < -1)]
        rewards_list = [0]

        for car in self.cars:
            waited_time = car.get_waited_time()
            car_next_position = car.next_step()
            new_grid = self.place_cars_in_grid(self.grid_without_cars)
            next_position_value_in_grid = new_grid[car_next_position]

            if next_position_value_in_grid == OBJECT_MAPPING["traffic_light_red"] or next_position_value_in_grid == OBJECT_MAPPING["cars"]:
                car.increment_waiting_time()
                if (waited_time > 30):
                    rewards_list.append(-5)
                elif (waited_time > 20):
                    rewards_list.append(-4)
                elif (waited_time > 10):
                    rewards_list.append(-3)
                else:
                    rewards_list.append(-1)

            else:
                if next_position_value_in_grid == OBJECT_MAPPING["traffic_light_green"]:
                    car.set_is_in_junction()
                    car.go_to_next_step()
                    rewards_list.append(10)
                else:
                    car.go_to_next_step()
                    if (waited_time > 30):
                        rewards_list.append(5)
                    elif (waited_time > 20):
                        rewards_list.append(4)
                    elif (waited_time > 10):
                        rewards_list.append(3)
                    else:
                        rewards_list.append(1)

        for car1 in self.cars:
            for car2 in self.cars:
                if car1 == car2:
                    continue
                if car1.get_coordinate() !=  car2.get_coordinate():
                    if car1.get_is_in_junction() == car2.get_is_in_junction() == True:
                        rewards_list=[-200]
        
        return sum(rewards_list) / len(rewards_list)
            
    def create_grid(self, grid_size):
        assert grid_size > 10, "grid size must be bigger than 10"

        if grid_size % 2 == 0:
            grid_size += 1
            print("Adding one to grid size so that it has odd dimensions.")

        grid = np.full((grid_size, grid_size), fill_value=OBJECT_MAPPING["abandoned_area"])
        self.place_streets(grid)
        self.place_traffic_lights(grid, self.traffic_lights_status[0], self.traffic_lights_status[1])
        self.grid_without_cars = grid

    def next_step(self, action):
        self.traffic_lights_status = ACTIONS_MAPPING[str(action)]
        self.place_traffic_lights(self.grid_without_cars, self.traffic_lights_status[0], self.traffic_lights_status[1])
        reward = self.move_cars_by_one_step()
        next_state = self.get_current_state()
        self.current_running_time += 1
        done = self.current_running_time > self.total_running_time
        for car1 in self.cars:
            for car2 in self.cars:
                if car1 == car2:
                    continue
                if car1.get_coordinate() !=  car2.get_coordinate():
                    if car1.get_is_in_junction() == car2.get_is_in_junction() == True:
                        done = True

        return next_state, reward, done

    def generate_network(self):
        return nn.Sequential(
            nn.Linear(self.states_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.actions_num))
    
    def step(self, state, action):
        state = state.numpy().flatten()
        state = tuple(int(x) for x in state)
        action = action.item()
        next_state, reward, done = self.next_step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done


    def get_current_state(self):
        waiting_time_in_horizontal = 0
        waiting_time_in_vertical = 0
        is_car_from_x_in_junction = 0
        is_car_from_y_in_junction = 0
        for car in self.cars:
            if car.get_coordinate() == "x":
                waiting_time_in_horizontal += car.get_waited_time()
                if car.get_is_in_junction():
                    is_car_from_x_in_junction = 1
            else:
                waiting_time_in_vertical += car.get_waited_time()
                if car.get_is_in_junction():
                    is_car_from_y_in_junction = 1
        state = np.array([waiting_time_in_vertical, waiting_time_in_horizontal, is_car_from_y_in_junction, is_car_from_x_in_junction])
        return state
        

    def reset(self):
        self.current_running_time = 0
        self.cars = []
        for i in range(self.cars_num):
            self.generate_car()
        self.traffic_lights_status = (-1, -1)
        initial_state = np.array([0, 0, 0, 0])
        return torch.from_numpy(initial_state).unsqueeze(dim=0).float()

    def create_video_from_frames(self, frames, output_filename, fps=5):
        print("Generating the video...")
        if not frames:
            raise ValueError("The frames list is empty")

        # Get frame size from the first frame
        height, width, layers = frames[0].shape
        size = (width, height)

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(output_filename, fourcc, fps, size)

        for frame in frames:
            out.write(frame)

        out.release()



if __name__ == "__main__":
    TSC = TrafficSignalControl()

