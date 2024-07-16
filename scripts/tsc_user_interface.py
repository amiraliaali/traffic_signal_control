import numpy as np
from traffic_signal_control import TrafficSignalControl, OBJECT_MAPPING
import cv2 as cv
import torch


class TrafficSignalControlUI(TrafficSignalControl):
    def __init__(self, frame_dim=(500, 500), grid_size=15, num_traffic_lights=4, frame_delay_in_ms=500, cars_num=3) -> None:
        super().__init__(grid_size=grid_size, num_traffic_lights=num_traffic_lights, cars_num=cars_num)
        assert cars_num <= num_traffic_lights*2, "Number of cars needs to be smaller than twice the traffic lights"
        self.frame_dim = self.adjust_frame_dim(frame_dim, grid_size)
        self.cell_size = 0
        self.frame_without_cars = None
        self.frame_delay = frame_delay_in_ms
        self.cars_num = cars_num
        self.generate_empty_frame()

    def adjust_frame_dim(self, frame_dim, grid_size):
        adjusted_dim = (
            (frame_dim[0] // grid_size) * grid_size,
            (frame_dim[1] // grid_size) * grid_size
        )
        return adjusted_dim

    def draw_lines(self, frame, cell_size):
        grid_centre = int(self.grid_size / 2)
        streets_range = int(self.num_traffic_lights / 2)
        for i in range(grid_centre-streets_range, grid_centre+streets_range+1):
            # horizontal line
            cv.line(
                frame,
                (0, i * cell_size),
                (self.frame_dim[1], i * cell_size),
                (255, 255, 255),
                cell_size // 15,
            )
            cv.line(
                frame,
                (i * cell_size, 0),
                (i * cell_size, self.frame_dim[0]),
                (255, 255, 255),
                cell_size // 15,
            )

    def draw_streets(self, frame, cell_size, grid):
        street = cv.imread("scripts/images/road.jpg", cv.IMREAD_UNCHANGED)
        street = cv.resize(street, (cell_size, cell_size))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == OBJECT_MAPPING["streets"]:
                    top_left_x = j * cell_size
                    bottom_right_x = top_left_x + cell_size
                    top_left_y = i * cell_size
                    bottom_right_y = top_left_y + cell_size
                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = street

    def draw_control_lights(self, frame, cell_size, grid):
        green_light = cv.imread("scripts/images/green_light.jpeg", cv.IMREAD_UNCHANGED)
        green_light = cv.resize(green_light, (cell_size, cell_size))
        red_light = cv.imread("scripts/images/red_light.jpeg", cv.IMREAD_UNCHANGED)
        red_light = cv.resize(red_light, (cell_size, cell_size))

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == OBJECT_MAPPING["traffic_light_green"] or grid[i, j] == OBJECT_MAPPING["traffic_light_red"]:
                    top_left_x = j * cell_size
                    bottom_right_x = top_left_x + cell_size
                    top_left_y = i * cell_size
                    bottom_right_y = top_left_y + cell_size
                    if grid[i, j] == OBJECT_MAPPING["traffic_light_green"]:
                        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = green_light
                    else:
                        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = red_light
    
    def draw_cars(self, frame, cell_size, grid):
        frame = frame.copy()
        car_1_h = cv.imread("scripts/images/cars/car1_horizontal.png", cv.IMREAD_UNCHANGED)
        car_1_h = cv.resize(car_1_h, (cell_size, cell_size))[:, :, :3]
        car_1_v = cv.imread("scripts/images/cars/car1_vertical.png", cv.IMREAD_UNCHANGED)
        car_1_v = cv.resize(car_1_v, (cell_size, cell_size))[:, :, :3]

        car_2_h = cv.imread("scripts/images/cars/car2_horizontal.png", cv.IMREAD_UNCHANGED)
        car_2_h = cv.resize(car_2_h, (cell_size, cell_size))[:, :, :3]
        car_2_v = cv.imread("scripts/images/cars/car2_vertical.png", cv.IMREAD_UNCHANGED)
        car_2_v = cv.resize(car_2_v, (cell_size, cell_size))[:, :, :3]

        car_3_h = cv.imread("scripts/images/cars/car3_horizontal.png", cv.IMREAD_UNCHANGED)
        car_3_h = cv.resize(car_3_h, (cell_size, cell_size))[:, :, :3]
        car_3_v = cv.imread("scripts/images/cars/car3_vertical.png", cv.IMREAD_UNCHANGED)
        car_3_v = cv.resize(car_3_v, (cell_size, cell_size))[:, :, :3]

        car_4_h = cv.imread("scripts/images/cars/car4_horizontal.png", cv.IMREAD_UNCHANGED)
        car_4_h = cv.resize(car_4_h, (cell_size, cell_size))[:, :, :3]
        car_4_v = cv.imread("scripts/images/cars/car4_vertical.png", cv.IMREAD_UNCHANGED)
        car_4_v = cv.resize(car_4_v, (cell_size, cell_size))[:, :, :3]

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == OBJECT_MAPPING["cars"]:
                    top_left_x = j * cell_size
                    bottom_right_x = top_left_x + cell_size 
                    top_left_y = i * cell_size
                    bottom_right_y = top_left_y + cell_size 

                    for car in self.cars:
                        if car.get_position() == (i, j):
                            if car.get_coordinate() == "x":
                                if car.get_num() == 1:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_1_h
                                elif car.get_num() == 2:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_2_h
                                elif car.get_num() == 3:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_3_h
                                else:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_4_h
                            else:
                                if car.get_num() == 1:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_1_v
                                elif car.get_num() == 2:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_2_v
                                elif car.get_num() == 3:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_3_v
                                else:
                                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = car_4_v
        return frame

    def render(self, frame):
        cv.imshow("frame", frame)
        cv.waitKey(self.frame_delay)
        cv.destroyAllWindows()

    def generate_empty_frame(self):
        maze_frame = np.ones((*self.frame_dim, 3), dtype=np.uint8) * 255
        cell_size = self.frame_dim[0] // self.grid_size
        self.create_grid(self.grid_size)

        self.draw_streets(maze_frame, cell_size, self.grid_without_cars)

        self.draw_lines(
            maze_frame, cell_size
        )
        self.cell_size = cell_size
        self.frame_without_cars = maze_frame

    def test_run(self, training_episodes, output_filename):
        maze_frame = self.frame_without_cars

        self.draw_control_lights(maze_frame, self.cell_size, self.grid_without_cars)

        for i in range(self.cars_num):
            self.generate_car()
        
        self.train_deep_sarsa(episodes=training_episodes)
        self.test_agent(np.array([0, 0]))
        self.create_video_from_frames(self.all_frames, output_filename)

    def test_agent(self, state):
        next_state = state
        self.reset()
        end = False
        while not end:
            frame_copy = np.copy(
                self.frame_without_cars
            )  # Create a new copy for each iteration

            state_tensor = np.array(next_state)
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(dim=0).float()
            action = torch.argmax(self.q_network(state_tensor)).item()

            next_state, reward, end = self.next_step(action)
            self.draw_control_lights(frame_copy, self.cell_size, self.grid_without_cars)
            new_grid = self.place_cars_in_grid(self.grid_without_cars)
            frame = self.draw_cars(frame_copy, self.cell_size, new_grid)
            self.render(frame)  # Render the frame_copy
            self.all_frames.append(frame)
            if end:
                print()

if __name__ == "__main__":
    tsc = TrafficSignalControlUI(frame_dim=(700, 700), grid_size=15, num_traffic_lights=4, frame_delay_in_ms=500, cars_num=4)
    tsc.test_run(750, "output_video.mp4")

