import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import logging
import sys

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 80  # pixels
# HEIGHT = 5  # grid height
# WIDTH = 5  # grid width

IMAGE_ICON_SIZE = 25

GO_UP = 0
GO_DOWN = 1
GO_LEFT = 2
GO_RIGHT = 3

STEP_PENALTY = -1
INVALID_STEP_PENALTY = -10


class Victim(object):
    def __init__(self, victim_id, reward):
        self.id = victim_id
        self.pos = None
        self.resource_id = None
        self.rescued = False
        self.reward = reward

        if self.reward < 0:
            self.rescued = True

    def get_id(self):
        return self.id

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def set_resource_id(self, resource_id):
        if self.resource_id is not None:
            raise Exception("Resource id for this victim has been assigned to " + self.resource_id)

        self.resource_id = resource_id

    def get_resource_id(self):
        return self.resource_id

    def set_rescued(self):
        self.rescued = True

    def reset_rescued(self):
        self.rescued = False

    def is_rescued(self):
        return self.rescued

    def get_reward(self):
        return self.reward

class Env(tk.Tk):
    def __init__(self, max_agent_count, max_victim_count, info):
        super(Env, self).__init__()

        # Environment info
        self.env_info = info["env"]

        self.Ny = self.env_info["Ny"]
        self.Nx = self.env_info["Nx"]

        self.WIDTH = self.Nx
        self.HEIGHT = self.Ny
        # ********* ***********

        self.action_space = ['u', 'd', 'l', 'r']
        self.max_a_count = max_agent_count
        self.max_v_count = max_victim_count
        self.n_actions = len(self.action_space)

        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(self.WIDTH * UNIT, self.HEIGHT * UNIT))
        self.shapes = self.load_images()

        self.canvas = self._build_canvas()

        # self._reset_agents()
        self.texts = []
        self.agents = []
        self.agent_positions = {}

        self.victims = []
        self.victim_positions = {}

        # ************* from other file ******************


        # State and action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int)
        self.N_actions = len(self.action_coords)

        self.state_dim = (self.Ny, self.Nx)  # tuple of integers
        self.action_dim = (self.N_actions,)  # tuple of integers

        self.state_size = np.prod(np.array(list(self.state_dim), dtype=np.int))  # integer
        self.action_size = np.prod(np.array(list(self.action_dim), dtype=np.int))  # integer

        # Check
        if len(self.action_dict.keys()) != self.N_actions:
            raise IOError("Error: Inconsistent action dimensions!")

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=self.HEIGHT * UNIT,
                           width=self.WIDTH * UNIT)
        # create grids
        for c in range(0, self.WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, self.HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, self.HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        return canvas

    def pack_canvas(self):
        self.canvas.pack()

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((IMAGE_ICON_SIZE, IMAGE_ICON_SIZE)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((IMAGE_ICON_SIZE, IMAGE_ICON_SIZE)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((IMAGE_ICON_SIZE, IMAGE_ICON_SIZE)))

        return rectangle, triangle, circle

    # return state of all agents
    def reset_n(self):
        self.update()
        time.sleep(0.5)

        # set random position for agents
        self.reset_agents()
        #
        self.reset_victims_state()

        self.render()

        # get current state from agent positions
        return self.starting_state()

    # def reset_victims(self):
    #     for v in self.victims:
    #         pos = self.set_victim_random_position(v)
    #         x_coord = self.get_row_center_pixel(pos)
    #         y_coord = self.get_column_center_pixel(pos)
    #         v.set_position(pos)
    #         v.reset_rescued()
    #
    #         self.canvas.coords(v.get_resource_id(), [x_coord, y_coord])

    def reset_victims_state(self):
        for v in self.victims:
            if v.get_reward() > 0:
                v.reset_rescued()

    # ===================
    # Starting and terminal state
    # ===================
    def starting_state(self):

        self.reset_agents()

        return self.current_state()

    def current_state(self):
        state_n = []
        for a in self.agents:
            pos = a.get_position()
            row = self.get_row(pos)
            col = self.get_col(pos)
            state_n.append([row, col])

        return state_n

    def reset_agents(self):
        states = {}
        for a in self.agents:
            initial_pos = a.get_initial_position()
            a.set_position(initial_pos)
            a.reset_rescued_victims()

            x_coord = self.get_column_center_pixel(initial_pos)
            y_coord = self.get_row_center_pixel(initial_pos)

            self.canvas.coords(a.get_resource_id(), [x_coord, y_coord])

            states[a.get_id()] = [self.get_row(initial_pos), self.get_col(initial_pos)]

        return states

    def agent_step_collaborative(self, agent, action, state_n):
        i = 0

    def _calculate_step(self, agent, action):
        agent_resource_id = agent.get_resource_id()
        state = self.canvas.coords(agent_resource_id)
        base_action = np.array([0, 0])

        reward = 0

        if action == GO_UP:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
            else:
                reward = INVALID_STEP_PENALTY
        elif action == GO_DOWN:  # down
            if state[1] < (self.HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
            else:
                reward = INVALID_STEP_PENALTY

        elif action == GO_LEFT:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
            else:
                reward = INVALID_STEP_PENALTY
        elif action == GO_RIGHT:  # right
            if state[0] < (self.WIDTH - 1) * UNIT:
                base_action[0] += UNIT
            else:
                reward = INVALID_STEP_PENALTY

        return base_action, reward

    def get_reward_for_agent(self, agent):
        pos = agent.get_position()
        r = self.get_row(pos)
        c = self.get_col(pos)

        # if self.hit_walls(r, c):
        #     raise Exception('invalid position, agent=' + str(agent.get_id()))

        reward = STEP_PENALTY

        for v in self.victims:
            # if v.get_reward() < 0:
            #     continue
            if v.is_rescued():
                continue

            v_pos = v.get_position()
            v_r = self.get_row(v_pos)
            v_c = self.get_col(v_pos)

            if r == v_r and c == v_c:
                reward = v.get_reward()
                if reward > 0:
                    v.set_rescued()
                break

        return reward

    def agent_step(self, agent, action):
        agent_resource_id = agent.get_resource_id()

        next_state, shift_row, shift_col = agent.perform_action(action)
        reward = self.get_reward_for_agent(agent)

        # base_action, reward = self._calculate_step(agent, action)
        self.render()
        # move agent - with canvas, we must convert to canvas coordintes
        shift_x = shift_col * UNIT
        shift_y = shift_row * UNIT

        self.canvas.move(agent_resource_id, shift_x, shift_y)
        # move rectangle to top level of canvas
        self.canvas.tag_raise(agent_resource_id)

        next_state = self.canvas.coords(agent_resource_id)
        x_coord = int(next_state[0])
        y_coord = int(next_state[1])

        row = self.get_row_from_coord(y_coord)
        col = self.get_col_from_coord(x_coord)

        pos = self.get_pos_from_row_and_col(row, col)
        if pos != agent.get_position():
            raise Exception('Invalid move causing wrong position, agent=' + str(agent.get_id()))

        done = self.hit_walls(row, col)

        return [row, col], reward, done

    def is_terminated(self):
        for v in self.victims:
            if not v.is_rescued():
                return False

        return True

    def render(self):
        time.sleep(0.03)
        self.update()

    def _contain_agent(self, agent):
        for a in self.agents:
            if a.get_id() == agent.get_id():
                return True

        return False

    # get agent object. False otherwise
    def get_agent(self, agent_id):
        for a in self.agents:
            if agent_id == a.get_id():
                return a

        return False

    def get_agents(self):
        return self.agents

    def get_victims(self):
        return self.victims

    def add_agent_at_pos(self, agent, pos):
        if self._contain_agent(agent):
            return False

        if len(self.agents) >= self.max_a_count:
            return False

        pos = self.set_agent_at_position(agent, pos)

        agent.set_initial_position(pos)

        # add image
        r_pixel = self.get_row_center_pixel(pos)
        c_pixel = self.get_column_center_pixel(pos)

        resource_id = self.canvas.create_image(r_pixel, c_pixel, image=self.shapes[0])
        agent.set_resource_id(resource_id)

        self.agents.append(agent)

        return agent

    def add_agent_at_row_col(self, agent, row, col):
        pos = self.get_pos_from_row_and_col(row, col)

        return self.add_agent_at_pos(agent, pos)

    # agent position in the form (row, col)
    def allowed_agent_actions(self, agent_row, agent_col, agent_id):
        actions = []

        agent = self.get_agent(agent_id)

        last_action = agent.get_last_action()
        # move up
        out_bound = self.out_of_bound(agent_row-1, agent_col)
        if not out_bound and last_action != GO_DOWN:
            actions.append(GO_UP)
        # move down
        out_bound = self.out_of_bound(agent_row + 1, agent_col)
        if not out_bound and last_action != GO_UP:
            actions.append(GO_DOWN)

        # test move left
        out_bound = self.out_of_bound(agent_row, agent_col-1)
        if not out_bound and last_action != GO_RIGHT:
            actions.append(GO_LEFT)

        # test move right
        out_bound = self.out_of_bound(agent_row, agent_col + 1)
        if not out_bound and last_action != GO_LEFT:
            actions.append(GO_RIGHT)

        # # test move up
        # hit = self.hit_walls(agent_row-1, agent_col)
        # if not hit:
        #     actions.append(GO_UP)
        #
        # # test move down
        # hit = self.hit_walls(agent_row + 1, agent_col)
        # if not hit:
        #     actions.append(GO_DOWN)
        #
        # # test move left
        # hit = self.hit_walls(agent_row, agent_col-1)
        # if not hit:
        #     actions.append(GO_LEFT)
        #
        # # test move right
        # hit = self.hit_walls(agent_row, agent_col + 1)
        # if not hit:
        #     actions.append(GO_RIGHT)

        return np.array(actions)

    def out_of_bound(self, row, col):
        if col < 0 or row < 0 or col >= self.WIDTH or row >= self.HEIGHT:
            return True

        return False

    def hit_walls(self, row, col):
        # if col < 0 or row < 0 or col >= self.WIDTH or row >= self.HEIGHT:
        #     return True
        #
        for v in self.victims:
            if v.get_reward() >= 0:
                continue

            # obstacles have negative rewards
            pos = v.get_position()
            v_row = self.get_row(pos)
            v_col = self.get_col(pos)
            if v_row == row and v_col == col:
                return True

        return False

    def allowed_actions(self, state_n):
        rows_major_cols_style = np.nonzero(state_n > 0)
        rows = rows_major_cols_style[0]
        cols = rows_major_cols_style[1]

        possible_actions = []
        i = 0
        for row, col in zip(rows, cols):
            acts = self.allowed_agent_actions(agent_row=row, agent_col=col, agent_id=i)
            possible_actions.append(acts)
            i += 1

        return possible_actions

    def add_agent(self, agent):
        if self._contain_agent(agent):
            return False

        if len(self.agents) >= self.max_a_count:
            return False

        pos = self.set_agent_random_position(agent)

        agent.set_initial_position(pos)

        # add image
        r_pixel = self.get_row_center_pixel(pos)
        c_pixel = self.get_column_center_pixel(pos)

        resource_id = self.canvas.create_image(r_pixel, c_pixel, image=self.shapes[0])
        agent.set_resource_id(resource_id)

        self.agents.append(agent)

        return agent

    def set_agent_random_position(self, agent):
        key = str(agent.get_id())
        if key in self.agent_positions.keys():
            del self.agent_positions[agent.get_id()]
        positions = self.agent_positions.values()

        pos = self._generate_new_position(positions)
        self.agent_positions[agent.get_id()] = pos
        agent.set_position(pos)

        return pos

    def set_agent_at_position(self, agent, pos):
        key = str(agent.get_id())
        if key in self.agent_positions.keys():
            del self.agent_positions[agent.get_id()]

        self.agent_positions[agent.get_id()] = pos
        agent.set_position(pos)

        return pos

    def set_victim_random_position(self, victim):
        if victim.get_id() in self.victim_positions:
            del self.victim_positions[victim.get_id()]

        positions = self.victim_positions.values()

        pos = self._generate_new_position(positions)
        self.victim_positions[victim.get_id()] = pos
        victim.set_position(pos)

        return pos

    def set_victim_position(self, victim, pos):
        if victim.get_id() in self.victim_positions:
            del self.victim_positions[victim.get_id()]

        self.victim_positions[victim.get_id()] = pos
        victim.set_position(pos)

        return pos

    def _generate_new_position(self, existing_positions):
        while True:
            pos = np.random.randint(0, self.WIDTH * self.HEIGHT)
            if pos not in existing_positions:
                return pos

    def add_victim_at_row_col(self, row, col, reward):

        pos = self.get_pos_from_row_and_col(row, col)

        return self.add_victim_at_pos(pos, reward)

    def add_victim_at_pos(self, pos, reward):
        if len(self.victims) > self.max_v_count:
            return False

        v = Victim(len(self.victims), reward)
        pos = self.set_victim_position(v, pos)

        # print("Victim", v.get_id(), "; pos=", v.get_position())

        # add image
        y_pixel = self.get_row_center_pixel(pos)
        x_pixel = self.get_column_center_pixel(pos)

        if reward >= 0:
            resource_id = self.canvas.create_image(x_pixel, y_pixel, image=self.shapes[2])
        else:
            resource_id = self.canvas.create_image(x_pixel, y_pixel, image=self.shapes[1])

        v.set_resource_id(resource_id)

        self.victims.append(v)

        return v

    def add_victim(self):
        if len(self.victims) > self.max_v_count:
            return False

        v = Victim(len(self.victims), 100)
        pos = self.set_victim_random_position(v)

        print("Victim", v.get_id(), "; pos=", v.get_position())

        # add image
        y_pixel = self.get_row_center_pixel(pos)
        x_pixel = self.get_column_center_pixel(pos)
        resource_id = self.canvas.create_image(x_pixel, y_pixel, image=self.shapes[2])
        v.set_resource_id(resource_id)

        self.victims.append(v)

        return v

    def get_row(self, pos):
        return pos // self.WIDTH

    def get_col(self, pos):
        return pos % self.WIDTH

    def get_row_center_pixel(self, pos):
        row = self.get_row(pos)

        return int(row*UNIT + UNIT / 2)

    def get_column_center_pixel(self, pos):
        col = self.get_col(pos)

        return int(col*UNIT + UNIT / 2)

    def get_row_from_coord(self, row_pixel):

        return int((row_pixel - UNIT / 2) / UNIT)


    def get_col_from_coord(self, col_pixel):

        return int((col_pixel - UNIT / 2) / UNIT)

    def get_pos_from_row_and_col(self, row, col):
        return row * self.WIDTH + col

    def get_pos_from_coords(self, coord_x, coord_y):
        row = self.get_row_from_coord(coord_y)
        col = self.get_col_from_coord(coord_x)

        return row * self.WIDTH + col

    def get_unrescued_victims(self):
        unrescued_victims = []
        for v in self.victims:
            if not v.is_rescued():
                unrescued_victims.append(v)

        return unrescued_victims

    # unrescued_victims_only: whether to compare distance with victims in unrescued list only or all victims
    def get_closest_victim(self, agent, unrescued_victims_only):

        unrescued_victims = self.get_unrescued_victims()

        if unrescued_victims_only == False:
            unrescued_victims = self.get_unreached_victims_by_agent(agent)

        min_distance = 100000000
        closest_victim = None

        for v in unrescued_victims:
            distance = self.distance(agent=agent, victim=v)
            if min_distance > distance:
                min_distance = distance
                closest_victim = v

        return closest_victim

    def get_unreached_victims_by_agent(self, agent):

        visited_victims = agent.get_rescued_victims()
        result = []

        for vic in self.victims:
            if not self.list_contain_item(visited_victims, vic):
                result.append(vic)

        return result

    def list_contain_item(self, object_list, item):
        for i in object_list:
            if i.get_id() == item.get_id():
                return True

        return False


    # coordinate distance
    def distance(self, agent, victim):
        a_x = self.get_column_center_pixel(agent.get_position())
        v_x = self.get_column_center_pixel(victim.get_position())

        a_y = self.get_row_center_pixel(agent.get_position())
        v_y = self.get_row_center_pixel(victim.get_position())

        return abs(a_x - v_x) + abs(a_y - v_y)


    # get step distance
    def step_distance(self, agent, victim):
        a_r = self.get_row(agent.get_position())
        v_r = self.get_row(victim.get_position())

        a_c = self.get_col(agent.get_position())
        v_c = self.get_col(victim.get_position())

        return abs(a_r - v_r) + abs(a_c - v_c)

    def action_to_go_to_victim(self, agent, victim):

        a_x = self.get_column_center_pixel(agent.get_position())
        v_x = self.get_column_center_pixel(victim.get_position())
        a_y = self.get_row_center_pixel(agent.get_position())
        v_y = self.get_row_center_pixel(victim.get_position())

        if a_x > v_x:
            return GO_LEFT
        elif a_x < v_x:
            return GO_RIGHT
        else:
            if a_y > v_y:
                return GO_UP
            elif a_y < v_y:
                return GO_DOWN

        raise Exception('Unkown situation: ax=' + str(a_x) + ";ay=" + str(a_y) + ";apos=" + str(agent.get_position()) + ";vx=" + str(v_x) + ";vy=" + str(v_y)  + ";vpos=" + str(victim.get_position()))

    @staticmethod
    def setup_custom_logger(name, level, file_name='log.txt'):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(file_name, mode='w')
        handler.setFormatter(formatter)
        # screen_handler = logging.StreamHandler(stream=sys.stdout)
        # screen_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        # logger.addHandler(screen_handler)

        return logger

    # support printing for 1 agent only
    def print_value_all(self, q_table):
        if len(self.agents) > 1:
            return

        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                for action in range(self.n_actions):
                    temp = q_table[(i, j, action)]
                    self.text_value(i, j, round(temp, 2), action)
                        #
                        # state = str([i, j])
                        # if state in q_table.keys():
                        #     temp = q_table[str(state)][action]
                        #     self.text_value(i, j, round(temp, 2), action)

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):

        if action == GO_UP:
            origin_x, origin_y = int(UNIT/2)-5, 1
        elif action == GO_DOWN:
            origin_x, origin_y = int(UNIT/2)-5, UNIT - 15
        elif action == GO_LEFT:
            origin_x, origin_y = 1, int(UNIT / 2) - 8
        elif action == GO_RIGHT:
            origin_x, origin_y = UNIT - 30, int(UNIT / 2) - 8
        else:
            origin_x, origin_y = 0, 0

        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def turn_back(self, last_action, action):
        if last_action == GO_UP and action == GO_DOWN:
            return True

        if last_action == GO_DOWN and action == GO_UP:
            return True

        if last_action == GO_LEFT and action == GO_RIGHT:
            return True

        if last_action == GO_RIGHT and action == GO_LEFT:
            return True

        return False
