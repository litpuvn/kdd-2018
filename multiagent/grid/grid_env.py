import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width

GO_UP = 0
GO_DOWN = 1
GO_LEFT = 2
GO_RIGHT = 3

class Victim(object):
    def __init__(self, victim_id):
        self.id = victim_id
        self.pos = None
        self.resource_id = None
        self.rescued = False

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


class Env(tk.Tk):
    def __init__(self, max_agent_count, max_victim_count):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.max_a_count = max_agent_count
        self.max_v_count = max_victim_count
        self.n_actions = len(self.action_space)

        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()

        self.canvas = self._build_canvas()

        # self._reset_agents()
        self.texts = []
        self.agents = []
        self.agent_positions = {}

        self.victims = []
        self.victim_positions = {}

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        return canvas

    def pack_canvas(self):
        self.canvas.pack()

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    # return state of all agents
    def reset_n(self):
        self.update()
        time.sleep(0.5)

        # set random position for agents
        states = self.reset_agents()

        self.reset_victims_state()

        self.render()

        # get current state from agent positions
        return states

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
            v.reset_rescued()

    def reset_agents(self):
        states = {}
        for a in self.agents:
            initial_pos = a.get_initial_position()
            a.set_position(initial_pos)

            x_coord = self.get_column_center_pixel(initial_pos)
            y_coord = self.get_row_center_pixel(initial_pos)

            self.canvas.coords(a.get_resource_id(), [x_coord, y_coord])

            states[a.get_id()] = [x_coord, y_coord]

        return states

    def agent_step(self, agent, action):
        agent_resource_id = agent.get_resource_id()
        state = self.canvas.coords(agent_resource_id)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # move agent
        self.canvas.move(agent_resource_id, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(agent_resource_id)

        next_state = self.canvas.coords(agent_resource_id)
        x_coord = int(next_state[0])
        y_coord = int(next_state[1])

        row = self.get_row_from_coord(y_coord)
        col = self.get_col_from_coord(x_coord)

        pos = self.get_pos_from_row_and_col(row, col)
        agent.set_position(pos)

        unrescued_victims = self.get_unrescued_victims()
        reward = 0
        for v in unrescued_victims:
            if next_state == self.canvas.coords(v.get_resource_id()):
                reward = reward + 100
                v.set_rescued()

        # action does not save anyone will be discounted 100
        if reward == 0:
            reward = -100

        unrescued_victims = self.get_unrescued_victims()

        # done if all victims are rescued
        done = len(unrescued_victims) == 0

        return [x_coord, y_coord], reward, done

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

    def set_victim_random_position(self, victim):
        if victim.get_id() in self.victim_positions:
            del self.victim_positions[victim.get_id()]

        positions = self.victim_positions.values()

        pos = self._generate_new_position(positions)
        self.victim_positions[victim.get_id()] = pos
        victim.set_position(pos)

        return pos

    def _generate_new_position(self, existing_positions):
        while True:
            pos = np.random.randint(0, WIDTH * HEIGHT)
            if pos not in existing_positions:
                return pos

    def add_victim(self):
        if len(self.victims) > self.max_v_count:
            return False

        v = Victim(len(self.victims))
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
        return pos // WIDTH

    def get_col(self, pos):
        return pos % WIDTH

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
        return row * WIDTH + col

    def get_unrescued_victims(self):
        unrescued_victims = []
        for v in self.victims:
            if not v.is_rescued():
                unrescued_victims.append(v)

        return unrescued_victims

    def get_closest_victim(self, agent):

        unrescued_victims = self.get_unrescued_victims()
        min_distance = 100000000
        closest_victim = None

        for v in unrescued_victims:
            distance = self.distance(agent=agent, victim=v)
            if min_distance > distance:
                min_distance = distance
                closest_victim = v

        return closest_victim

    def distance(self, agent, victim):
        a_x = self.get_column_center_pixel(agent.get_position())
        v_x = self.get_column_center_pixel(victim.get_position())

        a_y = self.get_row_center_pixel(agent.get_position())
        v_y = self.get_column_center_pixel(victim.get_position())

        return abs(a_x - v_x) + abs(a_y - v_y)

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