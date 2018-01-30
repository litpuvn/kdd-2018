
class BaseAgent:

    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.resource_id = None
        self.pos = None
        self.init_pos = None
        self.env = env

    def get_id(self):
        return self.agent_id

    def set_resource_id(self, resource_id):
        if self.resource_id is not None:
            raise Exception("Resource id for this agent has been assigned to " + self.resource_id)

        self.resource_id = resource_id

    def get_resource_id(self):
        return self.resource_id

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def set_initial_position(self, pos):
        self.init_pos = pos

    def get_initial_position(self):
        return self.init_pos

    # by default, the agent does not have capability to learn anything
    def learn(self, state, action, reward, next_state):
        a =1
