
class BaseAgent:

    def __init__(self, agent_id, env, options):
        self.agent_id = agent_id
        self.resource_id = None
        self.pos = None
        self.init_pos = None
        self.env = env
        self.options = options
        self.rescued_victims = []
        self.last_action = None
        self.start_target_search_pos = None

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

    def set_start_target_search_position(self, pos):
        self.start_target_search_pos = pos

    def get_start_target_search_position(self):
        return self.start_target_search_pos

    # by default, the agent does not have capability to learn anything
    def learn(self, state, action, reward, next_state):
        a =1

    def is_distributed(self):
        if 'distributed' not in self.options:
            return False

        return self.options['distributed']

    def add_rescued_victims(self, victim):
        for v in self.rescued_victims:
            if v.get_id == victim.get_id():
                return False

        self.rescued_victims.append(victim)

        return self.rescued_victims

    def get_rescued_victims(self):
        return self.rescued_victims

    def reset_rescued_victims(self):
        self.rescued_victims = []

        return True

    def reset_last_action(self):
        self.last_action = None

    def get_last_action(self):
        return self.last_action

    def set_last_action(self, action):
        self.last_action = action
