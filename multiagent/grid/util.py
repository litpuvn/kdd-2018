import ast


class Util:

    def __init__(self):
        i = 0

    @staticmethod
    def read_q_table(file_path):
        dictionary = {}

        with open(file_path) as fp:
            line = fp.readline()
            while line:
                if line != '\n':
                    block1, need = line.split("] ")
                    tempArray = need[12:len(need) - 1]
                    testarray = ast.literal_eval(tempArray)
                    dictionary[need[0:12]] = testarray
                line = fp.readline()

        return dictionary

    def write_q_table(self, file_paht, q_table):
        i = 0