class Distribution:

    def __init__(self):

        # mapping
        self.volunteers = [
            [2, 30],
            [4, 2],
            [4, 34],
            [5, 45],
            [18, 30],
            [22, 35],
            [29, 29],
            [30, 26],
            [38, 22],
            [41, 43],
            [41, 49],
            [45, 50],
            [49, 44]
        ]

        self.victims = [
            [4, 27],
            [11, 31],
            [11, 32],
            [12, 32],
            [14, 3],
            [14, 20],
            [17, 16],
            [18, 13],
            [19, 13],
            [23, 34],
            [24, 21],
            [26, 33],
            [27, 25],
            [32, 19],
            [32, 39],
            [33, 23],
            [37, 21],
            [37, 22],
            [37, 23],
            [37, 24],
            [39, 23],
            [39, 27],
            [40, 18],
            [40, 22],
            [40, 28],
            [40, 35],
            [41, 9],
            [41, 15],
            [41, 22],
            [41, 25],
            [42, 9],
            [42, 20],
            [42, 21],
            [42, 22],
            [42, 23],
            [42, 24],
            [42, 25],
            [49, 2]
        ]

        # testing
        self.volunteers = [
            # [4, 0],
            [0, 0],
            [0, 0]
        ]

        # self.volunteers = [
        #     [0, 0]
        # ]
        self.victims = [
            [1, 2, 50],
            [1, 4, 50],
            [2, 2, 50],
            [4, 4, 100]
        ]

        self.victims = [
            [1, 2, -10],
            [2, 1, -10],
            [2, 2,  100],
            [4, 2,  100]
        ]

        # self.victims = [
        #     [1, 2, -100],
        #     [2, 1, -100],
        #     [4, 2,  50]
        # ]
    def get_distribution_of_vitims(self):
        return self.victims

    def get_distribution_of_volunteers(self):
        return self.volunteers


