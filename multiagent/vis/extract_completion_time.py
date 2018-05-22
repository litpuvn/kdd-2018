def extract_completion_time(episodes, policy_file):

    completion_times = []
    with open(policy_file) as fp:
        line = fp.readline()
        while line:
            block1, attributes = line.split("     ")
            block, clock, block8 = block1.split(" ")
            block2, episode, score, time, block3 = attributes.split(":")
            timeValue, block4, block5 = time.split(" ")
            epiValue, block6 = episode.split("  ")
            scoValue, block7 = score.split("  ")


            completion_times.append(timeValue)

            line = fp.readline()

        custom_completion_times = []
        for epi in episodes:
            time_val = completion_times[int(epi)]
            custom_completion_times.append(int(time_val))

        return custom_completion_times

episodes = ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000', '1050', '1100', '1150', '1200', '1250', '1300', '1350', '1400', '1450', '1500', '1550', '1600', '1650', '1700', '1750', '1800', '1850', '1900', '1950', '2000', '2050', '2100', '2150', '2200', '2250', '2300', '2350', '2400', '2450', '2500', '2550', '2600', '2650', '2700', '2750', '2800', '2850', '2900', '2950', '3000', '3050', '3100', '3150', '3200', '3250', '3300']
completion_times = extract_completion_time(episodes, policy_file='random.txt')

print(completion_times)