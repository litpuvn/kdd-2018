import datetime

def convert_number_to_segment(current_number, min_number):
    return current_number - min_number + 1

def create_converted_rewards(policy_file):
    with open(policy_file) as fp:
        line = fp.readline()
        scoreArray = []
        timeArray = []
        secArray = []
        normalized_score_array = []
        converted_rewards = []

        MIN_SCORE_OF_THREE_POLICIES = -7520
        MAX_SCORE_OF_THREE_POLICIES = 90
        last_time_in_second = None
        total_time_step = 0
        calc = []
        cumulative_time = 0
        while line:
            block1,need = line.split("     ")
            block, clock, block8 = block1.split(" ")
            block2, episode, score, time, block3 = need.split(":")
            timeValue, block4, block5 = time.split(" ")
            epiValue, block6 = episode.split("  ")
            scoValue, block7 = score.split("  ")

            converted_score = convert_number_to_segment(int(scoValue), MIN_SCORE_OF_THREE_POLICIES)
            converted_rewards.append(converted_score)

            a = datetime.datetime.strptime(clock, '%H:%M:%S')
            total_second = datetime.timedelta(hours=a.hour, minutes=a.minute, seconds=a.second).seconds
            if last_time_in_second is None:
                secArray.append(cumulative_time)
            else:
                temp = total_second - last_time_in_second
                cumulative_time = cumulative_time + temp
                secArray.append(cumulative_time)

            last_time_in_second = total_second



            scoreArray.append(int(scoValue))
            timeArray.append(int(timeValue))

            total_time_step = total_time_step + int(timeValue)
            if (int(epiValue) % 59 == 58):
                calc.append(total_time_step/59)
                total_time_step = 0

            line = fp.readline()

def extract_time_in_seconds(policy_file):
    with open(policy_file) as fp:
        line = fp.readline()
        scoreArray = []
        timeArray = []
        secArray = []
        normalized_score_array = []
        converted_rewards = []

        MIN_SCORE_OF_THREE_POLICIES = -7520
        MAX_SCORE_OF_THREE_POLICIES = 90
        last_time_in_second = None
        total_time_step = 0
        calc = []
        cumulative_time = 0
        while line:
            block1,need = line.split("     ")
            block, clock, block8 = block1.split(" ")
            block2, episode, score, time, block3 = need.split(":")
            timeValue, block4, block5 = time.split(" ")
            epiValue, block6 = episode.split("  ")
            scoValue, block7 = score.split("  ")

            converted_score = convert_number_to_segment(int(scoValue), MIN_SCORE_OF_THREE_POLICIES)
            converted_rewards.append(converted_score)

            a = datetime.datetime.strptime(clock, '%H:%M:%S')
            total_second = datetime.timedelta(hours=a.hour, minutes=a.minute, seconds=a.second).seconds
            if last_time_in_second is None:
                secArray.append(cumulative_time)
            else:
                temp = total_second - last_time_in_second
                cumulative_time = cumulative_time + temp
                secArray.append(cumulative_time)

            last_time_in_second = total_second



            scoreArray.append(int(scoValue))
            timeArray.append(int(timeValue))

            total_time_step = total_time_step + int(timeValue)
            if (int(epiValue) % 59 == 58):
                calc.append(total_time_step/59)
                total_time_step = 0

            line = fp.readline()

    return secArray

q_policy_converted_rewards = create_converted_rewards('q.txt')

base_times = extract_time_in_seconds('q.txt')

random_policy_rewards =  create_converted_rewards('random.txt')
times_in_random_policy = extract_time_in_seconds('random.txt')

def get_policy_time_index_that_has_higher_time_value(current_time_value, policy_times):
    for i in range(len(policy_times)):
        policy_time_value = policy_times[i]
        if policy_time_value >= current_time_value:
            return i
    return False

def create_converted_


print("score", format(scoreArray))
print()
print("time", format(timeArray))
print()
print("number of episodes", format(epiValue))

print("sec ",format(secArray))


category = []
for i in range(0, int(epiValue),50):
    category.append(str(i))


print (calc)
print (len(calc))
print(max(scoreArray))
print(min(scoreArray))

print(converted_rewards)


total_converted_reward = sum(converted_rewards)

for i in converted_rewards:
    normalized_score_array.append(i/total_converted_reward)

print(normalized_score_array)

