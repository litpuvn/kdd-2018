import datetime

def convert_number_to_segment(current_number, min_number):
    return current_number - min_number + 1

def create_converted_rewards(policy_file, convert = True):
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

            if convert == True:
                converted_score = convert_number_to_segment(int(scoValue), MIN_SCORE_OF_THREE_POLICIES)
            else:
                converted_score = int(scoValue)
            converted_rewards.append(converted_score)

            line = fp.readline()

        return converted_rewards

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

q_policy_rewards = create_converted_rewards(policy_file='q_policy_log.txt', convert=False)
q_policy_times = extract_time_in_seconds(policy_file='q_policy_log.txt')
q_policty_time_min_max = [min(q_policy_times), max(q_policy_times)]
print('Q policy time: ' + str(q_policty_time_min_max))

random_policy_rewards = create_converted_rewards(policy_file='random_policy_log.txt', convert=False)
times_in_random_policy = extract_time_in_seconds(policy_file='random_policy_log.txt')
random_policty_time_min_max = [min(times_in_random_policy), max(times_in_random_policy)]
print('Random policy time: ' + str(random_policty_time_min_max))

def get_policy_time_index_that_has_higher_time_value(current_time_value, policy_times):
    for i in range(len(policy_times)):
        policy_time_value = policy_times[i]
        if policy_time_value >= current_time_value:
            return i
    return None


def rescale_values(base_policy_times, policy_times, policy_values):

    rescale_rewards = []
    for i in range(len(base_policy_times)):
        base_time_value = base_policy_times[i]

        policy_time_index_higher = get_policy_time_index_that_has_higher_time_value(base_time_value, policy_times)
        if policy_time_index_higher is None:
            raise Exception('Cannot find index has higher time value: base_index=' + str(i) + ";base_val=" + str(base_time_value))

        policy_time_value = policy_times[policy_time_index_higher]
        policy_reward_at_time_value = policy_values[policy_time_index_higher]

        if policy_time_value == base_time_value:
            rescale_rewards.append(policy_reward_at_time_value)
            continue

        if policy_time_index_higher < 1:
            rescale_rewards.append((base_time_value * policy_reward_at_time_value) / policy_time_value)
            continue

        policy_time_pre_value = policy_times[policy_time_index_higher-1]
        policy_reward_at_time_pre_value = policy_values[policy_time_index_higher-1]

        # rescaled_reward = base_time_value * (policy_reward_at_time_value - policy_reward_at_time_pre_value) / (policy_time_value-policy_time_pre_value)
        rescaled_reward = (policy_reward_at_time_pre_value + policy_reward_at_time_value) / 2
        rescale_rewards.append(rescaled_reward)

    return rescale_rewards


rescaled_values_random_policy = rescale_values(base_policy_times=q_policy_times, policy_times=times_in_random_policy, policy_values=random_policy_rewards)
#

# print("x" + str(q_policy_times))
print(len(rescaled_values_random_policy))
print(rescaled_values_random_policy)
