import numpy as np
import random

eps = 0.0625
Gradient_Bandit_alpha = 0.1


def stationary_action_value_method():
    true_value = np.random.normal(0, 1, 10)
    print(true_value)
    best_action = np.argmax(true_value)
    print("真正的目标动作为： ", best_action)
    action_list = []
    reward_list = []
    # The first action must be random dur to no reward have before
    step = 1
    action = random.randint(0, 9)
    # print("第", step, "步采取的动作为：", action, " random")
    reward = np.random.normal(true_value[action], 2)
    action_list.append(action)
    reward_list.append(reward)
    step = step + 1
    is_leverage = False
    while not is_leverage:
        c = random.random()
        if c >= eps:
            count_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            reward_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            action_value_now = []
            for i in range(step - 1):
                count_now[action_list[i]] = count_now[action_list[i]] + 1
                reward_now[action_list[i]] = reward_now[action_list[i]] + reward_list[i]
            for i in range(10):
                if count_now[i] == 0:
                    action_value_now.append(0)
                    continue
                else:
                    action_value_now.append(reward_now[i] / count_now[i])
            # a = np.argmax(action_value_now)
            # print("a: ", a)
            # print(action_value_now)
            winner = np.argwhere(action_value_now == np.max(action_value_now))
            # print(winner)
            prob_action_num = len(winner)
            winner = winner.flatten().tolist()
            # print(winner)
            # print(prob_action_num)
            if prob_action_num > 1:
                index = random.randint(0, prob_action_num - 1)
                # print(index)
                action = winner[index]
                # print("第", step, "步采取的动作为： ", action, " not random but random select")
            else:
                action = winner[0]
                # print("第", step, "步采取的动作为： ", action, " not random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)

        else:
            action = random.randint(0, 9)
            # print("第", step, "步采取的动作为：", action, " random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)

        # judge whether the algorithm is leverage or not
        if step <= 50:
            step = step + 1
            continue
        else:
            if 1 == len(set(action_list[-10:])):
                print("目标已经收敛至：", action_list[step - 1], " it costs step:", step, " the result is:",
                      action_list[step - 1] == best_action)
                is_leverage = True
                break
            else:
                step = step + 1
    return step, best_action, action_list[step - 1]


def stationary_gradient_method():
    true_value = np.random.normal(0, 1, 10)
    print(true_value)
    best_action = np.argmax(true_value)
    print("真正的目标动作为： ", best_action)
    action_list = []
    reward_list = []
    # The first action must be random dur to no reward have before
    Q_estimation = np.zeros(10)
    action_prob = np.zeros(10)
    average_reward = 0.0
    is_leverage = False
    step = 1
    gradient_baseline = True
    while not is_leverage:
        # calculate the softmax distribution
        exp_est = np.exp(Q_estimation)
        action_prob = exp_est / np.sum(exp_est)
        action = np.random.choice(np.arange(10), p=action_prob)
        reward = np.random.randn() + true_value[action]
        action_list.append(action)
        reward_list.append(reward)
        if step <= 50:
            step += 1
            average_reward = (step - 1.0) / step * average_reward + reward / step
            one_hot = np.zeros(10)
            one_hot[action] = 1
            if gradient_baseline:
                baseline = average_reward
            else:
                baseline = 0
            ''' preference update formula! The one_hot array is used to ensure that
            the selected and unselected actions are updated in different directions '''
            Q_estimation = Q_estimation + Gradient_Bandit_alpha * (reward - baseline) * (one_hot - action_prob)
        else:
            if 1 == len(set(action_list[-10:])):
                print("目标已经收敛至：", action_list[step - 1], " it costs step:", step, " the result is:",
                      action_list[step - 1] == best_action)
                is_leverage = True
                break
            else:
                step = step + 1
                average_reward = (step - 1.0) / step * average_reward + reward / step
                one_hot = np.zeros(10)
                one_hot[action] = 1
                if gradient_baseline:
                    baseline = average_reward
                else:
                    baseline = 0
                ''' preference update formula! The one_hot array is used to ensure that
                the selected and unselected actions are updated in different directions '''
                Q_estimation = Q_estimation + Gradient_Bandit_alpha * (reward - baseline) * (one_hot - action_prob)
    return step, best_action, action_list[step - 1]


def nonstationary_action_value_method():
    true_value = np.random.normal(0, 1, 10)
    # print(true_value)
    best_action = np.argmax(true_value)
    # print("真正的目标动作为： ", best_action)
    action_list = []
    reward_list = []
    # The first action must be random dur to no reward have before
    step = 1
    action = random.randint(0, 9)
    # print("第", step, "步采取的动作为：", action, " random")
    reward = np.random.normal(true_value[action], 2)
    action_list.append(action)
    reward_list.append(reward)
    step = step + 1
    is_leverage = False
    while step <= 1000:
        update = np.random.normal(0, 0.1, 10)
        true_value = list(map(lambda x, y: x + y, true_value, update))
        c = random.random()
        if c >= eps:
            count_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            reward_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            action_value_now = []
            for i in range(step - 1):
                count_now[action_list[i]] = count_now[action_list[i]] + 1
                reward_now[action_list[i]] = reward_now[action_list[i]] + reward_list[i]
            for i in range(10):
                if count_now[i] == 0:
                    action_value_now.append(0)
                    continue
                else:
                    action_value_now.append(reward_now[i] / count_now[i])
            # a = np.argmax(action_value_now)
            # print("a: ", a)
            # print(action_value_now)
            winner = np.argwhere(action_value_now == np.max(action_value_now))
            # print(winner)
            prob_action_num = len(winner)
            winner = winner.flatten().tolist()
            # print(winner)
            # print(prob_action_num)
            if prob_action_num > 1:
                index = random.randint(0, prob_action_num - 1)
                # print(index)
                action = winner[index]
                # print("第", step, "步采取的动作为： ", action, " not random but random select")
            else:
                action = winner[0]
                # print("第", step, "步采取的动作为： ", action, " not random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)
        else:
            action = random.randint(0, 9)
            # print("第", step, "步采取的动作为：", action, " random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)
        step += 1
    total = sum(reward_list)
    return total


def nonstationary_action_value_method_fixed_step():
    true_value = np.random.normal(0, 1, 10)
    # print(true_value)
    best_action = np.argmax(true_value)
    # print("真正的目标动作为： ", best_action)
    alpha = 0.1
    action_list = []
    reward_list = []
    reward_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # The first action must be random dur to no reward have before
    step = 1
    action = random.randint(0, 9)
    # print("第", step, "步采取的动作为：", action, " random")
    reward = np.random.normal(true_value[action], 2)
    action_list.append(action)
    reward_list.append(reward)
    reward_now[action] = reward_now[action] + alpha * (reward - reward_now[action])
    step = step + 1
    is_leverage = False
    while step <= 1000:
        update = np.random.normal(0, 0.1, 10)
        true_value = list(map(lambda x, y: x + y, true_value, update))
        c = random.random()
        if c >= eps:
            count_now = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            action_value_now = []
            for i in range(step - 1):
                count_now[action_list[i]] = count_now[action_list[i]] + 1
            for i in range(10):
                if count_now[i] == 0:
                    action_value_now.append(0)
                    continue
                else:
                    action_value_now.append(reward_now[i])
            # a = np.argmax(action_value_now)
            # print("a: ", a)
            # print(action_value_now)
            winner = np.argwhere(action_value_now == np.max(action_value_now))
            # print(winner)
            prob_action_num = len(winner)
            winner = winner.flatten().tolist()
            # print(winner)
            # print(prob_action_num)
            if prob_action_num > 1:
                index = random.randint(0, prob_action_num - 1)
                # print(index)
                action = winner[index]
                # print("第", step, "步采取的动作为： ", action, " not random but random select")
            else:
                action = winner[0]
                # print("第", step, "步采取的动作为： ", action, " not random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)
            reward_now[action] = reward_now[action] + alpha * (reward - reward_now[action])
        else:
            action = random.randint(0, 9)
            # print("第", step, "步采取的动作为：", action, " random")
            reward = np.random.normal(true_value[action], 2)
            action_list.append(action)
            reward_list.append(reward)
            reward_now[action] = reward_now[action] + alpha * (reward - reward_now[action])
        step += 1
    total = sum(reward_list)
    return total


def nonstationary_gradient_method(isNormal):
    true_value = np.random.normal(0, 1, 10)
    # print(true_value)
    best_action = np.argmax(true_value)
    # print("真正的目标动作为： ", best_action)
    action_list = []
    reward_list = []
    # The first action must be random dur to no reward have before
    Q_estimation = np.zeros(10)
    action_prob = np.zeros(10)
    average_reward = 0.0
    is_leverage = False
    step = 1
    gradient_baseline = True
    while step <= 1000:
        # calculate the softmax distribution
        exp_est = np.exp(Q_estimation)
        action_prob = exp_est / np.sum(exp_est)
        action = np.random.choice(np.arange(10), p=action_prob)
        reward = np.random.randn() + true_value[action]
        action_list.append(action)
        reward_list.append(reward)
        step += 1
        if isNormal:
            average_reward = (step - 1.0) / step * average_reward + reward / step
        else:
            average_reward = (1 - Gradient_Bandit_alpha) * (step - 1.0) / step * average_reward + reward / step
        one_hot = np.zeros(10)
        one_hot[action] = 1
        if gradient_baseline:
            baseline = average_reward
        else:
            baseline = 0
            ''' preference update formula! The one_hot array is used to ensure that
            the selected and unselected actions are updated in different directions '''
        Q_estimation = Q_estimation + Gradient_Bandit_alpha * (reward - baseline) * (one_hot - action_prob)
        update = np.random.normal(0, 0.1, 10)
        true_value = list(map(lambda x, y: x + y, true_value, update))
    total = sum(reward_list)
    return total


if __name__ == '__main__':
    step_list = []
    converge_list = []
    # for i in range(100000):
    #     step, best, actual = stationary_gradient_method()
    #     step_list.append(step)
    #     if best == actual:
    #         leverage = 1
    #     else:
    #         leverage = 0
    #     converge_list.append(leverage)
    # print("method: stationary_gradient_method", " alpha= ", Gradient_Bandit_alpha)
    # print("average:", np.mean(step_list), " var: ", np.var(step_list, ddof=1), " std: ", np.std(step_list, ddof=1),
    #       " success rate: ", np.mean(converge_list))
    total_list = []
    # for i in range(10000):
    #     total = nonstationary_action_value_method()
    #     # total = nonstationary_action_value_method_fixed_step()
    #     print(i, " total: ", total)
    #     total_list.append(total)
    # print("method: nonstationary_action_value_method", " eps= ", eps)
    # print("average:", np.mean(total_list), " var: ", np.var(total_list, ddof=1), " std: ", np.std(total_list, ddof=1), )
    for i in range(10000):
        total = nonstationary_gradient_method(False)
        print(i, " total: ", total)
        total_list.append(total)
    print("method: nonstationary_gradient_method", " alpha= ", Gradient_Bandit_alpha)
    print("average:", np.mean(total_list), " var: ", np.var(total_list, ddof=1), " std: ", np.std(total_list, ddof=1))
    # nonstationary_action_value_method()
    # stationary_gradient_method()
    # nonstationary_gradient_method()
