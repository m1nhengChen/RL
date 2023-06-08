import numpy as np

# 定义状态空间大小和动作空间大小
state_space_size = 200
action_space = ['stick', 'hit']

# 初始化Q表格和策略表格
Q = np.zeros((state_space_size, len(action_space)))
policy = np.zeros(state_space_size, dtype=int)

# 定义超参数
num_episodes = 100000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1

# 定义状态编码函数
def encode_state(current_sum, dealer_card, usable_ace):
    return (current_sum - 12) * 10 + dealer_card - 1 + int(usable_ace) * 100

# 定义策略选择函数
def choose_action(state):
    if np.random.uniform() < epsilon:
        # 以epsilon的概率随机选择动作
        return np.random.choice(action_space)
    else:
        # 以1-epsilon的概率选择Q值最大的动作
        return action_space[np.argmax(Q[state])]

# 更新Q值和策略
def update_q_value(state, action, reward, new_state):
    action_index = action_space.index(action)
    Q[state][action_index] += learning_rate * (reward + discount_factor * np.max(Q[new_state]) - Q[state][action_index])
    policy[state] = action_index

# Q-learning算法
for episode in range(num_episodes):
    # 初始化环境和状态
    current_sum = np.random.randint(12, 22)
    dealer_card = np.random.randint(1, 11)
    usable_ace = bool(np.random.randint(0, 2))
    state = encode_state(current_sum, dealer_card, usable_ace)

    for step in range(max_steps_per_episode):
        # 选择动作
        action = choose_action(state)

        # 执行动作并观察新的状态和奖励
        if action == 'hit':
            current_sum += np.random.randint(1, 11)
            current_sum = min(current_sum, 21)  # 点数总和不超过21

        # 更新新的状态
        new_state = encode_state(current_sum, dealer_card, usable_ace)

        # 终止条件
        if current_sum > 21:
            reward = -1
            update_q_value(state, action, reward, new_state)
            break

        if current_sum == 21:
            reward = 1
            update_q_value(state, action, reward, new_state)
            break

        # 庄家的策略
        dealer_sum = dealer_card
        while dealer_sum < 17:
            dealer_sum += np.random.randint(1, 11)
            dealer_sum = min(dealer_sum, 21)

        # 根据庄家的策略更新Q值
        if dealer_sum < current_sum:
            reward = 1
        elif dealer_sum > current_sum:
            reward = -1
        else:
            reward = 0

        update_q_value(state, action, reward, new_state)

        # 跳转到新的状态
        state = new_state

    # 衰减epsilon值
    epsilon *= 0.999

# 打印策略表格
print("Policy:")
for i in range(12, 22):
    for j in range(1, 11):
        for k in range(2):
            state = encode_state(i, j, k)
            action = action_space[policy[state]]
            print(f"State: {i}, Dealer Card: {j}, Usable Ace: {k}, Action: {action}")

# 打印状态-动作价值表格
print("State-Action Value Table:")
for i in range(12, 22):
    for j in range(1, 11):
        for k in range(2):
            state = encode_state(i, j, k)
            print(f"State: {i}, Dealer Card: {j}, Usable Ace: {k}")
            for action in action_space:
                action_index = action_space.index(action)
                print(f"Action: {action}, Value: {Q[state][action_index]}")
            print()
