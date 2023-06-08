import numpy as np

# 定义状态和动作空间
states = range(12, 22)  # 当前点数总和
dealer_showing = range(1, 11)  # 庄家的明牌
usable_ace = [False, True]  # 是否拥有可用的Ace牌
actions = ['stick', 'hit']

# 初始化状态-动作价值表
Q = {}  # 状态-动作价值表

# 初始化状态-动作价值表的值为随机小数
for state in states:
    for dealer in dealer_showing:
        for ace in usable_ace:
            for action in actions:
                Q[(state, dealer, ace, action)] = np.random.random()


# 定义混合策略函数，根据状态选择动作
def mixed_policy(state, dealer, ace, epsilon, temperature):
    if np.random.random() < epsilon:
        # 使用ε-greedy策略进行探索
        action = np.random.choice(actions)
    else:
        # 使用Softmax策略进行利用
        action_values = np.array([Q[(state, dealer, ace, a)] for a in actions])
        probabilities = np.exp(action_values / temperature)
        probabilities /= np.sum(probabilities)
        action = np.random.choice(actions, p=probabilities)
    return action


# 进行蒙特卡洛控制
num_episodes = 100000  # 游戏的总场次
epsilon = 0.1  # ε-greedy策略的ε值
temperature = 0.1  # Softmax策略的温度参数
alpha = 0.1  # 学习率

for episode in range(num_episodes):
    episode_states = []  # 存储每一场游戏的状态
    episode_actions = []  # 存储每一场游戏的动作
    episode_rewards = []  # 存储每一场游戏的奖励

    # 初始化游戏状态
    state = np.random.choice(states)
    dealer = np.random.choice(dealer_showing)
    ace = np.random.choice(usable_ace)

    while True:
        action = mixed_policy(state, dealer, ace, epsilon, temperature)

        episode_states.append((state, dealer, ace))
        episode_actions.append(action)

        if action == 'stick':
            break

        # 请求一张新牌
        card = np.random.randint(1, 11)
        state += card

        # 如果玩家爆牌
        if state > 21:
            # 判断是否拥有可用的Ace牌
            if ace:
                state -= 10
                ace = False
            else:
                # 玩家输
                reward = -1
                episode_rewards.append(reward)
                break

    # 庄家根据固定策略进行操作
    while dealer < 17:
        card = np.random.randint(1, 11)
        dealer += card

    # 判断游戏结果
    if state > dealer or dealer > 21:
        reward = 1
    elif state == dealer:
        reward = 0
    else:
        reward = -1

    episode_rewards.append(reward)

    # 更新状态-动作价值表
    for i in range(len(episode_states)):
        state = episode_states[i]
        action = episode_actions[i]
        G = sum(episode_rewards[i:])
        Q[state + (action,)] += alpha * (G - Q[state + (action,)])

# 输出策略和状态-动作价值表
for state in states:
    for dealer in dealer_showing:
        for ace in usable_ace:
            action = mixed_policy(state, dealer, ace, 0, 0.1)  # 使用纯利用的Softmax策略
            print(f"State: {state}, Dealer: {dealer}, Ace: {ace}, Action: {action}")
