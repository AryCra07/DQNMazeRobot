# AIMazeRobot

> 2023 RoboCom AI Competition 提交作品

主要采用Deep QLearning 算法构建机器人走迷宫的模型，并进行优化工作。关于题目要求与算法描述参见 `main.ipynb`，我实现的机器人参见 `main.py`，`test.ipynb` 是测试版代码。

除了实现基本功能之外，主要有以下几点优化：

- 采用 **epsilon-greedy** 策略来增加机器人的探索部分。
- 在 `torch_py/QNetwork.py` 中调整网络结构以优化模型性能。
- 在 `torch_py/MinDQNRobot.py` 中使用经验回放、固定Q目标等优化方法，并在训练循环中对机器人进行学习更新。
- 对一些超参数进行了调整优化。
