import metaworld
import random

# 初始化 Meta-World benchmark
ml1 = metaworld.ML1()  # 可替换为其他任务

env = ml1.train_classes['pick-place-v2']()  # 加载环境
task = random.choice(ml1.train_tasks)  # 随机选择一个任务

# 设置环境任务
env.set_task(task)

# 进行一次随机动作的仿真
obs = env.reset()  # 重置环境
for _ in range(100):
    action = env.action_space.sample()  # 随机动作
    re = env.step(action)
    env.render()
    if re[-2]:
        env.reset()
