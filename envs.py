# @source: https://github.com/openai/gym/issues/106#issuecomment-226675545

from gym import envs
class NullE:
    def __init__(self):
        self.observation_space = self.action_space = self.reward_range = "N/A"

envall = envs.registry.all()

table = "|Environment Id|Observation Space|Action Space|Reward Range|\n"
table += "|---|---|---|---|---|---|---|\n"

for e in envall:
    try:
        env = e.make()
    except:
        env = NullE()
        continue
    table += '| {}|{}|{}|{}|\n'.format(e.id, env.observation_space, env.action_space, env.reward_range) # ,

print(table)