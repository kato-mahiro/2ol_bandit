from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id = 'static_cyclic_env-v0',
    entry_point = 'myenvs.myenvs:StaticCyclicEnv0'
)