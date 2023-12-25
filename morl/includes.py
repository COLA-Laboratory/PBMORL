import os,sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir,'externals/baselines/'))


#import env based on mujoco
from environments.ant import AntEnv
from environments.half_cheetah import HalfCheetahEnv
from environments.hopper import HopperEnv
from environments.hopper_v3 import HopperEnv
from environments.humanoid import HumanoidEnv
from environments.swimmer import SwimmerEnv
from environments.walker2d import Walker2dEnv
from environments.mmsd import  MMSDENV

def get_env(env_name):
    global_symbols = globals()
    try:
        env = global_symbols[env_name]
        return env
    except:
        raise ValueError("no such env_name")
