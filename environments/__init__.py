import os,sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
sys.path.append(base_dir)
from environments.base import *
from environments.balance import *
from environments.manipulate import *
from environments.climb import *
from environments.flip import *
from environments.jump import *
from environments.multi_goal import *
from environments.change_shape import *
from environments.traverse import *
from environments.walk import *

from gym.envs.registration import register

## SIMPLE ##
register(
    id = 'Walker-v0',
    entry_point = 'environments.walk:WalkingFlat',
    max_episode_steps=500
)

register(
    id = 'BridgeWalker-v0',
    entry_point = 'environments.walk:SoftBridge',
    max_episode_steps=500
)

register(
    id = 'CaveCrawler-v0',
    entry_point = 'environments.walk:Duck',
    max_episode_steps=1000
)

register(
    id = 'Jumper-v0',
    entry_point = 'environments.jump:StationaryJump',
    max_episode_steps=500
)

register(
    id = 'Flipper-v0',
    entry_point = 'environments.flip:Flipping',
    max_episode_steps=600
)

register(
    id = 'Balancer-v0',
    entry_point = 'environments.balance:Balance',
    max_episode_steps=600
)

register(
    id = 'Balancer-v1',
    entry_point = 'environments.balance:BalanceJump',
    max_episode_steps=600
)

register(
    id = 'UpStepper-v0',
    entry_point = 'environments.traverse:StepsUp',
    max_episode_steps=600
)

register(
    id = 'DownStepper-v0',
    entry_point = 'environments.traverse:StepsDown',
    max_episode_steps=500
)

register(
    id = 'ObstacleTraverser-v0',
    entry_point = 'environments.traverse:WalkingBumpy',
    max_episode_steps=1000
)

register(
    id = 'ObstacleTraverser-v1',
    entry_point = 'environments.traverse:WalkingBumpy2',
    max_episode_steps=1000
)

register(
    id = 'Hurdler-v0',
    entry_point = 'environments.traverse:VerticalBarrier',
    max_episode_steps=1000
)

register(
    id = 'GapJumper-v0',
    entry_point = 'environments.traverse:Gaps',
    max_episode_steps=1000
)

register(
    id = 'PlatformJumper-v0',
    entry_point = 'environments.traverse:FloatingPlatform',
    max_episode_steps=1000
)

register(
    id = 'Traverser-v0',
    entry_point = 'environments.traverse:BlockSoup',
    max_episode_steps=600
)

## PACKAGE ##
register(
    id = 'Lifter-v0',
    entry_point = 'environments.manipulate:LiftSmallRect',
    max_episode_steps=300
)

register(
    id = 'Carrier-v0',
    entry_point = 'environments.manipulate:CarrySmallRect',
    max_episode_steps=500
)

register(
    id = 'Carrier-v1',
    entry_point = 'environments.manipulate:CarrySmallRectToTable',
    max_episode_steps=1000
)

register(
    id = 'Pusher-v0',
    entry_point = 'environments.manipulate:PushSmallRect',
    max_episode_steps=500
)

register(
    id = 'Pusher-v1',
    entry_point = 'environments.manipulate:PushSmallRectOnOppositeSide',
    max_episode_steps=600
)

register(
    id = 'BeamToppler-v0',
    entry_point = 'environments.manipulate:ToppleBeam',
    max_episode_steps=1000
)

register(
    id = 'BeamSlider-v0',
    entry_point = 'environments.manipulate:SlideBeam',
    max_episode_steps=1000
)

register(
    id = 'Thrower-v0',
    entry_point = 'environments.manipulate:ThrowSmallRect',
    max_episode_steps=300
)

register(
    id = 'Catcher-v0',
    entry_point = 'environments.manipulate:CatchSmallRect',
    max_episode_steps=400
)

### SHAPE ###
register(
    id = 'AreaMaximizer-v0',
    entry_point = 'environments.change_shape:MaximizeShape',
    max_episode_steps=600
)

register(
    id = 'AreaMinimizer-v0',
    entry_point = 'environments.change_shape:MinimizeShape',
    max_episode_steps=600
)

register(
    id = 'WingspanMazimizer-v0',
    entry_point = 'environments.change_shape:MaximizeXShape',
    max_episode_steps=600
)

register(
    id = 'HeightMaximizer-v0',
    entry_point = 'environments.change_shape:MaximizeYShape',
    max_episode_steps=500
)

### CLIMB ###
register(
    id = 'Climber-v0',
    entry_point = 'environments.climb:Climb0',
    max_episode_steps=400
)

register(
    id = 'Climber-v1',
    entry_point = 'environments.climb:Climb1',
    max_episode_steps=600
)

register(
    id = 'Climber-v2',
    entry_point = 'environments.climb:Climb2',
    max_episode_steps=1000
)

### MULTI GOAL ###
register(
    id = 'BidirectionalWalker-v0',
    entry_point = 'environments.multi_goal:BiWalk',
    max_episode_steps=1000
)

register(
    id = 'MO-Ant-v2',
    entry_point = 'environments.ant:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'environments.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = 'environments.hopper_v3:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = 'environments.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = 'environments.walker2d:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = 'environments.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Humanoid-v2',
    entry_point = 'environments.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)

