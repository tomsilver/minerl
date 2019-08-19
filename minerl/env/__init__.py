# ------------------------------------------------------------------------------------------------
# Copyright (c) 2018 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------
import os

# import gym
# Perform the registration.
from gym.envs.registration import register
from collections import OrderedDict
from minerl.env import spaces
from minerl.env.core import MineRLEnv, missions_dir

import numpy as np

  
register(
    id='MineRLTreechop-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'treechop.xml'),
        'observation_space': spaces.Dict({
            'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        }),
        'action_space': spaces.Dict(spaces={
            "forward": spaces.Discrete(2), 
            "back": spaces.Discrete(2), 
            "left": spaces.Discrete(2), 
            "right": spaces.Discrete(2), 
            "jump": spaces.Discrete(2), 
            "sneak": spaces.Discrete(2), 
            "sprint": spaces.Discrete(2), 
            "attack": spaces.Discrete(2),
            "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
        }),
        'docstr': """
.. image:: ../assets/treechop1.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/treechop2.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/treechop3.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/treechop4.mp4.gif
  :scale: 100 %
  :alt: 
In treechop, the agent must collect 64 `minercaft:log`. This replicates a common scenario in Minecraft, as logs are necessary to craft a large amount of items in the game, and are a key resource in Minecraft.

The agent begins in a forest biome (near many trees) with an iron axe for cutting trees. The agent is given +1 reward for obtaining each unit of wood, and the episode terminates once the agent obtains 64 units.\n"""
    },
    max_episode_steps=8000,
    reward_threshold=64.0,
)


#######################
#      NAVIGATE       #
#######################

def make_navigate_text(top, dense):
    navigate_text = """
.. image:: ../assets/navigate{}1.mp4.gif
    :scale: 100 %
    :alt: 

.. image:: ../assets/navigate{}2.mp4.gif
    :scale: 100 %
    :alt: 

.. image:: ../assets/navigate{}3.mp4.gif
    :scale: 100 %
    :alt: 

.. image:: ../assets/navigate{}4.mp4.gif
    :scale: 100 %
    :alt: 

In this task, the agent must move to a goal location denoted by a diamond block. This represents a basic primitive used in many tasks throughout Minecraft. In addition to standard observations, the agent has access to a “compass” observation, which points near the goal location, 64 meters from the start location. The goal has a small random horizontal offset from the compass location and may be slightly below surface level. On the goal location is a unique block, so the agent must find the final goal by searching based on local visual features.

The agent is given a sparse reward (+100 upon reaching the goal, at which point the episode terminates). """
    if dense:
        navigate_text += "**This variant of the environment is dense reward-shaped where the agent is given a reward every tick for how much closer (or negative reward for farther) the agent gets to the target.**\n"
    else: 
        navigate_text += "**This variant of the environment is sparse.**\n"

    if top is "normal":
        navigate_text += "\nIn this environment, the agent spawns on a random survival map.\n"
        navigate_text = navigate_text.format(*["" for _ in range(4)])
    else:
        navigate_text += "\nIn this environment, the agent spawns in an extreme hills biome.\n"
        navigate_text = navigate_text.format(*["extreme" for _ in range(4)])
    return navigate_text


navigate_action_space = spaces.Dict({
    "forward": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
    "place": spaces.Enum('none', 'dirt')})

navigate_observation_space = spaces.Dict({
    'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
    'inventory': spaces.Dict(spaces={
        'dirt': spaces.Box(low=0, high=2304, shape=(), dtype=np.int)
    }),
    'compassAngle': spaces.Box(low=-180.0, high=180.0, shape=(), dtype=np.float32)
})

register(
    id='MineRLNavigate-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'navigation.xml'),
        'observation_space': navigate_observation_space,
        'action_space': navigate_action_space,
        'docstr': make_navigate_text('normal', False)
    },
    max_episode_steps=6000,
)

register(
    id='MineRLNavigateDense-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'navigationDense.xml'),
        'observation_space': navigate_observation_space,
        'action_space': navigate_action_space,
        'docstr': make_navigate_text('normal', True)
    },
    max_episode_steps=6000,
)


register(
    id='MineRLNavigateExtreme-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'navigationExtreme.xml'),
        'observation_space': navigate_observation_space,
        'action_space': navigate_action_space,
        'docstr': make_navigate_text('extreme', False) 
    },
    max_episode_steps=6000,
)

register(
    id='MineRLNavigateExtremeDense-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'navigationExtremeDense.xml'),
        'observation_space': navigate_observation_space,
        'action_space': navigate_action_space,
        'docstr': make_navigate_text('extreme', True)  
    },
    max_episode_steps=6000,
)


#######################
#     Obtain Iron     #
#######################

obtain_observation_space = spaces.Dict({
    'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
    'inventory': spaces.Dict({
        'dirt': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'coal': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'torch': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'log': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'planks': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'stick': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'crafting_table': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'wooden_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'wooden_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'stone': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'cobblestone': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'furnace': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'stone_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'stone_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'iron_ore': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'iron_ingot': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'iron_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
        'iron_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
    }),
    'equipped_items': spaces.Dict({
        'mainhand': spaces.Dict({
            'type': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe',
                                'iron_axe', 'iron_pickaxe', 'other'),
            'damage': spaces.Box(low=-1, high=1562, shape=(), dtype=np.int),
            'maxDamage': spaces.Box(low=-1, high=1562, shape=(), dtype=np.int),
        })
    })
})

obtain_action_space = spaces.Dict({
    "forward": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),  # Pitch, Yaw
    "place": spaces.Enum('none', 'dirt', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'),
    "equip": spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'),
    "craft": spaces.Enum('none', 'torch', 'stick', 'planks', 'crafting_table'),
    "nearbyCraft": spaces.Enum('none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'),
    "nearbySmelt": spaces.Enum('none', 'iron_ingot', 'coal')})


register(
    id='MineRLObtainIronPickaxe-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'obtainIronPickaxe.xml'),
        'observation_space': obtain_observation_space,
        'action_space': obtain_action_space,
        'docstr': """
.. image:: ../assets/orion1.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion2.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion3.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion4.mp4.gif
  :scale: 100 %
  :alt: 
In this environment the agent is required to obtain an iron pickaxe. The agent begins in a random starting location, on a random survival map, without any items, matching the normal starting conditions for human players in Minecraft.
The agent is given access to a selected view of its inventory and GUI free
crafting, smelting, and inventory management actions.


During an episode **the agent is rewarded only once per item the first time it obtains that item
in the requisite item hierarchy for obtaining an iron pickaxe.** The reward for each
item is given here::
    <Item amount="1" reward="1" type="log" />
    <Item amount="1" reward="2" type="planks" />
    <Item amount="1" reward="4" type="stick" />
    <Item amount="1" reward="4" type="crafting_table" />
    <Item amount="1" reward="8" type="wooden_pickaxe" />
    <Item amount="1" reward="16" type="cobblestone" />
    <Item amount="1" reward="32" type="furnace" />
    <Item amount="1" reward="32" type="stone_pickaxe" />
    <Item amount="1" reward="64" type="iron_ore" />
    <Item amount="1" reward="128" type="iron_ingot" />
    <Item amount="1" reward="256" type="iron_pickaxe" />

\n"""
    },
    max_episode_steps=6000,
)


register(
    id='MineRLObtainIronPickaxeDense-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'obtainIronPickaxeDense.xml'),
        'observation_space': obtain_observation_space,
        'action_space': obtain_action_space,
        'docstr': """
.. image:: ../assets/orion1.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion2.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion3.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/orion4.mp4.gif
  :scale: 100 %
  :alt: 
In this environment the agent is required to obtain an iron pickaxe. The agent begins in a random starting location, on a random survival map, without any items, matching the normal starting conditions for human players in Minecraft.
The agent is given access to a selected view of its inventory and GUI free
crafting, smelting, and inventory management actions.


During an episode the agent is rewarded **every time ** it obtains an item
in the requisite item hierarchy for obtaining an iron pickaxe. The rewards for each
item are given here::
    <Item amount="1" reward="1" type="log" />
    <Item amount="1" reward="2" type="planks" />
    <Item amount="1" reward="4" type="stick" />
    <Item amount="1" reward="4" type="crafting_table" />
    <Item amount="1" reward="8" type="wooden_pickaxe" />
    <Item amount="1" reward="16" type="cobblestone" />
    <Item amount="1" reward="32" type="furnace" />
    <Item amount="1" reward="32" type="stone_pickaxe" />
    <Item amount="1" reward="64" type="iron_ore" />
    <Item amount="1" reward="128" type="iron_ingot" />
    <Item amount="1" reward="256" type="iron_pickaxe" />

\n"""
    },
    max_episode_steps=6000,
)


#######################
#   Obtain Diamond    #
#######################
register(
    id='MineRLObtainDiamond-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'obtainDiamond.xml'),
        'observation_space': obtain_observation_space,
        'action_space': obtain_action_space,
        'docstr': """
.. image:: ../assets/odia1.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia2.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia3.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia4.mp4.gif
  :scale: 100 %
  :alt: 

.. caution::
    **This is the evaluation environment of the MineRL Competition!** Specifically, you are allowed
    to train your agents on any environment (including `MineRLObtainDiamondDense-v0`_) however,
    your agent will only be evaluated on this environment..

In this environment the agent is required to obtain a diamond in 18000 steps. The agent begins in a random starting location, on a random survival map, without any items, matching the normal starting conditions for human players in Minecraft.
The agent is given access to a selected view of its inventory and GUI free
crafting, smelting, and inventory management actions.


During an episode **the agent is rewarded only once per item the first time it obtains that item
in the requisite item hierarchy for obtaining an iron pickaxe.** The reward for each
item is given here::

    <Item reward="1" type="log" />
    <Item reward="2" type="planks" />
    <Item reward="4" type="stick" />
    <Item reward="4" type="crafting_table" />
    <Item reward="8" type="wooden_pickaxe" />
    <Item reward="16" type="cobblestone" />
    <Item reward="32" type="furnace" />
    <Item reward="32" type="stone_pickaxe" />
    <Item reward="64" type="iron_ore" />
    <Item reward="128" type="iron_ingot" />
    <Item reward="256" type="iron_pickaxe" />
    <Item reward="1024" type="diamond" />

\n"""
    },
    max_episode_steps=18000,
)


register(
    id='MineRLObtainDiamondDense-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'obtainDiamondDense.xml'),
        'observation_space': obtain_observation_space,
        'action_space': obtain_action_space,
        'docstr': """
.. image:: ../assets/odia1.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia2.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia3.mp4.gif
  :scale: 100 %
  :alt: 

.. image:: ../assets/odia4.mp4.gif
  :scale: 100 %
  :alt: 

In this environment the agent is required to obtain a diamond. The agent begins in a random starting location on a random survival map without any items, matching the normal starting conditions for human players in Minecraft.
The agent is given access to a selected summary of its inventory and GUI free
crafting, smelting, and inventory management actions.


During an episode the agent is rewarded **every** time it obtains an item
in the requisite item hierarchy to obtaining a diamond. The rewards for each
item are given here::

    <Item reward="1" type="log" />
    <Item reward="2" type="planks" />
    <Item reward="4" type="stick" />
    <Item reward="4" type="crafting_table" />
    <Item reward="8" type="wooden_pickaxe" />
    <Item reward="16" type="cobblestone" />
    <Item reward="32" type="furnace" />
    <Item reward="32" type="stone_pickaxe" />
    <Item reward="64" type="iron_ore" />
    <Item reward="128" type="iron_ingot" />
    <Item reward="256" type="iron_pickaxe" />
    <Item reward="1024" type="diamond" />

\n"""
    },
    max_episode_steps=18000,
)


#######################
#        DEBUG        #
#######################

register(
    id='MineRLNavigateDenseFixed-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'navigationDenseFixedMap.xml'),
        'observation_space': navigate_observation_space,
        'action_space': navigate_action_space,
    },
    max_episode_steps=6000,
)

register(
    id='MineRLObtainTest-v0',
    entry_point='minerl.env:MineRLEnv',
    kwargs={
        'xml': os.path.join(missions_dir, 'obtainDebug.xml'),
        'observation_space': obtain_observation_space,
        'action_space':  spaces.Dict({
            "forward": spaces.Discrete(2),
            "back": spaces.Discrete(2),
            "left": spaces.Discrete(2),
            "right": spaces.Discrete(2),
            "jump": spaces.Discrete(2),
            "sneak": spaces.Discrete(2),
            "sprint": spaces.Discrete(2),
            "attack": spaces.Discrete(2),
            "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),  # Pitch, Yaw
            "place": spaces.Enum('none', 'dirt', 'log', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'),
            "equip": spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'),
            "craft": spaces.Enum('none', 'torch', 'stick', 'planks', 'crafting_table'),
            "nearbyCraft": spaces.Enum('none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'),
            "nearbySmelt": spaces.Enum('none', 'iron_ingot', 'coal')})
    },
    max_episode_steps=2000,
)


#######################
#     CONTRIBUTED     #
#######################
contributed_env_infos = [
    ('MineRLFlatGrid-v0', 'flatgrid.xml'),
    ('MineRLForaging-v0', 'foraging.xml'),
    ('MineRLGridUnitTest-v0', 'grid_unit_test.xml'),
    ('MineRLStairsUnitTest-v0', 'stairs_unit_test.xml'),
    ('MineRLSafetyUnitTest-v0', 'safety_unit_test.xml'),
    ('MineRLSafetyUnitTest2-v0', 'safety_unit_test2.xml'),
    ('MineRLSafetyUnitTest3-v0', 'safety_unit_test3.xml'),
    ('MineRLMazeTest-v0', 'maze_test.xml'),
    ('MineRLAscendingMazeTest-v0', 'ascending_maze_test.xml'),
    ('MineRLOpenRoomTest-v0', 'open_room_test.xml'),
    ('MineRLBumpyRoomTest-v0', 'bumpy_room_test.xml'),
    ('MineRLWoodUnitTest-v0', 'wood_unit_test.xml'),
    ('MineRLWoodUnitTest2-v0', 'wood_unit_test2.xml'),
    ('MineRLCraftingTableTest-v0', 'crafting_table_test.xml'),
    ('MineRLLeveling-v0', 'leveling.xml'),
]

for env_id, xml in contributed_env_infos:
    register(
        id=env_id,
        entry_point='minerl.env:MineRLEnv',
        kwargs={
            'xml': os.path.join(missions_dir, xml),
            'observation_space': spaces.Dict({
                'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                'XPos': spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
                'YPos': spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
                'ZPos': spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
                'Yaw' : spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                'grid': spaces.MultiDiscrete([5, 5, 5]),
                'inventory': spaces.Dict({
                    'dirt': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'coal': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'torch': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'log': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'planks': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'stick': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'crafting_table': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'wooden_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'wooden_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'stone': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'cobblestone': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'furnace': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'stone_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'stone_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'iron_ore': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'iron_ingot': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'iron_axe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                    'iron_pickaxe': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
                }),
                'equipped_items': spaces.Dict({
                    'mainhand': spaces.Dict({
                        'type': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe',
                                            'iron_axe', 'iron_pickaxe', 'other'),
                        'damage': spaces.Box(low=-1, high=1562, shape=(), dtype=np.int),
                        'maxDamage': spaces.Box(low=-1, high=1562, shape=(), dtype=np.int),
                    })
                })
            }),
            'action_space': spaces.Dict(spaces={
                # "forward": spaces.Discrete(2), 
                # "back": spaces.Discrete(2), 
                # "left": spaces.Discrete(2), 
                # "right": spaces.Discrete(2),
                # "forward": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                # "back": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                # "left": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                # "right": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "move" : spaces.Box(low=-1, high=1, shape=(), dtype=np.float32),
                "strafe" : spaces.Box(low=-1, high=1, shape=(), dtype=np.float32),
                # "turn" : spaces.Box(low=-1, high=1, shape=(), dtype=np.int),
                "jump" : spaces.Discrete(2),
                "attack" : spaces.Discrete(2),
                # "setYaw": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.int),
                "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
                "place": spaces.Enum('none', 'dirt', 'log', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'),
                "equip": spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'),
                "craft": spaces.Enum('none', 'torch', 'stick', 'planks', 'crafting_table'),
                "nearbyCraft": spaces.Enum('none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'),
                "nearbySmelt": spaces.Enum('none', 'iron_ingot', 'coal')
            }),
            'docstr': """TODO"""
        },
    )



layout0 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences0 = np.inf

layout1 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]
num_fences1 = np.inf

layout2 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]
num_fences2 = np.inf

layout3 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]
num_fences3 = np.inf

layout4 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
]
num_fences4 = np.inf

layout5 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx'],
]
num_fences5 = np.inf

layout6 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences6 = np.inf

layout7 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences7 = np.inf

layout8 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences8 = np.inf

layout9 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'sheep', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences9 = np.inf

layout10 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences10 = np.inf

layout11 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'sheep', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
]
num_fences11 = np.inf

layout12 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence',  'fence'],
]
num_fences12 = np.inf

layout13 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]
num_fences13 = np.inf

layout14 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence', 'fence', 'fence'],
]
num_fences14 = np.inf

layout15 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx'],
]
num_fences15 = np.inf

layout16 = [
    ['xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences16 = np.inf

layout17 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
]
num_fences17 = np.inf

layout18 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]
num_fences18 = np.inf


layout19 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'sheep', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
]
num_fences19 = np.inf


LAYOUTS = [(0, layout0, num_fences0), (1, layout1, num_fences1), (2, layout2, num_fences2), 
    (3, layout3, num_fences3), (4, layout4, num_fences4), (5, layout5, num_fences5), (6, layout6, num_fences6), 
    (7, layout7, num_fences7), (8, layout8, num_fences8), (9, layout9, num_fences9), (10, layout10, num_fences10), 
    (11, layout11, num_fences11), (12, layout12, num_fences12), (13, layout13, num_fences13), (14, layout14, num_fences14), 
    (15, layout15, num_fences15), (16, layout16, num_fences16), (17, layout17, num_fences17), (18, layout18, num_fences18), 
    (19, layout19, num_fences19)]

# from minerl.env.minecraft_enclose
for layout_id, layout, num_fences in LAYOUTS:

    register(
        id='MinecraftEnclose{}-v0'.format(layout_id),
        entry_point='minerl.env.minecraft_enclose:MinecraftEnclose',
        kwargs={'layout' : layout, 'num_fences' : num_fences},
    )

    register(
        id='MiniMinecraftEnclose{}-v0'.format(layout_id),
        entry_point='minerl.env.minecraft_enclose:MiniMinecraftEnclose',
        kwargs={'layout' : layout, 'num_fences' : num_fences},
    )

