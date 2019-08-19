from core import MineRLEnv

import spaces

import gym
import numpy as np
import tempfile
import imageio

import logging
logging.basicConfig(level=logging.INFO)


class MinecraftChase(MineRLEnv):

    def __init__(self, layout, render_inner=False, video_outfile='out.mp4', fps=20, *args, **kwargs):
        self.layout = np.array(layout, dtype=object)
        self.render_inner = render_inner
        self.video_outfile = video_outfile
        self.fps = fps
        self.inner_images = []

        self.grid_mins = (0, 1, 0)
        self.grid_maxs = (self.layout.shape[0] - 1, 1, self.layout.shape[1] - 1)

        self.agent_start = (self.layout.shape[1] // 2, -4)
        sheep_r, sheep_c = np.argwhere(self.layout == 'sheep')[0]
        sheep_z, sheep_x = self.layout.shape[0] - sheep_r - 1, self.layout.shape[1] - sheep_c - 1
        self.sheep_start = (sheep_x, sheep_z)

        xml = self.create_xml_file()

        observation_space = spaces.Dict({
            'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'grid': spaces.MultiDiscrete([5, 5, 5]),
            'inventory': spaces.Dict({
                'fence': spaces.Box(low=0, high=2304, shape=(), dtype=np.int),
            }),
        })

        action_space = spaces.Dict(spaces={
            "place": spaces.Enum('none', 'fence'),
            "tpx": spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
            "tpy": spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
            "tpz": spaces.Box(low=-2**63, high=2**63, shape=(1,), dtype=np.int64),
        })

        super().__init__(xml, observation_space, action_space, *args, **kwargs)

    def reset(self):
        obs = super().reset()
        if self.render_inner:
            self.inner_images.append(self.render())
        return self.post_process_obs(obs)

    def step(self, action):
        # Get a fresh obs
        obs, reward, done, debug_info = self._step({})

        # Check that the target pos is empty
        if obs[action[0], action[1]] != None or action[0] == 0:
            logging.debug("Return 1")
            # import pdb; pdb.set_trace()
            return obs, reward, done, debug_info

        # Check that the target+1 pos is empty
        if obs[action[0]-1, action[1]] != None:
            logging.debug("Return 2")
            return obs, reward, done, debug_info

        # Go to the target+1 pos
        move_action = {
            'tpx' : self.layout.shape[1] - action[1] - 0.5,
            'tpy' : 1.0,
            'tpz' : self.layout.shape[0] - action[0] - 1.5,
        }

        logging.debug("Taking move action")
        obs, reward, done, debug_info = self._step(move_action)

        # Let env settle
        for _ in range(2):
            obs, reward, done, debug_info = self._step({})

        # Try to build the fence
        logging.debug("Trying to build fence")

        build_action = {'place' : 'fence'}
        fence_count = np.sum(obs == 'fence')
        for _ in range(5):
            obs, reward, done, debug_info = self._step(build_action)
            if np.sum(obs == 'fence') > fence_count:
                break

            # Let env settle
            for _ in range(2):
                obs, reward, done, debug_info = self._step({})

        if np.sum(obs == 'fence') == fence_count:
            logging.debug("Failed to build fence")
            import pdb; pdb.set_trace()

        return obs, reward, done, debug_info

    def _step(self, action):
        obs, reward, done, debug_info = super().step(action)
        if self.render_inner:
            self.inner_images.append(self.render())
        obs = self.post_process_obs(obs)
        return obs, reward, done, debug_info

    def post_process_obs(self, obs):
        grid = obs['grid']
        shape = 1 + np.subtract(self.grid_maxs, self.grid_mins)
        arr = np.array(grid, dtype='object').reshape((shape[2], shape[0], shape[1]), order='F')
        arr = np.moveaxis(arr, 0, -1)
        arr = np.rot90(arr, k=2, axes=(0, 2))

        assert arr.shape[1] == 1
        arr = arr.squeeze()
        assert len(arr.shape) == 2

        arr[arr == 'air'] = None

        return arr

    def create_xml_file(self):
        layout_blocks = self.create_layout_xml_blocks()

        xml = self.create_xml({
            'GRID_MIN_X' : self.grid_mins[2],
            'GRID_MIN_Y' : self.grid_mins[1],
            'GRID_MIN_Z' : self.grid_mins[0],
            'GRID_MAX_X' : self.grid_maxs[2],
            'GRID_MAX_Y' : self.grid_maxs[1],
            'GRID_MAX_Z' : self.grid_maxs[0],
            'AGENT_X' : self.agent_start[0],
            'AGENT_Z' : self.agent_start[1],
            'SHEEP_X' : self.sheep_start[0],
            'SHEEP_Z' : self.sheep_start[1],
            'LAYOUT_BLOCKS' : layout_blocks,
        })
        new_f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        new_f.write(xml)
        return new_f.name

    def create_layout_xml_blocks(self):
        xml_blocks = ''

        # Create fences
        for r, c in np.argwhere(self.layout=='fence'):
            x = self.layout.shape[1] - c - 1
            z = self.layout.shape[0] - r - 1
            fence_block = self.create_fence_xml_block(x, z)
            xml_blocks += fence_block

        return xml_blocks

    def create_fence_xml_block(self, x, z):
        return '<DrawBlock type="fence" x="{}" y="1" z="{}"/>'.format(x, z)

    def create_xml(self, fill_ins):
        xml = '''
        <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <About>
                    <Summary>MinecraftChase</Summary>
                </About>

                <ModSettings>
                    <MsPerTick>50</MsPerTick>
                </ModSettings>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>6000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                        <AllowSpawning>false</AllowSpawning>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="2;2;1;"/>
                        <DrawingDecorator>
                            $(LAYOUT_BLOCKS)
                            <DrawEntity type="Sheep" x="$(SHEEP_X)" y="1" z="$(SHEEP_Z)" />
                        </DrawingDecorator>
                        <ServerQuitFromTimeUp timeLimitMs="300000" description="out_of_time"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>MineRLAgent</Name>
                    <AgentStart>
                        <Placement x="$(AGENT_X)" y="1" z="$(AGENT_Z)" pitch="60" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="0" type="fence" quantity="50"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <VideoProducer want_depth="false" viewpoint="1">
                            <Width>512</Width>
                            <Height>512</Height>
                        </VideoProducer>
                        <FileBasedPerformanceProducer/>
                        <ObservationFromFullInventory flat="false"/>
                        <ObservationFromFullStats/>
                        <ObservationFromEquippedItem/>
                        <HumanLevelCommands/>
                        <AbsoluteMovementCommands/>
                        <CameraCommands/>
                        <PlaceCommands/>
                        <EquipCommands/>
                        <ObservationFromGrid>
                           <Grid name="grid" absoluteCoords="true">
                              <min x="$(GRID_MIN_X)" y="$(GRID_MIN_Y)" z="$(GRID_MIN_Z)" />
                              <max x="$(GRID_MAX_X)" y="$(GRID_MAX_Y)" z="$(GRID_MAX_Z)" />
                           </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                </AgentSection>
            </Mission>
        '''

        for placeholder, fill_in in fill_ins.items(): 
            xml = xml.replace('$({})'.format(placeholder), str(fill_in))

        # print(xml)
        # assert False
        # import pdb; pdb.set_trace()

        return xml

    def close(self):
        if self.render_inner:
            imageio.mimsave(self.video_outfile, self.inner_images, fps=self.fps)
            print("Wrote out video to {}.".format(self.video_outfile))
        return super().close()


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
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence', 'fence'],
]

layout1 = [
    ['fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'xxxxx', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'sheep', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence'],
    ['fence', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'fence', 'fence'],
]


def demo():
    env = MinecraftChase(layout1, render_inner=True, video_outfile='MinecraftChase_demo.mp4', fps=20)

    obs = env.reset()

    T = 250

    actions = [(obs.shape[0]-1, 2), (obs.shape[0]-1, 3), (obs.shape[0]-1, 4), (obs.shape[0]-1, 5)]

    for t in range(T):
        if t < len(actions):
            action = actions[t]
        else:
            action = (0, 0)
        obs, reward, done, debug_info = env.step(action)

    env.close()


if __name__ == '__main__':
    demo()
