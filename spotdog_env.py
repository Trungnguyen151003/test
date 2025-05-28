import gym
import pybullet as p
import pybullet_data
import numpy as np
import os
from gym import spaces

class SpotDogEnv(gym.Env):
    def __init__(self, render=False):
        super(SpotDogEnv, self).__init__()

        self.render_mode = render
        self.time_step = 1.0 / 240.0

        if p.getConnectionInfo()['isConnected'] == 0:
            if self.render_mode:
                p.connect(p.GUI)
            else:
                p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)

        self.plane = p.loadURDF("plane.urdf")

        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../simulation/SpotDog2305/urdf/SpotDog2305.urdf")
        )
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False)

        self.num_joints = p.getNumJoints(self.robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = self.num_joints * 2 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Thiết lập giới hạn bước ngẫu nhiên
        self.min_steps = 250
        self.max_steps = 500
        self._max_episode_steps = np.random.randint(self.min_steps, self.max_steps + 1)
        self.current_step = 0

        self.cam_dist = 2.0
        self.cam_yaw = 45
        self.cam_pitch = -30
        self.cam_target = [0, 0, 0.2]

    def reset(self):
        if p.getConnectionInfo()['isConnected'] == 0:
            if self.render_mode:
                p.connect(p.GUI)
            else:
                p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = p.loadURDF("plane.urdf")
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../simulation/SpotDog2305/urdf/SpotDog2305.urdf")
        )
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False)

        self.num_joints = p.getNumJoints(self.robot)
        self.current_step = 0
        self._max_episode_steps = np.random.randint(self.min_steps, self.max_steps + 1)
        return self._get_obs()

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array(joint_positions + joint_velocities + list(base_pos), dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL,
                targetPosition=action[i],
                force=20
            )

        p.stepSimulation()
        obs = self._get_obs()
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)

        reward = base_pos[0]
        done = base_pos[2] < 0.15 or self.current_step >= self._max_episode_steps

        return obs, reward, done, {}

    def close(self):
        if p.getConnectionInfo()['isConnected']:
            p.disconnect()
