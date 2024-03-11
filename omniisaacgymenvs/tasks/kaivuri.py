from omniisaacgymenvs.robots.articulations.kaivuri import Kaivuri
from omni.isaac.core.articulations import Articulation, ArticulationView  # test both!

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask

import torch

#from time import sleep


class KaivuriTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        self._max_episode_length = 200

        self.actions_multiplier = 2

        self.update_config(sim_config)

        self._num_observations = 3  # arm_end_ghetto_location but why 4...
        self._num_actions = 3  # joint_1(lift), joint_2(tilt), joint_3(rotate)

        self._kaivuri_position = torch.tensor([0.0, 0.0, 1.0])
        self._ball_position = torch.tensor([0.3, 0, 1.0])
        self.ball_radius = 0.03

        RLTask.__init__(self, name=name, env=env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        #self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

    def set_up_scene(self, scene) -> None:
        self.get_kaivuri()
        self.get_target()

        # self.get_boomEnd()

        RLTask.set_up_scene(self, scene)

        self._kaivurit = ArticulationView(prim_paths_expr="/World/envs/.*/Kaivuri", name="kaivuri_view")

        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view",reset_xform_properties=False)
        self._balls._non_root_link = True  # do not set states for kinematics

        self._boomEnds = RigidPrimView(prim_paths_expr="/World/envs/.*/Kaivuri/armEnd", name="boom_view", )
        self._boomEnds._non_root_link = True

        scene.add(self._kaivurit)
        scene.add(self._balls)
        scene.add(self._boomEnds)


    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("kaivuri_view"):
            scene.remove_object("kaivuri_view", registry_only=True)

        if scene.object_exists("targets_view"):
            scene.remove_object("targets_view", registry_only=True)
        self._kaivurit = ArticulationView(prim_paths_expr="/World/envs/.*/Kaivuri", name="kaivuri",)
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view",
                                    reset_xform_properties=False)

        self._boomEnds = RigidPrimView(prim_paths_expr="/World/envs/.*/Kaivuri/armEnd", name="boom_view", reset_xform_properties=False)

        scene.add(self._kaivurit)
        scene.add(self._balls)
        scene.add(self._boomEnds)


    def get_kaivuri(self):
        kaivuri = Kaivuri(prim_path=self.default_zero_env_path + "/Kaivuri", name="kaivuri", translation=self._kaivuri_position)

        self._sim_config.apply_articulation_settings("kaivuri", get_prim_at_path(kaivuri.prim_path), self._sim_config.parse_actor_config("kaivuri"))



    def get_boomEnd(self):
        boomEnd = RigidPrimView(prim_paths_expr="/World/envs/.*/Kaivuri/armEnd", name="end_view")
        self._sim_config.apply_articulation_settings("boomEnd", get_prim_at_path(boomEnd.prim_path), self._sim_config.parse_actor_config("boomEnd")) #ei oo configgia tälle vie
        boomEnd.set_collision_enabled(False)


    def get_target(self):
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=self.ball_radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path),
                                                     self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)


    def get_observations(self) -> dict:

        self.root_pos, self.root_rot = self._kaivurit.get_world_poses(clone=False)  # origo XYZ
        self.root_velocities = self._kaivurit.get_velocities(clone=False)

        self.boomEnd_pos, self.boomEnd_rot = self._boomEnds.get_world_poses(clone=False)

        # the most important thing on the whole script!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # this gets the individual local env position, not measured from the general center!"!
        self.root_positions = self.root_pos - self._env_pos
        self.boomEnd_positions = self.boomEnd_pos - self._env_pos

        #print(f"env0 kaivuri positions: {self.root_positions[0]}")
        #print(f"env0 boom end position: {self.boomEnd_positions[0]}")


        self.obs_buf[..., 0:3] = (self.target_positions - self.boomEnd_positions) / 3

        observations = {
            self._kaivurit.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:

        #print(f"env0 actions: {actions[0]}")
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)

        # apply actions
        self._kaivurit.set_joint_velocities(actions * self.actions_multiplier)


    def post_reset(self):
        print(f"Found joints: {self._kaivurit.dof_names}")

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1

        #root pos or root positions!!!!!1

        self.root_pos, self.root_rot = self._kaivurit.get_world_poses()
        self.root_velocities = self._kaivurit.get_velocities()
        self.dof_pos = self._kaivurit.get_joint_positions()
        self.dof_vel = self._kaivurit.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()


    """
    def set_targets(self, env_ids):

        # target spawn limits
        # x skips some values to not spawn the ball inside excavator
        x_min_front, x_max_front = 0.4, 0.7
        x_min_back, x_max_back = -0.7, -0.4
        y_min, y_max = -0.1, 0.1
        z_min, z_max = 0.7, 1.3


        num_sets = len(env_ids)
        envs_long = env_ids.long()

        # Randomly decide for each target whether to spawn in front or back
        for i in range(num_sets):
            if torch.rand(1).item() < 0.5:  # 50% chance to spawn in front, adjust as needed
                self.target_positions[envs_long[i], 0] = torch.rand(1, device=self._device) * (
                            x_max_front - x_min_front) + x_min_front
                self.target_positions[envs_long[i], 1] = torch.rand(1, device=self._device) * (
                            y_max - y_min) + y_min
                self.target_positions[envs_long[i], 2] = torch.rand(1, device=self._device) * (
                            z_max - z_min) + z_min
            else:  # Spawn behind
                self.target_positions[envs_long[i], 0] = torch.rand(1, device=self._device) * (
                            x_max_back - x_min_back) + x_min_back
                self.target_positions[envs_long[i], 1] = torch.rand(1, device=self._device) * (
                            y_max - y_min) + y_min
                self.target_positions[envs_long[i], 2] = torch.rand(1, device=self._device) * (
                            z_max - z_min) + z_min

        # Set the values
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)
        """


    def set_targets(self, env_ids):

        # target spawn limits
        x_min, x_max = 0.4,0.7
        y_min, y_max = -0.1, 0.1
        z_min, z_max = 0.7,1.3


        num_sets = len(env_ids)
        envs_long = env_ids.long()


        # Set target position randomly within specified limits for X, Y, Z
        self.target_positions[envs_long, 0] = torch.rand(num_sets, device=self._device) * (x_max - x_min) + x_min
        self.target_positions[envs_long, 1] = torch.rand(num_sets, device=self._device) * (y_max - y_min) + y_min
        self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) * (z_max - z_min) + z_min

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        #ball_pos[:, 2] += 0.4
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)



    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # Randomize joint angles within specified ranges
        self.dof_pos[env_ids, 1] = torch_rand_float(-10, 110 , (num_resets, 1), device=self._device).squeeze()
        self.dof_pos[env_ids, 2] = torch_rand_float(-40, 100, (num_resets, 1), device=self._device).squeeze()

        # Set joint velocities to 0 to start from a standstill
        self.dof_vel[env_ids, :] = 0

        # Keep kaivuri position the same
        root_pos = self.initial_root_pos.clone()
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # Apply resets
        #disabled for now
        #self._kaivurit.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        #self._kaivurit.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        #self._kaivurit.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        #self._kaivurit.set_velocities(root_velocities[env_ids], indices=env_ids)

        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:

        # Calculate distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions - self.boomEnd_positions).sum(-1))

        # Penalize being far from the target to encourage movement towards it
        # The penalty increases with distance
        distance_penalty = -0.2 * target_dist

        reward = distance_penalty

        # Check if within threshold distance and increase reward
        threshold_distance = self.ball_radius
        close_to_target = target_dist < threshold_distance
        hit_target_reward = 5  # Reward for hitting the target

        # Apply reward for hitting the target
        reward[close_to_target] += hit_target_reward

        # print target hit
        #if close_to_target.any():
            #print("target hit!")

        #print(f"env0 target distance: {target_dist[0]}")

        # Update the metrics
        self.target_dist = target_dist
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 20.0, ones, die)
        #die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)

        # Resets due to achieving goal (being very close to the target)
        achieved_goal = self.target_dist < self.ball_radius
        reset_due_to_success = torch.where(achieved_goal, ones, torch.zeros_like(self.reset_buf))

        # Combine conditions for reset: misbehavior, episode length, or success
        combined_reset_conditions = torch.max(die, reset_due_to_success)
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones,
                                        combined_reset_conditions)