// Copyright (c) 2026 Agile Robots SE
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gmock/gmock.h>

#include <array>
#include <cmath>
#include <memory>

#include <controller_manager/controller_manager.hpp>
#include <hardware_interface/resource_manager.hpp>
#include <rclcpp/executor.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/utilities.hpp>
#include <ros2_control_test_assets/descriptions.hpp>

#include <franka_trajectory_replay/trajectory_replay_controller.hpp>

using franka_trajectory_replay::TrajectoryReplayController;

TEST(TestLoadTrajectoryReplayController, load_controller) {
  rclcpp::init(0, nullptr);
  std::shared_ptr<rclcpp::Executor> executor =
      std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  controller_manager::ControllerManager cm(
      std::make_unique<hardware_interface::ResourceManager>(
          ros2_control_test_assets::minimal_robot_urdf),
      executor, "test_controller_manager");
  ASSERT_NE(cm.load_controller("test_trajectory_replay_controller",
                               "franka_trajectory_replay/TrajectoryReplayController"),
            nullptr);
  rclcpp::shutdown();
}

TEST(TrajectoryReplayControllerMath, quintic_blend_endpoints) {
  EXPECT_DOUBLE_EQ(TrajectoryReplayController::quintic_blend(0.0), 0.0);
  EXPECT_DOUBLE_EQ(TrajectoryReplayController::quintic_blend(1.0), 1.0);
  EXPECT_DOUBLE_EQ(TrajectoryReplayController::quintic_blend_derivative(0.0), 0.0);
  EXPECT_DOUBLE_EQ(TrajectoryReplayController::quintic_blend_derivative(1.0), 0.0);
  // Peak velocity of the quintic is 1.875 at s = 0.5.
  EXPECT_NEAR(TrajectoryReplayController::quintic_blend_derivative(0.5), 1.875, 1e-12);
}

TEST(TrajectoryReplayControllerMath, velocity_limits_match_libfranka_shape) {
  std::array<double, 7> home{0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
  const auto upper = TrajectoryReplayController::upper_velocity_limits(home);
  const auto lower = TrajectoryReplayController::lower_velocity_limits(home);
  for (int i = 0; i < 7; ++i) {
    EXPECT_GT(upper[i], 0.0) << "joint " << i + 1;
    EXPECT_LT(lower[i], 0.0) << "joint " << i + 1;
    EXPECT_LE(upper[i], 5.26);
    EXPECT_GE(lower[i], -5.26);
  }
  // Well inside the range the limits are the datasheet values minus the packet-loss tolerance.
  EXPECT_NEAR(upper[0], 2.62 - 0.031, 1e-9);
  EXPECT_NEAR(lower[0], -2.62 + 0.031, 1e-9);
  // Near the upper position limit of joint 1 the allowed positive velocity collapses.
  std::array<double, 7> near_limit = home;
  near_limit[0] = 2.70;
  EXPECT_LT(TrajectoryReplayController::upper_velocity_limits(near_limit)[0], 0.6);
}

TEST(TrajectoryReplayControllerMath, sample_trajectory_hermite_hits_knots) {
  TrajectoryReplayController::Trajectory trajectory;
  trajectory.times = {0.0, 1.0, 2.0};
  trajectory.positions = {{0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0}};
  trajectory.velocities = {{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}};
  trajectory.has_velocities = true;
  size_t hint = 0;
  std::array<double, 7> sample{};
  TrajectoryReplayController::sample_trajectory(trajectory, 1.0, hint, sample);
  EXPECT_NEAR(sample[0], 1.0, 1e-12);
  TrajectoryReplayController::sample_trajectory(trajectory, 0.5, hint, sample);
  EXPECT_NEAR(sample[0], 0.5, 1e-12);  // zero-velocity Hermite midpoint is the smoothstep value
  TrajectoryReplayController::sample_trajectory(trajectory, 5.0, hint, sample);
  EXPECT_NEAR(sample[0], 0.0, 1e-12);  // clamped to the end
  TrajectoryReplayController::sample_trajectory(trajectory, -1.0, hint, sample);
  EXPECT_NEAR(sample[0], 0.0, 1e-12);  // clamped to the start
}
