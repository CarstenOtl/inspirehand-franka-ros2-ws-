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

#include <cmath>
#include <memory>

#include <controller_manager/controller_manager.hpp>
#include <hardware_interface/resource_manager.hpp>
#include <rclcpp/executor.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/utilities.hpp>
#include <ros2_control_test_assets/descriptions.hpp>

#include <franka_forge_cartesian_impedance/cartesian_impedance_controller.hpp>

using franka_forge_cartesian_impedance::CartesianImpedanceController;

TEST(TestLoadCartesianImpedanceController, load_controller) {
  rclcpp::init(0, nullptr);
  std::shared_ptr<rclcpp::Executor> executor =
      std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  rclcpp::Logger logger = rclcpp::get_logger("load_controller");
  controller_manager::ControllerManager cm(
      std::make_unique<hardware_interface::ResourceManager>(
          ros2_control_test_assets::minimal_robot_urdf,
          std::make_shared<rclcpp::Clock>(), logger),
      executor, "test_controller_manager");
  ASSERT_NE(cm.load_controller("test_cartesian_impedance_controller",
                               "franka_forge_cartesian_impedance/CartesianImpedanceController"),
            nullptr);
  rclcpp::shutdown();
}

TEST(CartesianImpedanceControllerMath, sample_trajectory_slerps_and_hermites) {
  CartesianImpedanceController::Trajectory trajectory;
  trajectory.times = {0.0, 1.0, 2.0};
  trajectory.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}};
  // 0, 90 and 180 degrees about z, with the last one on the opposite hemisphere on purpose.
  trajectory.orientations = {{0, 0, 0, 1},
                             {0, 0, std::sin(M_PI / 4), std::cos(M_PI / 4)},
                             {0, 0, -1, 0}};
  trajectory.velocities = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  trajectory.has_velocities = true;
  trajectory.nullspace = {{0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2}};
  size_t hint = 0;
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  Eigen::Matrix<double, 7, 1> n;
  CartesianImpedanceController::sample_trajectory(trajectory, 1.0, hint, p, q, n);
  EXPECT_NEAR(p.x(), 1.0, 1e-12);
  EXPECT_NEAR(n(0), 1.0, 1e-12);
  CartesianImpedanceController::sample_trajectory(trajectory, 0.5, hint, p, q, n);
  EXPECT_NEAR(p.x(), 0.5, 1e-12);  // zero-velocity Hermite midpoint is the smoothstep value
  EXPECT_NEAR(n(0), 0.5, 1e-12);   // nullspace is linear
  EXPECT_NEAR(2.0 * std::atan2(q.z(), q.w()), M_PI / 4, 1e-12);  // slerp midpoint: 45 degrees
  CartesianImpedanceController::sample_trajectory(trajectory, 1.5, hint, p, q, n);
  // Slerp across the hemisphere flip must take the short way: 135 degrees, not -45.
  const double angle = 2.0 * std::atan2(q.z(), q.w());
  EXPECT_NEAR(std::abs(std::fmod(angle + 2 * M_PI, 2 * M_PI)), 3 * M_PI / 4, 1e-9);
  CartesianImpedanceController::sample_trajectory(trajectory, 5.0, hint, p, q, n);
  EXPECT_NEAR(p.y(), 1.0, 1e-12);  // clamped to the end
  CartesianImpedanceController::sample_trajectory(trajectory, -1.0, hint, p, q, n);
  EXPECT_NEAR(p.x(), 0.0, 1e-12);  // clamped to the start
  EXPECT_DOUBLE_EQ(CartesianImpedanceController::quintic_blend(0.5), 0.5);
  EXPECT_STREQ(
      CartesianImpedanceController::phase_name(CartesianImpedanceController::Phase::kPolicy),
      "policy");
}
