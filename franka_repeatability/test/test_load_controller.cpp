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

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "controller_manager/controller_manager.hpp"
#include "hardware_interface/resource_manager.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "ros2_control_test_assets/descriptions.hpp"

class TestLoadRepeatabilityController : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

// Catches the mistakes that otherwise only surface on hardware: a plugin description that does
// not match the exported class, or a library that fails to load.
TEST_F(TestLoadRepeatabilityController, LoadsFromPluginDescription) {
  std::shared_ptr<rclcpp::Executor> executor =
      std::make_shared<rclcpp::executors::SingleThreadedExecutor>();

  controller_manager::ControllerManager controller_manager(
      std::make_unique<hardware_interface::ResourceManager>(
          ros2_control_test_assets::minimal_robot_urdf),
      executor, "test_controller_manager_repeatability");

  auto controller = controller_manager.load_controller(
      "test_repeatability_ik_controller",
      "franka_repeatability/CartesianTargetImpedanceIKController");

  ASSERT_NE(controller, nullptr);
}
