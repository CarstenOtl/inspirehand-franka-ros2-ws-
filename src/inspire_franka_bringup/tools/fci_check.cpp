// Minimal FCI reachability check: no ROS, no controllers, no motion, no torque.
//
//     ros2 run inspire_franka_bringup fci_check 172.16.0.2
//
// It connects, reads one robot state, prints the joint angles and exits. That is the
// smallest possible answer to "is the FCI actually reachable and speaking my protocol",
// and it separates the three failures that otherwise all look like "it doesn't work":
//
//   NETWORK      - nothing listening / packets dropped / wrong address. Nothing to do with
//                  ROS, libfranka versions or controller configuration.
//   INCOMPATIBLE - TCP reached the FCI, but the robot's server version and this libfranka
//                  disagree. Only reachable AFTER a successful handshake.
//   FRANKA       - connected and compatible, but the robot refused (no control token, FCI
//                  not activated, brakes, an active error).
//
// libfranka's own communication_test would cover this too, but docker/Dockerfile builds
// libfranka with -DBUILD_EXAMPLES=OFF, so it is not in the image.
//
// RealtimeConfig::kIgnore on purpose: reading one state needs no real-time thread, and this
// must stay usable on a machine that has no PREEMPT_RT kernel yet.

#include <franka/exception.h>
#include <franka/robot.h>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fci_check <robot-ip-or-hostname>\n";
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const franka::RobotState state = robot.readOnce();

    std::cout << "OK: connected to " << argv[1] << " and read a robot state.\n";
    std::cout << std::fixed << std::setprecision(4) << "  q       = [";
    for (size_t i = 0; i < state.q.size(); ++i) {
      std::cout << state.q[i] << (i + 1 < state.q.size() ? ", " : "");
    }
    std::cout << "]\n  control = " << static_cast<int>(state.robot_mode) << " (robot_mode)\n";
    return 0;
  } catch (const franka::IncompatibleVersionException& e) {
    std::cerr << "INCOMPATIBLE: " << e.what() << "\n"
              << "  The FCI answered, but its server version and this libfranka disagree.\n";
    return 1;
  } catch (const franka::NetworkException& e) {
    std::cerr << "NETWORK: " << e.what() << "\n"
              << "  Never reached the FCI. Check that the address is right, that FCI is\n"
              << "  activated in Desk, and that tcp/1337 is actually open on that address.\n";
    return 1;
  } catch (const franka::Exception& e) {
    std::cerr << "FRANKA: " << e.what() << "\n"
              << "  Connected and compatible, but the robot refused. Usually the control\n"
              << "  token is held elsewhere, or the robot is not in Execution mode.\n";
    return 1;
  }
}
