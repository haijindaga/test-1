#!/usr/bin/env python3
"""Phase 2 of the cross-layer demo: execute a plan.json on the Gazebo Franka.

Reads a plan.json (produced by the brain's make_plan.py) and a layout.yaml
(region poses + obstacle box), then for each go-action:

  1. set the PlanningScene obstacles = a collision box for each obstacle region
     in this step (the FindObstacle output),
  2. move the end-effector to the target region; MoveIt plans a path that keeps
     the *whole arm* out of the boxes.

grab/place actions are skipped in this go-only Tier-1 demo.

Termination/start: the arm first moves to ``scan_pose`` (consistent start), then
each subsequent move begins from wherever the previous one ended (continuous).

Usage (run the launch first, then):
    ros2 run crosslayer_motion motion_executor --ros-args \
        -p plan:=/abs/path/to/nav_demo_with.json \
        -p layout:=/abs/path/to/layout.yaml

Run it once with ``..._with.json`` (arm avoids the boxes) and once with
``..._without.json`` (no boxes -> arm cuts straight through) to see the
with/without-supervisor contrast.
"""

import json
from threading import Thread

import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda

GO_PREFIX = "go_to_"


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MotionExecutor(Node):
    def __init__(self) -> None:
        super().__init__("motion_executor")

        self.declare_parameter("plan", "")
        self.declare_parameter("layout", "")
        plan_path = self.get_parameter("plan").get_parameter_value().string_value
        layout_path = self.get_parameter("layout").get_parameter_value().string_value
        assert plan_path, "missing -p plan:=/abs/path/to/plan.json"
        assert layout_path, "missing -p layout:=/abs/path/to/layout.yaml"

        self.plan = _load_json(plan_path)
        layout = _load_yaml(layout_path)
        self.regions = layout["regions"]
        self.ee_quat = layout.get("ee_orientation_xyzw", [0.0, 1.0, 0.0, 0.0])
        obstacle = layout.get("obstacle", {})
        self.obstacle_size = list(obstacle.get("default_size", [0.12, 0.12, 0.30]))
        self.obstacle_zbase = float(obstacle.get("z_base", 0.0))
        planning = layout.get("planning", {})

        self._cb = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self._cb,
        )
        self.moveit2.max_velocity = float(planning.get("max_velocity", 0.1))
        self.moveit2.max_acceleration = float(planning.get("max_acceleration", 0.1))
        self._active_boxes: set = set()

        self.get_logger().info(
            f"plan={plan_path} (scenario={self.plan.get('scenario')}, "
            f"supervisor={self.plan.get('supervisor')}, task={self.plan.get('task')!r})"
        )

    # -- planning scene ---------------------------------------------------- #
    def _region_pose(self, region: str):
        r = self.regions[region]
        return [float(r["x"]), float(r["y"]), float(r["z"])]

    def _box_position(self, region: str):
        """Box centred on the region's (x, y), sitting on the table from z_base up."""
        r = self.regions[region]
        return [float(r["x"]), float(r["y"]), self.obstacle_zbase + self.obstacle_size[2] / 2.0]

    def _set_obstacles(self, region_names) -> None:
        # remove the previous step's boxes, then add this step's
        for box_id in list(self._active_boxes):
            self.moveit2.remove_collision_object(id=box_id)
        self._active_boxes.clear()
        for region in region_names:
            if region not in self.regions:
                self.get_logger().warn(f"obstacle region '{region}' not in layout; skipped")
                continue
            box_id = f"obstacle_{region}"
            self.moveit2.add_collision_box(
                id=box_id,
                size=self.obstacle_size,
                position=self._box_position(region),
                quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
            self._active_boxes.add(box_id)

    # -- motion ------------------------------------------------------------ #
    def _go(self, region: str) -> bool:
        avoiding = ", ".join(sorted(self._active_boxes)) or "nothing"
        self.get_logger().info(f"-> move to {region}  (avoiding: {avoiding})")
        self.moveit2.move_to_pose(position=self._region_pose(region), quat_xyzw=self.ee_quat)
        ok = self.moveit2.wait_until_executed()
        if not ok:
            self.get_logger().error(f"FAILED to plan/execute move to {region}")
        return bool(ok)

    def execute(self) -> None:
        # consistent start: clear the scene and move to the scan pose
        self._set_obstacles([])
        if "scan_pose" in self.regions:
            self._go("scan_pose")

        for i, step in enumerate(self.plan.get("steps", [])):
            action = step.get("action", "")
            if not action.startswith(GO_PREFIX):
                self.get_logger().info(f"[{i}] skip non-go action: {action}")
                continue
            region = action[len(GO_PREFIX):]
            if region not in self.regions:
                self.get_logger().error(f"[{i}] region '{region}' not in layout; stopping")
                return
            self._set_obstacles(step.get("obstacles", []))
            if not self._go(region):
                self.get_logger().error(f"[{i}] stopping at failed step: {action}")
                return
        self.get_logger().info("plan complete")


def main() -> None:
    rclpy.init()
    node = MotionExecutor()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    node.create_rate(1.0).sleep()  # let the MoveIt2 interface initialise

    try:
        node.execute()
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
