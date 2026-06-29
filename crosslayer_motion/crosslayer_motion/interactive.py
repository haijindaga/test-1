#!/usr/bin/env python3
"""Interactive cross-layer demo: talk to the Panda over colored areas.

A live REPL. The 4 colored areas are shown (RViz markers; optional Gazebo
tiles). You type a natural-language instruction such as

    青を避けて赤へ      /   go to red avoiding blue
    次は黄色を避けて白へ /   now go to white avoiding yellow

A local LLM (Ollama) parses it into {target, avoid:[...]}; the named "avoid"
areas become MoveIt collision boxes, and the arm moves to the target FROM ITS
CURRENT POSE. Then it waits for the next instruction -- state persists, so you
chain commands and the arm continues from wherever it is.

Run the panda launch first (Gazebo + MoveIt), then:

    ros2 run crosslayer_motion interactive --ros-args \
        -p layout:=/abs/path/to/layout.yaml

Needs: a running Ollama server (env OLLAMA_MODEL / OLLAMA_BASE_URL),
`pip install openai`, pymoveit2, and the panda launch running.
Gazebo tiles are opt-in: add -p spawn_gazebo:=true (experimental).
"""

import json
import os
import subprocess
import threading

import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _system_prompt(regions) -> str:
    return (
        "You convert a navigation instruction for a robot arm into JSON.\n"
        f"The arm can go to these areas: {', '.join(regions)}.\n"
        "Color words may be Japanese: 白=white, 赤=red, 黄/黄色=yellow, 青=blue.\n"
        'Output ONLY JSON: {"target": "<area>", "avoid": ["<area>", ...]} '
        "where each <area> is exactly one of the area names above "
        "(e.g. red_area). If there is nothing to avoid, use an empty list. "
        "No prose, no markdown."
    )


class Interactive(Node):
    def __init__(self) -> None:
        super().__init__("interactive")
        self.declare_parameter("layout", "")
        self.declare_parameter("spawn_gazebo", False)
        self.declare_parameter("world", "empty_world")
        layout_path = self.get_parameter("layout").get_parameter_value().string_value
        assert layout_path, "missing -p layout:=/abs/path/to/layout.yaml"
        self.spawn_gazebo = self.get_parameter("spawn_gazebo").get_parameter_value().bool_value
        self.world = self.get_parameter("world").get_parameter_value().string_value

        layout = _load_yaml(layout_path)
        self.regions = layout["regions"]
        self.ee_quat = layout.get("ee_orientation_xyzw", [0.0, 1.0, 0.0, 0.0])
        obstacle = layout.get("obstacle", {})
        self.obstacle_size = list(obstacle.get("default_size", [0.10, 0.10, 0.50]))
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

        self.marker_pub = self.create_publisher(MarkerArray, "crosslayer_areas", 10)

        from openai import OpenAI  # lazy

        self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )

    # -- visualisation ----------------------------------------------------- #
    def publish_area_markers(self) -> None:
        arr = MarkerArray()
        for i, (name, r) in enumerate(self.regions.items()):
            m = Marker()
            m.header.frame_id = panda.base_link_name()
            m.ns = "areas"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(r["x"])
            m.pose.position.y = float(r["y"])
            m.pose.position.z = self.obstacle_zbase + 0.005
            m.pose.orientation.w = 1.0
            m.scale.x = 0.12
            m.scale.y = 0.12
            m.scale.z = 0.01
            c = r.get("color", [0.5, 0.5, 0.5])
            m.color.r, m.color.g, m.color.b = float(c[0]), float(c[1]), float(c[2])
            m.color.a = 0.85
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def spawn_gazebo_tiles(self) -> None:
        if not self.spawn_gazebo:
            return
        for name, r in self.regions.items():
            c = r.get("color", [0.5, 0.5, 0.5])
            sdf = (
                '<?xml version="1.0"?>'
                f'<sdf version="1.6"><model name="area_{name}"><static>true</static>'
                '<link name="link"><visual name="v"><geometry><box><size>0.12 0.12 0.01</size>'
                f"</box></geometry><material><ambient>{c[0]} {c[1]} {c[2]} 0.9</ambient>"
                f"<diffuse>{c[0]} {c[1]} {c[2]} 0.9</diffuse></material></visual></link></model></sdf>"
            )
            try:
                subprocess.run(
                    [
                        "ros2", "run", "ros_gz_sim", "create",
                        "-world", self.world, "-string", sdf, "-name", f"area_{name}",
                        "-x", str(r["x"]), "-y", str(r["y"]),
                        "-z", str(self.obstacle_zbase + 0.005),
                    ],
                    timeout=20, check=False,
                )
            except Exception as exc:  # never let a spawn failure break the REPL
                self.get_logger().warn(f"gazebo spawn failed for {name}: {exc}")

    # -- obstacles + motion ------------------------------------------------ #
    def set_obstacles(self, region_names) -> None:
        for box_id in list(self._active_boxes):
            self.moveit2.remove_collision_object(id=box_id)
        self._active_boxes.clear()
        for region in region_names:
            if region not in self.regions:
                self.get_logger().warn(f"avoid region '{region}' unknown; skipped")
                continue
            r = self.regions[region]
            box_id = f"obstacle_{region}"
            self.moveit2.add_collision_box(
                id=box_id,
                size=self.obstacle_size,
                position=[
                    float(r["x"]), float(r["y"]),
                    self.obstacle_zbase + self.obstacle_size[2] / 2.0,
                ],
                quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
            self._active_boxes.add(box_id)

    def go(self, region: str) -> bool:
        r = self.regions[region]
        self.moveit2.move_to_pose(
            position=[float(r["x"]), float(r["y"]), float(r["z"])],
            quat_xyzw=self.ee_quat,
        )
        return bool(self.moveit2.wait_until_executed())

    # -- LLM --------------------------------------------------------------- #
    def parse_instruction(self, text: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _system_prompt(list(self.regions))},
                {"role": "user", "content": text},
            ],
        )
        content = resp.choices[0].message.content or ""
        s, e = content.find("{"), content.rfind("}")
        obj = json.loads(content[s : e + 1])
        target = str(obj.get("target", "")).strip()
        avoid = [str(a).strip() for a in obj.get("avoid", []) if str(a).strip()]
        return target, avoid

    # -- REPL -------------------------------------------------------------- #
    def repl(self) -> None:
        self.publish_area_markers()
        self.spawn_gazebo_tiles()
        print("\n=== cross-layer interactive demo ===")
        print("areas:", ", ".join(self.regions.keys()))
        print("例: 青を避けて赤へ  /  go to white avoiding yellow  /  quit で終了\n")
        while rclpy.ok():
            try:
                text = input("命令 > ").strip()
            except EOFError:
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            self.publish_area_markers()
            try:
                target, avoid = self.parse_instruction(text)
            except Exception as exc:
                print(f"  (解釈できなかった: {exc})")
                continue
            if target not in self.regions:
                print(f"  (目標 '{target}' が領域にない。{list(self.regions)} のどれか)")
                continue
            print(f"  -> target={target}, avoid={avoid}")
            self.set_obstacles(avoid)
            print("  OK 到達" if self.go(target) else "  失敗（経路なし/届かない）")
        print("終了")


def main() -> None:
    rclpy.init()
    node = Interactive()
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    node.create_rate(1.0).sleep()  # let MoveIt2 + publishers initialise
    try:
        node.repl()
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
