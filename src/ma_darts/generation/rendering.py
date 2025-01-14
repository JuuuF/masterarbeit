import os
import bpy
import cv2
import sys
import numpy as np
import pandas as pd
import pickle
import random
from mathutils import Vector, Euler
from contextlib import contextmanager, nullcontext

# -----------------------------------------------
# Variables

BLEND_FILE = "data/generation/darts.blend"
HDRI_DIR = "data/generation/hdri/"
SUPPRESS_RENDER_OUTPUT = True

# -----------------------------------------------
# Load and setup


def init_project():
    bpy.ops.wm.open_mainfile(filepath=BLEND_FILE)

    # Set render engine
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 32

    # Enable GPU rendering
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_cpu = False
    bpy.context.preferences.addons["cycles"].preferences.get_devices_for_type("CUDA")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if not "NVIDIA" in d["name"]:
            continue
        d.use = True
        print("Using", d["name"])

    Compositor.init_vars()


WARNINGS = []


class SampleInfo:

    def reset(id: int, out_dir: str):
        # Clear everything
        for name, attr in SampleInfo.get_attributes().items():
            delattr(SampleInfo, name)

        # Set defaults
        SampleInfo.OUT_DIR = out_dir
        if id is None:
            # Get id automatically
            samples = [int(f) for f in os.listdir(SampleInfo.OUT_DIR) if f.isnumeric()]
            if len(samples) == 0:
                id = "0"
            else:
                id = str(max(samples) + 1)
        SampleInfo.sample_id = str(id)
        SampleInfo.out_file_template = os.path.join(
            SampleInfo.OUT_DIR, SampleInfo.sample_id, "{filename}"
        )

    def get_attributes():
        attrs = dict()
        for a in dir(SampleInfo):
            if a.startswith("__"):
                continue
            if callable(getattr(SampleInfo, a)):
                continue
            attrs[a] = getattr(SampleInfo, a)
        return attrs

    def to_series():
        series = pd.Series(SampleInfo.get_attributes())
        series.sort_index(inplace=True)
        return series

    def __repr__(self):
        max_len = max([len(k) for k in SampleInfo.get_attributes().keys()])
        res = [
            f"{k.ljust(max_len)} = {v}" for k, v in SampleInfo.get_attributes().items()
        ]
        return "\n".join(res)


class Utils:

    @contextmanager
    def suppress_output():
        fd = sys.stdout.fileno()

        def _redirect_stdout(file):
            sys.stdout.close()
            os.dup2(file.fileno(), fd)
            sys.stdout = os.fdopen(fd, "w")

        with os.fdopen(os.dup(fd), "w") as old_stdout:
            with open(os.devnull, "w") as null:
                _redirect_stdout(null)
            try:
                yield
            finally:
                _redirect_stdout(old_stdout)

    def get_out_file_template() -> str:

        if SampleInfo.sample_id is not None:
            # Read given sample ID
            sample_id = SampleInfo.sample_id
        else:
            # Get next sample ID
            sample_ids = [
                int(f) for f in os.listdir(SampleInfo.OUT_DIR) if f.isnumeric()
            ]
            if sample_ids:
                sample_id = str(max(sample_ids) + 1)
            else:
                sample_id = "0"

        os.makedirs(os.path.join(SampleInfo.OUT_DIR, sample_id), exist_ok=True)
        out_file_template = os.path.join(
            SampleInfo.OUT_DIR, sample_id, "{filename}.png"
        )
        SampleInfo.OUT_DIR = os.path.dirname(out_file_template)
        SampleInfo.sample_id = int(sample_id)
        return out_file_template

    def randomize_looks():
        max_frame = bpy.context.scene.frame_end
        min_frame = bpy.context.scene.frame_start
        frame = np.random.randint(min_frame, max_frame + 1)
        bpy.context.scene.frame_set(frame)

    def render_to_file(filename: str):
        if not filename.endswith(".png"):
            filename += ".png"

        filepath = SampleInfo.out_file_template.format(
            filename=filename,
        )
        bpy.context.scene.render.filepath = filepath
        print(f"Rendering file '{filepath}'... ", end=" ", flush=True)
        with Utils.suppress_output() if SUPPRESS_RENDER_OUTPUT else nullcontext():
            bpy.ops.render.render(write_still=True)
        print("Done!", flush=True)

    def render_sample_with_masks():
        # Render original
        Utils.render_to_file("render")
        # Render masks
        MaskRendering.render_masks(
            [
                "Darts Board Area",
                "Dart 1",
                "Dart 2",
                "Dart 3",
                "Intersections",
                "Board Orientation",
            ]
        )

        # Check if masks are valid and if darts board is fully visible
        board_mask = cv2.imread(
            SampleInfo.out_file_template.format(filename="mask_Darts_Board_Area.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        _, board_mask = cv2.threshold(board_mask, 127, 255, cv2.THRESH_BINARY)
        edge_sum = (
            np.sum(board_mask[0])
            + np.sum(board_mask[-1])
            + np.sum(board_mask[:, 0])
            + np.sum(board_mask[:, -1])
        )
        if edge_sum != 0:
            raise AssertionError("Invalid render. Board is not fully visible.")


class SceneUtils:
    def get_geometry_nodes(obj):

        if type(obj) == str:
            obj = SceneUtils.get_object(obj)

        for modifier in obj.modifiers:
            if modifier.type == "NODES":
                return modifier.node_group
        raise RuntimeError(f"Object {obj} does not contain Geometry Nodes.")

    def get_object(name: str):
        return bpy.data.objects.get(name)

    def get_board_radii():
        board = SceneUtils.get_object("Darts Board")
        db_geonodes = SceneUtils.get_geometry_nodes(board)
        radii = []
        for i in range(1, 7):
            r = db_geonodes.interface.items_tree.get(f"Radius {i}").default_value
            radii.append(r)
        return radii

    def record_dart_score():

        radii = SceneUtils.get_board_radii()
        numbers = [6,13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10]  # fmt: skip
        board = SceneUtils.get_object("Darts Board")

        center_dists = []

        def get_dart_score(i):
            dart = SceneUtils.get_object(f"Dart {i}")
            if dart.hide_render:
                return 0, "HIDDEN"
            x = dart.location[0] - board.location[0]
            y = dart.location[2] - board.location[2]

            r = np.sqrt(x**2 + y**2)
            theta = np.degrees(np.arctan2(y, x))
            center_dists.append(round(r, 4))

            number_idx = round(theta / (360 / 20))
            number = numbers[number_idx]
            score_str = str(number)

            if r < radii[0]:
                score = 50
                score_str = "DBull"
            elif r < radii[1]:
                score = 25
                score_str = "Bull"
            elif r < radii[2]:
                score = number
            elif r < radii[3]:
                score = 3 * number
                score_str = "T" + score_str
            elif r < radii[4]:
                score = number
            elif r < radii[5]:
                score = 2 * number
                score_str = "D" + score_str
            else:
                score = 0
                score_str = "OUT"

            return score, score_str

        scores = [get_dart_score(i) for i in range(1, 4)]
        total_score = sum(s[0] for s in scores)

        SampleInfo.scores = scores
        SampleInfo.total_score = total_score
        SampleInfo.dart_radii = center_dists

        return scores, total_score

    def random_motion_blur(shift_range: float = 0.02) -> None:

        def random_translation() -> np.ndarray:  # (3,)
            d = np.random.normal(0, shift_range / 3, (3,))
            return np.clip(d, -shift_range, shift_range)

        # Get info
        camera = bpy.context.scene.camera
        start_frame = bpy.context.scene.frame_current

        # Clear existing keyframes
        camera.animation_data_clear()

        # Only blur 10%
        if np.random.random() > 0.1:
            SampleInfo.camera_motion_blur = 0
            return

        # Keyframe start
        camera.keyframe_insert(data_path="location", frame=start_frame)

        # Update position
        dx, dy, dz = random_translation()
        camera.location.x += dx
        camera.location.y += dy
        camera.location.z += dz

        # Keyframe end
        camera.keyframe_insert(data_path="location", frame=start_frame + 2)
        bpy.context.scene.frame_set(start_frame + 1)

        # Save info
        SampleInfo.camera_motion_blur = round(Vector([dx, dy, dz]).length, 3)

    def random_env_texture():

        def random_texture() -> bool:
            hdri_paths = [
                os.path.join(HDRI_DIR, f)
                for f in os.listdir(HDRI_DIR)
                if f.endswith(".exr")
            ]

            # Return if there are no textures
            if len(hdri_paths) == 0:
                return False

            hdri_path = random.choice(hdri_paths)
            hdri = bpy.data.images.load(hdri_path)
            env_tex_node.image = hdri
            return True

        def random_strength(min: float = 0.5, max: float = 1.1):
            for bg_node in nodes:
                if bg_node.name == "Background":
                    break
            else:
                # No Background node
                WARNINGS.append(
                    "Could not find Background node in Compositor to set random environment texture."
                )
                return
            strength = np.random.uniform(min, max)
            bg_node.inputs["Strength"].default_value = strength
            SampleInfo.env_texture_strength = round(strength, 2)

        def random_z_rotation():
            for node in nodes:
                if node.label == "Z rotation":
                    break
            else:
                WARNINGS.append(
                    "Could not find Z rotation node for environment texture in Compositor."
                )
                return
            node.outputs["Value"].default_value = np.random.uniform(0, 360)

        # Collect data
        world = bpy.data.worlds["World"]
        bpy.context.scene.world = world
        nodes = world.node_tree.nodes

        # Get environment texture node
        for env_tex_node in nodes:
            if env_tex_node.name == "Environment Texture":
                break
        else:
            WARNINGS.append(
                "Could not set random environment texture as there is no Environment Texture node."
            )
            return

        if not random_texture():
            return
        random_strength()
        random_z_rotation()

        SampleInfo.env_texture = env_tex_node.image.filepath

    def rotation_to_board(obj_rotation: Vector | Euler, obj_normal=(0, 0, 1)):

        if type(obj_rotation) != Euler:
            obj_rotation = Euler(obj_rotation)

        # Get normals
        board_normal = Vector([0, -1, 0])
        obj_normal = Vector(obj_normal)
        obj_normal.rotate(obj_rotation)

        # Calculate angle
        dot = board_normal.dot(obj_normal)
        angle = np.rad2deg(np.arccos(dot))

        # Correct angles >90°
        if angle > 90:
            angle = 180 - angle
        return angle


class ObjectPlacement:

    min_dart_dist = 0.005  # 5mm
    min_dart_y_displacement = 0.0075  # 7.5mm
    max_dart_y_displacement = 0.025  # 2.5cm

    def place_darts():

        board_center = SceneUtils.get_object("Darts Board").location

        def get_random_dart_rotation():
            drx = np.random.normal(0, 3)
            dry = np.random.normal(0, 1)
            drz = np.random.normal(0, 1)
            return drx, dry, drz

        def get_better_random_dart_rotation():
            # x rotation: up/down
            min_x = 5
            max_x = -10
            if np.random.uniform(min_x, max_x) < 0:
                drx = abs(np.random.normal(0, min_x / 3))
            else:
                drx = -abs(np.random.normal(0, abs(max_x) / 3))

            # y rotation: spin
            min_y = -180
            max_y = 180
            dry = np.random.uniform(min_y, max_y)

            # z rotation: left/right
            max_z = 15
            drz = np.random.normal(0, max_z / 3)
            drz = np.clip(drz, -max_z, max_z)

            return drx, dry, drz

        def get_random_dart_displacement():
            r = board_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi

            dx = r * np.cos(theta)
            dy = np.random.normal(0, 0.025)
            dz = r * np.sin(theta)
            return dx, dy, dz

        def get_weighted_dart_displacement() -> tuple[float, float]:
            weight_map_obj = SceneUtils.get_object("Darts Weights")
            vertex_group = weight_map_obj.vertex_groups[0]

            coordinates = [v.co for v in weight_map_obj.data.vertices]
            weights = [
                vertex_group.weight(v.index) for v in weight_map_obj.data.vertices
            ]
            normalized_weights = np.array(weights) / np.sum(weights)

            coordinate_idx = np.random.choice(
                list(range(len(coordinates))),
                # size=1,
                # replace=True,
                p=normalized_weights,
            )
            dx, _, dz = coordinates[coordinate_idx]
            return dx, dz

        def displace_intersection(dx, dz) -> tuple[float, float]:
            """
            If the darts intersect the dividing bars, move them a little.
            """
            dart_r = np.sqrt(dx**2 + dz**2)
            dart_theta = np.arctan2(dz, dx)
            tip_radius = 0.00_2 / 2  # 2mm tip diameter
            bar_radius = 0.00_1 / 2  # 0.1cm bar thickness

            # Check for radial intersection
            board_radii = SceneUtils.get_board_radii()
            for board_r in board_radii:
                # Check inside
                if board_r - bar_radius > dart_r + tip_radius:
                    continue

                # Check outside
                if board_r + bar_radius < dart_r - tip_radius:
                    continue

                if board_r > dart_r:
                    # Move dart inside
                    dart_r = board_r - bar_radius - tip_radius
                else:
                    # Move dart outside
                    dart_r = board_r + bar_radius + tip_radius
                break

            # Check for line intersection
            def dart_bar_distance(dart_r, dart_theta):
                bar_theta_mod = (np.pi * 2) / 40
                dart_theta_mod = dart_theta % (np.pi / 10)

                dart_x = np.cos(dart_theta_mod) * dart_r
                dart_y = np.sin(dart_theta_mod) * dart_r

                bar_x = np.cos(bar_theta_mod) * dart_r
                bar_y = np.sin(bar_theta_mod) * dart_r

                distance = np.sqrt((dart_x - bar_x) ** 2 + (dart_y - bar_y) ** 2)
                return distance

            if board_radii[1] < dart_r < board_radii[5] + 0.01:
                before = dart_theta
                dist = dart_bar_distance(dart_r, dart_theta)
                # Move it a little and remove
                for i in range(10):
                    if dart_bar_distance(dart_r, dart_theta) < bar_radius + tip_radius:
                        dart_theta += (np.pi * 2) / (20 * 10)
                    else:
                        break

            dx = dart_r * np.cos(dart_theta)
            dz = dart_r * np.sin(dart_theta)

            return dx, dz

        def random_y_displacement() -> float:
            disp_range = (
                ObjectPlacement.max_dart_y_displacement
                - ObjectPlacement.min_dart_y_displacement
            )

            dy = np.random.normal(0, disp_range / 3)  # normal distribution
            dy = abs(dy)  # cut off negatives
            dy = min(dy, disp_range)  # clip values
            dy = disp_range - dy  # invert curve -> bigger displacements more likely
            dy += ObjectPlacement.min_dart_y_displacement  # shift to correct range
            return dy

        dart_count = 0
        SampleInfo.dart_rotations = []
        for i in range(1, 4):
            dart = SceneUtils.get_object(f"Dart {i}")

            # Randomly skip dart
            if i > 1 and np.random.random() < 1 / 6:
                dart.location = (0, 0, 0)
                dart.hide_render = True
                continue
            dart_count += 1

            # Location
            dx, dz = get_weighted_dart_displacement()
            dx, dz = displace_intersection(dx, dz)
            dy = random_y_displacement()
            dart.location = (
                board_center[0] + dx,
                board_center[1] + dy,
                board_center[2] + dz,
            )

            # Check against other darts
            for j in range(1, i):
                other_dart = SceneUtils.get_object(f"Dart {j}")
                dist = (dart.location - other_dart.location).length
                # If they are too close, remove this dart
                if dist < ObjectPlacement.min_dart_dist:
                    dart.location = (0, 0, 0)
                    dart.hide_render = True
                    break

            # Rotation
            drx, dry, drz = get_better_random_dart_rotation()
            dart.rotation_euler = (
                np.radians(drx),
                np.radians(dry),
                np.radians(drz),
            )
            dart_rot = SceneUtils.rotation_to_board(
                obj_rotation=dart.rotation_euler,
                obj_normal=(0, 1, 0),
            )
            SampleInfo.dart_rotations.append(round(dart_rot))

        # Save dart stats
        dists = []
        for i in range(1, 4):
            dart_1 = SceneUtils.get_object(f"Dart {i}")
            if dart_1.hide_render:
                continue
            for j in range(i + 1, 4):
                dart_2 = SceneUtils.get_object(f"Dart {j}")
                if dart_2.hide_render:
                    continue
                dist = dart_1.location - dart_2.location
                dists.append(round(dist.length, 3))

        SampleInfo.dart_distances = dists
        SampleInfo.dart_count = dart_count

    def randomize_camera_parameters():
        cam = SceneUtils.get_object("Camera")
        darts_board = SceneUtils.get_object("Darts Board")

        def get_min_max_dist():
            cam_space = SceneUtils.get_object("Camera Space")
            gnode_mod = cam_space.modifiers.get("GeometryNodes")
            min_dist = gnode_mod["Socket_2"]  # this is ugly, but this works for now
            max_dist = gnode_mod["Socket_3"]
            return min_dist, max_dist

        # Get focal lengths
        min_focal = 18
        max_focal = 60
        mean_focal = (max_focal + min_focal) / 2

        # Get distances
        min_dist, max_dist = get_min_max_dist()
        dist = (cam.location - darts_board.location).length
        dist_fac = (dist - min_dist) / (max_dist - min_dist)

        # Calculate lower and upper distance bounds
        lower = min_focal + dist_fac * (mean_focal - min_focal)
        upper = mean_focal + dist_fac * (mean_focal - min_focal)

        # Set focal length
        focal_fac = np.random.random()
        focal = lower + focal_fac * (upper - lower)
        cam.data.lens = focal

        # Aperture
        min_aperture = 1.8
        max_aperture = 5
        aperture = min_aperture + focal_fac * (max_aperture - min_aperture)
        aperture += np.random.normal(0, 1)
        aperture = np.clip(aperture, min_aperture, max_aperture)
        f_stop = focal / (2 * aperture)
        cam.data.dof.aperture_fstop = f_stop

        # Set resolution
        img_format = np.random.choice(
            [4 / 3, 16 / 9, 1 / 1, 3 / 2, 2 / 1, 21 / 9, 5 / 4]
        )
        dim_a = int(np.random.uniform(1000, 4000))
        dim_b = int(dim_a / img_format)
        vertical = np.random.random() > 0.3
        bpy.context.scene.render.resolution_x = dim_b if vertical else dim_a
        bpy.context.scene.render.resolution_y = dim_a if vertical else dim_b

        # Flash light
        use_flash = np.random.random() > SampleInfo.env_texture_strength
        SceneUtils.get_object("Flash").hide_render = not use_flash

        # Exposure
        exposure = np.random.uniform(0.2, 1.0)
        bpy.context.scene.view_settings.exposure = exposure

        # Save cam info
        SampleInfo.camera_distance = round(dist, 2)
        SampleInfo.camera_focal_mm = round(focal, 2)
        SampleInfo.camera_aperture = round(aperture, 2)
        SampleInfo.camera_flash = int(use_flash)
        SampleInfo.camera_exposure = round(exposure, 2)
        SampleInfo.img_width = bpy.context.scene.render.resolution_x
        SampleInfo.img_height = bpy.context.scene.render.resolution_y

    def place_camera():

        def get_random_cam_position():
            bbox_min = cam_space.matrix_world @ Vector(
                cam_space.bound_box[0]
            )  # lowest corner at 0
            bbox_max = cam_space.matrix_world @ Vector(
                cam_space.bound_box[-2]
            )  # highest corner at -2

            def is_inside(p_w):
                p_l = cam_space.matrix_world.inverted() @ p_w
                result, point_on_mesh_l, normal_l, _index = (
                    cam_space.closest_point_on_mesh(p_l, distance=10)
                )

                if not result:
                    return False

                # Back to world space
                point_on_mesh_w = cam_space.matrix_world @ point_on_mesh_l
                normal_w = cam_space.matrix_world.to_quaternion() @ normal_l

                # Check if point inside
                p2 = point_on_mesh_w - p_w
                v = p2.dot(normal_w)
                return v >= 0

            while True:
                p = Vector(
                    [np.random.uniform(bbox_min[i], bbox_max[i]) for i in range(3)]
                )
                if is_inside(p):
                    break
            return p

        def get_cam_rotation(cam_pos):
            darts_board = SceneUtils.get_object("Darts Board")
            view_center = darts_board.location.copy()
            board_r = darts_board.dimensions.x / 2.2

            view_center.x += np.clip(
                np.random.normal(0, board_r / 3), -board_r, board_r
            )
            view_center.z += np.clip(
                np.random.normal(0, board_r / 3), -board_r, board_r
            )

            view_dir = view_center - cam_pos
            rot_quat = view_dir.to_track_quat("-Z", "Y")

            return rot_quat.to_euler()

        def set_focus_point():
            board = SceneUtils.get_object("Darts Board")
            board_pos = board.location
            board_r = (
                SceneUtils.get_geometry_nodes(board)
                .interface.items_tree.get("Board Diameter")
                .default_value
                / 2
            )

            focus_pos = board_pos.copy()
            focus_pos[0] += np.random.normal(0, board_r / 3)
            focus_pos[1] += np.random.normal(0, 0.02)
            focus_pos[2] += np.random.normal(0, board_r / 3)

            SceneUtils.get_object("Camera Focus").location = focus_pos

            # Save focus distance
            focus_dist = (
                SceneUtils.get_object("Darts Board").location.y
                - SceneUtils.get_object("Camera Focus").location.y
            )
            SampleInfo.camera_focus_distance = round(focus_dist, 2)

        # Get info
        cam = SceneUtils.get_object("Camera")
        cam_space = SceneUtils.get_object("Camera Space")

        # Set position
        cam_pos = get_random_cam_position()
        cam.location = cam_pos

        # Set rotation
        cam_rot = get_cam_rotation(cam_pos)
        cam.rotation_euler = cam_rot
        SampleInfo.camera_angle = round(
            SceneUtils.rotation_to_board(cam.rotation_euler)
        )

        # Set focus
        set_focus_point()


class Compositor:
    tree = None

    def init_vars():
        Compositor.tree = bpy.context.scene.node_tree

    def get_node(name: str):
        # Get by name
        node = Compositor.tree.nodes.get(name)
        if node:
            return node

        # Get by label
        for node in Compositor.tree.nodes:
            if node.label == name:
                return node

        raise RuntimeError(
            f"No node with name {name} found in Compositor node tree."
            " Check the node name or set a name manually if unsure."
        )

    def check_link_src(
        link,
        name: str,
        socket: str,
    ):
        # Name check
        if name not in [link.from_node.name, link.from_node.label]:
            return False
        # Socket check
        if link.from_socket.name != socket:
            return False
        return True

    def check_link_dst(
        link,
        name: str,
        socket: str,
    ):
        # Name check
        if name not in [link.to_node.name, link.to_node.label]:
            return False
        # Socket check
        if link.to_socket.name != socket:
            return False
        return True

    def get_link(
        src_node: str,
        src_socket: str,
        dst_node: str,
        dst_socket: str,
    ):
        for link in Compositor.tree.links:
            if Compositor.check_link_src(
                link,
                src_node,
                src_socket,
            ) and Compositor.check_link_dst(
                link,
                dst_node,
                dst_socket,
            ):
                return link
        raise RuntimeError(
            f"No link found from '{src_node}'[{src_socket}] -> '{dst_node}'[{dst_socket}] in Compositor."
            " Check the node name or set a name manually if unsure."
        )

    def remove_link(
        state,
        src_node: str,
        src_socket: str,
        dst_node: str,
        dst_socket: str,
    ):
        # Bookkeeping
        if state is not None:
            state["removed_links"].append((src_node, src_socket, dst_node, dst_socket))

        # Remove link
        link = Compositor.get_link(src_node, src_socket, dst_node, dst_socket)
        Compositor.tree.links.remove(link)

        return state

    def add_link(
        state,
        src_node: str,
        src_socket: str,
        dst_node: str,
        dst_socket: str,
    ):
        # Bookkeeping
        if state is not None:
            state["added_links"].append((src_node, src_socket, dst_node, dst_socket))

        # Add link
        src_node = Compositor.get_node(src_node)
        src_socket = src_node.outputs.get(src_socket)
        dst_node = Compositor.get_node(dst_node)
        dst_socket = dst_node.inputs.get(dst_socket)
        Compositor.tree.links.new(src_socket, dst_socket)

        return state


class MaskRendering:

    def prepare_objects() -> dict:
        state = {}
        for obj in bpy.data.objects:
            state[obj.name] = obj.hide_render
            obj.hide_render = True
        return state

    def restore_objects(state: dict):
        for obj in bpy.data.objects:
            obj.hide_render = state[obj.name]

    def prepare_env_texture() -> dict:
        state = {}

        # Get node tree
        node_tree = bpy.context.scene.world.node_tree
        if not node_tree:
            raise RuntimeError(
                "No world node tree containing environment texture to clear."
            )

        # Get Background node
        for bg_node in node_tree.nodes:
            if bg_node.type == "BACKGROUND":
                break
        else:
            raise RuntimeError(
                "No background node in world node tree. Could not remove environment texture."
            )

        # Set state
        state["default_value"] = bg_node.inputs["Strength"].default_value
        bg_node.inputs["Strength"].default_value = 0

        return state

    def restore_env_texture(state: dict):
        # Get node tree
        node_tree = bpy.context.scene.world.node_tree
        if not node_tree:
            raise RuntimeError(
                "No world node tree containing environment texture to clear."
            )

        # Get Background node
        for bg_node in node_tree.nodes:
            if bg_node.type == "BACKGROUND":
                break
        else:
            raise RuntimeError(
                "No background node in world node tree. Could not remove environment texture."
            )

        # Set state
        bg_node.inputs["Strength"].default_value = state["default_value"]

    def prepare_compositor() -> dict:
        state = dict(removed_links=[], added_links=[])

        render_layers_node = Compositor.get_node("Render Layers")
        composite_node = Compositor.get_node("Composite")
        lens_distortion = Compositor.get_node("Lens Distortion")

        # Get original compositor node link
        Compositor.remove_link(
            state, "Film Grain Output", "Image", "Composite", "Image"
        )
        Compositor.remove_link(state, "Bloom Mix", "Image", "Lens Distortion", "Image")
        Compositor.add_link(state, "Render Layers", "Alpha", "Lens Distortion", "Image")
        Compositor.add_link(state, "Lens Distortion", "Image", "Composite", "Image")

        return state

    def restore_compositor(state: dict):
        for src_node, src_socket, dst_node, dst_socket in state["added_links"]:
            Compositor.remove_link(None, src_node, src_socket, dst_node, dst_socket)
        for src_node, src_socket, dst_node, dst_socket in state["removed_links"]:
            Compositor.add_link(None, src_node, src_socket, dst_node, dst_socket)

    def prepare_settings() -> dict:
        state = {}
        scene = bpy.context.scene

        state["file_format"] = scene.render.image_settings.file_format
        scene.render.image_settings.file_format = "PNG"

        state["filepath"] = scene.render.filepath
        scene.render.filepath = f"dump/mask.png"

        state["engine"] = bpy.context.scene.render.engine
        bpy.context.scene.render.engine = "BLENDER_WORKBENCH"

        return state

    def restore_settings(state: dict):
        scene = bpy.context.scene

        scene.render.image_settings.file_format = state["file_format"]
        scene.render.filepath = state["filepath"]
        bpy.context.scene.render.engine = state["engine"]

    def render_masks(mask_obj_names: str | list[str]):

        if type(mask_obj_names) == str:
            mask_obj_names = [mask_obj_names]

        # Prepare state
        obj_state = MaskRendering.prepare_objects()
        bpy.context.scene.render.film_transparent = True
        env_state = MaskRendering.prepare_env_texture()
        compositor_state = MaskRendering.prepare_compositor()
        settings_state = MaskRendering.prepare_settings()

        # Render the scene
        for mask_obj_name in mask_obj_names:
            # Prepare mask object
            mask_obj = SceneUtils.get_object(mask_obj_name)
            mask_obj.hide_render = False

            # Render
            Utils.render_to_file(f"mask_{mask_obj_name.replace(' ', '_')}")

            # Restore mask object
            mask_obj.hide_render = True

        # Restore state
        MaskRendering.restore_settings(settings_state)
        MaskRendering.restore_compositor(compositor_state)
        MaskRendering.restore_env_texture(env_state)
        bpy.context.scene.render.film_transparent = False
        MaskRendering.restore_objects(obj_state)


# -------------------------------------------------------------------------------------------------


def render_image(id=None, out_dir: str = "data/generation/out"):
    init_project()

    # Reset sample info
    SampleInfo.reset(id=id, out_dir=out_dir)

    # Randomize Scene
    Utils.randomize_looks()

    # Randomize HDRI
    SceneUtils.random_env_texture()

    # Place Darts
    ObjectPlacement.place_darts()
    scores, total_score = SceneUtils.record_dart_score()

    # Place Camera
    ObjectPlacement.place_camera()
    SceneUtils.random_motion_blur()
    ObjectPlacement.randomize_camera_parameters()

    # Rendering
    Utils.render_sample_with_masks()

    # Save info
    sample_info = SampleInfo.to_series()
    with open(SampleInfo.out_file_template.format(filename="info.pkl"), "wb") as f:
        pickle.dump(sample_info, f)

    # Print warnings
    if len(WARNINGS) > 0:
        print("=" * 120)
        print(f"Encontered {len(WARNINGS)} warnings:")
        for i, warning in enumerate(WARNINGS, 1):
            print(f"\t{i}. {warning}")
        print("=" * 120)

    # print(SampleInfo(), flush=True)
    return sample_info

    # -------------------------------


if __name__ == "__main__":
    render_image()
