import os
import bpy
import numpy as np
import random
from mathutils import Vector

# -----------------------------------------------
# Variables

BLEND_FILE = "data/generation/darts.blend"
OUT_FILE = "dump/test.png"
OUT_DIR = "data/generation/out/"
HDRI_DIR = "data/generation/hdri/"

# -----------------------------------------------
# Load and setup
bpy.ops.wm.open_mainfile(filepath=BLEND_FILE)

# Set render engine
bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 1

# Enable GPU rendering
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
bpy.context.scene.cycles.device = "GPU"
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    if "4060" in d["name"]:
        d["use"] = 1
    else:
        d["use"] = 0

# Set output path
bpy.context.scene.render.filepath = OUT_FILE

# -----------------------------------------------
# Do stuff


class SceneUtils:
    def get_geometry_nodes(obj):
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

    def calculate_dart_score():

        radii = SceneUtils.get_board_radii()
        numbers = [6,13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10]  # fmt: skip
        board = SceneUtils.get_object("Darts Board")

        def get_dart_score(i):
            dart = SceneUtils.get_object(f"Dart {i}")
            x = dart.location[0] - board.location[0]
            y = dart.location[2] - board.location[2]

            r = np.sqrt(x**2 + y**2)
            theta = np.degrees(np.arctan2(y, x))

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

        return scores, total_score


def random_env_texture():
    world = bpy.data.worlds["World"]

    bpy.context.scene.world = world

    nodes = world.node_tree.nodes
    for env_tex_node in nodes:
        if env_tex_node.name == "Environment Texture":
            break
    else:
        raise ValueError("No Environment Texture node in world node tree.")

    hdri_paths = [
        os.path.join(HDRI_DIR, f) for f in os.listdir(HDRI_DIR) if f.endswith(".exr")
    ]

    # Return if there are no textures
    if len(hdri_paths) == 0:
        return

    hdri_path = random.choice(hdri_paths)
    hdri = bpy.data.images.load(hdri_path)
    env_tex_node.image = hdri

    # Random image strength
    for bg_node in nodes:
        if bg_node.name == "Background":
            break
    else:
        # No Background node
        return

    bg_node.inputs["Strength"].default_value = np.random.uniform(0.1, 0.8)


class ObjectPlacement:

    min_dart_dist = 0.01

    def place_darts():

        def get_random_dart_rotation():
            drx = np.random.normal(0, 3)
            dry = np.random.normal(0, 1)
            drz = np.random.normal(0, 1)
            return drx, dry, drz

        def get_position_based_dart_rotation(x, y):
            drx = 0
            dry = 0
            drz = 0

            return drx, dry, drz

        def get_random_dart_displacement():
            r = board_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi

            dx = r * np.cos(theta)
            dy = np.random.normal(0, 0.025)
            dz = r * np.sin(theta)
            return dx, dy, dz

        def get_weighted_dart_displacement():
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
            return coordinates[coordinate_idx]

        def displace_intersection(dx, dy, dz):
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

            return dx, dy, dz

        for i in range(1, 4):
            dart = SceneUtils.get_object(f"Dart {i}")

            # Randomly skip dart
            if i > 1 and True and np.random.random() < 1 / 6:
                dart.location = (0, 0, 0)
                dart.hide_render = True
                continue

            # Location
            dx, dy, dz = get_weighted_dart_displacement()
            dx, dy, dz = displace_intersection(dx, dy, dz)
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
                print(dist)
                if dist < ObjectPlacement.min_dart_dist:
                    dart.location = (0, 0, 0)
                    dart.hide_render = True
                    break

            # Rotation
            drx, dry, drz = get_position_based_dart_rotation(dx, dz)
            dart.rotation_euler = (
                np.radians(-90 + drx),
                np.radians(dry),
                np.radians(drz),
            )

    def place_camera():
        cam = SceneUtils.get_object("Camera")
        cam_space = SceneUtils.get_object("Camera Space")

        # -------------------------------------------
        # Position
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

        cam_pos = get_random_cam_position()
        cam.location = cam_pos

        # -------------------------------------------
        # Focus
        def get_focus_point():
            board = SceneUtils.get_object("Darts Board")
            board_pos = board.location
            board_r = (
                SceneUtils.get_geometry_nodes(darts_board)
                .interface.items_tree.get("Board Diameter")
                .default_value
                / 2
            )

            focus_pos = board_pos.copy()
            focus_pos[0] += np.random.normal(0, board_r / 3)
            focus_pos[1] += np.random.normal(0, 0.02)
            focus_pos[2] += np.random.normal(0, board_r / 3)

            SceneUtils.get_object("Camera Focus").location = focus_pos

            return focus_pos

        focus_point = get_focus_point()

        # -------------------------------------------
        # Rotation
        def get_cam_rotation(cam_pos, focus_point):
            view_dir = focus_point - cam_pos
            rot_quat = view_dir.to_track_quat("-Z", "Y")
            return rot_quat.to_euler()

        cam_rot = get_cam_rotation(cam_pos, focus_point)
        cam.rotation_euler = cam_rot


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
        state = {}

        # Get basic objects
        tree = bpy.context.scene.node_tree
        render_layers_node = tree.nodes.get("Render Layers")
        composite_node = tree.nodes.get("Composite")
        if not render_layers_node or not composite_node:
            raise RuntimeError(
                "No Render Layers node or Composite node in Compositor node tree"
            )

        # Get original compositor node input
        image_input = composite_node.inputs.get("Image")
        for link in tree.links:
            if link.to_node == composite_node and link.to_socket == image_input:
                break
        else:
            raise RuntimeError(
                "No link to Composite node's 'Image' input socket in Compositor node tree."
            )
        state["link_from_socket"] = link.from_socket
        tree.links.remove(link)

        # Replace with new link
        alpha_output = render_layers_node.outputs.get("Alpha")
        tree.links.new(alpha_output, image_input)

        return state

    def restore_compositor(state: dict):
        # Get basic objects
        tree = bpy.context.scene.node_tree
        render_layers_node = tree.nodes.get("Render Layers")
        composite_node = tree.nodes.get("Composite")
        if not render_layers_node or not composite_node:
            raise RuntimeError(
                "No Render Layers node or Composite node in Compositor node tree"
            )

        # Remove new link
        image_input = composite_node.inputs.get("Image")
        for link in tree.links:
            if link.from_node == render_layers_node and link.to_node == composite_node:
                break
        else:
            raise RuntimeError(
                "New Link from Render Layers node to Composite node not found in Compositor node tree."
            )
        tree.links.remove(link)

        # Restore old link
        tree.links.new(state["link_from_socket"], image_input)

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
            out_name = f"mask_{mask_obj_name.replace(' ', '_')}.png"
            bpy.context.scene.render.filepath = os.path.join("dump", out_name)
            bpy.ops.render.render(write_still=True)

            # Restore mask object
            mask_obj.hide_render = True

        # Restore state
        MaskRendering.restore_settings(settings_state)
        MaskRendering.restore_compositor(compositor_state)
        MaskRendering.restore_env_texture(env_state)
        bpy.context.scene.render.film_transparent = False
        MaskRendering.restore_objects(obj_state)


# Get Scene Infos
darts_board = SceneUtils.get_object("Darts Board")
db_geonodes = SceneUtils.get_geometry_nodes(darts_board)

board_radius = db_geonodes.interface.items_tree.get("Radius 6").default_value
board_center = darts_board.location

# Place Darts
ObjectPlacement.place_darts()
scores, total_score = SceneUtils.calculate_dart_score()

# Place Camera
ObjectPlacement.place_camera()

# Randomize HDRI
random_env_texture()

# Render
# bpy.ops.render.render(write_still=True)
MaskRendering.render_masks(["Darts Board Area", "Dart 1", "Dart 2", "Dart 3"])

print(scores, total_score)

exit()

# -------------------------------
# move to out directory
id = max(int(f.split(".")[0]) for f in os.listdir(OUT_DIR) if not "mask" in f) + 1
print(id)
os.rename("dump/test.png", os.path.join(OUT_DIR, f"{id:04d}.png"))
os.rename("dump/mask.png", os.path.join(OUT_DIR, f"{id:04d}_mask.png"))
