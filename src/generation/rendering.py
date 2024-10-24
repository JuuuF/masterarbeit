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


def get_geometry_nodes(obj):
    for modifier in obj.modifiers:
        if modifier.type == "NODES":
            return modifier.node_group
    raise RuntimeError(f"Object {obj} does not contain Geometry Nodes.")


def get_object(name: str):
    return bpy.data.objects.get(name)


def get_board_radii():
    board = get_object("Darts Board")
    db_geonodes = get_geometry_nodes(board)
    radii = []
    for i in range(1, 7):
        r = db_geonodes.interface.items_tree.get(f"Radius {i}").default_value
        radii.append(r)
    return radii


def calculate_dart_score():

    radii = get_board_radii()
    numbers = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

    def get_dart_score(i):
        dart = get_object(f"Dart {i}")
        board = get_object("Darts Board")
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


def darts_placement():

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
        weight_map_obj = get_object("Darts Weights")
        vertex_group = weight_map_obj.vertex_groups[0]

        coordinates = [v.co for v in weight_map_obj.data.vertices]
        weights = [vertex_group.weight(v.index) for v in weight_map_obj.data.vertices]
        normalized_weights = np.array(weights) / np.sum(weights)

        coordinate_idx = np.random.choice(
            list(range(len(coordinates))),
            # size=1,
            # replace=True,
            p=normalized_weights,
        )
        return coordinates[coordinate_idx]

    for i in range(1, 4):
        dart = get_object(f"Dart {i}")

        # Randomly skip dart
        if i > 1 and True and np.random.random() < 1 / 6:
            dart.location = (0, 0, 0)
            continue

        # Location
        dx, dy, dz = get_weighted_dart_displacement()
        dart.location = (
            board_center[0] + dx,
            board_center[1] + dy,
            board_center[2] + dz,
        )

        # Rotation
        drx, dry, drz = get_position_based_dart_rotation(dx, dz)
        dart.rotation_euler = (
            np.radians(-90 + drx),
            np.radians(dry),
            np.radians(drz),
        )


def place_camera():
    cam = get_object("Camera")
    cam_space = get_object("Camera Space")

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
            result, point_on_mesh_l, normal_l, _index = cam_space.closest_point_on_mesh(
                p_l, distance=10
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
            p = Vector([np.random.uniform(bbox_min[i], bbox_max[i]) for i in range(3)])
            if is_inside(p):
                break
        return p

    cam_pos = get_random_cam_position()
    cam.location = cam_pos

    # -------------------------------------------
    # Focus
    def get_focus_point():
        board = get_object("Darts Board")
        board_pos = board.location
        board_r = (
            get_geometry_nodes(darts_board)
            .interface.items_tree.get("Board Diameter")
            .default_value
            / 2
        )

        focus_pos = board_pos.copy()
        focus_pos[0] += np.random.normal(0, board_r / 3)
        focus_pos[1] += np.random.normal(0, 0.02)
        focus_pos[2] += np.random.normal(0, board_r / 3)

        get_object("Camera Focus").location = focus_pos

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


def render_board_mask():
    def clear_tree(tree):
        for node in tree.nodes:
            tree.nodes.remove(node)

    # Hide all objects
    for obj in bpy.data.objects:
        obj.hide_render = True

    board_area = get_object("Darts Board Area")
    board_area.hide_render = False

    # clear environment texture
    node_tree = bpy.context.scene.world.node_tree
    if node_tree:
        for node in node_tree.nodes:
            if node.type == "TEX_ENVIRONMENT":
                node_tree.nodes.remove(node)
    bpy.context.scene.render.film_transparent = True

    # Clear compositor tree
    tree = bpy.context.scene.node_tree
    clear_tree(tree)

    # Create render layers
    render_layer = tree.nodes.new("CompositorNodeRLayers")

    composite_node = tree.nodes.new("CompositorNodeComposite")

    # Link Nodes
    links = tree.links
    links.new(render_layer.outputs["Alpha"], composite_node.inputs["Image"])

    # Set render settings
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = f"dump/mask.png"

    # Render the scene
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.ops.render.render(write_still=True)


# Get Scene Infos
darts_board = get_object("Darts Board")
db_geonodes = get_geometry_nodes(darts_board)

board_radius = db_geonodes.interface.items_tree.get("Radius 6").default_value
board_center = darts_board.location

# Place Darts
darts_placement()
scores, total_score = calculate_dart_score()

# Place Camera
place_camera()

# Randomize HDRI
random_env_texture()

# Render
bpy.ops.render.render(write_still=True)
render_board_mask()

print(scores, total_score)

exit()

# -------------------------------
# move to out directory
id = max(int(f.split(".")[0]) for f in os.listdir(OUT_DIR) if not "mask" in f) + 1
print(id)
os.rename("dump/test.png", os.path.join(OUT_DIR, f"{id:04d}.png"))
os.rename("dump/mask.png", os.path.join(OUT_DIR, f"{id:04d}_mask.png"))
