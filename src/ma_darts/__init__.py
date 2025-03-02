dart_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
img_size = 800
classes = ["nothing", "black", "white", "red", "green", "out"]

# Board radii in images
margin = 100
_board_radius_px = img_size / 2 - margin
_board_radius_scaling = _board_radius_px / 17.0  # double outer radius [cm]
radii = {
    "r_do": _board_radius_scaling * 17.0,  # double outer
    "r_di": _board_radius_scaling * 16.2,  # double inner
    "r_to": _board_radius_scaling * 10.7,  # triple outer
    "r_ti": _board_radius_scaling * 9.8,  # triple inner
    "r_bo": _board_radius_scaling * 1.6,  # bull outer
    "r_bi": _board_radius_scaling * 0.635,  # bull inner
}
radii[0] = radii["r_bi"]
radii[1] = radii["r_bo"]
radii[2] = radii["r_ti"]
radii[3] = radii["r_to"]
radii[4] = radii["r_di"]
radii[5] = radii["r_do"]
del _board_radius_px, _board_radius_scaling

__all__ = []
__all__.append("dart_order")
__all__.append("img_size")
__all__.append("classes")
__all__ += [f"r_{p}{l}" for p in "dtb" for l in "oi"]
