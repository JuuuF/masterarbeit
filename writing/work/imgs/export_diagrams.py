import os
from subprocess import run

imgs_dir = "writing/work/imgs"
diagram_mappings = {
    "Project": os.path.join(imgs_dir, "ma_project_structure.pdf"),
    "|| Data ||": os.path.join(imgs_dir, "rendering", "rendering_pipeline.pdf"),
    "Data - Heatmaps": os.path.join(imgs_dir, "rendering", "methodik", "heatmaps.pdf"),
    "|| CV  ||": os.path.join(imgs_dir, "cv", "cv_pipeline.pdf"),
    "CV - Edges": os.path.join(imgs_dir, "cv", "methodik", "edges.pdf"),
    "CV - Lines": os.path.join(imgs_dir, "cv", "methodik", "lines.pdf"),
    "CV - Orientation": os.path.join(imgs_dir, "cv", "methodik", "orientation.pdf"),
    # "|| AI ||": os.path.join(imgs_dir, ".pdf"),
    "AI - augmentation": os.path.join(imgs_dir, "ai", "augmentation_pipeline.pdf"),
}

drawio_file = "writing/work/imgs/charts.drawio"

with open(drawio_file, "r") as f:
    contents = f.read()

pages = contents.split('<diagram id="')[1:]
pages = [p.split('name="')[1] for p in pages]
pages = [p.split('">')[0] for p in pages]

for i, page_name in enumerate(pages):
    output_file = diagram_mappings.get(page_name, None)
    if output_file is None:
        print(
            f"WARNING: no output file specified for diagram page {page_name}. Skipping."
        )
        continue

    run(
        [
            "sh",
            "-c",
            "drawio"
            " --export"
            " --output"
            f" {output_file}"
            f" {drawio_file}"
            " --format"
            " pdf"
            " --transparent"
            " --crop"
            " --border"
            " 0"
            " --page-index"
            f" {i}",
        ],
        check=True,
    )
