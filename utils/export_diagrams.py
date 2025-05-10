import os
from subprocess import run

def export_diagrams(imgs_dir):

    files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(imgs_dir)
        for f in filenames
        if os.path.splitext(f)[1] == ".drawio"
    ]

    for file in files:
        output_file = file.replace(".drawio", ".pdf")
        run(
            [
                "sh",
                "-c",
                "drawio"
                " --export"
                " --output"
                f" {output_file}"
                f" {file}"
                " --format"
                " pdf"
                " --transparent"
                " --crop"
                " --border"
                " 0",
            ],
            check=True,
        )

# export_diagrams(imgs_dir = "writing/work/imgs/")
export_diagrams(imgs_dir = "writing/presentation/imgs/")
