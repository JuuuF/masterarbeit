import os
from subprocess import run


imgs_dir = "writing/work/imgs/"
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
