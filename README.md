# Masterarbeit

Hallo! Das hier ist meine Masterarbeit. Bisschen was mit CV und KI und Blender und Spaß. Hoffentlich.

Sieh' dich gerne ein wenig um, aber mach' bitte nichts anfassen, das ist alles sehr wackelig und kippt schnell um. Danke!

## Setup

### 1. Create conda environment

```bash
conda create -n masterarbeit python=3.11.5 -y
conda activate masterarbeit
```

### 2. Install packages

Install either CPU or GPU packages.

```bash
pip install -r setup/requirements_cpu.txt
OR
pip install -r setup/requirements_gpu.txt
```

### 3. Install project package

Execute the following at project root directory `Masterarbeit/`:

```bash
pip install -e .
```
