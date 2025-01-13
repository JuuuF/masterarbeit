# Masterarbeit

Hallo! Das hier ist meine Masterarbeit. Bisschen was mit CV und KI und Blender und Spaß. Hoffentlich.

Sieh' dich gerne ein wenig um, aber mach' bitte nichts anfassen, das ist alles sehr wackelig und kippt schnell um. Danke!

## Setup

### 1. Python Environment

```bash
conda create -n masterarbeit python=3.11.5 -y
conda activate masterarbeit

pip install -r setup/requirements_cpu.txt
OR
pip install -r setup/requirements_gpu.txt
```

### 2. Install project package

Execute the following at project root directory `Masterarbeit/`:

```bash
pip install -e .
```

## Paper Setup

### 1. Python environment

```bash
conda create -n ma_deepdarts python=3.7
conda activate ma_deepdarts
pip install -r setup/requirements_deepdarts.txt
```
