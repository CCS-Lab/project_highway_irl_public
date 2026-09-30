# Highway Task IRL Analysis

This repository contains the code and analysis pipeline used to reproduce the behavioral and inverse reinforcement learning (IRL) analyses for the highway task.

The code works best in a Linux environment and has been tested on:

- Ubuntu 24.04.4 LTS
- Anaconda 25.5.1

## 1. Install Anaconda

Install Anaconda before proceeding.

Tested version:

```text
Anaconda 25.5.1
```

## 2. Create the Conda Environment

From the repository directory, create the Conda environment using the provided `irl_env.yaml` file.

Because the environment includes several large packages, increasing the pip timeout and retry limit is recommended:

```bash
export PIP_DEFAULT_TIMEOUT=600
export PIP_RETRIES=10

conda env create -f irl_env.yaml
```

The installation may take some time.

Activate the environment:

```bash
conda activate irl-original
```

> **Note:** The provided environment is not intended to be a minimal reproduction environment. It may contain additional packages used for internal testing.

## 3. Register the Environment as an IPython/Jupyter Kernel

To make the Conda environment available in Jupyter or IPython:

```bash
python -m ipykernel install \
    --user \
    --name irl-original \
    --display-name "Python (irl-original)"
```

When running the notebooks in this repository, select:

```text
Python (irl-original)
```

as the kernel.

## 4. Install the Analysis Package

The current directory should be the root of the cloned repository:

```text
project_highway_irl_public/
```

Install the `behavior_v2` package in editable mode:

```bash
pip install -e behavior_v2
```

Equivalently, this step can be performed by running the corresponding code block in:

```text
1_install_env.ipynb
```

using the `irl-original` kernel.

## 5. Configure CUDA for TensorFlow

The analysis environment uses TensorFlow 2.7.4, which requires CUDA 11.2 and cuDNN 8.x.

Install the compatible CUDA runtime and cuDNN inside the Conda environment:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

Then make the Conda environment libraries available to the runtime:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

This does not modify the system-wide CUDA installation.

You can verify that TensorFlow detects the GPU with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

# Analysis Reproduction

## 1. AIRL Training

Run:

```text
6_run_AIRL_data.py
```

Before running the script, set the `sub_list` variable to the subject IDs you want to analyze.

Valid subject IDs range from:

```text
302-348
```

The script uses behavioral data stored in:

```text
behavior_v2/data_v3/
```

and generates the corresponding AIRL models.

The version labels are internal development identifiers:

- `behavior_v2`: second version of the behavioral analysis environment
- `data_v3`: third version of the experiment, following pilot versions `v1` and `v2`

## 2. Behavioral Data Analysis

Run the code blocks in:

```text
descriptive_analysis_final.ipynb
```

using the `irl-original` kernel.

This notebook reproduces the behavioral analyses corresponding to:

- Figure 2 in the main paper
- Figure S1 in the Supplement

## 3. Group-Level IRL Analysis

Run the code blocks in:

```text
IRL_group.ipynb
IRL_raw_reward.ipynb
```

using the `irl-original` kernel.

These notebooks reproduce the group-level IRL analyses corresponding to:

- Figure 3 in the main paper
- Figure S2 in the Supplement

## 4. IRL Trajectory Analysis

Run the code blocks in:

```text
reward_trajectory.ipynb
```

using the `irl-original` kernel.

This notebook reproduces the IRL trajectory analyses, including:

- Figure 4 in the main paper
- Figure S3 in the Supplement
