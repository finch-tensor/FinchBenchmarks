---
author:
- Willow Ahrens
- Teodoro Fields Collin
- Radha Patel
- Kyle Deeds
- Changwan Hong
- Saman Amarasinghe
bibliography:
- FinchOOPSLAWillow.bib
title: "Finch: Sparse and Structured Tensor Programming with Control
  Flow: The Artifact"
---

# Artifact Appendix

## Abstract

In this artifact, we provide an archive of the Finch compiler at the
time of writing and instructions to replicate all benchmarks in this
paper. We note that the Finch compiler is separately available as
open-source software. We claim that the results in this paper are
reproducible with the provided artifact on an identical machine. Some
results, especially those regarding MKL, may be architecture-specific.

## Artifact check-list (meta-information)

*Obligatory. Use just a few informal keywords in all fields applicable
to your artifacts and remove the rest. This information is needed to
find appropriate reviewers and gradually unify artifact meta information
in Digital Libraries.*

-   **Algorithm:** Sparse Tensors, Compilers, Image Processing,
    Scientific Computing, Graph Analytics

-   **Program:** Julia, C++, Python

-   **Compilation:** Makefile, CMake, gcc, Julia, LLVM

-   **Transformations:** Sparsity and Structural Specialization

-   **Binary:** x86-64, also ARM

-   **Data set:** Synthetic, SuiteSparse, MNIST, Omniglot, HumanSketches

-   **Run-time environment:** Ubuntu 22.04.5 LTS, Linux
    5.15.0-119-generic, Root Access

-   **Hardware:** 12-core 2-socket Intel Xeon CPU E5-2680 v3 running at
    2.50GHz with 128GB of memory.

-   **Run-time state:** Sensitive to memory bandwidth, cache size

-   **Execution:** Requires exclusive access to node for repeatable
    results

-   **Metrics:** Execution time, Speedup

-   **Output:** JSON table, PNG plot

-   **Experiments:** SpMV, SpGEMM, Erosion, Histogram, Breadth-First
    Search, Shortest Path

-   **How much disk space required (approximately)?:**

-   **How much time is needed to prepare workflow (approximately)?:**

-   **How much time is needed to complete experiments
    (approximately)?:**

-   **Publicly available?:** yes

-   **Code licenses (if publicly available)?:** MIT

-   **Data licenses (if publicly available)?:** MIT

-   **Workflow framework used?:** SLURM, Shell

-   **Archived (provide DOI)?:** 10.5281/zenodo.14597755

## Description

### How delivered

The artifact may be downloaded from zeonodo at
[doi.org/10.5281/zenodo.14597755](doi.org/10.5281/zenodo.14597755){.uri},
or cloned from the `oopsla-25-artifact` branch of the FinchBenchmarks
repository on GitHub at
<https://github.com/finch-tensor/FinchBenchmarks>. The artifact contains
a copy of the Finch.jl compiler version v1.1.0, and all benchmarks used
in the paper. The artifact takes 1.6 MB to download, and 17 GB of disk
space once it has been extracted and built and datasets have been
downloaded and generated.

The `deps` subdirectory contains major dependencies required. The
`spmv`, `spgemm`, `graphs`, and `images` directories contain the
benchmarks corresponding to the SpMV, SpGEMM, Graphs, and Image
Processing sections of the paper, respectively.

### Hardware dependencies

This artifact was run on a 12-core 2-socket Intel Xeon CPU E5-2680 v3
running at 2.50GHz with 128GB of memory. The Intel MKL and CORA
benchmarks require an x86-64 machine to build and run, but we believe
that the other benchmarks can be built on ARM hardware.

### Software dependencies

The results in the paper concern the following software dependencies,
and special notes for building are included. Although we include the
sources of the artifact, the artifact itself contains these repositories
as submodules.

1.  Julia [@bezanson_julia:_2017] v1.10.7 Julia can be installed via
    'juliaup' at <https://github.com/JuliaLang/juliaup> or at
    <https://julialang.org/downloads/>.

2.  Finch 1.1.0 Finch is a registered Julia package and will be
    installed automatically during setup. However, if for whatever
    reason you would like to use the copy included with the artifact,
    you may run

              julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path="./deps/Finch.jl"))'

    from the root of the artifact.

3.  TACO [@kjolstad_tensor_2017] at commit `1278503a1` from
    <https://github.com/tensor-compiler/taco>, corresponding to the
    "benchmark" branch. Taco requires CMake.

4.  Eigen 3.4.0 [@guennebaud_eigen_2010] from
    <https://gitlab.com/libeigen/eigen.git>

5.  GraphBLAS 9.4.2 from
    <https://github.com/DrTimothyAldenDavis/GraphBLAS>.

6.  LAGraph 1.1.4 from <https://github.com/GraphBLAS/LAGraph>.

7.  Graphs.jl 1.9 from <https://github.com/JuliaGraphs/Graphs.jl>.

8.  Intel MKL 2024.2 [@noauthor_developer_2024], available from
    <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html>.
    This will need to be installed to the `deps/intel` folder.

9.  The Cora tensor compiler [@fegade_cora_2022] at commit `8e7de1d7c`
    from <https://github.com/pratikfegade/cora.git>. An artifact is also
    available at <https://doi.org/10.5281/zenodo.6326456>. Cora requires
    MKL, LLVM 9.0.0, Z3 4.8.8, CMake, and Python 3. Instructions for
    building are available in the `cora/ae_appendix_supplement.pdf`
    file.

Our build process for all comparison frameworks is automated in our
Makefile, included at the root of the directory. The `Project.toml` file
contains all of the required Julia dependecies (as well as their
versions). The `pyproject.toml` file contains all of the required python
dependecies (as well as their versions).

We built our artifact on Ubuntu 22.04.5 LTS, Linux 5.15.0-119-generic,
using the following dependencies:

1.  cmake 3.22.1

2.  gcc 11.4.0

3.  Python 3.10.12

4.  We used poetry 1.8.5 to manage python dependencies, which can be
    installed with `pip` or following
    <https://python-poetry.org/docs/#installation>.

5.  jq 1.6, git 2.34.1, curl 7.81.0, GNU tar 1.34, and UnZip 6.00

### Data sets

1.  SuiteSparse\
    The SuiteSparse Matrix Collection [@davis_university_2011] is
    available at <https://sparse.tamu.edu/>. We use the MatrixDepot.jl
    package at <https://github.com/JuliaLinearAlgebra/MatrixDepot.jl> to
    download matrices from this collection. The precise datasets used
    for each benchmark are listed in the test harnesses.

2.  MNIST\
    The MNIST dataset [@lecun_gradient-based_1998] is available at
    <http://yann.lecun.com/exdb/mnist/>. We use the MLDatasets.jl
    package at <https://github.com/JuliaML/MLDatasets.jl> to download
    this dataset.

3.  Omniglot\
    The Omniglot dataset [@lake_human-level_2015] is available at
    <https://www.omniglot.com/>. We use the MLDatasets.jl package at
    <https://github.com/JuliaML/MLDatasets.jl> to download this dataset.

4.  HumanSketches\
    We evaluate on a dataset of human line drawings [@eitz_how_2012],
    available at
    <https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip>.

5.  Dip3masks\
    We also hand-selected a subset of mask images from a digital image
    processing textbook [@gonzalez_digital_2006]. The precise set of
    images used is included with the artifact.

6.  Synthetic Data\
    Scripts are included to generate synthetic data. For spmv, we
    generate banded, triangular, and a reverse permutation matrix. For
    SpGEMM, we generate a series of increasingly larger uniformly random
    sparse matrices. For the Graphs dataset, we generate a few RMAT
    [@chakrabarti_r-mat_2004] graphs to match the dataset used by Yang
    et. al. [@yang_implementing_2018].

## Installation

### 1. Install system dependecies

Install Julia, Python, CMake, and other system dependencies as described
in the software dependencies section.

### 2. Download the core dependencies

Several dependencies are included as submodules in the `deps/` folder,
referencing the precise commits we used to build the artifact. Most of
these can be installed via

      git submodule update --init --recursive

MKL must be manually installed to the `deps/intel` folder.
[Install Intel MKL version 2024.2, available from https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html](Install Intel MKL version 2024.2, available from https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html){.uri}.
You'll need to request an academic license on the website, then download
an offline installer. There should be instructions on the website for
how to run the install script. When asked, you can install to the
`deps/intel` folder.

The makefile also contains instructions to clone the submodules.

### 3. Build the dependencies and benchmarks

The makefile contains targets to build all of the benchmarks and
dependencies. We refer to each individual dependency for more detailed
instructions. We expect that some system-specific adjustments to the
makefile may be necessary to build CORA, as it has many complex
subdependencies such as LLVM and Z3.

When all dependencies have been successfully installed, from the root of
the artifact, run

      make

### 4. Instantiate Runtime Environments

In this step, we will setup and install the Julia and Python
environments. This can be achieved by running from the root of the
artifact:

      bash instantiate_environments.sh

which simply runs the following commands:

      julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
      poetry install --no-root

## Experiment workflow

### 1. Running the dataset generators

To generate the synthetic data used in the benchmarks, run

      bash generate_data.sh

### 2. Running the benchmarks

To run all benchmarks, run

      bash run_benchmarks.sh

This script will run all benchmarks and generate the JSON tables and PNG
plots used in the paper.

A more fine-grained approach may be taken to run the benchmarks
individually. Each experiment subdirectory contains a `run_*.sh` script
that will run that particular benchmark, and the commands contained
within may be modified to run different subsets of experiments. We
expect that running the whole set of experiments may take in excess of
24 hours, so some experiments may be commented out. Additionally, if
evaluators have access to a SLURM cluster, we include SLURM scripts that
help accelerate the process, which may need adaptation to your
particular cluster. It is convenient to use `jq` to combine json outputs
from parallel runs, for example,

      jq -s 'add' results_*.json > combined_results.json

### 3. Plotting the output

To generate the plots used for each experiment, run the corresponding
chart.py script in each experiment directory. For example, to generate
the plots for the SpMV experiment, run from within the `spmv` directory

      poetry run python chart.py

Plots will be generated in the corresponding `charts/` directory of each
experiment.

## Evaluation and expected result

Reference JSON and PNG plot results are stored for each experiement with
the prefix `reference_`. Evaluators can compare the reference plots with
the result plots. To generate our geomean speedup claims, evaluators may
run

      poetry run get_geomean.py

in the corresponding experiment directory and compare to the text.

## Reusability Guide

All of the julia run scripts support a `--help` flag describing
parameters which allowing one to customize the experiments. Separately,
the Finch compiler is a featureful Julia package, and the evaluators are
encouraged to experiment with the compiler using the documentation at
<https://finch-tensor.github.io/Finch.jl/stable/> and
<https://finch-tensor.github.io/Finch.jl/stable/docs/language/calling_finch/>.
Reviewers may also build the documentation locally by running
`julia docs/make.jl` from the root of the `Finch.jl` directory. Some
example Finch programs are given in the `docs/examples` directory.
