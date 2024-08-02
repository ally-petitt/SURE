# Chevron SURE Program - GPU Parallelization Data and Program
This repository contains code and data collected during the Chevron Summer Undergraduate Research Experience (SURE) program on 7/8/24-8/1/24. 

![Chevron Logo](https://www.coastkeeper.org/wp-content/uploads/2022/08/Chevron-Logo.png)


## Table of Contents
* [Example](#sure-project-objective)
2. [Example2](#example2)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)


# SURE Project Overview
## Background
Central Processing Units (CPUs) traditionally execute operations serially on few high-powered cores. This is effective for handling many computing operations, however, the throughput of a CPU can often become a limitation in High Performance Computing (HPC) environments when execution speed is critical. Many HPC setups use a combination of CPUs, GPUs, and specialized accelerators such as a Tensor Processing Units (TPUs) to remedy this. This project focuses on accelerating the execution speed of GPUs, specifically.

Graphics Processing Units (GPUs) excel at providing higher throughput than CPUs by utilizing thousands of lower-powered threads to break large problems down into smaller sub-problems that can be atomically calculated in parallel. In particular, GPUs are well-equipped for matrix manipulation operations such as matrix inversion and matrix multiplication.


## Objective
The primary objective of this research project was to increase the performance of the GPU. Three optimization pathways were explored as a means of accomplishing this:

1. Parallelizing GPU computations over multiple nodes in a cluster within a LAN via CUDA-aware OpenMPI.
2. The use of optimization algorithms such as [gradient descent](https://www.ibm.com/topics/gradient-descent) and [Branch & Bound](https://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf).
3. Effectively managing matrix sparsity to ensure optimal execution speed.

Combined, the data collected from this research informs the ideal matrix sparsity to arrange for when performing parallel GPU computations that can implement optimization algorithms as an avenue for performance acceleration. A CUDA kernel was developed to test the impact of each optimization pathway and collect data for comparison against a control. 

### Technologies Used
* [CUDA v12.5](https://docs.nvidia.com/cuda/) to facillitate lower-level interaction with the GPU.
* Matrix manipulation operations imported from the [cuBLAS API](https://docs.nvidia.com/cuda/cublas/).
* [PyTorch] built-in matrix inversion functionality as a benchmark to compare developed CUDA kernel against.
* [OpenMPI](https://www.open-mpi.org/) to parallelize GPU computations across multiple nodes.


# Matrix Manipulation and GPU Parallelization
This section was written assuming a current working directory of `matrix_manipulation/`.

## Program Overview
The set of programs stored within the `matrix_manipulation/` directory serve two distinct, yet related purposes:

1. Collect control data on matrix inversion performance in PyTorch and CUDA (with cuBLAS).
2. Create a CUDA-aware OpenMPI program to profile matrix multiplication on:
    * A single, local GPU
    * 2 remote, parallel GPUs


The collected data was stored in `data/matrix_manipulation_data.ods`.


## Build and Run Instructions
### `time_matrix_multiplication_mpi.cu`
This program uses the MPI protocol, abstracted by the OpenMPI library, in order to coordinate the parallel execution of matrix multiplication among two remote instances. The combined execution times are printed to stdout in CSV format.

An example for the `time_matrix_multiplication_mpi.cu` kernel is shown below. `$PROGRAM` is exported to the name of the CUDA kernel file excluding the file extension. An object file is first compiled with `nvcc` to add proper CUDA functionality and is then linked by `mpic++`.

```
export PROGRAM=time_matrix_multiplication_mpi
nvcc -c $PROGRAM.cu -o $PROGRAM.o && mpic++ -o $PROGRAM $PROGRAM.o -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
mpirun -n2 -hostfile hosts.txt ./$PROGRAM
```

In the above example, `hosts.txt` contains the IP addresses along with any cooresponding slot limitations for each node. The host file used to collect data was as follows:

```
worker   slots=1
manager  slots=1
```

The configuration steps used to setup the worker and manager instances in `hosts.txt` is documented by [Dwaraka Nath](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/). More information on `mpirun` and its hostfile can be found [here](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile).

### `time_matrix_muliplication_single.cu`

For the purpose of keeping the control data consistent with the non-control data from `time_matrix_multiplication_mpi.cu`, `time_matrix_muliplication_single.cu` also uses OpenMPI, however, it makes no attempt at parallelizing the operations among separate cloud instances. The following code snippet contains the commands used to build and execute the kernel:

```
export PROGRAM=time_matrix_multiplication_single
nvcc -c $PROGRAM.cu -o $PROGRAM.o && mpic++ -o $PROGRAM $PROGRAM.o -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
mpirun -n1 ./$PROGRAM
```


### `time_matrix_inversion.cu`
This program measures the execution time of performing matrix inversion operations on matrices of sizes 1 to 4250 in increments of 50.
The build process of `time_matrix_inversion.cu` has fewer dependencies than the previously documented files, which results in fewer build steps:

```
nvcc time_matrix_inversion.cu -o time_matrix_inversion
./time_matrix_inversion
```

### `pytorch_time_matrix_inversion.py`

This Python program measures the execution time of performing matrix inversion operations on different sized matrices. The results are then compared against the execution time of performing the same operation in the developed CUDA kernel `time_matrix_inversion.cu` in `../data/matrix_manipulation_data.ods`.

The dependencies for this program can be downloaded through `pip`.
```
pip install torch csv
```

The Python file can be executed as follows:
```
python ./pytorch_time_matrix_inversion.py
```