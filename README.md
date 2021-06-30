# High Performance Matrix Multiplication on CPU


## Introduction

Matrix multiplication is the mathematical operation that defines product of two matrices and stores the result in another matrix. Mathematically:

```math
	
	C(m, n) = A(m, k) * B(k, n)

```

It is implemented as the dot product between the row of matrix A and the column of matrix B. The naive implementation in C is:

```C
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			for (int r = 0; r < k; r++){
				C(i, j) += A(i, r) * B(r, j);
			}
		}
	}
```


## CPU Performance Metrics

The naive implementation of the matrix multiplication is very inefficient. How would we measure the efficiency of the Matrix multiplication? The performance of the matrix multiplication is measured with respect to the CPU machine peak performance. Then question arises what is the CPU machine peak and how would we compute it? 

* First determine the CPU machine peak perfomance:

	- Let's consider the case for `Intel Cascade Lake processor with AVX3 enabled`

	- Cascade Lake has `2 Sockets` and each socket has `28 cores`

	- The AVX clock frequecy of the cascade lake is `2.4 * 10^9 cylces/sec` or `2.4 GHz`

	- This processor contains `2 vector processing units (VPUs)` or ports

	- Each VPU has `2 Fused Multiply-Add (FMADD)`  	 

	- The SIMD width of the processor is `16`, i.e. its capable of running 16-wide FMADD instructions on each port every cycle

 To calculate the throughput of the machine per core we need to multiply these numbers together i.e. `16 floating point numbers, times two VPUs, times two FMADD, times 2.4 GHz`. The throughput of the processor for 1 core is ~ `153 GFlops`. For 1 socket i.e. 28 cores it is ~ `4300.8 GFlops`.

 The AVX clock frequency can be determined by this C code:

 ```C
 	
 	int64_t procf = 1;
    int64_t startTick = __rdtsc();
    sleep(1);
    int64_t endTick = __rdtsc();
    procf = endTick - startTick;
    printf("procf = %ld\n", procf);

 ```







