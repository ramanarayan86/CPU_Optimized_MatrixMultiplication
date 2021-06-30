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

The naive implementation of the matrix multiplication is very inefficient. How would we measure the efficiency of the Matrix multiplication? The performance of the matrix multiplication is measured with respect to the CPU machine peak performance. Then question arises what is the CPU machine peak and how would we determine it? 

* Let's first determine the CPU machine peak perfomance:

Let's consider the case for `Intel Cascade Lake processor`





