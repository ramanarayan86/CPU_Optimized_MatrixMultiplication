# High Performance Matrix Multiplication on CPU


Matrix multiplication is the mathematical operation that defines product of two matrices and stores the result in another matrix. Mathematically:

```math
	
	C(m, n) = A(m, k) * B(k, n)

```

It is implemented as the dot product between the row of matrix A and the column of matrix B. The naive implementation in C is:

```C
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int r = 0; r < k; r++)
			{
				C(i, j) += A(i, r) * B(r, j);
			}
		}
	}
```


## Basic CPU Evaluation Metrics

 


