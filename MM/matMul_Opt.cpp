//#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <x86intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mkl.h>
#include <cstring>
#include <immintrin.h>


#ifdef VTUNE_ANALYSIS
    #include <ittnotify.h>
#endif

#define USE_PERF_COUNTERS 0

#if USE_PERF_COUNTERS
	#include "counters.h"
#endif


using namespace std;

#define MKL 1
#define BASELINE 0
#define WRITE 1

// #define M 1000
// #define K 1000
// #define N 1000

#define A_(i, j) A[(i)*lda + j]
#define B_(i, j) B[(i)*ldb + j]
#define C_(i, j) C[(i)*ldc + j]


#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


typedef struct  dense_matrix
{
        int rows;
        int cols;
        float* values;
        float **values2D;
        bool is_rowmajor;
}dense_matrix;


// *************************************** Genearate Dense Matrix for Multiplication *************************************************

dense_matrix generate_matrix(int rows, int cols, float ** values_){

    // float* values = new float[rows * cols]();
    float* values = (float*) mkl_malloc (rows * cols * sizeof(float), 64);

    for(int i=0; i< rows*cols; i++){
        values[i] = ( static_cast<float>(rand()) + 1.0 ) / static_cast<float>(RAND_MAX);
    }

    // float ** matrix = (float*) mkl_malloc (rows * cols * sizeof(float), 64);
    float ** matrix = (float **)malloc(rows * sizeof(float *));
    for (int i=0; i<rows; i++)
         matrix[i] = (float *)malloc(cols * sizeof(float));

    for(int i=0; i< rows; i++){
    	for(int j=0; j < cols; j++){
    		matrix[i][j] = values[i*cols + j];
    	}
    }
    // Packaging dense matrix into structure.
    dense_matrix dense_mat;

    dense_mat.rows = rows;
    dense_mat.cols = cols;
    dense_mat.values = values;
    dense_mat.values2D = matrix;

    *values_ = values;

    return dense_mat;
}


// ********************************************** print functionalities *********************************************************
void print_matrix(float* data, int rows, int cols, bool is_rowmajor){
    for (int i = 0; i < min(rows, 20); ++i)
    {
        for (int j = 0; j < min(cols, 20); ++j)
        {
            if(is_rowmajor){
                std::cout << std::setw(3) << data[i*cols + j] << " ";
            }else{
                std::cout << std::setw(3) << data[j*rows + i] << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ********************************************************************************************************************************
void print_2Dmatrix(float** data, int rows, int cols, bool is_rowmajor){
    for (int i = 0; i < min(rows, 15); ++i)
    {
        for (int j = 0; j < min(cols, 15); ++j)
        {
            // if(is_rowmajor){
                std::cout << std::setw(3) << data[i][j] << " ";
            // }else{
                // std::cout << std::setw(3) << data[j*rows + i] << " ";
            // }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ********************************************************************************************************************************

void print_m512Var (__m512 in)
{
    alignas(16) float v[16];
    _mm512_store_ps((__m512*) v, in);
    std::cout << "v16_flt:" << v[0] << " " << v[1] << " " <<v[2] << " " <<v[3] << " | "
                            <<  v[4] << " " << v[5] << " " << v[6] << " " << v[7] << " | " <<
                                v[8] << " " << v[9] << " " << v[10] << " " << v[11] << " | " <<
                                v[12] << " " << v[13] << " " << v[14] << " " << v[15] << " " << std::endl;
}

// ********************************************************************************************************************************

// UNROLL & JAM

void MM_VEC_INNER_JAM(float* A, float* B, float* C, int M, int N, int K, int m0, int m1, int n0, int n1, int k0, int k1)
{
    int m,n,k;

    __m512 vec_A1 = _mm512_setzero_ps();
    __m512 vec_A2 = _mm512_setzero_ps();
    __m512 vec_A3 = _mm512_setzero_ps();
    __m512 vec_A4 = _mm512_setzero_ps();

    __m512 vec_B = _mm512_setzero_ps();

    __m512 vec_C1 = _mm512_setzero_ps();
    __m512 vec_C2 = _mm512_setzero_ps();
    __m512 vec_C3 = _mm512_setzero_ps();
    __m512 vec_C4 = _mm512_setzero_ps();

    for (m = m0; m <= m1; m+=4){
        for (k = k0; k <= k1; k++){

            vec_A1 = _mm512_set1_ps (A[m*K+k] ) ;
            vec_A2 = _mm512_set1_ps (A[ (m+1)*K + k] ) ;
            vec_A3 = _mm512_set1_ps (A[ (m+2)*K + k] ) ;
            vec_A4 = _mm512_set1_ps (A[ (m+3)*K + k] ) ;

            _mm_prefetch (&B [ (k+2)*N + n], _MM_HINT_T0 ) ;
            _mm_prefetch (&C [ (m+4)*N + n] , _MM_HINT_T0 ) ;
            _mm_prefetch (&C [ (m+5)*N + n ] , _MM_HINT_T0 ) ;
            _mm_prefetch (&C [ (m+6)*N + n ] , _MM_HINT_T0 ) ;

            for(n = n0; n <= n1; n+=16){

                vec_B = _mm512_load_ps((__m512*)&B[k*N + n]);

                vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                vec_C3 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]) ;
                vec_C4 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]) ;

                vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B, vec_C1);
                _mm512_store_ps (( __m512*)&C[(m+0)*N + n] , vec_C1 ) ;

                vec_C2 = _mm512_fmadd_ps(vec_A2, vec_B, vec_C2);
                _mm512_store_ps (( __m512*)&C[(m+1)*N + n] , vec_C2 ) ;

                vec_C3 = _mm512_fmadd_ps(vec_A3, vec_B, vec_C3);
                _mm512_store_ps (( __m512*)&C[(m+2)*N + n] , vec_C3 ) ;

                vec_C4 = _mm512_fmadd_ps(vec_A4, vec_B, vec_C4);
                _mm512_store_ps (( __m512*)&C[(m+3)*N + n] , vec_C4 ) ;

            }
        }
    }
}


void MM_L2CB(float* A, float* B, float* C, int M, int N, int K)
{
    int L2SIZE = 512 * 1024 / 4 ;  // L2 cache = 131072 elements 

    int mb = 128; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int kb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int nb = 128; //1024; // 512;

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
        m1 = m0 + mb - 1;
        if( m1 >= M){m1 = M -1;}
        for(int n0=0; n0 < N; n0 +=nb){
            n1 = n0 + nb - 1;
            if( n1 >= N){n1 = N -1;}
            for (int k0=0; k0 < K; k0 += kb){
                k1 = k0 + kb - 1;
                if( k1 >= K){k1 = K -1;} 
                // MM_INNER(A, B, C, m0, m1, n0, n1, k0, k1);
                // MM_VEC_INNER_IKJ(A, B, C, M, N, K, m0, m1, n0, n1, k0, k1);
                
                MM_VEC_INNER_JAM(A, B, C, M, N, K, m0, m1, n0, n1, k0, k1);
                // MM_L1CB_IKJ(A, B, C, M, N, K, m0, m1, n0, n1, k0, k1);
                // MM_L1CB_REG_IKJ(A, B, C, M, N, K, m0, m1, n0, n1, k0, k1);

            }
        }
    }
}

//***************************************************************************************************************************************************

// template <int regsA, int regsB>
void MM_RB(int k, float** A, float** B, float** C, int lda, int ldb, int ldc){

    int regsA = 4;
    int regsB = 3;
    
    // float CC[regsA][regsB] = {0.0};
    // memset(&CC[0][0], 0, regsA * regsB * sizeof(float));

    for (int p = 0; p < k; p++){
        for (int ai = 0; ai < regsA; ai++)
        {
            __m512 AA = _mm512_load_ps((__m512*)&A[ai*16][p]);
            for (int bi = 0; bi < regsB; bi++)
            {
                __m512 BB = _mm512_set1_ps(B[p][bi]);
                // CC[ai][bi] = _mm512_fmadd_ps() AA * BB; 
            }
        }
    }

    for (int bi = 0; bi < regsB; bi++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
            // _mm512_store_ps (( __m512*)&C[ai*16][bi] , CC[ai][bi]);
        }
    }
}


//***************************************************************************************************************************************************

void MM_SIMD(float* A, float* B, float* C, int M, int N, int K)
{
	int mb = 256; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int kb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int nb = 1024; //1024; 

    int regsA = 4;
    int regsB = 3;

    int mr = regsA * 16;
    int nr = regsB;

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
    	int M1 = m0 + mb;
    	for (int k0=0; k0 < K; k0 += kb){
    			int K1 = k0 + kb;
	    	for (int n0=0; n0 < N; n0 +=nb){
	    		int N1 = n0 + nb ;    		
    			for(int m=m0; m < M1; m+=4){
    				for(int k=k0; k< K1; k+=2){
    					 __m512 A1 = _mm512_set1_ps (A[ m * K + k ] ) ;
					     __m512 A2 = _mm512_set1_ps (A[ m * K + (k+1)] ) ;
					     __m512 A3 = _mm512_set1_ps (A[ (m+1) * K + k ] ) ;
					     __m512 A4 = _mm512_set1_ps (A[ (m+1) * K + (k+1)] ) ;
					     __m512 A5 = _mm512_set1_ps (A[ (m+2) * K + k] ) ;
					     __m512 A6 = _mm512_set1_ps (A[ (m+2) * K + (k+1)] ) ;
					     __m512 A7 = _mm512_set1_ps (A[ (m+3) * K + k] ) ;
					     __m512 A8 = _mm512_set1_ps (A[ (m+3) * K + (k+1)] ) ;

					    // _mm_prefetch (&B [ (k+2)*N + n], _MM_HINT_T0 ) ;
					    // _mm_prefetch (&B [ (k+3)*N + n], _MM_HINT_T0 ) ;

			      //       _mm_prefetch (&C [ (m+4)*N + n] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+5)*N + n ] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+6)*N + n ] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+7)*N + n ] , _MM_HINT_T0 ) ;

    					for(int n=n0; n< N1; n+=16){
    						__m512 C1 = _mm512_load_ps((__m512*)&C[m*N + n]);
    						__m512 B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
    						__m512 B2 = _mm512_load_ps((__m512*)&B[(k+1)*N + n]);

    						C1 = _mm512_add_ps(C1, _mm512_mul_ps(A1, B1));
    						C1 = _mm512_add_ps(C1, _mm512_mul_ps(A2, B2));
    						_mm512_store_ps (( __m512*)&C[(m)*N + n] , C1);

    						__m512 C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]);
    						C2 = _mm512_add_ps(C2, _mm512_mul_ps(A3, B1));
    						C2 = _mm512_add_ps(C2, _mm512_mul_ps(A4, B2));
    						_mm512_store_ps (( __m512*)&C[(m+1)*N + n] , C2);

    						__m512 C3 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]);
    						C3 = _mm512_add_ps(C3, _mm512_mul_ps(A5, B1));
    						C3 = _mm512_add_ps(C3, _mm512_mul_ps(A6, B2));
    						_mm512_store_ps (( __m512*)&C[(m+2)*N + n] , C3);

    						__m512 C4 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]);
    						C4 = _mm512_add_ps(C4, _mm512_mul_ps(A7, B1));
    						C4 = _mm512_add_ps(C4, _mm512_mul_ps(A8, B2));
    						_mm512_store_ps (( __m512*)&C[(m+3)*N + n] , C4);
    					}
    				}
    			}
    		}
    	}
    }
}


void MM_SIMD_ROW(float* A, float* B, float* C, int M, int N, int K)
{
    int mb = 256; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int kb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int nb = 1024; //1024; 

    
    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
        int M1 = m0 + mb;
        for (int k0=0; k0 < K; k0 += kb){
                int K1 = k0 + kb;
            for (int n0=0; n0 < N; n0 +=nb){
                int N1 = n0 + nb ;          
                for(int m=m0; m < M1; m+=2){
                    for(int k=k0; k< K1; k+=4){

                         register __m512 vec_A1 = _mm512_set1_ps (A[ m * K + k ] ) ;
                         register __m512 vec_A2 = _mm512_set1_ps (A[ m * K + (k+1)] ) ;
                         register __m512 vec_A3 = _mm512_set1_ps (A[ m * K + (k+2)] ) ;
                         register __m512 vec_A4 = _mm512_set1_ps (A[ m * K + (k+3)] ) ;

                         register __m512 vec_A5 = _mm512_set1_ps (A[ (m+1) * K + k ] ) ;
                         register __m512 vec_A6 = _mm512_set1_ps (A[ (m+1) * K + (k+1)] ) ;
                         register __m512 vec_A7 = _mm512_set1_ps (A[ (m+1) * K + (k+2)] ) ;
                         register __m512 vec_A8 = _mm512_set1_ps (A[ (m+1) * K + (k+3)] ) ;

                         for(int n = n0; n < N1; n+=16){

                            register __m512 vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            register __m512 vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;

                            register __m512 vec_B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
                            register __m512 vec_B2 = _mm512_load_ps((__m512*)&B[(k+1)*N + n]);
                            register __m512 vec_B3 = _mm512_load_ps((__m512*)&B[(k+2)*N + n]);
                            register __m512 vec_B4 = _mm512_load_ps((__m512*)&B[(k+3)*N + n]);

                            // vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B1, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A2, vec_B2, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A3, vec_B3, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A4, vec_B4, vec_C1);
                            _mm512_store_ps (( __m512*)&C[(m)*N + n] , vec_C1 ) ;

                            // vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            vec_C2 = _mm512_fmadd_ps(vec_A5, vec_B1, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A6, vec_B2, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A7, vec_B3, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A8, vec_B4, vec_C2);
                            _mm512_store_ps (( __m512*)&C[(m+1)*N + n] , vec_C2 ) ;

                         }

                    }
                }
            }
        }
    }
}


void MM_SIMD_ROW2(float* A, float* B, float* C, int M, int N, int K)
{
    int mb = 256; //256;                 
    int kb = 128; //128;                 
    int nb = 1024; //1024; 

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
        int M1 = m0 + mb;
        for (int k0=0; k0 < K; k0 += kb){
                int K1 = k0 + kb;
            for (int n0=0; n0 < N; n0 +=nb){
                int N1 = n0 + nb ;          
                for(int m=m0; m < M1; m+=4){
                    for(int k=k0; k< K1; k+=16){

                         register __m512 vec_A1 = _mm512_set1_ps (A[ m * K + k ] ) ;
                         register __m512 vec_A2 = _mm512_set1_ps (A[ m * K + (k+1)] ) ;
                         register __m512 vec_A3 = _mm512_set1_ps (A[ m * K + (k+2)] ) ;
                         register __m512 vec_A4 = _mm512_set1_ps (A[ m * K + (k+3)] ) ;

                         // register __m512 vec_A5 = _mm512_set1_ps (A[ (m+1) * K + k ] ) ;
                         // register __m512 vec_A6 = _mm512_set1_ps (A[ (m+1) * K + (k+1)] ) ;
                         // register __m512 vec_A7 = _mm512_set1_ps (A[ (m+1) * K + (k+2)] ) ;
                         // register __m512 vec_A8 = _mm512_set1_ps (A[ (m+1) * K + (k+3)] ) ;

                         // register __m512 vec_A9 = _mm512_set1_ps (A[ (m+2) * K + k ] ) ;
                         // register __m512 vec_A10 = _mm512_set1_ps (A[ (m+2) * K + (k+1)] ) ;
                         // register __m512 vec_A11 = _mm512_set1_ps (A[ (m+2) * K + (k+2)] ) ;
                         // register __m512 vec_A12 = _mm512_set1_ps (A[ (m+2) * K + (k+3)] ) ;

                         // register __m512 vec_A13 = _mm512_set1_ps (A[ (m+3) * K + k ] ) ;
                         // register __m512 vec_A14 = _mm512_set1_ps (A[ (m+3) * K + (k+1)] ) ;
                         // register __m512 vec_A15 = _mm512_set1_ps (A[ (m+3) * K + (k+2)] ) ;
                         // register __m512 vec_A16 = _mm512_set1_ps (A[ (m+3) * K + (k+3)] ) ;

                         // #pragma unroll(2)
                         for(int n = n0; n < N1; n+=4){

                            register __m512 vec_C00 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            register __m512 vec_C01 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            register __m512 vec_C02 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]) ;
                            register __m512 vec_C03 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]) ;

                            register __m512 vec_C10 = _mm512_load_ps((__m512*)&C[m*N + (n +1)]) ;
                            register __m512 vec_C11 = _mm512_load_ps((__m512*)&C[(m+1)*N + (n + 1)]) ;
                            register __m512 vec_C12 = _mm512_load_ps((__m512*)&C[(m+2)*N + (n + 1)]) ;
                            register __m512 vec_C13 = _mm512_load_ps((__m512*)&C[(m+3)*N + (n + 1)]) ;

                            register __m512 vec_C20 = _mm512_load_ps((__m512*)&C[m*N + (n + 2)]) ;
                            register __m512 vec_C21 = _mm512_load_ps((__m512*)&C[(m+1)*N + (n+ 2)]) ;
                            register __m512 vec_C22 = _mm512_load_ps((__m512*)&C[(m+2)*N + (n+ 2)]) ;
                            register __m512 vec_C23 = _mm512_load_ps((__m512*)&C[(m+3)*N + (n+ 2)]) ;

                            register __m512 vec_C30 = _mm512_load_ps((__m512*)&C[m*N + (n+ 3)]) ;
                            register __m512 vec_C31 = _mm512_load_ps((__m512*)&C[(m+1)*N + (n+ 3)]) ;
                            register __m512 vec_C32 = _mm512_load_ps((__m512*)&C[(m+2)*N + (n+ 3)]) ;
                            register __m512 vec_C33 = _mm512_load_ps((__m512*)&C[(m+3)*N + (n+ 3)]) ;

                            register __m512 vec_B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
                            register __m512 vec_B2 = _mm512_load_ps((__m512*)&B[(k+1)*N + n]);
                            register __m512 vec_B3 = _mm512_load_ps((__m512*)&B[(k+2)*N + n]);
                            register __m512 vec_B4 = _mm512_load_ps((__m512*)&B[(k+3)*N + n]);

                           
                            // vec_C01 = _mm512_fmadd_ps(vec_A1, vec_B1, vec_C01);
                            // vec_C01 = _mm512_fmadd_ps(vec_A2, vec_B2, vec_C01);
                            // vec_C01 = _mm512_fmadd_ps(vec_A3, vec_B3, vec_C01);
                            // vec_C01 = _mm512_fmadd_ps(vec_A4, vec_B4, vec_C01);
                            // _mm512_store_ps (( __m512*)&C[(m)*N + n] , vec_C01 ) ;

                            
                            // vec_C02 = _mm512_fmadd_ps(vec_A5, vec_B1, vec_C02);
                            // vec_C02 = _mm512_fmadd_ps(vec_A6, vec_B2, vec_C02);
                            // vec_C02 = _mm512_fmadd_ps(vec_A7, vec_B3, vec_C02);
                            // vec_C02 = _mm512_fmadd_ps(vec_A8, vec_B4, vec_C02);
                            // _mm512_store_ps (( __m512*)&C[(m+1)*N + n] , vec_C02 ) ;

                            // vec_C03 = _mm512_fmadd_ps(vec_A9, vec_B1, vec_C03);
                            // vec_C03 = _mm512_fmadd_ps(vec_A10, vec_B2, vec_C03);
                            // vec_C03 = _mm512_fmadd_ps(vec_A11, vec_B3, vec_C03);
                            // vec_C03 = _mm512_fmadd_ps(vec_A12, vec_B4, vec_C03);
                            // _mm512_store_ps (( __m512*)&C[(m+2)*N + n] , vec_C03 ) ;

                            // vec_C04 = _mm512_fmadd_ps(vec_A13, vec_B1, vec_C04);
                            // vec_C04 = _mm512_fmadd_ps(vec_A14, vec_B2, vec_C04);
                            // vec_C04 = _mm512_fmadd_ps(vec_A15, vec_B3, vec_C04);
                            // vec_C04 = _mm512_fmadd_ps(vec_A16, vec_B4, vec_C04);
                            // _mm512_store_ps (( __m512*)&C[(m+3)*N + n] , vec_C04 ) ;

                         }

                    }
                }
            }
        }
    }
}


void MM_SIMD_ROW3(float* A, float* B, float* C, int M, int N, int K)
{
    int mb = 1024; //256;                
    int kb = 128; //128;                 
    int nb = 1024; //1024; 

    // int regsA = 4;
    // int regsB = 3;

    // int mr = regsA * 16;
    // int nr = regsB;

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
        int M1 = m0 + mb;
        for (int k0=0; k0 < K; k0 += kb){
                int K1 = k0 + kb;
            for (int n0=0; n0 < N; n0 +=nb){
                int N1 = n0 + nb ;          
                for(int m=m0; m < M1; m+=4){
                    for(int k=k0; k< K1; k+=4){

                         register __m512 vec_A1 = _mm512_set1_ps (A[ m * K + k ] ) ;
                         register __m512 vec_A2 = _mm512_set1_ps (A[ m * K + (k+1)] ) ;
                         register __m512 vec_A3 = _mm512_set1_ps (A[ m * K + (k+2)] ) ;
                         register __m512 vec_A4 = _mm512_set1_ps (A[ m * K + (k+3)] ) ;

                         register __m512 vec_A5 = _mm512_set1_ps (A[ (m+1) * K + k ] ) ;
                         register __m512 vec_A6 = _mm512_set1_ps (A[ (m+1) * K + (k+1)] ) ;
                         register __m512 vec_A7 = _mm512_set1_ps (A[ (m+1) * K + (k+2)] ) ;
                         register __m512 vec_A8 = _mm512_set1_ps (A[ (m+1) * K + (k+3)] ) ;

                         register __m512 vec_A9 = _mm512_set1_ps (A[ (m+2) * K + k ] ) ;
                         register __m512 vec_A10 = _mm512_set1_ps (A[ (m+2) * K + (k+1)] ) ;
                         register __m512 vec_A11 = _mm512_set1_ps (A[ (m+2) * K + (k+2)] ) ;
                         register __m512 vec_A12 = _mm512_set1_ps (A[ (m+2) * K + (k+3)] ) ;

                         register __m512 vec_A13 = _mm512_set1_ps (A[ (m+3) * K + k ] ) ;
                         register __m512 vec_A14 = _mm512_set1_ps (A[ (m+3) * K + (k+1)] ) ;
                         register __m512 vec_A15 = _mm512_set1_ps (A[ (m+3) * K + (k+2)] ) ;
                         register __m512 vec_A16 = _mm512_set1_ps (A[ (m+3) * K + (k+3)] ) ;

                         for(int n = n0; n < N1; n+=32){

                            register __m512 vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            register __m512 vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            register __m512 vec_C3 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]) ;
                            register __m512 vec_C4 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]) ;

                            register __m512 vec_C5 = _mm512_load_ps((__m512*)&C[m*N + (n + 16)]) ;
                            register __m512 vec_C6 = _mm512_load_ps((__m512*)&C[(m+1)*N + (n + 16)]) ;
                            register __m512 vec_C7 = _mm512_load_ps((__m512*)&C[(m+2)*N + (n + 16)]) ;
                            register __m512 vec_C8 = _mm512_load_ps((__m512*)&C[(m+3)*N + (n + 16)]) ;

                            register __m512 vec_B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
                            register __m512 vec_B2 = _mm512_load_ps((__m512*)&B[(k+1)*N + n]);
                            register __m512 vec_B3 = _mm512_load_ps((__m512*)&B[(k+2)*N + n]);
                            register __m512 vec_B4 = _mm512_load_ps((__m512*)&B[(k+3)*N + n]);

                            register __m512 vec_B5 = _mm512_load_ps((__m512*)&B[k*N + (n + 16)]);
                            register __m512 vec_B6 = _mm512_load_ps((__m512*)&B[(k+1)*N + (n + 16)]);
                            register __m512 vec_B7 = _mm512_load_ps((__m512*)&B[(k+2)*N + (n + 16)]);
                            register __m512 vec_B8 = _mm512_load_ps((__m512*)&B[(k+3)*N + (n + 16)]);

                            // vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B1, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A2, vec_B2, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A3, vec_B3, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A4, vec_B4, vec_C1);
                            _mm512_store_ps (( __m512*)&C[(m)*N + n] , vec_C1 ) ;

                            // vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            vec_C2 = _mm512_fmadd_ps(vec_A5, vec_B1, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A6, vec_B2, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A7, vec_B3, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A8, vec_B4, vec_C2);
                            _mm512_store_ps (( __m512*)&C[(m+1)*N + n] , vec_C2 ) ;

                            vec_C3 = _mm512_fmadd_ps(vec_A9, vec_B1, vec_C3);
                            vec_C3 = _mm512_fmadd_ps(vec_A10, vec_B2, vec_C3);
                            vec_C3 = _mm512_fmadd_ps(vec_A11, vec_B3, vec_C3);
                            vec_C3 = _mm512_fmadd_ps(vec_A12, vec_B4, vec_C3);
                            _mm512_store_ps (( __m512*)&C[(m+2)*N + n] , vec_C3 ) ;

                            vec_C4 = _mm512_fmadd_ps(vec_A13, vec_B1, vec_C4);
                            vec_C4 = _mm512_fmadd_ps(vec_A14, vec_B2, vec_C4);
                            vec_C4 = _mm512_fmadd_ps(vec_A15, vec_B3, vec_C4);
                            vec_C4 = _mm512_fmadd_ps(vec_A16, vec_B4, vec_C4);
                            _mm512_store_ps (( __m512*)&C[(m+3)*N + n] , vec_C4 ) ;


                            vec_C5 = _mm512_fmadd_ps(vec_A1, vec_B5, vec_C5);
                            vec_C5 = _mm512_fmadd_ps(vec_A2, vec_B6, vec_C5);
                            vec_C5 = _mm512_fmadd_ps(vec_A3, vec_B7, vec_C5);
                            vec_C5 = _mm512_fmadd_ps(vec_A4, vec_B8, vec_C5);
                            _mm512_store_ps (( __m512*)&C[(m)*N + (n+16)] , vec_C5 ) ;

                            // vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            vec_C6 = _mm512_fmadd_ps(vec_A5, vec_B5, vec_C6);
                            vec_C6 = _mm512_fmadd_ps(vec_A6, vec_B6, vec_C6);
                            vec_C6 = _mm512_fmadd_ps(vec_A7, vec_B7, vec_C6);
                            vec_C6 = _mm512_fmadd_ps(vec_A8, vec_B8, vec_C6);
                            _mm512_store_ps (( __m512*)&C[(m+1)*N + (n+16)] , vec_C6 ) ;

                            vec_C7 = _mm512_fmadd_ps(vec_A9, vec_B5, vec_C7);
                            vec_C7 = _mm512_fmadd_ps(vec_A10, vec_B6, vec_C7);
                            vec_C7 = _mm512_fmadd_ps(vec_A11, vec_B7, vec_C7);
                            vec_C7 = _mm512_fmadd_ps(vec_A12, vec_B8, vec_C7);
                            _mm512_store_ps (( __m512*)&C[(m+2)*N + (n+16)] , vec_C7 ) ;

                            vec_C8 = _mm512_fmadd_ps(vec_A13, vec_B5, vec_C8);
                            vec_C8 = _mm512_fmadd_ps(vec_A14, vec_B6, vec_C8);
                            vec_C8 = _mm512_fmadd_ps(vec_A15, vec_B7, vec_C8);
                            vec_C8 = _mm512_fmadd_ps(vec_A16, vec_B8, vec_C8);
                            _mm512_store_ps (( __m512*)&C[(m+3)*N + (n+16)] , vec_C8 ) ;

                         }

                    }
                }
            }
        }
    }
}


#if 0

void MM_SIMD_ROW4(float* A, float* B, float* C, int M, int N, int K)
{
   int Lmb = 256; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int Lkb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int Lnb = 1024; //1024; 

    int Smb = 128;
    int Skb = 128;
    int Snb = 128;

    // int regsA = 4;
    // int regsB = 3;

    // int mr = regsA * 16;
    // int nr = regsB;

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += Lmb){
        int M1 = m0 + mb;
        for (int k0=0; k0 < K; k0 += Lkb){
                int K1 = k0 + kb;
            for (int n0=0; n0 < N; n0 += Lnb){
                int N1 = n0 + nb ;          
                for(int m=m0; m < M1; m+=Smb){
                    for(int k=k0; k< K1; k+=Skb){
                        for(int n = n0; n < N1; n+=Snb){
                        
                         register __m512 vec_A1 = _mm512_set1_ps (A[ m * K + (k+0)] ) ;
                         register __m512 vec_A2 = _mm512_set1_ps (A[ m * K + (k+16)] ) ;
                         register __m512 vec_A3 = _mm512_set1_ps (A[ m * K + (k+32)] ) ; 
                         register __m512 vec_A4 = _mm512_set1_ps (A[ m * K + (k+48)] ) ;



                            register __m512 vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            register __m512 vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            register __m512 vec_C3 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]) ;
                            register __m512 vec_C4 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]) ;

                            register __m512 vec_C5 = _mm512_load_ps((__m512*)&C[m*N + n+1]) ;
                            register __m512 vec_C6 = _mm512_load_ps((__m512*)&C[(m+1)*N + n+1]) ;
                            register __m512 vec_C7 = _mm512_load_ps((__m512*)&C[(m+2)*N + n+1]) ;
                            register __m512 vec_C8 = _mm512_load_ps((__m512*)&C[(m+3)*N + n+1]) ;

                            register __m512 vec_C9 = _mm512_load_ps((__m512*)&C[m*N + n+2]) ;
                            register __m512 vec_C10 = _mm512_load_ps((__m512*)&C[(m+1)*N + n+2]) ;
                            register __m512 vec_C11 = _mm512_load_ps((__m512*)&C[(m+2)*N + n+2]) ;
                            register __m512 vec_C12 = _mm512_load_ps((__m512*)&C[(m+3)*N + n+2]) ;

                            register __m512 vec_C13 = _mm512_load_ps((__m512*)&C[m*N + n+3]) ;
                            register __m512 vec_C14 = _mm512_load_ps((__m512*)&C[(m+1)*N + n+3]) ;
                            register __m512 vec_C15 = _mm512_load_ps((__m512*)&C[(m+2)*N + n+3]) ;
                            register __m512 vec_C16 = _mm512_load_ps((__m512*)&C[(m+3)*N + n+3]) ;


                            register __m512 vec_B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
                            register __m512 vec_B2 = _mm512_load_ps((__m512*)&B[(k+16)*N + n]);
                            register __m512 vec_B3 = _mm512_load_ps((__m512*)&B[(k+32)*N + n]);
                            register __m512 vec_B4 = _mm512_load_ps((__m512*)&B[(k+48)*N + n]);

                            // vec_C1 = _mm512_load_ps((__m512*)&C[m*N + n]) ;
                            vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B1, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A2, vec_B2, vec_C1);
                            vec_C1 = _mm512_fmadd_ps(vec_A3, vec_B3, vec_C1);
                            // vec_C1 = _mm512_fmadd_ps(vec_A4, vec_B4, vec_C1);
                            // _mm512_store_ps (( __m512*)&C[(m)*N + n] , vec_C1 ) ;

                            // vec_C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]) ;
                            vec_C2 = _mm512_fmadd_ps(vec_A4, vec_B1, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A5, vec_B2, vec_C2);
                            vec_C2 = _mm512_fmadd_ps(vec_A6, vec_B3, vec_C2);
                            // vec_C2 = _mm512_fmadd_ps(vec_A8, vec_B4, vec_C2);
                            // _mm512_store_ps (( __m512*)&C[(m+1)*N + n] , vec_C2 ) ;

                            vec_C3 = _mm512_fmadd_ps(vec_A7, vec_B1, vec_C3);
                            vec_C3 = _mm512_fmadd_ps(vec_A8, vec_B2, vec_C3);
                            vec_C3 = _mm512_fmadd_ps(vec_A9, vec_B3, vec_C3);
                            // vec_C3 = _mm512_fmadd_ps(vec_A12, vec_B4, vec_C3);
                            // _mm512_store_ps (( __m512*)&C[(m+2)*N + n] , vec_C3 ) ;

                            vec_C4 = _mm512_fmadd_ps(vec_A10, vec_B1, vec_C4);
                            vec_C4 = _mm512_fmadd_ps(vec_A11, vec_B2, vec_C4);
                            vec_C4 = _mm512_fmadd_ps(vec_A12, vec_B3, vec_C4);
                            // vec_C4 = _mm512_fmadd_ps(vec_A16, vec_B4, vec_C4);
                            // _mm512_store_ps (( __m512*)&C[(m+3)*N + n] , vec_C4 ) ;

                        }
                    }
                }
            }
        }
    }
}

#endif

void MM_SIMD2(float* A, float* B, float* C, int M, int N, int K)
{
	int mb = 256; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int kb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int nb = 1024; //1024; 

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
    	int M1 = m0 + mb;
    	for (int k0=0; k0 < K; k0 += kb){
    			int K1 = k0 + kb;
	    	for (int n0=0; n0 < N; n0 +=nb){
	    		int N1 = n0 + nb ;    		
    			for(int m=m0; m < M1; m+=4){
    				for(int k=k0; k< K1; k+=4){
    					 __m512 A1 = _mm512_set1_ps (A[ m * K + k ] ) ;
					     __m512 A2 = _mm512_set1_ps (A[ m * K + (k+1)] ) ;
					     __m512 A3 = _mm512_set1_ps (A[ (m+1) * K + k ] ) ;
					     __m512 A4 = _mm512_set1_ps (A[ (m+1) * K + (k+1)] ) ;
					     __m512 A5 = _mm512_set1_ps (A[ (m+2) * K + k] ) ;
					     __m512 A6 = _mm512_set1_ps (A[ (m+2) * K + (k+1)] ) ;
					     __m512 A7 = _mm512_set1_ps (A[ (m+3) * K + k] ) ;
					     __m512 A8 = _mm512_set1_ps (A[ (m+3) * K + (k+1)] ) ;

					    // _mm_prefetch (&B [ (k+2)*N + n], _MM_HINT_T0 ) ;
					    // _mm_prefetch (&B [ (k+3)*N + n], _MM_HINT_T0 ) ;

			      //       _mm_prefetch (&C [ (m+4)*N + n] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+5)*N + n ] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+6)*N + n ] , _MM_HINT_T0 ) ;
			      //       _mm_prefetch (&C [ (m+7)*N + n ] , _MM_HINT_T0 ) ;

    					for(int n=n0; n< N1; n+=16){
    						
    						__m512 B1 = _mm512_load_ps((__m512*)&B[k*N + n]);
    						__m512 B2 = _mm512_load_ps((__m512*)&B[(k+1)*N + n]);
    						__m512 B3 = _mm512_load_ps((__m512*)&B[k*N + (n+1)]);
    						__m512 B4 = _mm512_load_ps((__m512*)&B[(k+1)*N + (n+1)]);

    						__m512 C1 = _mm512_load_ps((__m512*)&C[m*N + n]);
    						C1 = _mm512_add_ps(C1, _mm512_mul_ps(A1, B1));
    						C1 = _mm512_add_ps(C1, _mm512_mul_ps(A2, B2));
    						_mm512_store_ps (( __m512*)&C[(m)*N + n] , C1);

    						__m512 C2 = _mm512_load_ps((__m512*)&C[(m+1)*N + n]);
    						C2 = _mm512_add_ps(C2, _mm512_mul_ps(A3, B1));
    						C2 = _mm512_add_ps(C2, _mm512_mul_ps(A4, B2));
    						_mm512_store_ps (( __m512*)&C[(m+1)*N + n] , C2);

    						__m512 C3 = _mm512_load_ps((__m512*)&C[(m+2)*N + n]);
    						C3 = _mm512_add_ps(C3, _mm512_mul_ps(A5, B1));
    						C3 = _mm512_add_ps(C3, _mm512_mul_ps(A6, B2));
    						_mm512_store_ps (( __m512*)&C[(m+2)*N + n] , C3);

    						__m512 C4 = _mm512_load_ps((__m512*)&C[(m+3)*N + n]);
    						C4 = _mm512_add_ps(C4, _mm512_mul_ps(A7, B1));
    						C4 = _mm512_add_ps(C4, _mm512_mul_ps(A8, B2));
    						_mm512_store_ps (( __m512*)&C[(m+3)*N + n] , C4);

    						__m512 C5 = _mm512_load_ps((__m512*)&C[m*N + (n+1)]);
    						C5 = _mm512_add_ps(C5, _mm512_mul_ps(A1, B3));
    						C5 = _mm512_add_ps(C5, _mm512_mul_ps(A2, B4));
    						_mm512_store_ps (( __m512*)&C[(m)*N + (n+1)] , C5);

    						__m512 C6 = _mm512_load_ps((__m512*)&C[(m+1)*N + (n+1)]);
    						C6 = _mm512_add_ps(C6, _mm512_mul_ps(A3, B3));
    						C6 = _mm512_add_ps(C6, _mm512_mul_ps(A4, B4));
    						_mm512_store_ps (( __m512*)&C[(m+1)*N + (n+1)] , C6);

    						__m512 C7 = _mm512_load_ps((__m512*)&C[(m+2)*N + (n+1)]);
    						C7 = _mm512_add_ps(C7, _mm512_mul_ps(A5, B3));
    						C7 = _mm512_add_ps(C7, _mm512_mul_ps(A6, B4));
    						_mm512_store_ps (( __m512*)&C[(m+2)*N + (n+1)] , C7);

    						__m512 C8 = _mm512_load_ps((__m512*)&C[(m+3)*N + (n+1)]);
    						C8 = _mm512_add_ps(C8, _mm512_mul_ps(A7, B3));
    						C8 = _mm512_add_ps(C8, _mm512_mul_ps(A8, B4));
    						_mm512_store_ps (( __m512*)&C[(m+3)*N + (n+1)] , C8);
    					}
    				}
    			}
    		}
    	}
    }
}


void MM_SIMD3(float* A, float* B, float* C, int M, int N, int K)
{
    int mb = 256; //256;                 // A = 256 x 128 = 32768, B = 128 x 256 = 32768, C = 256 x 256 = 65536
    int kb = 128; //128;                 // A = 256 x 128, B = 128 x 1024, C = 128 x 1024 Time: 0.0516 sec, GFLOPS = 36.59 for 1024x1024 matrix
    int nb = 1024; //1024; 

    int m1, n1, k1;

    for (int m0=0; m0 < M; m0 += mb){
        int M1 = m0 + mb;
        for (int k0=0; k0 < K; k0 += kb){
            int K1 = k0 + kb;
            for (int n0=0; n0 < N; n0 +=nb){
                int N1 = n0 + nb ;          
                for(int m=m0; m < M1; m+=4){
                    for(int k=k0; k< K1; k+=16){

                        float fps [16 * 12];
                        float accum[12];

                        const unsigned Aoff1 = (m + 0) * K;
                        const unsigned Aoff2 = (m + 1) * K;
                        const unsigned Aoff3 = (m + 2) * K;
                        const unsigned Aoff4 = (m + 3) * K;

                        const unsigned Boff1 = (k + 0) * N;
                        const unsigned Boff2 = (k + 1) * N;
                        const unsigned Boff3 = (k + 2) * N;

                        __m512 a, b1, b2, b3;

                        __m512 C1 = _mm512_setzero_ps();
                        __m512 C2 = _mm512_setzero_ps();
                        __m512 C3 = _mm512_setzero_ps();
                        __m512 C4 = _mm512_setzero_ps();
                        __m512 C5 = _mm512_setzero_ps();
                        __m512 C6 = _mm512_setzero_ps();
                        __m512 C7 = _mm512_setzero_ps();
                        __m512 C8 = _mm512_setzero_ps();
                        __m512 C9 = _mm512_setzero_ps();
                        __m512 C10 = _mm512_setzero_ps();
                        __m512 C11 = _mm512_setzero_ps();
                        __m512 C12 = _mm512_setzero_ps();
                            
                        _mm_prefetch (&A[Aoff1], _MM_HINT_T0 );
                        _mm_prefetch (&A[Aoff2], _MM_HINT_T0 );
                        _mm_prefetch (&A[Aoff3], _MM_HINT_T0 );
                        _mm_prefetch (&A[Aoff4], _MM_HINT_T0 );

                        _mm_prefetch (&B[Boff1], _MM_HINT_T0 );
                        _mm_prefetch (&B[Boff2], _MM_HINT_T0 );
                        _mm_prefetch (&B[Boff3], _MM_HINT_T0 );

                        for (int n = n0; n < N1; n+=3)
                        {
                            _mm_prefetch (&A[Aoff1 + k + 16], _MM_HINT_T0 );

                            b1 = _mm512_load_ps((__m512*)&B[Boff1 + n]);
                            b2 = _mm512_load_ps((__m512*)&B[Boff2 + n]);
                            b3 = _mm512_load_ps((__m512*)&B[Boff3 + n]);

                            _mm_prefetch (&A[Aoff2 + k + 16], _MM_HINT_T0 );

                            a = _mm512_load_ps((__m512*)&A[Aoff1 + k]);
                            C1 = _mm512_fmadd_ps(a, b1, C1);
                            C2 = _mm512_fmadd_ps(a, b2, C2);
                            C3 = _mm512_fmadd_ps(a, b3, C3);

                            _mm_prefetch (&A[Aoff3 + k + 16], _MM_HINT_T0 );

                            a = _mm512_load_ps((__m512*)&A[Aoff2 + k]);
                            C4 = _mm512_fmadd_ps(a, b1, C4);
                            C5 = _mm512_fmadd_ps(a, b2, C5);
                            C6 = _mm512_fmadd_ps(a, b3, C6);                            

                            _mm_prefetch (&A[Aoff4 + k + 16], _MM_HINT_T0 );

                            a = _mm512_load_ps((__m512*)&A[Aoff3 + k]);
                            C7 = _mm512_fmadd_ps(a, b1, C7);
                            C8 = _mm512_fmadd_ps(a, b2, C8);
                            C9 = _mm512_fmadd_ps(a, b3, C9);

                            _mm_prefetch (&B[Boff1 + n + 16], _MM_HINT_T0 );
                            _mm_prefetch (&B[Boff2 + n + 16], _MM_HINT_T0 );
                            _mm_prefetch (&B[Boff3 + n + 16], _MM_HINT_T0 );

                             a = _mm512_load_ps((__m512*)&A[Aoff4 + k]);
                            C10 = _mm512_fmadd_ps(a, b1, C10);
                            C11 = _mm512_fmadd_ps(a, b2, C11);
                            C12 = _mm512_fmadd_ps(a, b3, C12);
                        

                         memset(accum, 0, sizeof(accum));

                         _mm512_store_ps(( __m512*)&fps[0], C1);
                         _mm512_store_ps(( __m512*)&fps[16], C2);
                         _mm512_store_ps(( __m512*)&fps[32], C3);
                         _mm512_store_ps(( __m512*)&fps[48], C4);
                         _mm512_store_ps(( __m512*)&fps[64], C5);
                         _mm512_store_ps(( __m512*)&fps[96], C6);
                         _mm512_store_ps(( __m512*)&fps[112], C7);
                         _mm512_store_ps(( __m512*)&fps[128], C8);
                         _mm512_store_ps(( __m512*)&fps[144], C9);
                         _mm512_store_ps(( __m512*)&fps[160], C10);
                         _mm512_store_ps(( __m512*)&fps[176], C11);
                         _mm512_store_ps(( __m512*)&fps[192], C12);

                         for (int i = 0; i < 12; ++i)
                         {
                             for (int j = 0; j < 16; ++j)
                             {
                                 accum[i] += fps[i*16 + j];
                             }
                         }

                         C[(m+0)*N + n + 0] = accum[0];
                         C[(m+0)*N + n + 1] = accum[1];
                         C[(m+0)*N + n + 2] = accum[2];

                         C[(m+1)*N + n + 0] = accum[3];
                         C[(m+1)*N + n + 1] = accum[4];
                         C[(m+1)*N + n + 2] = accum[5];

                         C[(m+2)*N + n + 0] = accum[6];
                         C[(m+2)*N + n + 1] = accum[7];
                         C[(m+2)*N + n + 2] = accum[8];

                         C[(m+3)*N + n + 0] = accum[9];
                         C[(m+3)*N + n + 1] = accum[10];
                         C[(m+3)*N + n + 2] = accum[11];

                        }
                    }
                }
            }
        }
    }
}

//************************************************************************************************************
//************************************************************************************************************

// *************************************************************************************************************************************************

void naive_mm(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc){

    #pragma omp parallel for //collapse(3)//schedule(static,2) collapse(2) //private(i,j,k) shared(A, B, C)
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {   
            for (int j = 0; j < N; j++)
            {
                C_(i,j) += A_(i, k) * B_(k, j);
            }
        }
    }
// }
}




void mm_mul(int k, const float* A, const float* B,  float* C, int lda, int ldb, int ldc, int regsA, int regsB)
{

    __m512 Csum[regsA][regsB] = {{0.0}};
       
    for (int p = 0; p < k; p++)
    {
        for (int bi = 0; bi < regsB; bi++)
        {
            __m512 bb = _mm512_load_ps((__m512*)&B_(p, bi*16));
            
            for (int ai = 0; ai < regsA; ai++)
            {
                __m512 aa = _mm512_set1_ps(A_(ai, p));
                Csum[ai][bi] = _mm512_fmadd_ps(aa, bb, Csum[ai][bi]);                
            }
        }
    }

    for (int ai = 0; ai < regsA; ai++)
        {
        for (int bi = 0; bi < regsB; bi++)
            {
            _mm512_store_ps (( __m512*)&C_(ai, bi*16), _mm512_add_ps(_mm512_load_ps(( __m512*)&C_(ai, bi*16)), Csum[ai][bi]));
        }
    }
}

void mm_inner(const float* A, const float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){

    int procf = 2693749694;

    int regsA = 3; //3;  //5;
    int regsB = 4;  //4;  //2;

    int mr = regsA ; //* 16;
    int nr = regsB * 16;
    int i, j;

    // #pragma omp parallel for private(i,j) shared(A, B, C) collapse(2)
    for ( i = 0; i < M - mr +1; i+=mr){
        for (j = 0; j < N - nr +1; j+=nr){
            mm_mul(K, &A_(i,0), &B_(0,j), &C_(i,j), lda, ldb, ldc, regsA, regsB);
        }
    }

// #if 1
    int mi = (M / mr) * mr;
    int nj = (N / nr) * nr;

    if (mi < M) {
        naive_mm(&A_(mi, 0), &B_(0,0), &C_(mi,0), M-mi, nj, K, lda, ldb, ldc);
       
    }
    if (nj < N){
        // std::cout << " nj ---------=> " << nj << std::endl;
        naive_mm(&A_(0, 0), &B_(0,nj), &C_(0, nj), mi, N-nj, K, lda, ldb, ldc);
    }
    if (mi < M && nj < N){
        // std::cout << " mi & nj---------=> " << mi << " " << nj << std::endl;
        naive_mm(&A_(mi, 0), &B_(0,nj), &C_(mi,nj), M-mi, N-nj, K, lda, ldb, ldc);
    }
// #endif
}


void MM_OPTIM(const float* A, const float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){
    int mb; //= M/32;//1024; //2048;//4096;//8192;                 
    int kb; //= K/64; //128;//32;//64;//128;//32;
     //64;                 
    int nb;// = N/16;//1024; //2048;//8192;//8192;

    // Tiergarten: works well for  regsA = 3, regsB=4,  mb = M/16, kb = K/8 & nb = N/8 time: 0.001307 sec (MKL = 0.001325 sec) for 28 cores
    if (M == 1024){
        mb = M/16;
        kb = K/8;
        nb = N/8;
    }
    // Tiergarten: works well for  regsA = 3, regsB=4,  mb = M/16, kb = K/32 & nb = N/8 time: 0.00907 sec (MKL = 0.00653 sec) for 28 cores
    else if(M == 2048){
        mb = M/16;
        kb = K/32;
        nb = N/8;
    }

    // Tiergarten: works well for  regsA = 3, regsB=4,  mb = M/16, kb = K/64 & nb = N/8 time: 0.0801 sec  (MKL = 0.0442 sec) for 28 cores
    else if (M == 4096 ){ 
        mb = M/16;
        kb = K/64;
        nb = N/8;
    }
    
    // Tiergarten: Best Value for  regsA = 5, regsB=2,  mb = M/32, kb = K/64 & nb = N/8 time: 1.035 sec  (MKL = 0.351 sec) for 28 cores
    // Tiergarten: Best Value for  regsA = 3, regsB=4,  mb = M/64, kb = K/256 & nb = N/8 time: 0.83 sec  (MKL = 0.351 sec) for 28 cores
    else if(M == 8192){
        mb = M/64; //128 //256
        kb = K/256; //32
        nb = N/8;   //1024
    }
   
   int m0, n0, k0, mb1, nb1, kb1;

    #pragma omp parallel for schedule(static) private(m0,n0,k0, mb1, nb1, kb1) shared(A, B, C) collapse(3)
   // #pragma omp parallel for num_threads(omp_get_max_threads()) private(m0,n0,k0, mb1, nb1, kb1) shared(A, B, C) collapse(3)
    for (m0=0; m0 < M; m0 += mb){    
        for (n0=0; n0 < N; n0 +=nb){
            for (k0=0; k0 < K; k0+=kb){
                    mb1 = MIN(M-m0, mb);          
                    nb1 = MIN(N-n0, nb);
                    kb1 = MIN(K-k0, kb);                                           
                    mm_inner(&A_(m0,k0), &B_(k0, n0), &C_(m0, n0), mb1, nb1, kb1, lda, ldb, ldc);
                }
            }
        }
    // }
}


//************************************************************************************************************
//************************************************************************************************************

void mm_dot(int k1, float* A, float* B, float* C, int M, int N, int K, int regsA, int regsB){

    // __m512 Csum = _mm512_setzero_ps();

    // float Csum[regsA][regsB] = {{0.0}};

    __m512 Csum[regsA][regsB];
    for (int ia = 0; ia < regsA; ia++)
    {
        for (int ib = 0; ib < regsB; ib++)
        {
            Csum[ia][ib] = _mm512_setzero_ps();
        }
    }
    // __m512 Csum[regsA][regsB] = _mm512_setzero_ps();
    // __m512 C00 = _mm512_setzero_ps();
    // __m512 C01 = _mm512_setzero_ps();
    // __m512 C02 = _mm512_setzero_ps();
    // __m512 C03 = _mm512_setzero_ps();

    // __m512 C10 = _mm512_setzero_ps();
    // __m512 C11 = _mm512_setzero_ps();
    // __m512 C12 = _mm512_setzero_ps();
    // __m512 C13 = _mm512_setzero_ps();

    // __m512 C20 = _mm512_setzero_ps();
    // __m512 C21 = _mm512_setzero_ps();
    // __m512 C22 = _mm512_setzero_ps();
    // __m512 C23 = _mm512_setzero_ps();

    // __m512 C30 = _mm512_setzero_ps();
    // __m512 C31 = _mm512_setzero_ps();
    // __m512 C32 = _mm512_setzero_ps();
    // __m512 C33 = _mm512_setzero_ps();    

    for (int k = 0; k < k1; k++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
            __m512 aa = _mm512_load_ps((__m512*)&A[(ai * 16) * K + k]);
            for (int bi = 0; bi < regsB; bi++)
            {
                __m512 bb = _mm512_set1_ps(B[k * N + bi]);
                Csum[ai][bi] = _mm512_fmadd_ps(aa, bb, Csum[ai][bi]);
                // Csum[ai][bi] =  aa * bb; 
            }
        }
    }

    for (int bi = 0; bi < regsB; bi++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
            _mm512_store_ps (( __m512*)&C[(ai * 16)*N + bi], _mm512_add_ps(_mm512_load_ps(( __m512*)&C[(ai * 16)*N + bi]), Csum[ai][bi]));
        }
    }
}

// void matmul_inner_L1(float* A, float* B, float* C, int M, int N, int K, int m1, int k1, int n1, int mr, int nr, int regsA, int regsB){
//     for (int i = 0; i < m1 - mr + 1; i+=mr)
//     {
//         for (int j = 0; j < n1 - nr + 1; j+=nr)
//         {
//             mm_dot(k1, A, B, C, M, N, K, regsA, regsB);
//         }
//     }
// }

void matmul_inner(float* A, float* B, float* C, int M, int N, int K, int m1, int k1, int n1){

    int regsA = 4;
    int regsB = 4;

    int mr = regsA * 16;
    int nr = regsB;

    // matmul_inner_L1(A, B, C, M, N, K, m1, k1, n1, mr, nr, regsA, regsB);
    for (int i = 0; i < m1 - mr; i+=mr)
    {
        for (int j = 0; j < n1 - nr; j+=nr)
        {
            mm_dot(k1, A, B, C, M, N, K, regsA, regsB);
        }
    }



}          

void matmul_outer(float* A, float* B, float* C, int M, int N, int K){
    int mb = 256; //256;                 
    int kb = 128; //128;                 
    int nb = 1024; //1024; 

    // int m1, n1, k1;


    for (int m0=0; m0 < M; m0 += mb){
        // int M1 = m0 + mb;
        int m1 = MIN(M-m0, mb);
        for (int k0=0; k0 < K; k0 += kb){
            // int K1 = k0 + kb;
            int k1 = MIN(K-k0, kb);
            for (int n0=0; n0 < N; n0 +=nb){
                // int N1 = n0 + nb ;
                int n1 = MIN(N-n0, nb);
                matmul_inner(A, B, C, M, N, K, m1, k1, n1);          
            }
        }
    }
}

void mm_mul(int k0, int k1, float* A, float* B, float* C, int M, int N, int K, int regsA, int regsB){

     __m512 Csum[regsA][regsB];
    for (int ia = 0; ia < regsA; ia++)
    {
        for (int ib = 0; ib < regsB; ib++)
        {
            Csum[ia][ib] = _mm512_setzero_ps();
            // printf("%f\n", Csum[ia][ib]);
        }
    }
    // exit(0);

    for (int k = k0; k < k1; k++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
            __m512 aa = _mm512_load_ps((__m512*)&A[(ai * 1) * K + k]);
            // printf("%f\t", aa);
            // std::cout << "aa =============-------------------->"<<std::endl;
            // print_m512Var(aa);
            for (int bi = 0; bi < regsB; bi++)
            {
                __m512 bb = _mm512_set1_ps(B[k * N + bi]);
                // std::cout << "bb =====-------------------->"<<std::endl;
                // print_m512Var(bb);
                Csum[ai][bi] = _mm512_fmadd_ps(aa, bb, Csum[ai][bi]);
                // Csum[ai][bi] +=  aa * bb; 
                // std::cout << "CSUM -------------------->"<<std::endl;
                // print_m512Var(Csum[ai][bi]);
                
            }
        }
        // exit(0);
    }
    // exit(0);
    for (int bi = 0; bi < regsB; bi++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
            _mm512_store_ps (( __m512*)&C[(ai * 1)*N + bi], _mm512_add_ps(_mm512_load_ps(( __m512*)&C[(ai * 1)*N + bi]), Csum[bi][ai]));
        }
    }

}


void MM_OPT2(float* A, float* B, float* C, int M, int N, int K){
    int mb = 128;                 
    int kb = 128;                 
    int nb = 128;


    int regsA = 2;
    int regsB = 2;

    int mr = regsA * 16;
    int nr = regsB;

    for (int m0=0; m0 < M; m0 += mb){
        int m1 = m0 + mb;
        // int m1 = MIN(M-m0, mb);
        for (int k0=0; k0 < K; k0 += kb){
            int k1 = k0 + kb;
            // int k1 = MIN(K-k0, kb);
            for (int n0=0; n0 < N; n0 +=nb){
                int n1 = n0 + nb ;
                // int n1 = MIN(N-n0, nb);
                for (int i = m0; i < m1 - mr; i+=mr){
                    for (int j = n0; j < n1 - nr; j+=nr){
                        mm_mul(k0, k1, A, B, C, M, N, K, regsA, regsB);
                    }
                }
            }
        }
    }

}


void mm_mul(int k, const float* A, const float* B, const float* C, int lda, int ldb, int ldc, int regsA, int regsB)
{

    __m512 Csum[regsA][regsB] = {{0.0}};
       
    for (int p = 0; p < k; p++)
    {
        for (int bi = 0; bi < regsB; bi++)
        {
            __m512 bb = _mm512_load_ps((__m512*)&B_(p, bi*16));
            
            for (int ai = 0; ai < regsA; ai++)
            {
                __m512 aa = _mm512_set1_ps(A_(ai, p));
                Csum[ai][bi] = _mm512_fmadd_ps(aa, bb, Csum[ai][bi]);                
            }
        }
    }

    for (int ai = 0; ai < regsA; ai++)
        {
        for (int bi = 0; bi < regsB; bi++)
            {
            _mm512_store_ps (( __m512*)&C_(ai, bi*16), _mm512_add_ps(_mm512_load_ps(( __m512*)&C_(ai, bi*16)), Csum[ai][bi]));
        }
    }
}

void mm_inner(const float* A, const float* B, const float* C, int M, int N, int K, int lda, int ldb, int ldc){

    int regsA = 4;
    int regsB = 4;

    int mr = regsA ; //* 16;
    int nr = regsB * 16;

    for (int i = 0; i < M - mr +1; i+=mr){
        for (int j = 0; j < N - nr +1; j+=nr){
            mm_mul(K, &A_(i,0), &B_(0,j), &C_(i,j), lda, ldb, ldc, regsA, regsB);
        }
    }
}

void MM_OPT(const float* A, const float* B, const float* C, int M, int N, int K, int lda, int ldb, int ldc){
    int mb = 512;                 
    int kb = 128;                 
    int nb = 1024;
    
    for (int m0=0; m0 < M; m0 += mb){
        int mb1 = MIN(M-m0, mb);   
        for (int k0=0; k0 < K; k0 += kb){
            int kb1 = MIN(K-k0, kb);
            for (int n0=0; n0 < N; n0 +=nb){
                int nb1 = MIN(N-n0, nb);         
                mm_inner(&A_(m0,k0), &B_(k0, n0), &C_(m0, n0), mb1, nb1, kb1, lda, ldb, ldc);
            }
        }
    }
}

//----------------------------------------------------------------- Performance 73% of MKL-------------------------------------------------------------------------------------------------

#if 1
void naive_mm(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc){

    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {   
            for (int j = 0; j < N; j++)
            {
                C_(i,j) += A_(i, k) * B_(k, j);
            }
        }
    }
}


#endif 

void mm_mul(int k, const float* A, const float* B,  float* C, int lda, int ldb, int ldc, int regsA, int regsB)
{

    __m512 Csum[regsA][regsB] = {{0.0}};
       
    for (int p = 0; p < k; p++)
    {
        for (int bi = 0; bi < regsB; bi++)
        {
            __m512 bb = _mm512_load_ps((__m512*)&B_(p, bi*16));
            
            for (int ai = 0; ai < regsA; ai++)
            {
                __m512 aa = _mm512_set1_ps(A_(ai, p));
                Csum[ai][bi] = _mm512_fmadd_ps(aa, bb, Csum[ai][bi]);                
            }
        }
    }

    for (int ai = 0; ai < regsA; ai++)
        {
        for (int bi = 0; bi < regsB; bi++)
            {
            _mm512_store_ps (( __m512*)&C_(ai, bi*16), _mm512_add_ps(_mm512_load_ps(( __m512*)&C_(ai, bi*16)), Csum[ai][bi]));
        }
    }
}

void mm_inner(const float* A, const float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){

    int procf = 2693749694;

    int regsA = 5;  //5;
    int regsB = 2;  //2;

    int mr = regsA ; //* 16;
    int nr = regsB * 16;

    for (int i = 0; i < M - mr +1; i+=mr){
        for (int j = 0; j < N - nr +1; j+=nr){
            mm_mul(K, &A_(i,0), &B_(0,j), &C_(i,j), lda, ldb, ldc, regsA, regsB);
        }
    }

#if 1
    int mi = (M / mr) * mr;
    int nj = (N / nr) * nr;

    if (mi < M) {
        // std::cout << " mi ---------=> " << mi << std::endl;
        // uint64_t startTick1 = __rdtsc();
        naive_mm(&A_(mi, 0), &B_(0,0), &C_(mi,0), M-mi, nj, K, lda, ldb, ldc);
        // uint64_t endTick1 = __rdtsc();
        
        // uint64_t tot_rc_ticks =  endTick1 - startTick1;
        // double total_time = tot_rc_ticks*1.0/procf;
        // float time_taken_rc = ((tot_rc_ticks*1.0)/procf)/NUM_ITERS;
        // printf("\t\tCPU: Time taken for Naive MM : %lf\n", total_time);

    }
    if (nj < N){
        // std::cout << " nj ---------=> " << nj << std::endl;
        naive_mm(&A_(0, 0), &B_(0,nj), &C_(0, nj), mi, N-nj, K, lda, ldb, ldc);
    }
    if (mi < M && nj < N){
        // std::cout << " mi & nj---------=> " << mi << " " << nj << std::endl;
        naive_mm(&A_(mi, 0), &B_(0,nj), &C_(mi,nj), M-mi, N-nj, K, lda, ldb, ldc);
    }
#endif
}


void MM_OPTIM(const float* A, const float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){
    int mb = 2048;                 
    int kb = 32;                 
    int nb = 4096;
   
   // #pragma omp parallel //private(m0,n0,k0) shared(C) 
   //  #pragma omp for schedule(static)
    
    for (int m0=0; m0 < M; m0 += mb){
        int mb1 = MIN(M-m0, mb);          
        for (int n0=0; n0 < N; n0 +=nb){
            int nb1 = MIN(N-n0, nb);
            for (int k0=0; k0 < K; k0+=kb){
                int kb1 = MIN(K-k0, kb);                                           
                    mm_inner(&A_(m0,k0), &B_(k0, n0), &C_(m0, n0), mb1, nb1, kb1, lda, ldb, ldc);
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------


int main (int argc, char const *argv[])
{
    
    int M, K, N;
    int NUM_ITERS = 10;
    
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = atoi(argv[3]);

    #ifdef VTUNE_ANALYSIS
        __itt_pause();
    #endif

    #if USE_PERF_COUNTERS
	  ctrs_skx_uc a, b, s;
	  bw_gibs bw_min, bw_max, bw_avg;

	  //setup_skx_uc_ctrs( CTRS_EXP_DRAM_ACT );
	  setup_skx_uc_ctrs( CTRS_EXP_DRAM_CAS );
	  //setup_skx_uc_ctrs( CTRS_EXP_CHA_BL_VERT );
	  //setup_skx_uc_ctrs( CTRS_EXP_CHA_BL_HORZ );
	  zero_skx_uc_ctrs( &a );
	  zero_skx_uc_ctrs( &b );
	  zero_skx_uc_ctrs( &s );
	#endif

	int64_t procf = 1;
    int64_t startTick = __rdtsc();
    sleep(1);
    int64_t endTick = __rdtsc();
    procf = endTick - startTick;
    printf("procf = %ld\n", procf);


    struct timeval st, ed;

    dense_matrix A_mat, B_mat;

    int max_threads = mkl_get_max_threads();
    printf("Available max MKL threads: %d\n", max_threads);

    printf("\tGenerating dense matrix A of size %d x %d\n", M, K);
    float * values_MatA=NULL;
    A_mat = generate_matrix(M, K, &values_MatA);
    print_matrix(values_MatA, M, K, 1);

    printf("\tGenerating dense matrix B of size %d x %d\n", K, N);
    float * values_MatB=NULL;
    B_mat = generate_matrix(K, N, &values_MatB);
    // print_matrix(values_MatB, K, N, 1);

// printf("\tGenerating dense matrix Amat of size %d x %d\n", M, K);
//     float * valuesMatA=NULL;
//     dense_matrix Amat = generate_matrix(M, K, &valuesMatA);
//     print_matrix(valuesMatA, M, K, 1);
//     print_2Dmatrix(Amat.values2D, M, K, 1);

   
    

    // *****************************************  Optimized Matrix Multiplication ************************************
    int L2Size = 1 * 1024 * 1024;
    // dense_matrix C_1;
   float* C_1 = (float*) mkl_malloc (M * N * sizeof(float), 64);
   for (int i1 = 0; i1 < M*N; ++i1)
    {
        C_1[i1] = 0.0;
    }

   float ** C1 = (float **)malloc(M * sizeof(float *));
   for (int i2=0; i2<M; i2++)
         C1[i2] = (float *)malloc(N * sizeof(float));

   for (int v1=0; v1< M; v1++){
   	for(int v2=0; v2 < N; v2++){
   		C1[v1][v2] = 0.0;
   	}
   }
    int TileSize = 64;
    // int TileHeight = 64 ;
    // int TileSize = L2Size / 2 / TileHeight / sizeof(float);
    uint64_t startTick1, endTick1;

    #ifdef VTUNE_ANALYSIS
        __itt_resume();
    #endif

    #if USE_PERF_COUNTERS
        read_skx_uc_ctrs( &a );
	#endif


   startTick1 = __rdtsc();
   for (int i=0; i<NUM_ITERS; i++)
   {
   		// MM_SIMD2D(A_mat.values2D, B_mat.values2D, C1, M, N, K);
   		// MM_SIMD3(values_MatA, values_MatB, C_1, M, N, K);
        // MM_SIMD_ROW2(values_MatA, values_MatB, C_1, M, N, K);
        // MM_L2CB(values_MatA, values_MatB, C_1, M, N, K);
        // MM_L2CB_KIJ(values_MatA, values_MatB, C_1, M, N, K); 

        // matmul_outer(values_MatA, values_MatB, C_1, M, N, K);
        // MM_OPT(values_MatA, values_MatB, C_1, M, N, K); 
        MM_SIMD_ROW3(values_MatA, values_MatB, C_1, M, N, K);  
    }

    endTick1 = __rdtsc();

    #if USE_PERF_COUNTERS
        read_skx_uc_ctrs( &b );
        difa_skx_uc_ctrs( &a, &b, &s );
        divi_skx_uc_ctrs( &s, 10);
	#endif


    #ifdef VTUNE_ANALYSIS
            __itt_pause();
    #endif

    uint64_t tot_rc_ticks =  endTick1 - startTick1;
    double total_time = tot_rc_ticks*1.0/procf;

    #if USE_PERF_COUNTERS
        get_cas_ddr_bw_skx( &s, total_time, &bw_avg );
        //get_llc_bw_skx( &s, total_time, &bw_avg );
        printf("AVG RD GiB/s: %f\n", bw_avg.rd);
        printf("AVG WR GiB/s: %f\n", bw_avg.wr);
	#endif

    printf("\tGenerating dense matrix C of size %d x %d\n", M, N);
    // print_matrix(C_mat.values, C_mat.rows, C_mat.cols, C_mat.is_rowmajor);
    // print_2Dmatrix(C1, M, K, 1);
    print_matrix(C_1, M, N, 1);

    float time_taken_rc = ((tot_rc_ticks*1.0)/procf)/NUM_ITERS;

    printf("\t\tCPU: Ticks taken for Optimized MatMul 1 Thread : %ld\n", tot_rc_ticks);
    printf("\t\tCPU: Time taken for Optimized MatMul 1 Thread : %lf\n", time_taken_rc);

    float Rc_freq =  2.0 * M * N * K / (time_taken_rc);
    printf(" GFLOPS for RC  %0.2lf \n", Rc_freq / 1e9 );

    float Effy_opt = Rc_freq / 153 / 1e9 ; //(4.3 * 1e12);
    printf(" Efficiency wrt Peak %0.2lf \n", Effy_opt );


#if MKL

    // // ****************************************Matrix Multiplication using MKL Library***************************************

    // Time for 1024 x 1024 Matrix: 0.015sec

    printf (" \n\n Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    
    float alpha, beta;
    alpha = 1.0; beta = 0.0;

    float* C = (float*) mkl_malloc (M * N * sizeof(float), 64);
    #pragma omp parallel for
    for (int i = 0; i < M*N; ++i)
    {
        C[i] = 0.0;
    }
     // gettimeofday(&st, NULL);
    uint64_t startTick_mkl, endTick_mkl;
    startTick_mkl = __rdtsc();
   for (int i=0; i<NUM_ITERS; i++)
   {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, values_MatA, K, values_MatB, N, beta, C, N);
    }
    // gettimeofday(&ed, NULL);
    endTick_mkl = __rdtsc();
    uint64_t tot_mkl_ticks =  endTick_mkl - startTick_mkl;


    printf("\tGenerating dense matrix C of size %d x %d\n", M, N);
    print_matrix(C, M, N, 1);

   // int time_taken_mkl = (ed.tv_sec*1e6 + ed.tv_usec) - (st.tv_sec* 1e6 + st.tv_usec);
   float time_taken_mkl = ((tot_mkl_ticks*1.0)/procf)/NUM_ITERS;
   
   printf("\t\tCPU: Time taken for MatMul MKL 1 Thread: %lf\n", time_taken_mkl );

    float Gflops_mkl =  2.0 * M * N * K / time_taken_mkl;
   printf(" GFLOPS for MKL  %0.2lf \n", Gflops_mkl / 1e9);

   float Effy_mkl = Gflops_mkl / 153 / 1e9; //(4.3 * 1e12);
   printf(" Efficiency wrt Peak %0.2lf \n", Effy_mkl);

    // printf(" Number of MKL MM Operations per Sec for 1 Thread\n", 153 / Effy_mkl);

//---------------------------------------------------------------------------------------------------------------------------
#endif

	for(int j = 0; j < M * N; j++)
        {
            float perc_diff = abs(C[j] - (C_1[j])/NUM_ITERS) / C[j];
            if(perc_diff > 0.0001)
            {
                printf("ERROR! j = %d, incorrect output: C_mkl[j] = %E, C_opt[j] = %E\n", j, C[j], C_1[j]);
                exit(0);
            }
        }


}