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
#include <string>
#include <immintrin.h>

using namespace std;

#define MKL 1
#define BASELINE 0
#define WRITE 1

// #define M 1000
// #define K 1000
// #define N 1000

typedef struct  dense_matrix
{
        int rows;
        int cols;
        float* values;
        bool is_rowmajor;
}dense_matrix;


// *************************************** Genearate Dense Matrix for Multiplication *************************************************

dense_matrix generate_matrix(int rows, int cols, float ** values_){

    // float* values = new float[rows * cols]();
    float* values = (float*) mkl_malloc (rows * cols * sizeof(float), 64);

    for(int i=0; i< rows*cols; i++){
        values[i] = ( static_cast<float>(rand()) + 1.0 ) / static_cast<float>(RAND_MAX);
    }

    // Packaging dense matrix into structure.
    dense_matrix dense_mat;

    dense_mat.rows = rows;
    dense_mat.cols = cols;
    dense_mat.values = values;
    
    *values_ = values;
    return dense_mat;
}

#if WRITE

void write_matrix(int rows, int cols, float ** values_){
	// std::ofstream mat(FILE_NAME);
	float* values = (float*) mkl_malloc (rows * cols * sizeof(float), 64);
    FILE *fp = fopen("matrixB.txt", "a");

    for(int i=0; i< rows*cols; i++){
        values[i] = ( static_cast<float>(rand()) + 1.0 ) / static_cast<float>(RAND_MAX);
        fwrite(values, sizeof(float), 1, fp);
    }

    

    fclose(fp);
    	// for(int j=0; j< cols; j++){
    		// values[i][j] = ( static_cast<float>(rand()) + 1.0 ) / static_cast<float>(RAND_MAX);
    		// matA << values[i][j] << " "; 
    
    		// mat << ( static_cast<float>(rand()) + 1.0 ) / static_cast<float>(RAND_MAX) << " ";
        // 
    	// matA << ";" << endl;
}


#else
    dense_matrix read_matrix(int rows, int cols, float ** values_, string FILE_NAME){
        std::ifstream file(FILE_NAME);

        float* values = (float*) mkl_malloc (rows * cols * sizeof(float), 64);
        if (!file)
        {  
          cout<<"Cannot open file\n";
          return; 
        }  
        for(int i=0; i< rows; i++){
            for(int j=0; j< cols; j++){
                // values[i] = mat[i][j];
                file >> mat[i][j];
            }
        }
        file.close();

        dense_matrix dense_mat;
        dense_mat.rows = rows;
        dense_mat.cols = cols;
        dense_mat.values = values;
        
        *values_ = values;
    return dense_mat;
    }

#endif

// ********************************************** print functionalities *********************************************************
void print_matrix(float* data, int rows, int cols, bool is_rowmajor){
    for (int i = 0; i < min(rows, 15); ++i)
    {
        for (int j = 0; j < min(cols, 15); ++j)
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


// **************************************** Serial Matrix multiplication (base line) ***********************************************
/*
	C[M, N]  = A[M, K] * B[K, N]

*/

// float * matMulBase1(float* A, float* B, int M, int N, int K)
// {
    
//     float* values = (float*) malloc (M * N * sizeof(float));

//     for (int v1=0; v1 < M; v1++)
//     {
//         for (int v2=0; v2< N; v2++)
//         {
//             for (int v3=0; v3 < K; v3++)
//             {
//                 float val = A[v1*K + v3] * B[v3*N + v2];
//                 values[v1*N + v2] += val; 
//             }
//         }
//     }
//     return values;
// }


void matMulBase (float* A, float* B, float* values, int M, int N, int K)         // Time for 1024 x 1024 Matrix: 3.46 sec
{
    
    // float* values = (float*) malloc (M * N * sizeof(float));

    for (int row=0; row < M; row++)
    {
        for (int col=0; col< N; col++)
        {
            for (int dim=0; dim < K; dim++)
            {
                values[row*N + col] += A[row*K + dim] * B[dim*N + col];
                // values[v1*N + v2] += val; 
            }
        }
    }
    // return values;
}


void matMulColTrans(float* A, float* B, float* values, int M, int N, int K)            // Time for 1024 x 1024 Matrix: 0.181 sec but ans is not matching with Baseline
{
     for (int row=0; row < M; row++)
    {
        for (int col=0; col< N; col++)
        {
            for (int dim=0; dim < K; dim++)
            {
                values[row*N + col] += A[row*K + dim] * B[dim + K*col];
                // values[v1*N + v2] += val; 
            }
        }
    }
}

// **************************************** Serial Matrix multiplication (Loop Interchange (i,j,k) <-> (j,i,k)) ***********************************************

void matMulLoopIntChng0(float* A, float* B, float* values, int M, int N, int K)
{
    

    for (int col=0; col< N; col++)
    {
        for (int row=0; row < M; row++)
        {
            for (int dim=0; dim < K; dim++)
            {
                values[row*N + col] += A[row*K + dim] * B[dim*N + col]; 
            }
        }
    }
}

// **************************************** Serial Matrix multiplication (Loop Interchange (i,j,k) <-> (i,k,j)) ***********************************************

void matMulLoopIntChng1(float* A, float* B, float* values, int M, int N, int K)          // Time for 1024 x 1024 Matrix: 0.202sec          
{
    
    for (int row=0; row < M; row++)          
    {
        for (int dim=0; dim < K; dim++)
        {
            // #pragma unroll(10)
            for (int col=0; col< N; col++)       
            {
                values[row*N + col] += A[row*K + dim] * B[dim*N + col]; 
            }
        }
    }
}


// ******************************************** Cache blocking Multiplication ***********************************************************************************


void matMulLoopIntChngCacheBlock(float* A, float* B, float* values, int M, int N, int K, int BlockSize)      // Time for 1024 x 1024 Matrix: 0.190sec
{
    
    for (int row=0; row < M; row+= BlockSize)          
    {
        for (int dim=0; dim < K; dim++)
        {
            for (int blockR = row; blockR < row + BlockSize; blockR++)
            {
                // #pragma unroll(10)
                for (int col=0; col< N; col+= BlockSize)       
                {
                    for (int blockC = col; blockC < col + BlockSize; blockC++)
                    {
                        values[blockR*N + blockC] += A[blockR*K + dim] * B[dim*N + blockC]; 
                        // values[blockR*N + blockC] += A[blockR*K + dim] * B[dim + K*blockC];
                    } 
                }

            }
        }
    }
}



void matMulCacheBlock(float* A, float* B, float* values, int M, int N, int K, int BlockSize)              // Time for 1024 x 1024 Matrix: 3.41 sec
{
    
    for (int row=0; row < M; row+= BlockSize)          
    {
        for (int col=0; col< N; col+= BlockSize) 
        {
            for (int blockR = row; blockR < row + BlockSize; blockR++)
            {    
                // #pragma unroll(10)
                 for (int blockC = col; blockC < col + BlockSize; blockC++)     
                {
                    for (int dim=0; dim < K; dim++)
                    {
                        values[blockR*N + blockC] += A[blockR*K + dim] * B[dim*N + blockC]; 
                    } 
                }

            }
        }
    }
}




void matMul_LoopIntChng2_blocking(float* A, float* B, float* values, int M, int N, int K, int TILE_SIZE)        // Time for 1024 x 1024 Matrix: 0.21 sec 
{  
    // float* values = (float*) malloc (M * N * sizeof(float));

    for(int a=0; a < M; a += TILE_SIZE)
    {
        for (int b = 0; b < N; b += TILE_SIZE)
        {
            for (int c = 0; c < K; c += TILE_SIZE)
            {
                for (int v1=a; v1 < a + TILE_SIZE; v1++)
                {
                    for (int v3=c; v3 < c + TILE_SIZE; v3++)
                    {   
                        // #pragma ivdep
                        // #pragma vector always
            // --------------------------- or ------------------------
                        // #pragma unroll(10)
                        for (int v2=b; v2< b + TILE_SIZE; v2++)
                        {
                            values[v1*N + v2] += A[v1*K + v3] * B[v3*N + v2];
                            
                        }
                    }
                }
            }
        }
    }
    // return values;
}

// ********************** Serial Matrix multiplication (Vector Intrinsics) Compiler derivatives for vectorization ***********************************************

void matMulVectorIntrinsics_ST(float* A, float* B, float* values, int M, int N, int K)                        // Time for 1024 x 1024 Matrix: 0.178 sec
{
    
  
    __m512 vec_C = _mm512_setzero_ps();
    __m512 vec_A = _mm512_setzero_ps();
    __m512 vec_B = _mm512_setzero_ps();

    
    #pragma omp parallel for 
    for (int v1=0; v1 < M; v1++)
    {
        for (int v3=0; v3 < K; ++v3)
        {   
            vec_A = _mm512_set1_ps(A[v1*K+v3]);
            for (int v2=0; v2< N; v2 += 16)
            {
                vec_B = _mm512_load_ps((__m512*)&B[v3*N + v2]);
                vec_C = _mm512_load_ps((__m512*)&values[v1*N + v2]) ;
                vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);

                _mm512_store_ps((__m512*)&values[v1*N + v2], vec_C);
            }
        }
    }
}


// ********************** Opt Matrix multiplication with Cache Blocking and Loop interchanging (Vector Intrinsics) Compiler derivatives for vectorization ***********************************************

void MM_LIC_CB_VectorIntrinsics(float* A, float* B, float* values, int M, int N, int K, int BlockSize)      // Time for 1024 x 1024 Matrix with Blocksize 64 : 0.096 sec
{

    __m512 vec_C = _mm512_setzero_ps();
    __m512 vec_A = _mm512_setzero_ps();
    __m512 vec_B = _mm512_setzero_ps();


    for (int row=0; row < M; row+= BlockSize)          
    {
        for (int dim=0; dim < K; dim++)
        {
            for (int blockR = row; blockR < row + BlockSize; blockR++)
            {
                vec_A = _mm512_set1_ps(A[blockR*K + dim]);
                // #pragma unroll(10)
                for (int col=0; col< N; col+= BlockSize)       
                {
                    for (int blockC = col; blockC < col + BlockSize; blockC+=16)
                    {
                        vec_B = _mm512_load_ps((__m512*)&B[dim*N + blockC]);
                        // vec_B = _mm512_load_ps((__m512*)&B[dim + K*blockC]);        // Testing with column major order read
                        vec_C = _mm512_load_ps((__m512*)&values[blockR*N + blockC]) ;
                        vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);

                        _mm512_store_ps((__m512*)&values[blockR*N + blockC], vec_C);
                    } 
                }

            }
        }
    }
}


// ********************** Opt Matrix multiplication with variable BlockSize and Loop interchanging (Vector Intrinsics) Compiler derivatives for vectorization ***********************************************

void MM_LIC_CB_VectorIntrinsics2(float* A, float* B, float* values, int M, int N, int K)      // Time for 1024 x 1024 Matrix: tile1 = 8, tile2 = 16, tile3 = 64 is 0.097 sec
{

    const int tile1 = 8, tile2 = 16, tile3 = 32;

    __m512 vec_C = _mm512_setzero_ps();
    __m512 vec_A = _mm512_setzero_ps();
    __m512 vec_B = _mm512_setzero_ps();

    int BlockSize1 = tile1 * tile2;
    int BlockSize2 = tile1 * tile3;


    for (int row=0; row < M; row+= BlockSize1)          
    {
        for (int dim=0; dim < K; dim++)
        {
            for (int blockR = row; blockR < row + BlockSize1; blockR++)
            {
                vec_A = _mm512_set1_ps(A[blockR*K + dim]);
                // #pragma unroll(10)
                for (int col=0; col< N; col+= BlockSize2)       
                {
                    for (int blockC = col; blockC < col + BlockSize2; blockC+=16)
                    {
                        vec_B = _mm512_load_ps((__m512*)&B[dim*N + blockC]);
                        vec_C = _mm512_load_ps((__m512*)&values[blockR*N + blockC]) ;
                        vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);

                        _mm512_store_ps((__m512*)&values[blockR*N + blockC], vec_C);
                    } 
                }

            }
        }
    }
}



// ************************************************************************************************************************************

int main (int argc, char const *argv[])
{
	
    int M, K, N;
	int NUM_ITERS = 10;
	
	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);

    struct timeval st, ed;

    dense_matrix A_mat, B_mat;

    int max_threads = mkl_get_max_threads();
    printf("Available max MKL threads: %d\n", max_threads);

	printf("\tGenerating dense matrix A of size %d x %d\n", M, K);
    float * values_MatA=NULL;
	A_mat = generate_matrix(M, K, &values_MatA);

	printf("\tGenerating dense matrix B of size %d x %d\n", K, N);
    float * values_MatB=NULL;
	B_mat = generate_matrix(K, N, &values_MatB);

 //    #if WRITE

 //       // float * values_MatA=NULL;
 //       // float * values_Mat=NULL;
 //        // float* mat = (float*) mkl_malloc (K * N * sizeof(float), 64);
	//    // write_matrix(K, N, &values_MatB);
 //    dense_matrix mat;
 //     FILE *fp ;
 //        fp = fopen("matrixB.txt", "a");
        
 //        fwrite(&B_mat, sizeof(float), 1, fp);

 //       // FILE *fp 
 //       fp = fopen("matrixB.txt", "r");


 //       while(!feof(fp)) {
 //            fread(mat, sizeof(float), 1, fp);
 //        }
 //        fclose(fp);
 //        print_matrix(mat.values, K, N, 1);
	// #else
       
 //       // read_matrix(M, K, A_mat, 'mat.txt');
 //       // read_matrix(K, N, B_mat, 'matB.txt');

 //    #endif

    // print_matrix(A_mat, M, K, 1);
    // print_matrix(B_mat, K, N, 1);

 // *****************************************  Optimized Matrix Multiplication ************************************

    // dense_matrix C_1;
   float* C_1 = (float*) mkl_malloc (M * N * sizeof(float), 64);
   for (int i1 = 0; i1 < M*N; ++i1)
    {
        C_1[i1] = 0.0;
    }

    int TileSize = 64;
    uint64_t startTick1, endTick1;
   startTick1 = __rdtsc();
   for (int i=0; i<NUM_ITERS; i++)
   {
        // matMulColTrans(values_MatA, values_MatB, C_1, M, N, K);
        // matMulLoopIntChng0(values_MatA, values_MatB, C_1, M, N, K);
        // matMulLoopIntChng1(values_MatA, values_MatB, C_1, M, N, K);
        // matMulLoopIntChngCacheBlock(values_MatA, values_MatB, C_1, M, N, K, TileSize);
        // matMulCacheBlock(values_MatA, values_MatB, C_1, M, N, K, TileSize);
        // matMul_LoopIntChng2_blocking(values_MatA, values_MatB, C_1, M, N, K, TileSize);


        // matMulVectorIntrinsics_ST(values_MatA, values_MatB, C_1, M, N, K);
        // MM_LIC_CB_VectorIntrinsics(values_MatA, values_MatB, C_1, M, N, K, TileSize);
        MM_LIC_CB_VectorIntrinsics2(values_MatA, values_MatB, C_1, M, N, K);
    }
   
    endTick1 = __rdtsc();
    uint64_t tot_rc_ticks =  endTick1 - startTick1;

    printf("\tGenerating dense matrix C of size %d x %d\n", M, N);
    // print_matrix(C_mat.values, C_mat.rows, C_mat.cols, C_mat.is_rowmajor);
    print_matrix(C_1, M, N, 1);

    float time_taken_rc = ((tot_rc_ticks*1.0)/2.7/1e9)/NUM_ITERS;

    printf("\t\tCPU: Time taken for Optimized MatMul 1 Thread : %lf\n", time_taken_rc);

    float Rc_freq =  2.0 * M * N * K / (time_taken_rc);
    printf(" GFLOPS for RC  %0.2lf \n", Rc_freq / 1e9 );

    float Effy_opt = Rc_freq / 153 / 1e9 ; //(4.3 * 1e12);
    printf(" Efficiency wrt Peak %0.2lf \n", Effy_opt );

    // printf(" Number of Optimized MM Operations per Sec for 1 Thread\n", 153 / Effy_opt);



	// ***************************************   Baseline Matrix Multiplication *************************************************
#if BASELINE    
    dense_matrix C_mat;
   float* res = (float*) mkl_malloc (M * N * sizeof(float), 64);
   for (int i1 = 0; i1 < M*N; ++i1)
    {
        res[i1] = 0.0;
    }

    printf (" \n\n Computing Matrix Mulitplication Baseline \n\n");
    // gettimeofday(&st, NULL);
   uint64_t startTick, endTick;
   startTick = __rdtsc();
   for (int i=0; i<NUM_ITERS; i++)
   {
        matMulBase(values_MatA, values_MatB, res, M, N, K);
    }
   
    endTick = __rdtsc();
    uint64_t tot_base_ticks =  endTick - startTick;

    C_mat.rows = M;
    C_mat.cols = N;
    C_mat.values = res;

    // gettimeofday(&ed, NULL);
   // int time_taken = (ed.tv_sec*1e6 + ed.tv_usec) - (st.tv_sec* 1e6 + st.tv_usec);

    printf("\tGenerating dense matrix C of size %d x %d\n", M, N);
    // print_matrix(C_mat.values, C_mat.rows, C_mat.cols, C_mat.is_rowmajor);
    print_matrix(res, M, N, 1);

    float time_taken = ((tot_base_ticks*1.0)/2.7/1e9)/NUM_ITERS;

    printf("\t\tCPU: Time taken for MatMul Baseline : %lf\n", time_taken);

    float Baseline_freq =  2.0 * M * N * K / (time_taken);
    printf(" GFLOPS for Baseline  %0.2lf \n", Baseline_freq /1e9 );

    // float Effy_base = Baseline_freq / (4.3 * 1e12);
    // printf(" Efficiency wrt Peak %0.2lf \n", Effy_base);
#endif

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
   float time_taken_mkl = ((tot_mkl_ticks*1.0)/2.7/1e9)/NUM_ITERS;
   
   printf("\t\tCPU: Time taken for MatMul MKL 1 Thread: %lf\n", time_taken_mkl );

    float Gflops_mkl =  2.0 * M * N * K / time_taken_mkl;
   printf(" GFLOPS for MKL  %0.2lf \n", Gflops_mkl / 1e9);

   float Effy_mkl = Gflops_mkl / 153 / 1e9; //(4.3 * 1e12);
   printf(" Efficiency wrt Peak %0.2lf \n", Effy_mkl);

    // printf(" Number of MKL MM Operations per Sec for 1 Thread\n", 153 / Effy_mkl);

//---------------------------------------------------------------------------------------------------------------------------
#endif

 


}
