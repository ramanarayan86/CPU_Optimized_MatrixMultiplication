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
// #define B_(i, j) B[(j)*ldb + i]
#define C_(i, j) C[(i)*ldc + j]

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// int64_t procf = 1;
//     int64_t startTick = __rdtsc();
//     sleep(1);
//     int64_t endTick = __rdtsc();
//     procf = endTick - startTick;
//     printf("procf = %ld\n", procf);


typedef struct  dense_matrix
{
        int rows;
        int cols;
        float* values;
        float **values2D;
        bool is_rowmajor;
}dense_matrix;

// ********************************************************************************************************************************
// ********************************************************************************************************************************



// ********************************************************************************************************************************
// ********************************************************************************************************************************
double* floatToDouble(int row, int col , float* x){

    double *y;
    y =(double*)malloc(row * col *sizeof(double));
    for (int i=0; i<row*col ; i++){
        y[i]=x[i];
    }
    return y;
}

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
//============================================================================================================================


inline void GeMM_4x32(int k, float *A, int lda, float *B, int ldb, float *C, int ldc){
    __m512 C_016_0 = _mm512_load_ps(&C_(0,0));
    __m512 C_1632_0 = _mm512_load_ps(&C_(0,16));

     __m512 C_016_1 = _mm512_load_ps(&C_(1,0));
    __m512 C_1632_1 = _mm512_load_ps(&C_(1,16));

     __m512 C_016_2 = _mm512_load_ps(&C_(2,0));
    __m512 C_1632_2 = _mm512_load_ps(&C_(2,16));

     __m512 C_016_3 = _mm512_load_ps(&C_(3,0));
    __m512 C_1632_3 = _mm512_load_ps(&C_(3,16));
    

    for (int p = 0; p < k; p++)
    {
        __m512 A_jp;

        // //Row Order
        // __m512 B_016_p = _mm512_load_ps(&B_(p,0));
        // __m512 B_1632_p = _mm512_load_ps(&B_(p,16));

        //Col Order
        __m512 B_016_p = _mm512_load_ps(&B_(0,p));
        __m512 B_1632_p = _mm512_load_ps(&B_(16,p));


        // A_jp = _mm512_set1_ps(A_(0,p));
        __m512 A_jp_0 = _mm512_load_ps(&A_(0,p));
        __m512 A_jp_16 = _mm512_load_ps(&A_(16,p));
        C_016_0 = _mm512_fmadd_ps(A_jp_0, B_016_p, C_016_0);
        C_1632_0 = _mm512_fmadd_ps(A_jp_16, B_1632_p, C_1632_0);

        
        // A_jp = _mm512_set1_ps(A_(1,p));
        A_jp_0 = _mm512_load_ps(&A_(0,p));
        A_jp_16 = _mm512_load_ps(&A_(16,p));
        C_016_1 = _mm512_fmadd_ps(A_jp, B_016_p, C_016_1);
        C_1632_1 = _mm512_fmadd_ps(A_jp, B_1632_p, C_1632_1);

        
        A_jp = _mm512_set1_ps(A_(2,p));
        C_016_2 = _mm512_fmadd_ps(A_jp, B_016_p, C_016_2);
        C_1632_2 = _mm512_fmadd_ps(A_jp, B_1632_p, C_1632_2);

        
        A_jp = _mm512_set1_ps(A_(3,p));
        C_016_3 = _mm512_fmadd_ps(A_jp, B_016_p, C_016_3);
        C_1632_3 = _mm512_fmadd_ps(A_jp, B_1632_p, C_1632_3);
    }

    _mm512_store_ps(&C_(0,0), C_016_0);
    _mm512_store_ps(&C_(0,16), C_1632_0);
    _mm512_store_ps(&C_(1,0), C_016_1);
    _mm512_store_ps(&C_(1,16), C_1632_1);
    _mm512_store_ps(&C_(2,0), C_016_2);
    _mm512_store_ps(&C_(2,16), C_1632_2);
    _mm512_store_ps(&C_(3,0), C_016_3);
    _mm512_store_ps(&C_(3,16), C_1632_3);
    
}

#define MR 4
#define NR 32

inline void MyGeMM_inner(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc){

    int i, j;
    // int MR = 4;
    // int NR = 2*16; //32;

    // #pragma unroll(2)
    for (i = 0; i < M; i+=MR){
        for (j = 0; j < N; j+=NR){
            GeMM_4x32(K, &A_(i,0), lda, &B_(0,j), ldb, &C_(i,j), ldc);
        }
    }
}

void MyGeMM(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc){

    int m0, n0, k0, mb1, kb1, nb1, i, j;
    int mb = 128;
    int kb = 32;
    int nb = 1024;


#if 1

    #pragma omp parallel for schedule(static) private(m0,n0,k0, mb1, nb1, kb1) shared(A, B, C) collapse(3)
    for (m0=0; m0 < M; m0+=mb){
        for (n0=0; n0 < N; n0 +=nb){
            for (k0=0; k0 < K; k0+=kb){
                mb1 = MIN(M-m0, mb);          
                nb1 = MIN(N-n0, nb);
                kb1 = MIN(K-k0, kb);
                MyGeMM_inner(mb1, nb1, kb1, &A_(m0,k0), lda, &B_(k0, n0), ldb, &C_(m0, n0), ldc);
            }
        }
    }

#endif
}

// ********************************************************************************************************************************
// ********************************************************************************************************************************


// ********************************************************************************************************************************
void AxBT_kernerl(int k, const float* A, const float* B,  float* C, int lda, int ldb, int ldc, int regsA, int regsB){

    __m512 Csum[regsA][regsB]={{0.0}};
    // float Csum[regsA][regsB]={{0.0}};
    __m512 Ar, Br, prd;
   float pd=0.0;

   for (int p = 0; p < k; p++)
    {
        for (int i=0; i < regsA; i++){
            prd = {0.0};
            Ar = _mm512_load_ps(&A_(i*16, p));
            for (int j = 0; j  < regsB; j++)
            {
                Br = _mm512_load_ps(&B_(p, j*16));
                prd = _mm512_mul_ps(Ar, Br);
                // pd = _mm512_reduce_add_ps(prd); 
                // std::cout << " Pd " << pd << std::endl;
                Csum[i][j] =  _mm512_reduce_add_ps(prd);
            }
        }    
    }

    for (int bi = 0; bi < regsB; bi++)
    {
        for (int ai = 0; ai < regsA; ai++)
        {
        _mm512_store_ps (( __m512*)&C_(ai*16, bi), _mm512_add_ps(_mm512_load_ps(( __m512*)&C_(ai*16, bi)), Csum[ai][bi]));
            // _mm512_store_ps (( __m512*)&C_(ai*16, bi), _mm512_add_ps(_mm512_load_ps(( __m512*)&C_(ai*16, bi)), (__m512*)Csum[ai][bi]));
        }        
    }
}


void mm_inner( float* A, float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){

    int procf = 2693749694;

    int regsA = 3; //3;  //5;
    int regsB = 4;  //4;  //2;

    int mr = regsA ; //* 16;
    int nr = regsB * 16;

    
    
    for (int j = 0; j < N - nr +1; j+=nr){
        for (int i = 0; i < M - mr +1; i+=mr){
            AxBT_kernerl(K, &A_(i,0), &B_(0,j), &C_(i,j), lda, ldb, ldc, regsA, regsB);
        }
    }
}

void MM_OPTIM( float* A, float* B,  float* C, int M, int N, int K, int lda, int ldb, int ldc){
    int mb = M/32;//1024; //2048;//4096;//8192;                 
    int kb= K/32; //128;//32;//64;//128;//32;
     //64;                 
    int nb= N/32;//1024; //2048;//8192;//8192;

    int m0, n0, k0, mb1, nb1, kb1;

    // #pragma omp parallel for schedule(static) private(m0,n0,k0, mb1, nb1, kb1) shared(A, B, C) collapse(3)
    // #pragma omp parallel for private(m0,n0,k0, mb1, nb1, kb1) shared(A, B, C) collapse(3)
    for (n0=0; n0 < N; n0 +=nb){
        for (k0=0; k0 < K; k0+=kb){
            for (m0=0; m0 < M; m0 += mb){    
                    mb1 = MIN(M-m0, mb);          
                    nb1 = MIN(N-n0, nb);
                    kb1 = MIN(K-k0, kb);                                           
                    mm_inner(&A_(m0,k0), &B_(k0, n0), &C_(m0, n0), mb1, nb1, kb1, lda, ldb, ldc);
                }
            }
        }
    // }
}

// ********************************************************************************************************************************

void matMulBase (float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc)         // Time for 1024 x 1024 Matrix: 3.46 sec
{
    int row, col, dim;
    // float* values = (float*) malloc (M * N * sizeof(float));
    // #pragma omp parallel for //schedule(static) private(row, col, dim) shared(A,B,C) collapse(3)
    for (row=0; row < M; row++)
    {
        for (col=0; col< N; col++)
        {
            for (dim=0; dim < K; dim++)
            {
                // C[row*N + col] += A[row*K + dim] * B[dim*N + col];
                C_(row, col) += A_(row, dim) * B_( col, dim);
                // values[v1*N + v2] += val; 
            }
        }
    }
    // return values;
}


void matMul_ABT_block (float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc)         // Time for 1024 x 1024 Matrix: 3.46 sec
{
    int row, col, dim;
    int BlockSizeR = 64;
    int BlockSizeC = 128;
    int BlockSizeD = 128;
    // float* values = (float*) malloc (M * N * sizeof(float));
    // #pragma omp parallel for //schedule(static) private(row, col, dim) shared(A,B,C) collapse(3)
    for (row=0; row < M; row+= BlockSizeR)
    {
        for (col=0; col< N; col+=BlockSizeC)
        {
            for (dim=0; dim < K; dim+=BlockSizeD)
            {
                for (int blockR = row; blockR < row + BlockSizeR; blockR++)
                {    
                    // #pragma unroll(10)
                     for (int blockC = col; blockC < col + BlockSizeC; blockC++)     
                    {
                        for (int blockD = dim; blockD < dim + BlockSizeD; blockD++)
                        {
                            C_(blockR, blockC) += A_(blockR, blockD) * B_( blockC, blockD);
                        }
                    }
                }
            }
        }
    }
    // return values;
}

// ********************************************************************************************************************************
// ********************************************************************************************************************************


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
    // print_matrix(values_MatA, M, K, 1);

    printf("\tGenerating dense matrix B of size %d x %d\n", K, N);
    float * values_MatB=NULL;
    B_mat = generate_matrix(K, N, &values_MatB);
    // print_matrix(values_MatB, K, N, 1);

// printf("\tGenerating dense matrix Amat of size %d x %d\n", M, K);
//     float * valuesMatA=NULL;
//     dense_matrix Amat = generate_matrix(M, K, &valuesMatA);
//     print_matrix(valuesMatA, M, K, 1);
//     print_2Dmatrix(Amat.values2D, M, K, 1);

   double *dA = floatToDouble(M, K, values_MatA);
   double *dB = floatToDouble(K, N, values_MatB);
    

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
   double *dC1 = floatToDouble(M, N, C_1); 
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

    int lda = K;
    int ldb = N;
    int ldc = N;

   startTick1 = __rdtsc();
   for (int i=0; i<NUM_ITERS; i++)
   {
        // MM_SIMD2D(A_mat.values2D, B_mat.values2D, C1, M, N, K);
        // MM_SIMD3(values_MatA, values_MatB, C_1, M, N, K);
        // MM_SIMD_ROW2(values_MatA, values_MatB, C_1, M, N, K);
        // MM_L2CB(values_MatA, values_MatB, C_1, M, N, K);
        // MM_L2CB_KIJ(values_MatA, values_MatB, C_1, M, N, K); 

        // matmul_outer(values_MatA, values_MatB, C_1, M, N, K);
        // MM_OPTIM(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc); 
        // MyGemm(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc); 
        // MM_SIMD_ROW2(values_MatA, values_MatB, C_1, M, N, K);
        // mult(values_MatA, values_MatB, C_1, M, N, K);
        // MM_SIMD_ROW3(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc);
        // MyGemm(M,N,K, values_MatA, lda, values_MatB, ldb, C_1, ldc) ;

    // MyGeMM(M, N, K, values_MatA, lda, values_MatB, ldb, C_1, ldc );
    // matMulBase(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc);
    // matMul_ABT_block(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc);
    MM_OPTIM(values_MatA, values_MatB, C_1, M, N, K, lda, ldb , ldc); 
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

    int num_threads=omp_get_max_threads(); 

    printf("\tGenerating dense matrix C of size %d x %d\n", M, N);
    // print_matrix(C_mat.values, C_mat.rows, C_mat.cols, C_mat.is_rowmajor);
    // print_2Dmatrix(C1, M, K, 1);
    print_matrix(C_1, M, N, 1);

    float time_taken_rc = ((tot_rc_ticks*1.0)/procf)/NUM_ITERS;

    printf("\t\tCPU: Ticks taken for Optimized MatMul 1 Thread : %ld\n", tot_rc_ticks);
    printf("\t\tCPU: Time taken for Optimized MatMul 1 Thread : %lf\n", time_taken_rc);

    float Rc_freq =  2.0 * M * N * K / (time_taken_rc);
    printf(" GFLOPS for RC  %0.2lf \n", Rc_freq / 1e9 );

    float Effy_opt = Rc_freq / (num_threads * 153) / 1e9 ; //(4.3 * 1e12);
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, values_MatA, K, values_MatB, N, beta, C, N);
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

   float Effy_mkl = Gflops_mkl / (num_threads * 153) / 1e9; //(4.3 * 1e12);
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