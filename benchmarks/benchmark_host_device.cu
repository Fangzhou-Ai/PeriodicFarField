#include "../include/puff.h"
#include <iostream>
#include <chrono>

using namespace puff;

void benchmark_SparseMatrix_Insertion_Host(int N)
{
    SparseMatrix_h<double> A;
    
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    A.make_matrix();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Make matrix on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;
}

void benchmark_SparseMatrix_Insertion_Device(int N)
{
    SparseMatrix_d<double> A;
    
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    A.make_matrix();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Make matrix on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;
}

template<typename T>
void benchmark_SpMV_Host(int N)
{
    Vector_h<T> x(N);
    Vector_h<T> y(N);
    SparseMatrix_h<T> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, T(1.0));

    A.make_matrix();

    // Warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);
    
    // Benchmark y = A * x
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 100; i++)
		A.SpMV(x, y);
    
    auto end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "SpMV on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;

    // warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    // Benchmark y = A' * x
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);
    
	end = std::chrono::high_resolution_clock::now();
	// output in microseconds
	std::cout << "Transpose SpMV on host of size " << N << ": " << \
		std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
		" us" << std::endl;
    return;
}

template<typename T>
void benchmark_SpMV_Device(int N)
{
    Vector_d<T> x(N);
    Vector_d<T> y(N);
    SparseMatrix_d<T> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, T(1.0));

    A.make_matrix();

    // Warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);

    // Benchmark y = A * x
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);

    auto end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "SpMV on device of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;


    // warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    // Benchmark y = A' * x
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "Transpose SpMV on device of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;
    return;
}


int main()
{
#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "**********Benchmark initialized with OpenMP**********" << std::endl;
#else
    std::cout << "**********Benchmark initialized without OpenMP**********" << std::endl;
#endif
    benchmark_SparseMatrix_Insertion_Host(1e6);
    benchmark_SparseMatrix_Insertion_Device(1e6);
    std::cout << "SpMV Benchmark: bcomplex" << std::endl;
    benchmark_SpMV_Host<puff::bcomplex>(1e6);
    benchmark_SpMV_Device<puff::bcomplex>(1e6);
    std::cout << "SpMV Benchmark: hcomplex" << std::endl;
    benchmark_SpMV_Host<puff::hcomplex>(1e6);
    benchmark_SpMV_Device<puff::hcomplex>(1e6);
    std::cout << "SpMV Benchmark: fcomplex" << std::endl;
    benchmark_SpMV_Host<puff::fcomplex>(1e6);
    benchmark_SpMV_Device<puff::fcomplex>(1e6);
    std::cout << "SpMV Benchmark: dcomplex" << std::endl;
    benchmark_SpMV_Host<puff::dcomplex>(1e6);
    benchmark_SpMV_Device<puff::dcomplex>(1e6);
    std::cout << "SpMV Benchmark: __nv_bfloat16" << std::endl;
    benchmark_SpMV_Host<__nv_bfloat16>(1e6);
    benchmark_SpMV_Device<__nv_bfloat16>(1e6);
    std::cout << "SpMV Benchmark: half" << std::endl;
    benchmark_SpMV_Host<half>(1e6);
    benchmark_SpMV_Device<half>(1e6);
    std::cout << "SpMV Benchmark: float" << std::endl;
    benchmark_SpMV_Host<float>(1e6);
    benchmark_SpMV_Device<float>(1e6);
    std::cout << "SpMV Benchmark: double" << std::endl;
    benchmark_SpMV_Host<double>(1e6);
    benchmark_SpMV_Device<double>(1e6);
    return 0;
}