#pragma once

#include <complex>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/sort.h>
#include <cusp/print.h>
#include <cusp/eigen/spectral_radius.h>
#include <cusp/krylov/gmres.h>
#include <cusp/monitor.h>
#include <cusp/functional.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if(err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT error in %s at line %d: %s\n", \
        __FILE__, __LINE__, cufftGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_DFTI(call) { \
    MKL_LONG err = call; \
    if(err && !DftiErrorClass(err, DFTI_NO_ERROR)) { \
        fprintf(stderr, "DFTI error in %s at line %d: %s\n", \
        __FILE__, __LINE__, DftiErrorMessage(err)); \
        exit(EXIT_FAILURE); \
    } \
}


namespace puff{

    template <typename T>
    struct conjugate_functor {
        __host__ __device__
        T operator()(const T& z) const {
            return T(z.real(), -z.imag());
        }
    };

    template<typename IndexType, typename ValueType, typename MemorySpace>
    using SparseMatrix = cusp::coo_matrix<IndexType, ValueType, MemorySpace>;        

    template<typename ValueType, typename MemorySpace>
    using Vector = cusp::array1d<ValueType, MemorySpace>;

    template<typename IndexType, typename ValueType, typename MemorySpace>
    struct SparseMatrixViewHelper;

    template<typename IndexType, typename ValueType>
    struct SparseMatrixViewHelper<IndexType, ValueType, cusp::host_memory> {
        using type = decltype(cusp::make_coo_matrix_view(
            std::declval<size_t>(), std::declval<size_t>(), std::declval<size_t>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<IndexType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<IndexType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<ValueType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>()
        ));
    };

    template<typename IndexType, typename ValueType>
    struct SparseMatrixViewHelper<IndexType, ValueType, cusp::device_memory> {
        using type = decltype(cusp::make_coo_matrix_view(
            std::declval<size_t>(), std::declval<size_t>(), std::declval<size_t>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<ValueType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>()
        ));
    };

    template<typename IndexType, typename ValueType, typename MemorySpace>
    using SparseMatrixView = typename SparseMatrixViewHelper<IndexType, ValueType, MemorySpace>::type;

    template <typename ValueType, typename MemorySpace>
    void Vector_element_wise_multiply_Vector(const Vector<ValueType, MemorySpace>& a,
                                            const Vector<ValueType, MemorySpace>& b,
                                            Vector<ValueType, MemorySpace>& c) {
        thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::multiplies<ValueType>());
    }

    template <typename ValueType, typename MemorySpace>
    void Vector_element_wise_multiply_Constant(const Vector<ValueType, MemorySpace>& in,
                                            const ValueType& value,
                                            Vector<ValueType, MemorySpace>& out) {
        thrust::transform(in.begin(), in.end(), thrust::make_constant_iterator(value), out.begin(), thrust::multiplies<ValueType>());
    }

    using dcomplex = thrust::complex<double>;
	using fcomplex = thrust::complex<float>;
	
    template <typename T>
    class puff_complex {
        public:
            using value_type = T;

            thrust::complex<value_type> value;

            // Default constructor
            __host__ __device__ puff_complex() : value(0, 0) {}

            // Constructor from value_type for real and imaginary parts
            __host__ __device__ puff_complex(value_type real, value_type imag) : value(real, imag) {}

            // Constructor that allows implicit conversion from int (for convenience with 0)
            __host__ __device__ puff_complex(int zero) : value(static_cast<value_type>(zero), static_cast<value_type>(zero)) {}

            // Getter for the real part
            __host__ __device__ value_type real() const { return value.real(); }
            
            // Setter for the real part
            __host__ __device__ void real(value_type r) { value = thrust::complex<value_type>(r, value.imag()); }

            // Getter for the imaginary part
            __host__ __device__ value_type imag() const { return value.imag(); }

            // Setter for the imaginary part
            __host__ __device__ void imag(value_type i) { value = thrust::complex<value_type>(value.real(), i); }

            // Operator overloads
            // Define the multiply operator for puff_complex
            friend puff_complex operator*(const puff_complex& a, const puff_complex& b) {
                return puff_complex(a.real() * b.real() - a.imag() * b.imag(),
                                a.real() * b.imag() + a.imag() * b.real());
            }

            // Define the addition operator for puff_complex
            friend puff_complex operator+(const puff_complex& a, const puff_complex& b) {
                return puff_complex(a.real() + b.real(), a.imag() + b.imag());
            }

            // Define the equality operator for puff_complex
            friend bool operator==(const puff_complex& a, const puff_complex& b) {
                return a.real() == b.real() && a.imag() == b.imag();
            }
    };

    using hcomplex = puff_complex<half>;
    using bcomplex = puff_complex<__nv_bfloat16>;

    
}