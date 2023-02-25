#include "device.hpp"
#include "cuda_def.cuh"

#include <cassert>
#include <cstddef>

#ifdef CUDA_PRINT_ERROR

#include <cstdio>
#include <cstring>

#endif


static void check_error(cudaError_t err, cstr label = "")
{
    if (err == cudaSuccess)
    {
        return;
    }

    #ifdef CUDA_PRINT_ERROR
    #ifndef	NDEBUG

    printf("\n*** CUDA ERROR ***\n\n");
    printf("%s", cudaGetErrorString(err));

    if (std::strlen(label))
    {
        printf("\n%s", label);
    }
    
    printf("\n\n******************\n\n");

    #endif
    #endif
}


namespace cuda
{
    bool device_malloc(ByteBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data_);

        if (!n_bytes || buffer.data_)
        {
            return false;
        }

        cudaError_t err = cudaMalloc((void**)&(buffer.data_), n_bytes);
        check_error(err, "malloc");

        bool result = err == cudaSuccess;

        if (result)
        {
            buffer.capacity_ = n_bytes;
            buffer.size_ = 0;
        }
        
        return result;
    }


    bool unified_malloc(ByteBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data_);

        if (!n_bytes || buffer.data_)
        {
            return false;
        }

        cudaError_t err = cudaMallocManaged((void**)&(buffer.data_), n_bytes);
        check_error(err, "unified_malloc");

        bool result = err == cudaSuccess;

        if (result)
        {
            buffer.capacity_ = n_bytes;
            buffer.size_ = 0;
        }
        
        return result;
    }


    bool free(ByteBuffer& buffer)
    {
        buffer.capacity_ = 0;
        buffer.size_ = 0;

        if (buffer.data_)
        {
            cudaError_t err = cudaFree(buffer.data_);
            check_error(err, "free");

            buffer.data_ = nullptr;

            return err == cudaSuccess;
        }

        return true;
    }
    

    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
    {
        cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
        check_error(err, "memcpy_to_device");

        return err == cudaSuccess;
    }


    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
    {
        cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
        check_error(err, "memcpy_to_host");

        return err == cudaSuccess;
    }


    bool no_errors(cstr label)
    {
        #ifndef	NDEBUG

        cudaError_t err = cudaGetLastError();
        check_error(err, label);

        return err == cudaSuccess;

        #else

        return true;

        #endif
    }


    bool launch_success(cstr label)
    {
        #ifndef	NDEBUG

        cudaError_t err = cudaDeviceSynchronize();
        check_error(err, label);

        return err == cudaSuccess;

        #else

        return true;

        #endif
    }
}