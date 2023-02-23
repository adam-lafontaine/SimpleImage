#pragma once

#include "../defines.hpp"



class ByteBuffer
{
public:
    u8* data_ = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;
};


namespace cuda
{
    bool device_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool unified_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool free(ByteBuffer& buffer);


    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


    bool no_errors(cstr label);

    bool launch_success(cstr label);    
}


template <typename T>
class DeviceBuffer
{
public:
	T* data_ = nullptr;
	u32 capacity_ = 0;
	u32 size_ = 0;
};


namespace cuda
{
    template <typename T>
    bool create_device_buffer(DeviceBuffer<T>& buffer, u32 n_elements)
    {
        auto const n_bytes = sizeof(T) * n_elements;

        ByteBuffer b{};
        auto result = device_malloc(b, n_bytes);
        if (result)
        {
            buffer.data_ = (T*)b.data_;
            buffer.capacity_ = n_elements;
            buffer.size_ = 0;
        }

        return result;
    }


    template <typename T>
    bool create_unified_buffer(DeviceBuffer<T>& buffer, u32 n_elements)
    {
        auto const n_bytes = sizeof(T) * n_elements;

        ByteBuffer b{};
        auto result = unified_malloc(b, n_bytes);
        if (result)
        {
            buffer.data_ = (T*)b.data_;
            buffer.capacity_ = n_elements;
            buffer.size_ = 0;
        }

        return result;
    }


    template <typename T>
	void destroy_buffer(DeviceBuffer<T>& buffer)
    {
        buffer.capacity_ = 0;
        buffer.size_ = 0;
        if (!buffer.data_)
        {
            return;
        }

        ByteBuffer b{};
        b.data_ = (u8*)buffer.data_;
        cuda::free(b);

        buffer.data_ = nullptr;
    }


    template <typename T>
	void reset_buffer(DeviceBuffer<T>& buffer)
	{
		buffer.size_ = 0;
	}


	template <typename T>
	T* push_elements(DeviceBuffer<T>& buffer, u32 n_elements)
	{
		assert(n_elements);

		if (n_elements == 0)
		{
			return nullptr;
		}

		assert(buffer.data_);
		assert(buffer.capacity_);

		auto is_valid =
			buffer.data_ &&
			buffer.capacity_ &&
			buffer.size_ < buffer.capacity_;

		auto elements_available = (buffer.capacity_ - buffer.size_) >= n_elements;
		assert(elements_available);

		if (!is_valid || !elements_available)
		{
			return nullptr;
		}

		auto data = buffer.data_ + buffer.size_;

		buffer.size_ += n_elements;

		return data;
	}


	template <typename T>
	void pop_elements(DeviceBuffer<T>& buffer, u32 n_elements)
	{
		if (!n_elements)
		{
			return;
		}

		assert(buffer.data_);
		assert(buffer.capacity_);
		assert(n_elements <= buffer.size_);

		if(n_elements > buffer.size_)
		{
			buffer.size_ = 0;
		}
		else
		{
			buffer.size_ -= n_elements;
		}
	}
}