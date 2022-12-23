#pragma once

#include "../defines.hpp"


namespace memory_buffer
{
	u8* malloc_bytes(size_t n_bytes);

	void free_bytes(void* data);


	template <typename T>
	bool create_buffer(MemoryBuffer<T>& buffer, size_t n_elements)
	{
		assert(n_elements);
		assert(!buffer.data_);

		if (n_elements == 0 || buffer.data_)
		{
			return false;
		}

		buffer.data_ = (T*)malloc_bytes(n_elements * sizeof(T));		
		assert(buffer.data_);

		if (!buffer.data_)
		{
			return false;
		}

		buffer.capacity_ = n_elements;
		buffer.size_ = 0;

		return true;
	}


	template <typename T>
	void destroy_buffer(MemoryBuffer<T>& buffer)
	{
		free_bytes(buffer.data_);

		buffer.data_ = nullptr;
		buffer.capacity_ = 0;
		buffer.size_ = 0;
	}


	template <typename T>
	void reset_buffer(MemoryBuffer<T>& buffer)
	{
		buffer.size_ = 0;
	}


	template <typename T>
	T* push_elements(MemoryBuffer<T>& buffer, size_t n_elements)
	{
		assert(n_elements);

		if (n_elements == 0)
		{
			return nullptr;
		}

		assert(buffer.data_);
		assert(buffer.capacity_);
		assert(buffer.size_ < buffer.capacity_);

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
}
