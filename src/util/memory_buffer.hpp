#pragma once


template <typename T>
class MemoryBuffer
{
private:
	T* data_ = nullptr;
	size_t capacity_ = 0;
	size_t size_ = 0;

public:
	MemoryBuffer(size_t n_elements)
	{
		auto data = std::malloc(sizeof(T) * n_elements);
		assert(data);

		data_ = (T*)data;
		capacity_ = n_elements;
	}


	T* push(size_t n_elements)
	{
		assert(data_);
		assert(capacity_);
		assert(size_ < capacity_);

		auto is_valid =
			data_ &&
			capacity_ &&
			size_ < capacity_;

		auto elements_available = (capacity_ - size_) >= n_elements;
		assert(elements_available);

		if (!is_valid || !elements_available)
		{
			return nullptr;
		}

		auto data = data_ + size_;

		size_ += n_elements;

		return data;
	}


	void reset()
	{
		size_ = 0;
	}


	void free()
	{
		if (data_)
		{
			std::free(data_);
			data_ = nullptr;
		}

		capacity_ = 0;
		size_ = 0;
	}
};
