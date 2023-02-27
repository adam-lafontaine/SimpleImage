#include "simage_cuda.hpp"
#include "../util/execute.hpp"


static void process_image_by_row(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
}


namespace simage
{
    template <typename T>
    static Matrix2D<T> as_host_matrix(DeviceMatrix2D<T> const& mat)
    {
        Matrix2D<T> dst{};

        dst.data_ = mat.data_;
        dst.width = mat.width;
        dst.height = mat.height;

        return dst;
    }
}


/* verify */

namespace simage
{
#ifndef NDEBUG

    template <typename T>
	static bool verify(Matrix2D<T> const& image)
	{
		return image.width && image.height && image.data_;
	}


	template <typename T>
	static bool verify(DeviceMatrix2D<T> const& image)
	{
		return image.width && image.height && image.data;
	}


    template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}


    template <typename T>
	static bool verify(DeviceBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}

#endif // NDEBUG
}


/* make_image */

namespace simage 
{
    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView image{};

        image.data_ = cuda::push_elements(buffer, width * height);
        image.width = width;
        image.height = height;

        assert(verify(image));

        return image;
    }
}


/* copy_to_device */

namespace simage
{
    void copy_to_device(Image const& src, DeviceView const& dst)
	{
        assert(verify(src, dst));

        auto const n_bytes = sizeof(Pixel) * src.width * src.height;

        if(!cuda::memcpy_to_device(src.data_, dst.data_, n_bytes)) { assert(false); }
	}


    void copy_to_device(View const& src, DeviceView const& dst)
	{
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(Pixel) * src.width;
        auto const device = as_host_matrix(dst);

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(src, y);
            auto d = row_begin(device, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_image_by_row(src.height, row_func);
	}
}


/* copy_to_host */

namespace simage
{
    void copy_to_host(DeviceView const& src, Image const& dst)
	{
        assert(verify(src, dst));

        auto const n_bytes = sizeof(Pixel) * src.width * src.height;

        if(!cuda::memcpy_to_host(src.data_, dst.data_, n_bytes)) { assert(false); }
	}


    void copy_to_host(DeviceView const& src, View const& dst)
	{
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(Pixel) * src.width;
        auto const device = as_host_matrix(src);

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(dst, y);
            auto d = row_begin(device, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_image_by_row(src.height, row_func);
	}
}