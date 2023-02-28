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
    using HostView = MatrixView<T>;


    template <typename T>
    static HostView<T> as_host_view(DeviceView2D<T> const& view)
    {
        HostView<T> dst{};

        dst.matrix_data_ = view.matrix_data_;
        dst.matrix_width = view.matrix_width;
        dst.width = view.width;        
        dst.height = view.height;
        dst.range = view.range;

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
	static bool verify(MatrixView<T> const& image)
	{
		return image.width && image.height && image.matrix_data_;
	}


	template <typename T>
	static bool verify(DeviceView2D<T> const& image)
	{
		return image.width && image.height && image.matrix_data_;
	}


    template <typename T, size_t N>
	static bool verify(DeviceChannelView2D<T, N> const& image)
	{
		return image.width && image.height && image.channel_data_[0];
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


/* make_view */

namespace simage 
{
    template <typename VIEW_T, typename BUFF_T, size_t N>
    static void make_channel_view(DeviceChannelView2D<VIEW_T, N>& view, u32 width, u32 height, DeviceBuffer<BUFF_T>& buffer)
    {
        static_assert(sizeof(VIEW_T) == sizeof(BUFF_T));

        view.channel_width_ = width;
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

        for (u32 ch = 0; ch < N; ++ch)
		{
			view.channel_data_[ch] = (VIEW_T*)cuda::push_elements(buffer, width * height);
		}
    }


    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView view;

        view.matrix_data_ = cuda::push_elements(buffer, width * height);
        view.matrix_width = width;
        view.width = width;
        view.height = height;

        view.range = make_range(width, height);

        assert(verify(view));

        return view;
    }


    DeviceView1r16 make_view_1(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

		DeviceView1r16 view;

		view.matrix_data_ = cuda::push_elements(buffer, width * height);
        view.width = width;
        view.height = height;

        view.range = make_range(width, height);

		assert(verify(view));

		return view;
    }


    DeviceView2r16 make_view_2(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView2r16 view;

        make_channel_view(view, width, height, buffer);

        assert(verify(view));

		return view;
    }


    DeviceView3r16 make_view_3(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView3r16 view;

        make_channel_view(view, width, height, buffer);

        assert(verify(view));

		return view;
    }


    DeviceView4r16 make_view_4(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView4r16 view;

        make_channel_view(view, width, height, buffer);

        assert(verify(view));

		return view;
    }
}


/* device copy */

namespace simage
{
    void copy_to_device(View const& src, DeviceView const& dst)
	{
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(Pixel) * src.width;
        auto const device = as_host_view(dst);

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(src, y);
            auto d = row_begin(device, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_image_by_row(src.height, row_func);
	}


    void copy_to_host(DeviceView const& src, View const& dst)
	{
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(Pixel) * src.width;
        auto const device = as_host_view(src);

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(dst, y);
            auto d = row_begin(device, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_image_by_row(src.height, row_func);
	}
}


/*  */

namespace simage
{
    
}