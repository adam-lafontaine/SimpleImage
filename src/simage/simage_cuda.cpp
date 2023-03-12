#include "simage_cuda.hpp"
#include "../util/execute.hpp"


static void process_image_by_row(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
}

/*
namespace simage
{
    template <typename T>
    using HostView = MatrixView<T>;


    template <typename T>
    static HostView<T> as_host_view(DeviceView2D<T> const& view)
    {
        HostView<T> dst;

        dst.matrix_data_ = view.matrix_data_;
        dst.matrix_width = view.matrix_width;
        dst.width = view.width;        
        dst.height = view.height;
        dst.range = view.range;

        return dst;
    }
}
*/

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


    template <class IMG>
	static bool verify(IMG const& image, Range2Du32 const& range)
	{
		return
			verify(image) &&
			range.x_begin < range.x_end &&
			range.y_begin < range.y_end &&
			range.x_begin < image.width &&
			range.x_end <= image.width &&
			range.y_begin < image.height &&
			range.y_end <= image.height;
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
    template <typename T>
    static void do_make_view(DeviceView2D<T>& view, u32 width, u32 height, DeviceBuffer<T>& buffer)
    {
        view.matrix_data_ = cuda::push_elements(buffer, width * height);
        view.matrix_width = width;
        view.width = width;
        view.height = height;

        view.range = make_range(width, height);
    }


    template <typename VIEW_T, typename BUFF_T, size_t N>
    static void do_make_channel_view(DeviceChannelView2D<VIEW_T, N>& view, u32 width, u32 height, DeviceBuffer<BUFF_T>& buffer)
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

        do_make_view(view, width, height, buffer);

        assert(verify(view));

        return view;
    }


    DeviceViewGray make_view(u32 width, u32 height, DeviceBuffer8& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceViewGray view;

        do_make_view(view, width, height, buffer);

        assert(verify(view));

        return view;
    }    


    DeviceView1u16 make_view_1(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

		DeviceView1u16 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
    }


    DeviceView2u16 make_view_2(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView2u16 view;

        do_make_channel_view(view, width, height, buffer);

        assert(verify(view));

		return view;
    }


    DeviceView3u16 make_view_3(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView3u16 view;

        do_make_channel_view(view, width, height, buffer);

        assert(verify(view));

		return view;
    }


    DeviceView4u16 make_view_4(u32 width, u32 height, DeviceBuffer16& buffer)
    {
        assert(verify(buffer, width * height));

        DeviceView4u16 view;

        do_make_channel_view(view, width, height, buffer);

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


    void copy_to_device(ViewGray const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(u8) * src.width;
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


    void copy_to_host(DeviceViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const bytes_per_row = sizeof(u8) * src.width;
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


/* sub_view */

namespace simage
{
    template <size_t N>
	static DeviceViewCHu16<N> do_sub_view(DeviceViewCHu16<N> const& view, Range2Du32 const& range)
	{
		DeviceViewCHu16<N> sub_view;

		sub_view.channel_width_ = view.channel_width_;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			sub_view.channel_data_[ch] = view.channel_data_[ch];
		}

		return sub_view;
	}


	DeviceView4u16 sub_view(DeviceView4u16 const& view, Range2Du32 const& range)
    {
        assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
    }


	DeviceView3u16 sub_view(DeviceView3u16 const& view, Range2Du32 const& range)
    {
        assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
    }


	DeviceView2u16 sub_view(DeviceView2u16 const& view, Range2Du32 const& range)
    {
        assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
    }


	DeviceView1u16 sub_view(DeviceView1u16 const& view, Range2Du32 const& range)
    {
        assert(verify(view, range));

		DeviceView1u16 sub_view;

		sub_view.matrix_data_ = view.matrix_data_;
		sub_view.matrix_width = view.matrix_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(verify(sub_view));

		return sub_view;
    }
}


/* select channel */

namespace simage
{
    template <size_t N>
	static DeviceView1u16 select_channel(DeviceViewCHu16<N> const& view, u32 ch)
	{
		DeviceView1u16 view1{};

		view1.matrix_width = view.channel_width_;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.matrix_data_ = view.channel_data_[ch];

		return view1;
	}


    DeviceView1u16 select_channel(DeviceViewRGBAu16 const& view, RGBA channel)
    {
        assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
    }


	DeviceView1u16 select_channel(DeviceViewRGBu16 const& view, RGB channel)
    {
        assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
    }

	//DeviceView1u16 select_channel(DeviceViewHSVu16 const& view, HSV channel);

	//DeviceView1u16 select_channel(DeviceView2u16 const& view, GA channel);

	DeviceView1u16 select_channel(DeviceView2u16 const& view, XY channel)
    {
        assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
    }


	DeviceViewRGBu16 select_rgb(DeviceViewRGBAu16 const& view)
    {
        assert(verify(view));

		DeviceViewRGBu16 rgb;

		rgb.channel_width_ = view.channel_width_;
		rgb.width = view.width;
		rgb.height = view.height;
		rgb.range = view.range;

		rgb.channel_data_[id_cast(RGB::R)] = view.channel_data_[id_cast(RGB::R)];
		rgb.channel_data_[id_cast(RGB::G)] = view.channel_data_[id_cast(RGB::G)];
		rgb.channel_data_[id_cast(RGB::B)] = view.channel_data_[id_cast(RGB::B)];

		return rgb;
    }
}