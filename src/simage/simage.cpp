#include "simage.hpp"
#include "../util/memory_buffer.hpp"


namespace mb = memory_buffer;


namespace simage
{
	void make_image(Image& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (Pixel*)malloc(sizeof(Pixel) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


    void make_image(ImageGray& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (u8*)malloc(sizeof(u8) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(ImageGray& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	void destroy_image(ImageYUV& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}
}


/* row begin */

namespace simage
{
	static Pixel* row_begin(View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	static u8* row_begin(ViewGray const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	static YUV2* row_begin(ViewYUV const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	static r32* row_begin(View1r32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
	static PixelCHr32<N> row_begin(ViewCHr32<N> const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelCHr32<N> p{};

		for (u32 ch = 0; ch < N; ++ch)
		{
			p.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return p;
	}


	static PixelRGBr32 rgb_row_begin(ViewRGBr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelRGBr32 p{};

		p.rgb.R = view.image_channel_data[id_cast(RGB::R)] + offset;
		p.rgb.G = view.image_channel_data[id_cast(RGB::G)] + offset;
		p.rgb.B = view.image_channel_data[id_cast(RGB::B)] + offset;

		return p;
	}


	static PixelHSVr32 hsv_row_begin(ViewHSVr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelHSVr32 p{};

		p.hsv.H = view.image_channel_data[id_cast(HSV::H)] + offset;
		p.hsv.S = view.image_channel_data[id_cast(HSV::S)] + offset;
		p.hsv.V = view.image_channel_data[id_cast(HSV::V)] + offset;

		return p;
	}


	template <size_t N>
	static r32* channel_row_begin(ViewCHr32<N> const& view, u32 y, u32 ch)
	{
		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


	static r32* row_offset_begin(View1r32 const& view, u32 y, int y_offset)
	{
		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
	static r32* channel_row_offset_begin(ViewCHr32<N> const& view, u32 y, int y_offset, u32 ch)
	{
		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


}


/* xy_at */

namespace simage
{
	static Pixel* xy_at(View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	static u8* xy_at(ViewGray const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	static YUV2* xy_at(ViewYUV const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	static r32* xy_at(View1r32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	template <size_t N>
	static PixelCHr32<N> xy_at_n(ViewCHr32<N> const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelCHr32<N> p{};

		for (u32 ch = 0; ch < N; ++ch)
		{
			p.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return p;
	}


	static Pixel4r32 xy_at(View4r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}


	static Pixel3r32 xy_at(View3r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}


	static Pixel2r32 xy_at(View2r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}
}


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <class IMAGE>
	static bool verify_image_1D(IMAGE const& image)
	{
		return image.width && image.height && image.data;
	}


	template <class VIEW>
	static bool verify_view_1D(VIEW const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	static bool verify(Image const& image)
	{
		return verify_image_1D(image);
	}


	static bool verify(View const& view)
	{
		return verify_view_1D(view);
	}


	static bool verify(ImageGray const& image)
	{
		return verify_image_1D(image);
	}


	static bool verify(ViewGray const& view)
	{
		return verify_view_1D(view);
	}


	static bool verify(ImageYUV const& image)
	{
		return verify_image_1D(image);
	}


	static bool verify(ViewYUV const& view)
	{
		return verify_view_1D(view);
	}


	static bool verify(View1r32 const& view)
	{
		return verify_view_1D(view);
	}


	template <typename T>
	static bool verify(MemoryBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}


	template <size_t N>
	static bool verify(ViewCHr32<N> const& view)
	{
		return view.image_width && view.width && view.height && view.image_channel_data[0];
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
}

#endif // !NDEBUG

/* make view */

namespace simage
{
	template <class IMAGE, class VIEW>
	static void do_make_view_1D(IMAGE const& image, VIEW& view)
	{
		view.image_data = image.data;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;
	}


	template <size_t N>
	static void do_make_view(ViewCHr32<N>& view, u32 width, u32 height, Buffer32& buffer)
	{
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.image_channel_data[ch] = mb::push_elements(buffer, width * height);
		}
	}


	View make_view(Image const& image)
	{
		assert(verify(image));

		View view{};

		do_make_view_1D(image, view);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(ImageGray const& image)
	{
		assert(verify(image));

		ViewGray view{};

		do_make_view_1D(image, view);
		assert(verify(view));

		return view;
	}


	ViewYUV make_view(ImageYUV const& image)
	{
		assert(verify(image));

		ViewYUV view{};

		do_make_view_1D(image, view);
		assert(verify(view));

		return view;
	}


	View1r32 make_view_1(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View1r32 view{};

		view.image_data = mb::push_elements(buffer, width * height);
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;

		assert(verify(view));

		return view;
	}


	View2r32 make_view_2(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2r32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View3r32 make_view_3(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3r32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View4r32 make_view_4(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4r32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}
}


/* sub_view */

namespace simage
{
	template <class IMAGE, class VIEW>
	static void do_sub_view_1D(IMAGE const& image, VIEW& sub_view, Range2Du32 const& range)
	{
		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;	
	}


	template <class VIEW>
	static void do_sub_view_1D(VIEW const& view, VIEW& sub_view, Range2Du32 const& range)
	{
		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;
	}


	template <size_t N>
	static ViewCHr32<N> do_sub_view(ViewCHr32<N> const& view, Range2Du32 const& range)
	{
		ViewCHr32<N> sub_view;

		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			sub_view.image_channel_data[ch] = view.image_channel_data[ch];
		}

		return sub_view;
	}


	View sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		View sub_view;

		do_sub_view_1D(image, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ImageGray const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		ViewGray sub_view;

		do_sub_view_1D(image, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewYUV sub_view(ImageYUV const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		ViewYUV sub_view;

		do_sub_view_1D(image, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View sub_view(View const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		View sub_view;

		do_sub_view_1D(view, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ViewGray const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		ViewGray sub_view;

		do_sub_view_1D(view, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewYUV sub_view(ViewYUV const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		ViewYUV sub_view;

		do_sub_view_1D(view, sub_view, range);

		assert(verify(sub_view));

		return sub_view;
	}



	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}