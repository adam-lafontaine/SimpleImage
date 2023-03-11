#pragma once

#include "simage_platform.hpp"
#include "../util/memory_buffer.hpp"

#include <array>
#include <functional>

namespace mb = memory_buffer;


/* view */

namespace simage
{
	template <typename T, size_t N>
	class ChannelView2D
	{
	public:

		u32 channel_width_ = 0;

		T* channel_data_[N] = {};		

		u32 width = 0;
		u32 height = 0;

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
	};


	//template <size_t N>
	//using ViewCHf32 = ChannelView2D<f32, N>;


	

	template <typename T>
	using View4 = ChannelView2D<T, 4>;

	template <typename T>
	using View3 = ChannelView2D<T, 3>;

	template <typename T>
	using View2 = ChannelView2D<T, 2>;

	template <typename T>
	using View1 = MatrixView<T>;

    using View4f32 = View4<f32>;
	using View3f32 = View3<f32>;
	using View2f32 = View2<f32>;
	using View1f32 = View1<f32>;

	using View4u16 = View4<u16>;
	using View3u16 = View3<u16>;
	using View2u16 = View2<u16>;
	using View1u16 = View1<u16>;


	template <typename T>
	using ChViewRGB = ChannelView2D<T, 3>;
	
}


/* make_view */

namespace simage
{
	using Buffer32 = MemoryBuffer<f32>;
	using Buffer16 = MemoryBuffer<u16>;

	using ViewRGBAf32 = View4f32;
	using ViewRGBf32 = View3f32;
	using ViewHSVf32 = View3f32;
	using ViewLCHf32 = View3f32;


	View1f32 make_view_1(u32 width, u32 height, Buffer32& buffer);

	View2f32 make_view_2(u32 width, u32 height, Buffer32& buffer);

	View3f32 make_view_3(u32 width, u32 height, Buffer32& buffer);

	View4f32 make_view_4(u32 width, u32 height, Buffer32& buffer);
}


/* map_gray */

namespace simage
{
	void map_gray(ViewGray const& src, View1f32 const& dst);

	void map_gray(View1f32 const& src, ViewGray const& dst);

	void map_gray(ViewYUV const& src, View1f32 const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAf32 const& dst);

	void map_rgba(ViewRGBAf32 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBf32 const& dst);

	void map_rgb(ViewRGBf32 const& src, View const& dst);

	void map_rgb(View1f32 const& src, View const& dst);
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst);

	void map_hsv_rgb(ViewHSVf32 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBf32 const& src, ViewHSVf32 const& dst);	

	void map_hsv_rgb(ViewHSVf32 const& src, ViewRGBf32 const& dst);
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHf32 const& dst);

	void map_lch_rgb(ViewLCHf32 const& src, View const& dst);

	void map_rgb_lch(ViewRGBf32 const& src, ViewLCHf32 const& dst);

	void map_lch_rgb(ViewLCHf32 const& src, ViewRGBf32 const& dst);
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst);	
}


/* map_bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBf32 const& dst);
}


/* sub_view */

namespace simage
{
	View4f32 sub_view(View4f32 const& view, Range2Du32 const& range);

	View3f32 sub_view(View3f32 const& view, Range2Du32 const& range);

	View2f32 sub_view(View2f32 const& view, Range2Du32 const& range);

	View1f32 sub_view(View1f32 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1f32 select_channel(ViewRGBAf32 const& view, RGBA channel);

	View1f32 select_channel(ViewRGBf32 const& view, RGB channel);

	View1f32 select_channel(ViewHSVf32 const& view, HSV channel);

	View1f32 select_channel(View2f32 const& view, GA channel);

	View1f32 select_channel(View2f32 const& view, XY channel);


	ViewRGBf32 select_rgb(ViewRGBAf32 const& view);
}


/* fill */

namespace simage
{
	void fill(View4f32 const& view, Pixel color);

	void fill(View3f32 const& view, Pixel color);

	void fill(View1f32 const& view, u8 gray);
}


/* transform */

namespace simage
{
	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func);

	void transform(View2f32 const& src, View1f32 const& dst, std::function<f32(f32, f32)> const& func);

	void transform(View3f32 const& src, View1f32 const& dst, std::function<f32(f32, f32, f32)> const& func);

	
	inline void transform_gray(ViewRGBf32 const& src, View1f32 const& dst)
	{
		return transform(src, dst, [](f32 red, f32 green, f32 blue) { return 0.299f * red + 0.587f * green + 0.114f * blue; });
	}
}


/* shrink */

namespace simage
{
	void shrink(View1f32 const& src, View1f32 const& dst);

	void shrink(View3f32 const& src, View3f32 const& dst);

	void shrink(ViewGray const& src, View1f32 const& dst);

	void shrink(View const& src, ViewRGBf32 const& dst);
}


/* gradients */

namespace simage
{
	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst);
}


/* blur */

namespace simage
{
	void blur(View1f32 const& src, View1f32 const& dst);

	void blur(View3f32 const& src, View3f32 const& dst);
}
