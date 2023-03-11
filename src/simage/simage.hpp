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


	template <size_t N>
	using ViewCHr32 = ChannelView2D<r32, N>;


	using View1r32 = MatrixView<r32>;

    using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;
	using View2r32 = ViewCHr32<2>;
}


/* make_view */

namespace simage
{
	using Buffer32 = MemoryBuffer<r32>;

	using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;
	using ViewHSVr32 = View3r32;
	using ViewLCHr32 = View3r32;


	View1r32 make_view_1(u32 width, u32 height, Buffer32& buffer);

	View2r32 make_view_2(u32 width, u32 height, Buffer32& buffer);

	View3r32 make_view_3(u32 width, u32 height, Buffer32& buffer);

	View4r32 make_view_4(u32 width, u32 height, Buffer32& buffer);
}


/* map_gray */

namespace simage
{
	void map_gray(ViewGray const& src, View1r32 const& dst);

	void map_gray(View1r32 const& src, ViewGray const& dst);

	void map_gray(ViewYUV const& src, View1r32 const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAr32 const& dst);

	void map_rgba(ViewRGBAr32 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBr32 const& dst);

	void map_rgb(ViewRGBr32 const& src, View const& dst);

	void map_rgb(View1r32 const& src, View const& dst);
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVr32 const& dst);

	void map_hsv_rgb(ViewHSVr32 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst);	

	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst);
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHr32 const& dst);

	void map_lch_rgb(ViewLCHr32 const& src, View const& dst);

	void map_rgb_lch(ViewRGBr32 const& src, ViewLCHr32 const& dst);

	void map_lch_rgb(ViewLCHr32 const& src, ViewRGBr32 const& dst);
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst);	
}


/* map_bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBr32 const& dst);
}


/* sub_view */

namespace simage
{
	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel);

	View1r32 select_channel(ViewRGBr32 const& view, RGB channel);

	View1r32 select_channel(ViewHSVr32 const& view, HSV channel);

	View1r32 select_channel(View2r32 const& view, GA channel);

	View1r32 select_channel(View2r32 const& view, XY channel);


	ViewRGBr32 select_rgb(ViewRGBAr32 const& view);
}


/* fill */

namespace simage
{
	void fill(View4r32 const& view, Pixel color);

	void fill(View3r32 const& view, Pixel color);

	void fill(View1r32 const& view, u8 gray);
}


/* transform */

namespace simage
{
	void transform(View1r32 const& src, View1r32 const& dst, std::function<r32(r32)> const& func);

	void transform(View2r32 const& src, View1r32 const& dst, std::function<r32(r32, r32)> const& func);

	void transform(View3r32 const& src, View1r32 const& dst, std::function<r32(r32, r32, r32)> const& func);

	
	inline void transform_gray(ViewRGBr32 const& src, View1r32 const& dst)
	{
		return transform(src, dst, [](r32 red, r32 green, r32 blue) { return 0.299f * red + 0.587f * green + 0.114f * blue; });
	}
}


/* shrink */

namespace simage
{
	void shrink(View1r32 const& src, View1r32 const& dst);

	void shrink(View3r32 const& src, View3r32 const& dst);

	void shrink(ViewGray const& src, View1r32 const& dst);

	void shrink(View const& src, ViewRGBr32 const& dst);
}


/* gradients */

namespace simage
{
	void gradients_xy(View1r32 const& src, View2r32 const& xy_dst);
}


/* blur */

namespace simage
{
	void blur(View1r32 const& src, View1r32 const& dst);

	void blur(View3r32 const& src, View3r32 const& dst);
}
