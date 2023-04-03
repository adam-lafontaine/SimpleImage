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
	using ViewCHu16 = ChannelView2D<u16, N>;

	template <size_t N>
	using ViewCHu16 = ChannelView2D<u16, N>;
	

	template <typename T>
	using View4 = ChannelView2D<T, 4>;

	template <typename T>
	using View3 = ChannelView2D<T, 3>;

	template <typename T>
	using View2 = ChannelView2D<T, 2>;

	template <typename T>
	using View1 = MatrixView<T>;

	using View4u16 = View4<u16>;
	using View3u16 = View3<u16>;
	using View2u16 = View2<u16>;
	using View1u16 = View1<u16>;

	using View1u8 = ViewGray;
	
	using Buffer16 = MemoryBuffer<u16>;

	using ViewRGBAu16 = View4u16;
	using ViewRGBu16 = View3u16;
	using ViewHSVu16 = View3u16;
	using ViewLCHu16 = View3u16;	
}


/* reinterpret_view */
#if 0

namespace simage
{
	template <typename ST, typename DT, size_t N>
	ChannelView2D<DT, N> reinterpret_view(ChannelView2D<ST, N> const& src, DT n)
	{
		static_assert(sizeof(ST) == sizeof(DT));

		ChannelView2D<DT, N> dst;

		dst.channel_width_ = src.channel_width_;
		dst.width = src.width;
		dst.height = src.height;
		dst.range = src.range;

		for (u32 ch = 0; ch < N; ++ch)
		{
			dst.channel_data_[ch] = (DT*)src.channel_data_[ch];
		}

		return dst;
	}


	template <typename ST, typename DT>
	View1<DT> reinterpret_view(View1<ST> const& src)
	{
		static_assert(sizeof(ST) == sizeof(DT));

		View1<DT> dst;

		dst.width = src.width;
		dst.height = src.height;
		dst.range = src.range;

		dst.matrix_data_ = (DT*)src.matrix_data_;

		return dst;
	}

}
#endif


/* make_view */

namespace simage
{
	View1u16 make_view_1(u32 width, u32 height, Buffer16& buffer);

	View2u16 make_view_2(u32 width, u32 height, Buffer16& buffer);

	View3u16 make_view_3(u32 width, u32 height, Buffer16& buffer);

	View4u16 make_view_4(u32 width, u32 height, Buffer16& buffer);
}


/* sub_view */

namespace simage
{
	View4u16 sub_view(View4u16 const& view, Range2Du32 const& range);

	View3u16 sub_view(View3u16 const& view, Range2Du32 const& range);

	View2u16 sub_view(View2u16 const& view, Range2Du32 const& range);

	View1u16 sub_view(View1u16 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1u16 select_channel(ViewRGBAu16 const& view, RGBA channel);

	View1u16 select_channel(ViewRGBu16 const& view, RGB channel);

	View1u16 select_channel(ViewHSVu16 const& view, HSV channel);

	View1u16 select_channel(View2u16 const& view, GA channel);

	View1u16 select_channel(View2u16 const& view, XY channel);


	ViewRGBu16 select_rgb(ViewRGBAu16 const& view);

}


/* map_gray */

namespace simage
{
	void map_gray(View1u8 const& src, View1u16 const& dst);

	void map_gray(View1u16 const& src, View1u8 const& dst);

	void map_gray(ViewYUV const& src, View1u16 const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAu16 const& dst);

	void map_rgba(ViewRGBAu16 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBu16 const& dst);

	void map_rgb(ViewRGBu16 const& src, View const& dst);

	void map_rgb(View1u16 const& src, View const& dst);


	inline void map_rgba(Image const& src, ViewRGBAu16 const& dst)
	{
		map_rgba(make_view(src), dst);
	}


	inline void map_rgb(Image const& src, ViewRGBu16 const& dst)
	{
		map_rgb(make_view(src), dst);
	}


	inline void map_rgb(ViewRGBu16 const& src, Image const& dst)
	{
		map_rgb(src, make_view(dst));
	}
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVu16 const& dst);

	void map_hsv_rgb(ViewHSVu16 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBu16 const& src, ViewHSVu16 const& dst);	

	void map_hsv_rgb(ViewHSVu16 const& src, ViewRGBu16 const& dst);
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHu16 const& dst);

	void map_lch_rgb(ViewLCHu16 const& src, View const& dst);

	void map_rgb_lch(ViewRGBu16 const& src, ViewLCHu16 const& dst);

	void map_lch_rgb(ViewLCHu16 const& src, ViewRGBu16 const& dst);
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBu16 const& dst);	
}


/* map_bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBu16 const& dst);
}


/* fill */

namespace simage
{
	void fill(View4u16 const& view, Pixel color);

	void fill(View3u16 const& view, Pixel color);

	void fill(View1u16 const& view, u8 gray);
}


/* transform */

namespace simage
{
	void transform(View1u16 const& src, View1u16 const& dst, std::function<f32(f32)> const& func32);

	void transform(View2u16 const& src, View1u16 const& dst, std::function<f32(f32, f32)> const& func32);

	void transform(View3u16 const& src, View1u16 const& dst, std::function<f32(f32, f32, f32)> const& func32);

	
	inline void transform_gray(ViewRGBu16 const& src, View1u16 const& dst)
	{
		return transform(src, dst, [](f32 red, f32 green, f32 blue) { return 0.299f * red + 0.587f * green + 0.114f * blue; });
	}


	void threshold(View1u16 const& src, View1u16 const& dst, f32 min);

	void threshold(View1u16 const& src, View1u16 const& dst, f32 min, f32 max);


	
}


/* alpha blend */

namespace simage
{
	void alpha_blend(ViewRGBAu16 const& src, ViewRGBu16 const& cur, ViewRGBu16 const& dst);
}


/* shrink */
/*
namespace simage
{
	void shrink(View1u16 const& src, View1u16 const& dst);

	void shrink(View3u16 const& src, View3u16 const& dst);

	void shrink(View1u8 const& src, View1u16 const& dst);

	void shrink(View const& src, ViewRGBu16 const& dst);
}
*/

/* gradients */

namespace simage
{
	void gradients_xy(View1u16 const& src, View2u16 const& xy_dst);
}


/* blur */

namespace simage
{
	void blur(View1u16 const& src, View1u16 const& dst);

	void blur(View3u16 const& src, View3u16 const& dst);
}
