#include "simage.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"
#define SIMAGE_NO_SIMD

#ifndef SIMAGE_NO_SIMD
#include "../util/simd.hpp"
#endif // !SIMAGE_NO_SIMD


#include <cmath>
#include <algorithm>

namespace cs = color_space;
namespace rng = std::ranges;



static void process_by_row(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
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
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data_;
	}


	template <typename T>
	static bool verify(MemoryBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}


	template <typename T, size_t N>
	static bool verify(ChannelView2D<T,N> const& view)
	{
		return view.channel_width_ && view.width && view.height && view.channel_data_[0];
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

#endif // !NDEBUG
}


/* pixels */

namespace simage
{
	template <typename T>
	class RGBp
	{
	public:
		T* R;
		T* G;
		T* B;
	};


	template <typename T>
	class RGBAp
	{
	public:
		T* R;
		T* G;
		T* B;
		T* A;
	};


	template <typename T>
	class HSVp
	{
	public:
		T* H;
		T* S;
		T* V;
	};


	template <typename T>
	class YUVp
	{
	public:
		T* Y;
		T* UV;
	};	


	template <typename T>
	class UVYp
	{
	public:
		T* UV;
		T* Y;
	};


	template <typename T>
	class LCHp
	{
	public:
		T* L;
		T* C;
		T* H;
	};


	using RGBu16p = RGBp<u16>;
	using RGBAu16p = RGBAp<u16>;

	using HSVu16p = HSVp<u16>;
	using YUVu16p = YUVp<u16>;
	using LCHu16p = LCHp<u16>;
}


/* row begin */

namespace simage
{
	static RGBu16p rgb_row_begin(ViewRGBu16 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		RGBu16p rgb{};

		rgb.R = view.channel_data_[id_cast(RGB::R)] + offset;
		rgb.G = view.channel_data_[id_cast(RGB::G)] + offset;
		rgb.B = view.channel_data_[id_cast(RGB::B)] + offset;

		return rgb;
	}


	static RGBAu16p rgba_row_begin(ViewRGBAu16 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		RGBAu16p rgba{};

		rgba.R = view.channel_data_[id_cast(RGBA::R)] + offset;
		rgba.G = view.channel_data_[id_cast(RGBA::G)] + offset;
		rgba.B = view.channel_data_[id_cast(RGBA::B)] + offset;
		rgba.A = view.channel_data_[id_cast(RGBA::A)] + offset;

		return rgba;
	}


	static HSVu16p hsv_row_begin(ViewHSVu16 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		HSVu16p hsv{};

		hsv.H = view.channel_data_[id_cast(HSV::H)] + offset;
		hsv.S = view.channel_data_[id_cast(HSV::S)] + offset;
		hsv.V = view.channel_data_[id_cast(HSV::V)] + offset;

		return hsv;
	}


	static LCHu16p lch_row_begin(ViewLCHu16 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		LCHu16p lch{};

		lch.L = view.channel_data_[id_cast(LCH::L)] + offset;
		lch.C = view.channel_data_[id_cast(LCH::C)] + offset;
		lch.H = view.channel_data_[id_cast(LCH::H)] + offset;

		return lch;
	}


	template <typename T>
	static T* row_offset_begin(View1<T> const& view, u32 y, int y_offset)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.matrix_width + view.x_begin);

		return view.matrix_data_ + offset;
	}


	template <typename T, size_t N>
	static T* channel_row_begin(ChannelView2D<T, N> const& view, u32 y, u32 ch)
	{
		assert(verify(view));

		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.channel_width_ + view.x_begin);

		return view.channel_data_[ch] + offset;
	}


	template <typename T, size_t N>
	static T* channel_row_offset_begin(ChannelView2D<T, N> const& view, u32 y, int y_offset, u32 ch)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.channel_width_ + view.x_begin);

		return view.channel_data_[ch] + offset;
	}

}


/* make view */

namespace simage
{
	template <typename T, size_t N>
	static void do_make_view_n(ChannelView2D<T, N>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		view.channel_width_ = width;
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.channel_data_[ch] = mb::push_elements(buffer, width * height);
		}
	}


	template <typename T>
	static void do_make_view_1(View1<T>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		view.matrix_data_ = mb::push_elements(buffer, width * height);
		view.matrix_width = width;		
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);
	}


	View1u16 make_view_1(u32 width, u32 height, Buffer16& buffer)
	{
		assert(verify(buffer, width * height));

		View1u16 view;

		do_make_view_1(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View2u16 make_view_2(u32 width, u32 height, Buffer16& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2u16 view;

		do_make_view_n(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View3u16 make_view_3(u32 width, u32 height, Buffer16& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3u16 view;

		do_make_view_n(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View4u16 make_view_4(u32 width, u32 height, Buffer16& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4u16 view;

		do_make_view_n(view, width, height, buffer);

		assert(verify(view));

		return view;
	}
}


/* sub_view */

namespace simage
{
	template <typename T, size_t N>
	static ChannelView2D<T, N> do_sub_view(ChannelView2D<T, N> const& view, Range2Du32 const& range)
	{
		ChannelView2D<T, N> sub_view;

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


	template <typename T>
	View1<T> do_sub_view(View1<T> const& view, Range2Du32 const& range)
	{
		View1<T> sub_view;

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


	View4u16 sub_view(View4u16 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3u16 sub_view(View3u16 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View2u16 sub_view(View2u16 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View1u16 sub_view(View1u16 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}
}


/* select_channel */

namespace simage
{
	template <typename T, size_t N, typename CH>
	static View1<T> select_channel(ChannelView2D<T, N> const& view, CH ch)
	{
		View1<T> view1{};

		view1.matrix_width = view.channel_width_;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.matrix_data_ = view.channel_data_[id_cast(ch)];

		return view1;
	}


	View1u16 select_channel(ViewRGBAu16 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1u16 select_channel(ViewRGBu16 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1u16 select_channel(ViewHSVu16 const& view, HSV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1u16 select_channel(View2u16 const& view, GA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1u16 select_channel(View2u16 const& view, XY channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	ViewRGBu16 select_rgb(ViewRGBAu16 const& view)
	{
		assert(verify(view));

		ViewRGBu16 rgb;

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


/* map */

namespace simage
{
	static void map_channel_row_u8_to_u16(u8* src, u16* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_u16(src[x]);
		}
	}


	static void map_channel_row_u16_to_u8(u16* src, u8* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_u8(src[x]);
		}
	}


	void map_gray(View1u8 const& src, View1u16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_u8_to_u16(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
	

	void map_gray(View1u16 const& src, View1u8 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_u16_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_gray(ViewYUV const& src, View1u16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_u16(s[x].y);
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* map_rgb */

namespace simage
{	
	static void map_rgba_row_u8_to_u16(Pixel* src, RGBAu16p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_u16(src[x].rgba.red);
			dst.G[x] = cs::to_channel_u16(src[x].rgba.green);
			dst.B[x] = cs::to_channel_u16(src[x].rgba.blue);
			dst.A[x] = cs::to_channel_u16(src[x].rgba.alpha);
		}
	}


	static void map_rgba_row_u16_to_u8(RGBAu16p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = cs::to_channel_u8(src.A[x]);
		}
	}


	static void map_rgb_row_u8_to_u16(Pixel* src, RGBu16p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_u16(src[x].rgba.red);
			dst.G[x] = cs::to_channel_u16(src[x].rgba.green);
			dst.B[x] = cs::to_channel_u16(src[x].rgba.blue);
		}
	}


	static void map_rgb_row_u16_to_u8(RGBu16p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = 255;
		}
	}


	void map_rgba(View const& src, ViewRGBAu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgba_row_begin(dst, y);

			map_rgba_row_u8_to_u16(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_rgba(ViewRGBAu16 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgba_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgba_row_u16_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}

	
	void map_rgb(View const& src, ViewRGBu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			map_rgb_row_u8_to_u16(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_rgb(ViewRGBu16 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgb_row_u16_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_rgb(View1u16 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const gray = cs::to_channel_u8(s[x]);

				d[x].rgba.red = gray;
				d[x].rgba.green = gray;
				d[x].rgba.blue = gray;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* map_hsv */

namespace simage
{	
	void map_rgb_hsv(View const& src, ViewHSVu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::u16_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVu16 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = hsv::u16_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_rgb_hsv(ViewRGBu16 const& src, ViewHSVu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto hsv = hsv::u16_from_rgb_u16(r, g, b);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVu16 const& src, ViewRGBu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::u16_to_rgb_u16(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto lch = lch::u16_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHu16 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = lch::u16_to_rgb_u8(s.L[x], s.C[x], s.H[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_rgb_lch(ViewRGBu16 const& src, ViewLCHu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto lch = lch::u16_from_rgb_u16(r, g, b);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_by_row(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHu16 const& src, ViewRGBu16 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = lch::u16_to_rgb_u16(s.L[x], s.C[x], s.H[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBu16 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		auto const row_func = [&](u32 y)
		{
			auto s422 = (YUV422u8*)row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgb = yuv::u8_to_rgb_u16(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::u8_to_rgb_u16(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_by_row(src.height, row_func);
	}


	void mipmap_yuv_rgb(ViewYUV const& src, ViewRGBu16 const& dst)
	{		
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d) 
		{
			auto val = 0.25f * ((u16)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((u16)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		auto const row_func = [&](u32 y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgb = yuv::u8_to_rgb_u16(y_avg, u_avg, v_avg);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_by_row(dst.height, row_func);
	}


	void map_yuv_rgb2(ViewYUV const& src, View const& dst)
	{
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d)
		{
			auto val = 0.25f * ((u16)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((u16)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		auto const row_func = [&](u32 y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgba = yuv::u8_to_rgb_u8(y_avg, u_avg, v_avg);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(dst.height, row_func);
	}
}


/* map bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBu16 const& dst)
	{		
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d.R[x] = cs::to_channel_u16(s[x].red);
				d.G[x] = cs::to_channel_u16(s[x].green);
				d.B[x] = cs::to_channel_u16(s[x].blue);
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* fill */

namespace simage
{
	template <typename T>
	static void fill_channel_row(T* d, T grayT, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			d[i] = grayT;
		}
	}


	template <typename T>
	static void fill_channel(View1<T> const& view, T grayT)
	{		
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_channel_row(d, grayT, view.width);
		};

		process_by_row(view.height, row_func);
	}


	template <size_t N>
	static void fill_n_channels(ViewCHu16<N> const& view, Pixel color)
	{
		u16 channels[N] = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			channels[ch] = cs::to_channel_u16(color.channels[ch]);
		}

		auto const row_func = [&](u32 y)
		{
			for (u32 ch = 0; ch < N; ++ch)
			{
				auto d = channel_row_begin(view, y, ch);
				fill_channel_row(d, channels[ch], view.width);
			}
		};

		process_by_row(view.height, row_func);
	}


	void fill(View4u16 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3u16 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1u16 const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, cs::to_channel_u16(gray));
	}
}


/* transform */

namespace simage
{
	void transform(View1u16 const& src, View1u16 const& dst, std::function<f32(f32)> const& func32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto p32 = cs::to_channel_f32(s[x]);
				d[x] = cs::to_channel_u16(func32(p32));
			}
		};

		process_by_row(src.height, row_func);
	}


	void transform(View2u16 const& src, View1u16 const& dst, std::function<f32(f32, f32)> const& func32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto a = cs::to_channel_f32(s0[x]);
				auto b = cs::to_channel_f32(s1[x]);

				d[x] = cs::to_channel_u16(func32(a, b));
			}
		};

		process_by_row(src.height, row_func);
	}


	void transform(View3u16 const& src, View1u16 const& dst, std::function<f32(f32, f32, f32)> const& func32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto s2 = channel_row_begin(src, y, 2);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto a = cs::to_channel_f32(s0[x]);
				auto b = cs::to_channel_f32(s1[x]);
				auto c = cs::to_channel_f32(s2[x]);

				d[x] = cs::to_channel_u16(func32(a, b, c));
			}
		};

		process_by_row(src.height, row_func);
	}


	void threshold(View1u16 const& src, View1u16 const& dst, u8 min)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min ? s[x] : 0;
			}
		};

		process_by_row(src.height, row_func);
	}


	void threshold(View1u16 const& src, View1u16 const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));

		auto mn = std::min(min, max);
		auto mx = std::max(min, max);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= mn && s[x] <= mx ? s[x] : 0;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* alpha blend */

namespace simage
{
	static void alpha_blend_row(RGBAu16p const& src, RGBu16p const& cur, RGBu16p const& dst, u32 width)
	{
		// TODO: simd here

		auto const blend = [](u16 s, u16 c, f32 a)
		{
			auto blended = a * s + (1.0f - a) * c;
			return (u16)(blended + 0.5f);
		};

		for (u32 x = 0; x < width; ++x)
		{
			auto a = cs::to_channel_f32(src.A[x]);

			dst.R[x] = blend(src.R[x], cur.R[x], a);
			dst.G[x] = blend(src.G[x], cur.G[x], a);
			dst.B[x] = blend(src.B[x], cur.B[x], a);
		}
	}


	void alpha_blend(ViewRGBAu16 const& src, ViewRGBu16 const& cur, ViewRGBu16 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const row_func = [&](u32 y)
		{			
			auto s = rgba_row_begin(src, y);
			auto c = rgb_row_begin(cur, y);
			auto d = rgb_row_begin(dst, y);

			alpha_blend_row(s, c, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
}


/* shrink_view */
#if 0
namespace simage
{
	static u16 average(View1u16 const& view)
	{
		u16 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += s[x];
			}
		}

		return total / (view.width * view.height);
	}


	static u16 average(ViewGray const& view)
	{
		u16 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += cs::to_channel_u16(s[x]);
			}
		}

		return total / (view.width * view.height);
	}


	template <size_t N>
	static std::array<f32, N> average(ViewCHu16<N> const& view)
	{
		std::array<f32, N> results = { 0 };
		for (u32 i = 0; i < N; ++i) { results[i] = 0.0f; }

		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 i = 0; i < N; ++i)
			{
				auto s = channel_row_begin(view, y, i);

				for (u32 x = 0; x < view.width; ++x)
				{
					results[i] += s[x];
				}
			}
		}

		for (u32 i = 0; i < N; ++i)
		{
			results[i] /= (view.width * view.height);
		}

		return results;
	}
	

	static cs::RGBu16 average(View const& view)
	{	
		u16 red = 0.0f;
		u16 green = 0.0f;
		u16 blue = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				auto p = s[x].rgba;
				red += cs::to_channel_u16(p.red);
				green += cs::to_channel_u16(p.green);
				blue += cs::to_channel_u16(p.blue);
			}
		}

		red /= (view.width * view.height);
		green /= (view.width * view.height);
		blue /= (view.width * view.height);

		return { red, green, blue };
	}


	template <class VIEW>
	static void do_shrink_1D(VIEW const& src, View1u16 const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;
				
				d[x] = average(sub_view(src, r));
			}
		};

		process_by_row(dst.height, row_func);
	}


	void shrink(View1u16 const& src, View1u16 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View3u16 const& src, View3u16 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		auto const row_func = [&](u32 y)
		{
			auto d0 = channel_row_begin(dst, y, 0);
			auto d1 = channel_row_begin(dst, y, 1);
			auto d2 = channel_row_begin(dst, y, 2);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;

				auto avg = average(sub_view(src, r));

				d0[x] = avg[0];
				d1[x] = avg[1];
				d2[x] = avg[2];
			}
		};

		process_by_row(dst.height, row_func);
	}


	void shrink(ViewGray const& src, View1u16 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View const& src, ViewRGBu16 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		auto const row_func = [&](u32 y)
		{
			auto d = rgb_row_begin(dst, y);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;

				auto avg = average(sub_view(src, r));
				d.R[x] = avg.red;
				d.G[x] = avg.green;
				d.B[x] = avg.blue;				
			}
		};

		process_by_row(dst.height, row_func);
	}
}
#endif

/* gradients */

namespace simage
{
	static void convolve(View1u16 const& src, View1u16 const& dst, Mat2Df32 const& kernel)
	{
		assert(kernel.width % 2 > 0);
		assert(kernel.height % 2 > 0);

		int const ry_begin = -(int)kernel.height / 2;
		int const ry_end = kernel.height / 2 + 1;
		int const rx_begin = -(int)kernel.width / 2;
		int const rx_end = kernel.width / 2 + 1;

		auto const row_func = [&](u32 y) 
		{			
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u32 w = 0;
				f32 total = 0.0f;
				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s = row_offset_begin(src, y, ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						total += (s + rx)[x] * kernel.data_[w];						
						++w;
					}

					d[x] = (u16)std::abs(total); // (i16)(total / 2);
				}
			}
		};

		process_by_row(src.height, row_func);
	}


	constexpr std::array<f32, 33> make_grad_x_11()
	{
		/*constexpr std::array<f32, 33> GRAD_X
		{
			-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
			-0.06f, -0.09f, -0.12f, -0.15f, -0.18f, 0.0f, 0.18f, 0.15f, 0.12f, 0.09f, 0.06f,
			-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.01f,
		};*/

		std::array<f32, 33> grad = { 0 };

		f32 values[] = { 0.08f, 0.06f, 0.04f, 0.02f, 0.01f };

		size_t w = 11;

		for (size_t i = 0; i < 5; ++i)
		{
			grad[6 + i] = values[i];
			grad[2 * w + 6 + i] = values[i];
			grad[w + 6 + i] = 3 * values[i];
			grad[4 - i] = -values[i];
			grad[2 * w + 4 - i] = -values[i];
			grad[w + 4 - i] = -3 * values[i];
		}

		return grad;
	}


	constexpr std::array<f32, 33> make_grad_y_11()
	{
		/*constexpr std::array<f32, 33> GRAD_Y
		{
			-0.02f, -0.06f, -0.02f,
			-0.03f, -0.09f, -0.03f,
			-0.04f, -0.12f, -0.04f,
			-0.05f, -0.15f, -0.05f,
			-0.06f, -0.18f, -0.06f,
			 0.00f,  0.00f,  0.00f,
			 0.06f,  0.18f,  0.06f,
			 0.05f,  0.15f,  0.05f,
			 0.04f,  0.12f,  0.04f,
			 0.03f,  0.09f,  0.03f,
			 0.02f,  0.06f,  0.02f,
		};*/

		std::array<f32, 33> grad = { 0 };

		f32 values[] = { 0.08f, 0.06f, 0.04f, 0.02f, 0.01f };

		size_t w = 3;

		for (size_t i = 0; i < 5; ++i)
		{
			grad[w * (6 + i)] = values[i];
			grad[w * (6 + i) + 2] = values[i];
			grad[w * (6 + i) + 1] = 3 * values[i];
			grad[w * (4 - i)] = -values[i];
			grad[w * (4 - i) + 2] = -values[i];
			grad[w * (4 - i) + 1] = -3 * values[i];
		}

		return grad;
	}


	constexpr std::array<f32, 15> make_grad_x_5()
	{
		std::array<f32, 15> grad
		{
			-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
			-0.16f, -0.24f, 0.0f, 0.24f, 0.16f
			-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
		};

		return grad;
	}


	constexpr std::array<f32, 15> make_grad_y_5()
	{
		std::array<f32, 15> grad
		{
			-0.08f, -0.16f, -0.08f,
			-0.12f, -0.24f, -0.12f,
			 0.00f,  0.00f,  0.00f,
			 0.12f,  0.24f,  0.12f,
			 0.08f,  0.16f,  0.08f,
		};

		return grad;
	}


	template <typename T>
	static void zero_outer(View1<T> const& view, u32 n_rows, u32 n_columns)
	{
		auto const top_bottom = [&]()
		{
			for (u32 r = 0; r < n_rows; ++r)
			{
				auto top = row_begin(view, r);
				auto bottom = row_begin(view, view.height - 1 - r);
				for (u32 x = 0; x < view.width; ++x)
				{
					top[x] = bottom[x] = (T)0;
				}
			}
		};

		auto const left_right = [&]()
		{
			for (u32 y = n_rows; y < view.height - n_rows; ++y)
			{
				auto row = row_begin(view, y);
				for (u32 c = 0; c < n_columns; ++c)
				{
					row[c] = row[view.width - 1 - c] = (T)0;
				}
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		execute(f_list);
	}

	
	static void gradients_x(View1u16 const& src, View1u16 const& dst)
	{
		constexpr auto grad_y = make_grad_x_11();
		constexpr u32 dim_max = 11;
		constexpr u32 dim_min = (u32)grad_y.size() / dim_max;

		Mat2Df32 kernel{};
		kernel.data_ = (f32*)grad_y.data();
		kernel.width = dim_max;
		kernel.height = dim_min;

		auto w = kernel.width / 2;
		auto h = kernel.height / 2;

		zero_outer(dst, w, h);

		Range2Du32 inner{};
		inner.x_begin = w;
		inner.x_end = src.width - w;
		inner.y_begin = h;
		inner.y_end = src.height - h;

		convolve(sub_view(src, inner), sub_view(dst, inner), kernel);
	}


	static void gradients_y(View1u16 const& src, View1u16 const& dst)
	{
		constexpr auto grad_y = make_grad_y_11();
		constexpr u32 dim_max = 11;
		constexpr u32 dim_min = (u32)grad_y.size() / dim_max;

		Mat2Df32 kernel{};
		kernel.data_ = (f32*)grad_y.data();
		kernel.width = dim_min;
		kernel.height = dim_max;

		auto w = kernel.width / 2;
		auto h = kernel.height / 2;

		zero_outer(dst, w, h);

		Range2Du32 inner{};
		inner.x_begin = w;
		inner.x_end = src.width - w;
		inner.y_begin = h;
		inner.y_end = src.height - h;

		convolve(sub_view(src, inner), sub_view(dst, inner), kernel);
	}


	void gradients_xy(View1u16 const& src, View2u16 const& xy_dst)
	{
		auto x_dst = select_channel(xy_dst, XY::X);
		auto y_dst = select_channel(xy_dst, XY::Y);

		assert(verify(src, x_dst));
		assert(verify(src, y_dst));

		gradients_x(src, x_dst);
		gradients_y(src, y_dst);
	}
}


/* blur */

namespace simage
{
	static constexpr std::array<f32, 9> make_gauss_3()
	{
		auto D3 = 16.0f;
		std::array<f32, 9> kernel = 
		{
			1.0f, 2.0f, 1.0f,
			2.0f, 4.0f, 2.0f,
			1.0f, 2.0f, 1.0f,
		};

		rng::for_each(kernel, [D3](f32& v) { v /= D3; });

		return kernel;
	}


	static constexpr std::array<f32, 25> make_gauss_5()
	{
		auto D5 = 256.0f;
		std::array<f32, 25> kernel =
		{
			1.0f, 4.0f,  6.0f,  4.0f,  1.0f,
			4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
			6.0f, 24.0f, 36.0f, 24.0f, 6.0f,
			4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
			1.0f, 4.0f,  6.0f,  4.0f,  1.0f,
		};

		rng::for_each(kernel, [D5](f32& v) { v /= D5; });

		return kernel;
	}


	template <typename T>
	static void copy_outer(View1<T> const& src, View1<T> const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const top_bottom = [&]()
		{
			auto s_top = row_begin(src, 0);
			auto s_bottom = row_begin(src, height - 1);
			auto d_top = row_begin(dst, 0);
			auto d_bottom = row_begin(dst, height - 1);
			for (u32 x = 0; x < width; ++x)
			{
				d_top[x] = s_top[x];
				d_bottom[x] = s_bottom[x];
			}
		};

		auto const left_right = [&]()
		{
			for (u32 y = 1; y < height - 1; ++y)
			{
				auto s_row = row_begin(src, y);
				auto d_row = row_begin(dst, y);

				d_row[0] = s_row[0];
				d_row[width - 1] = s_row[width - 1];
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		execute(f_list);
	}


	template <typename T>
	static void convolve_gauss_3x3_outer(View1<T> const& src, View1<T> const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto constexpr gauss = make_gauss_3();

		Mat2Df32 kernel{};
		kernel.width = 3;
		kernel.height = 3;
		kernel.data_ = (f32*)gauss.data();

		Range2Du32 top{};
		top.x_begin = 0;
		top.x_end = width;
		top.y_begin = 0;
		top.y_end = 1;

		auto bottom = top;
		bottom.y_begin = height - 1;
		bottom.y_end = height;

		auto left = top;
		left.x_end = 1;
		left.y_begin = 1;
		left.y_end = height - 1;

		auto right = left;
		right.x_begin = width - 1;
		right.x_end = width;

		convolve(sub_view(src, top), sub_view(dst, top), kernel);
		convolve(sub_view(src, bottom), sub_view(dst, bottom), kernel);
		convolve(sub_view(src, left), sub_view(dst, left), kernel);
		convolve(sub_view(src, right), sub_view(dst, right), kernel);			
	}


	template <typename T>
	static void convolve_gauss_5x5(View1<T> const& src, View1<T> const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto constexpr gauss = make_gauss_5();

		Mat2Df32 kernel{};
		kernel.width = 5;
		kernel.height = 5;
		kernel.data_ = (f32*)gauss.data();

		convolve(src, dst, kernel);
	}


	template <typename T>
	static void blur_1(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
		auto const height = src.height;

		copy_outer(src, dst);

		Range2Du32 r{};
		r.x_begin = 1;
		r.x_end = width - 1;
		r.y_begin = 1;
		r.y_end = height - 1;

		convolve_gauss_3x3_outer(sub_view(src, r), sub_view(dst, r));

		r.x_begin = 2;
		r.x_end = width - 2;
		r.y_begin = 2;
		r.y_end = height - 2;

		convolve_gauss_5x5(sub_view(src, r), sub_view(dst, r));
	}


	template <typename T, size_t N>
	static void blur_n(ChannelView2D<T, N> const& src, ChannelView2D<T, N> const& dst)
	{
		for (u32 ch = 0; ch < N; ++ch)
		{
			blur_1(select_channel(src, ch), select_channel(dst, ch));
		}
	}


	void blur(View1u16 const& src, View1u16 const& dst)
	{
		assert(verify(src, dst));

		blur_1(src, dst);
	}


	void blur(View3u16 const& src, View3u16 const& dst)
	{
		assert(verify(src, dst));

		blur_n(src, dst);
	}
}


