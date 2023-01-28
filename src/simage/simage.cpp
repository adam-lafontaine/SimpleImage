#include "simage.hpp"
#include "../util/memory_buffer.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"


#ifndef SIMAGE_NO_SIMD
#include "../util/simd.hpp"
#endif // !SIMAGE_NO_SIMD



#include <cmath>
#include <algorithm>


namespace mb = memory_buffer;
namespace cs = color_space;



static void process_image_rows(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
}


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <typename T>
	static bool verify(Matrix2D<T> const& image)
	{
		return image.width && image.height && image.data_;
	}


	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
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


/* pixels */

namespace simage
{
	template <size_t N>
	class PixelCHr32
	{
	public:

		static constexpr u32 n_channels = N;

		r32* channels[N] = {};
	};


	using Pixel4r32 = PixelCHr32<4>;
	using Pixel3r32 = PixelCHr32<3>;
	using Pixel2r32 = PixelCHr32<2>;


	class RGBr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
	};


	class HSVr32p
	{
	public:
		r32* H;
		r32* S;
		r32* V;
	};


	class LCHr32p
	{
	public:
		r32* L;
		r32* C;
		r32* H;
	};
}


/* row begin */

namespace simage
{
	template <size_t N>
	static PixelCHr32<N> row_begin(ViewCHr32<N> const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelCHr32<N> p{};

		for (u32 ch = 0; ch < N; ++ch)
		{
			p.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return p;
	}


	static RGBr32p rgb_row_begin(ViewRGBr32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		RGBr32p rgb{};

		rgb.R = view.image_channel_data[id_cast(RGB::R)] + offset;
		rgb.G = view.image_channel_data[id_cast(RGB::G)] + offset;
		rgb.B = view.image_channel_data[id_cast(RGB::B)] + offset;

		return rgb;
	}


	static HSVr32p hsv_row_begin(ViewHSVr32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		HSVr32p hsv{};

		hsv.H = view.image_channel_data[id_cast(HSV::H)] + offset;
		hsv.S = view.image_channel_data[id_cast(HSV::S)] + offset;
		hsv.V = view.image_channel_data[id_cast(HSV::V)] + offset;

		return hsv;
	}


	static LCHr32p lch_row_begin(ViewLCHr32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		LCHr32p lch{};

		lch.L = view.image_channel_data[id_cast(LCH::L)] + offset;
		lch.C = view.image_channel_data[id_cast(LCH::C)] + offset;
		lch.H = view.image_channel_data[id_cast(LCH::H)] + offset;

		return lch;
	}


	static r32* row_offset_begin(View1r32 const& view, u32 y, int y_offset)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.image_width + view.x_begin);

		return view.image_data + offset;
	}


	template <size_t N>
	static r32* channel_row_begin(ViewCHr32<N> const& view, u32 y, u32 ch)
	{
		assert(verify(view));

		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


	template <size_t N>
	static r32* channel_row_offset_begin(ViewCHr32<N> const& view, u32 y, int y_offset, u32 ch)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}

}


/* xy_at */

namespace simage
{
	template <size_t N>
	static PixelCHr32<N> xy_at(ViewCHr32<N> const& view, u32 x, u32 y)
	{
		assert(verify(view));
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
}


/* make view */

namespace simage
{
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


	View1r32 make_view_1(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View1r32 view;

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


/* map */

namespace simage
{
	void map(ViewGray const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_r32(s[x]);
			}
		};

		process_image_rows(src.height, row_func);
	}
	

	void map(View1r32 const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_u8(s[x]);
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map(ViewYUV const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_r32(s[x].y);
			}
		};

		process_image_rows(src.height, row_func);
	}
}


/* map_rgb */

namespace simage
{	
	static void map_rgba_no_simd(View const& src, ViewRGBAr32 const& dst)
	{
		constexpr auto r = id_cast(RGBA::R);
		constexpr auto g = id_cast(RGBA::G);
		constexpr auto b = id_cast(RGBA::B);
		constexpr auto a = id_cast(RGBA::A);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto dr = channel_row_begin(dst, y, r);
			auto dg = channel_row_begin(dst, y, g);
			auto db = channel_row_begin(dst, y, b);
			auto da = channel_row_begin(dst, y, a);

			for (u32 x = 0; x < src.width; ++x) // TODO: simd
			{
				dr[x] = cs::to_channel_r32(s[x].channels[r]);
				dg[x] = cs::to_channel_r32(s[x].channels[g]);
				db[x] = cs::to_channel_r32(s[x].channels[b]);
				da[x] = cs::to_channel_r32(s[x].channels[a]);
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_rgb_no_simd(View const& src, ViewRGBr32 const& dst)
	{
		constexpr auto r = id_cast(RGB::R);
		constexpr auto g = id_cast(RGB::G);
		constexpr auto b = id_cast(RGB::B);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto dr = channel_row_begin(dst, y, r);
			auto dg = channel_row_begin(dst, y, g);
			auto db = channel_row_begin(dst, y, b);

			for (u32 x = 0; x < src.width; ++x) // TODO: simd
			{
				dr[x] = cs::to_channel_r32(s[x].channels[r]);
				dg[x] = cs::to_channel_r32(s[x].channels[g]);
				db[x] = cs::to_channel_r32(s[x].channels[b]);
			}
		};

		process_image_rows(src.height, row_func);
	}


#ifdef SIMAGE_NO_SIMD

	static void do_map_rgba(View const& src, ViewRGBAr32 const& dst)
	{
		map_rgba_no_simd(src, dst);
	}


	static void do_map_rgb(View const& src, ViewRGBr32 const& dst)
	{
		map_rgb_no_simd(src, dst);
}

#else

	class Pixelr32Planar
	{
	public:
		r32 red[simd::VEC_LEN] = { 0 };
		r32 green[simd::VEC_LEN] = { 0 };
		r32 blue[simd::VEC_LEN] = { 0 };
		r32 alpha[simd::VEC_LEN] = { 0 };
	};


	Pixelr32Planar to_planar(Pixel* p_begin)
	{
		Pixelr32Planar planar;

		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			auto rgba = p_begin[i].rgba;

			planar.red[i] = cs::to_channel_r32(rgba.red);
			planar.green[i] = cs::to_channel_r32(rgba.green);
			planar.blue[i] = cs::to_channel_r32(rgba.blue);
			planar.alpha[i] = cs::to_channel_r32(rgba.alpha);
		}

		return planar;
	}


	static void map_rgba_row_simd(Pixel* src, r32* dr, r32* dg, r32* db, r32* da, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		auto const do_simd = [&](u32 i)
		{
			auto p = to_planar(src + i);

			auto vec = simd::load(p.red);
			simd::store(dr + i, vec);

			vec = simd::load(p.green);
			simd::store(dg + i, vec);

			vec = simd::load(p.blue);
			simd::store(db + i, vec);

			vec = simd::load(p.alpha);
			simd::store(da + i, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void map_rgb_row_simd(Pixel* src, r32* dr, r32* dg, r32* db, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		auto const do_simd = [&](u32 i)
		{
			auto p = to_planar(src + i);

			auto vec = simd::load(p.red);
			simd::store(dr + i, vec);

			vec = simd::load(p.green);
			simd::store(dg + i, vec);

			vec = simd::load(p.blue);
			simd::store(db + i, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void map_rgba_simd(View const& src, ViewRGBAr32 const& dst)
	{
		constexpr auto r = id_cast(RGBA::R);
		constexpr auto g = id_cast(RGBA::G);
		constexpr auto b = id_cast(RGBA::B);
		constexpr auto a = id_cast(RGBA::A);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto dr = channel_row_begin(dst, y, r);
			auto dg = channel_row_begin(dst, y, g);
			auto db = channel_row_begin(dst, y, b);
			auto da = channel_row_begin(dst, y, a);

			map_rgba_row_simd(s, dr, dg, db, da, src.width);
		};

		process_image_rows(src.height, row_func);
	}


	static void map_rgb_simd(View const& src, ViewRGBr32 const& dst)
	{
		constexpr auto r = id_cast(RGB::R);
		constexpr auto g = id_cast(RGB::G);
		constexpr auto b = id_cast(RGB::B);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto dr = channel_row_begin(dst, y, r);
			auto dg = channel_row_begin(dst, y, g);
			auto db = channel_row_begin(dst, y, b);

			map_rgb_row_simd(s, dr, dg, db, src.width);
		};

		process_image_rows(src.height, row_func);
	}


	static void do_map_rgba(View const& src, ViewRGBAr32 const& dst)
	{
		if (src.width < simd::VEC_LEN)
		{
			map_rgba_no_simd(src, dst);
		}
		else
		{
			map_rgba_simd(src, dst);
		}
	}


	static void do_map_rgb(View const& src, ViewRGBr32 const& dst)
	{
		if (src.width < simd::VEC_LEN)
		{
			map_rgb_no_simd(src, dst);
		}
		else
		{
			map_rgb_simd(src, dst);
		}
	}


#endif	


	void map_rgba(View const& src, ViewRGBAr32 const& dst)
	{
		assert(verify(src, dst));

		do_map_rgba(src, dst);
	}


	void map_rgba(ViewRGBAr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		constexpr auto r = id_cast(RGBA::R);
		constexpr auto g = id_cast(RGBA::G);
		constexpr auto b = id_cast(RGBA::B);
		constexpr auto a = id_cast(RGBA::A);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);
			auto sr = channel_row_begin(src, y, r);
			auto sg = channel_row_begin(src, y, g);
			auto sb = channel_row_begin(src, y, b);
			auto sa = channel_row_begin(src, y, a);

			for (u32 x = 0; x < src.width; ++x) // TODO: simd
			{
				d[x].channels[r] = cs::to_channel_u8(sr[x]);
				d[x].channels[g] = cs::to_channel_u8(sg[x]);
				d[x].channels[b] = cs::to_channel_u8(sb[x]);
				d[x].channels[a] = cs::to_channel_u8(sa[x]);
			}
		};

		process_image_rows(src.height, row_func);
	}

	
	void map_rgb(View const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));

		do_map_rgb(src, dst);
	}


	void map_rgb(ViewRGBr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		constexpr auto r = id_cast(RGBA::R);
		constexpr auto g = id_cast(RGBA::G);
		constexpr auto b = id_cast(RGBA::B);
		constexpr auto a = id_cast(RGBA::A);

		constexpr auto ch_max = cs::to_channel_u8(1.0f);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);
			auto sr = channel_row_begin(src, y, r);
			auto sg = channel_row_begin(src, y, g);
			auto sb = channel_row_begin(src, y, b);

			for (u32 x = 0; x < src.width; ++x) // TODO: simd
			{
				d[x].channels[r] = cs::to_channel_u8(sr[x]);
				d[x].channels[g] = cs::to_channel_u8(sg[x]);
				d[x].channels[b] = cs::to_channel_u8(sb[x]);
				d[x].channels[a] = ch_max;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_rgb(View1r32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		constexpr auto r = id_cast(RGBA::R);
		constexpr auto g = id_cast(RGBA::G);
		constexpr auto b = id_cast(RGBA::B);
		constexpr auto a = id_cast(RGBA::A);

		constexpr auto ch_max = cs::to_channel_u8(1.0f);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			for (u32 x = 0; x < src.width; ++x) // TODO: simd
			{
				auto const gray = cs::to_channel_u8(s[x]);

				d[x].channels[r] = gray;
				d[x].channels[g] = gray;
				d[x].channels[b] = gray;
				d[x].channels[a] = ch_max;
			}
		};

		process_image_rows(src.height, row_func);
	}
}


/* map_hsv */

namespace simage
{	
	void map_rgb_hsv(View const& src, ViewHSVr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::r32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = hsv::r32_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst)
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

				auto hsv = hsv::r32_from_rgb_r32(r, g, b);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::r32_to_rgb_r32(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_rows(src.height, row_func);
	}
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto lch = lch::r32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = lch::r32_to_rgb_u8(s.L[x], s.C[x], s.H[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_rgb_lch(ViewRGBr32 const& src, ViewLCHr32 const& dst)
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

				auto lch = lch::r32_from_rgb_r32(r, g, b);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHr32 const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = lch::r32_to_rgb_r32(s.L[x], s.C[x], s.H[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_rows(src.height, row_func);
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb_no_simd(ViewYUV const& src, ViewRGBr32 const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422*)s2;
			auto d = rgb_row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgb = yuv::u8_to_rgb_r32(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::u8_to_rgb_r32(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_rows(src.height, row_func);
	}


#ifdef SIMAGE_NO_SIMD

	static void do_map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst)
	{
		map_yuv_rgb_no_simd(src, dst);
	}

#else

	class YUV422r32Planar
	{
	public:
		r32 y1[simd::VEC_LEN] = { 0 };
		r32 y2[simd::VEC_LEN] = { 0 };
		r32 u[simd::VEC_LEN] = { 0 };
		r32 v[simd::VEC_LEN] = { 0 };
	};


	static YUV422r32Planar to_planar(YUV422* begin)
	{
		YUV422r32Planar planar;

		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			auto yuv = begin[i];

			planar.y1[i] = (r32)yuv.y1;
			planar.y2[i] = (r32)yuv.y2;
			planar.u[i] = (r32)yuv.u;
			planar.v[i] = (r32)yuv.v;
		}

		return planar;
	}


	static void map_yuv422_rgb_row(YUV422* yuv, r32* dr, r32* dg, r32* db, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		auto const do_simd = [&](u32 i)
		{
			auto p = to_planar(yuv);


		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void do_map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst)
	{
		if (src.width < simd::VEC_LEN)
		{
			map_yuv_rgb_no_simd(src, dst);
		}
		else
		{
			map_yuv_rgb_no_simd(src, dst);
		}		
	}


#endif


	

	void map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2) == 2);

		do_map_yuv_rgb(src, dst);
	}


	void mipmap_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst)
	{		
		static_assert(sizeof(YUV2) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d) 
		{
			auto val = 0.25f * ((r32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((r32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		auto const row_func = [&](u32 y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422*)row_begin(src, src_y1);
			auto s2 = (YUV422*)row_begin(src, src_y2);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgb = yuv::u8_to_rgb_r32(y_avg, u_avg, v_avg);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_rows(dst.height, row_func);
	}


	void map_yuv_rgb2(ViewYUV const& src, View const& dst)
	{
		static_assert(sizeof(YUV2) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d)
		{
			auto val = 0.25f * ((r32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((r32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		auto const row_func = [&](u32 y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422*)row_begin(src, src_y1);
			auto s2 = (YUV422*)row_begin(src, src_y2);
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

		process_image_rows(dst.height, row_func);
	}
}


/* sub_view */

namespace simage
{
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


	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		View1r32 sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
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


/* select_channel */

namespace simage
{
	template <size_t N>
	static View1r32 select_channel(ViewCHr32<N> const& view, u32 ch)
	{
		View1r32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		return view1;
	}


	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(ViewRGBr32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(ViewHSVr32 const& view, HSV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(View2r32 const& view, GA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(View2r32 const& view, XY channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	ViewRGBr32 select_rgb(ViewRGBAr32 const& view)
	{
		assert(verify(view));

		ViewRGBr32 rgb;

		rgb.image_width = view.image_width;
		rgb.width = view.width;
		rgb.height = view.height;
		rgb.range = view.range;

		rgb.image_channel_data[id_cast(RGB::R)] = view.image_channel_data[id_cast(RGB::R)];
		rgb.image_channel_data[id_cast(RGB::G)] = view.image_channel_data[id_cast(RGB::G)];
		rgb.image_channel_data[id_cast(RGB::B)] = view.image_channel_data[id_cast(RGB::B)];

		return rgb;
	}
}


/* fill */

namespace simage
{
	template <class VIEW, typename COLOR>
	static void fill_no_simd(VIEW const& view, COLOR color)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 i = 0; i < view.width; ++i)
			{
				d[i] = color;
			}
		};

		process_image_rows(view.height, row_func);
	}


	template <size_t N>
	static void fill_n_channels_no_simd(ViewCHr32<N> const& view, Pixel color)
	{
		r32 channels[N] = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			channels[ch] = cs::to_channel_r32(color.channels[ch]);
		}

		auto const row_func = [&](u32 y)
		{
			for (u32 ch = 0; ch < N; ++ch)
			{
				auto d = channel_row_begin(view, y, ch);
				for (u32 x = 0; x < view.width; ++x)
				{
					d[x] = channels[ch];
				}
			}
		};

		process_image_rows(view.height, row_func);
	}


#ifdef SIMAGE_NO_SIMD

	template <class VIEW, typename COLOR>
	static void do_fill(VIEW const& view, COLOR color)
	{
		fill_no_simd(view, color);
	}


	template <size_t N>
	static void do_fill_n_channels(ViewCHr32<N> const& view, Pixel color)
	{
		fill_n_channels_no_simd(view, color);
	}

#else

	static void fill_row_simd(r32* dst_begin, r32 value, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		r32* dst = 0;
		r32* val = &value;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i) 
		{
			dst = dst_begin + i;
			vec = simd::load_broadcast(val);
			simd::store(dst, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void fill_simd(View1r32 const& view, r32 gray32)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd(d, gray32, view.width);
		};

		process_image_rows(view.height, row_func);
	}


	


	template <size_t N>
	static void fill_n_channels_simd(ViewCHr32<N> const& view, Pixel color)
	{
		r32 channels[N] = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			channels[ch] = cs::to_channel_r32(color.channels[ch]);
		}

		auto const row_func = [&](u32 y)
		{
			for (u32 ch = 0; ch < N; ++ch)
			{
				auto d = channel_row_begin(view, y, ch);
				fill_row_simd(d, channels[ch], view.width);
			}
		};

		process_image_rows(view.height, row_func);
	}


	template <class VIEW, typename COLOR>
	static void do_fill(VIEW const& view, COLOR color)
	{
		auto len32 = view.width * sizeof(COLOR) / sizeof(r32);
		if (len32 < simd::VEC_LEN)
		{
			fill_no_simd(view, color);
		}
		else
		{
			fill_simd(view, color);
		}		
	}


	template <size_t N>
	static void do_fill_n_channels(ViewCHr32<N> const& view, Pixel color)
	{
		if (view.width < simd::VEC_LEN)
		{
			fill_n_channels_no_simd(view, color);
		}
		else
		{
			fill_n_channels_simd(view, color);
		}		
	}

#endif	


	


	void fill(View4r32 const& view, Pixel color)
	{
		assert(verify(view));

		do_fill_n_channels(view, color);
	}


	void fill(View3r32 const& view, Pixel color)
	{
		assert(verify(view));

		do_fill_n_channels(view, color);
	}


	void fill(View1r32 const& view, u8 gray)
	{
		assert(verify(view));		

		auto const gray32 = cs::to_channel_r32(gray);

		do_fill(view, gray32);
	}
}


/* transform */

namespace simage
{
	void transform(View1r32 const& src, View1r32 const& dst, std::function<r32(r32)> const& func)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		};

		process_image_rows(src.height, row_func);
	}


	void transform(View2r32 const& src, View1r32 const& dst, std::function<r32(r32, r32)> const& func)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y).channels;
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[0][x], s[1][x]);
			}
		};

		process_image_rows(src.height, row_func);
	}


	void transform(View3r32 const& src, View1r32 const& dst, std::function<r32(r32, r32, r32)> const& func)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y).channels;
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[0][x], s[1][x], s[2][x]);
			}
		};

		process_image_rows(src.height, row_func);
	}
}


/* shrink_view */

namespace simage
{
	static r32 average(View1r32 const& view)
	{
		r32 total = 0.0f;

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


	static r32 average(ViewGray const& view)
	{
		r32 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += cs::to_channel_r32(s[x]);
			}
		}

		return total / (view.width * view.height);
	}


	template <size_t N>
	static std::array<r32, N> average(ViewCHr32<N> const& view)
	{
		std::array<r32, N> results = { 0 };
		for (u32 i = 0; i < N; ++i) { results[i] = 0.0f; }

		for (u32 y = 0; y < view.height; ++y)
		{
			PixelCHr32<N> s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				for (u32 i = 0; i < N; ++i)
				{
					results[i] += s.channels[i][x];
				}
			}
		}

		for (u32 i = 0; i < N; ++i)
		{
			results[i] /= (view.width * view.height);
		}

		return results;
	}
	

	static cs::RGBr32 average(View const& view)
	{	
		r32 red = 0.0f;
		r32 green = 0.0f;
		r32 blue = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				auto p = s[x].rgba;
				red += cs::to_channel_r32(p.red);
				green += cs::to_channel_r32(p.green);
				blue += cs::to_channel_r32(p.blue);
			}
		}

		red /= (view.width * view.height);
		green /= (view.width * view.height);
		blue /= (view.width * view.height);

		return { red, green, blue };
	}


	template <class VIEW>
	static void do_shrink_1D(VIEW const& src, View1r32 const& dst)
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

		process_image_rows(dst.height, row_func);
	}


	void shrink(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View3r32 const& src, View3r32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

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

				auto avg = average(sub_view(src, r));

				d.channels[0][x] = avg[0];
				d.channels[1][x] = avg[1];
				d.channels[2][x] = avg[2];
			}
		};

		process_image_rows(dst.height, row_func);
	}


	void shrink(ViewGray const& src, View1r32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View const& src, ViewRGBr32 const& dst)
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

		process_image_rows(dst.height, row_func);
	}
}


/* gradients */

namespace simage
{
	static void convolve(View1r32 const& src, View1r32 const& dst, Matrix2D<r32> const& kernel)
	{
		assert(verify(src, dst));
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
				d[x] = 0.0f;
				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s = row_offset_begin(src, y, ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						d[x] += (s + rx)[x] * kernel.data_[w];
						++w;
					}
				}
			}
		};

		process_image_rows(src.height, row_func);
	}


	constexpr std::array<r32, 33> make_grad_x_11()
	{
		/*constexpr std::array<r32, 33> GRAD_X
		{
			-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
			-0.06f, -0.09f, -0.12f, -0.15f, -0.18f, 0.0f, 0.18f, 0.15f, 0.12f, 0.09f, 0.06f,
			-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.01f,
		};*/

		std::array<r32, 33> grad = { 0 };

		r32 values[] = { 0.08f, 0.06f, 0.04f, 0.02f, 0.01f };

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


	constexpr std::array<r32, 33> make_grad_y_11()
	{
		/*constexpr std::array<r32, 33> GRAD_Y
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

		std::array<r32, 33> grad = { 0 };

		r32 values[] = { 0.08f, 0.06f, 0.04f, 0.02f, 0.01f };

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


	constexpr std::array<r32, 15> make_grad_x_5()
	{
		std::array<r32, 15> grad
		{
			-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
			-0.16f, -0.24f, 0.0f, 0.24f, 0.16f
			-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
		};

		return grad;
	}


	constexpr std::array<r32, 15> make_grad_y_5()
	{
		std::array<r32, 15> grad
		{
			-0.08f, -0.16f, -0.08f,
			-0.12f, -0.24f, -0.12f,
			 0.00f,  0.00f,  0.00f,
			 0.12f,  0.24f,  0.12f,
			 0.08f,  0.16f,  0.08f,
		};

		return grad;
	}


	static void zero_outer(View1r32 const& view, u32 n_rows, u32 n_columns)
	{
		auto const top_bottom = [&]()
		{
			for (u32 r = 0; r < n_rows; ++r)
			{
				auto top = row_begin(view, r);
				auto bottom = row_begin(view, view.height - 1 - r);
				for (u32 x = 0; x < view.width; ++x)
				{
					top[x] = bottom[x] = 0.0f;
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
					row[c] = row[view.width - 1 - c] = 0.0f;
				}
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		execute(f_list);
	}


	void gradients_xy(View1r32 const& src, View2r32 const& xy_dst)
	{
		auto x_dst = select_channel(xy_dst, XY::X);
		auto y_dst = select_channel(xy_dst, XY::Y);

		assert(verify(src, x_dst));
		assert(verify(src, y_dst));

		/*constexpr auto grad_x = make_grad_x_11();
		constexpr auto grad_y = make_grad_y_11();
		constexpr u32 kernel_dim_a = 11;*/

		constexpr auto grad_x = make_grad_x_5();
		constexpr auto grad_y = make_grad_y_5();
		constexpr u32 kernel_dim_a = 5;

		constexpr u32 kernel_dim_b = (u32)grad_x.size() / kernel_dim_a;

		Matrix2D<r32> x_kernel{};
		x_kernel.width = kernel_dim_a;
		x_kernel.height = kernel_dim_b;
		x_kernel.data_ = (r32*)grad_x.data();

		Matrix2D<r32> y_kernel{};
		y_kernel.width = kernel_dim_b;
		y_kernel.height = kernel_dim_a;
		y_kernel.data_ = (r32*)grad_y.data();

		zero_outer(x_dst, kernel_dim_b / 2, kernel_dim_a / 2);
		zero_outer(y_dst, kernel_dim_a / 2, kernel_dim_b / 2);

		Range2Du32 x_inner{};
		x_inner.x_begin = kernel_dim_a / 2;
		x_inner.x_end = src.width - kernel_dim_a / 2;
		x_inner.y_begin = kernel_dim_b / 2;
		x_inner.y_end = src.height - kernel_dim_b / 2;

		Range2Du32 y_inner{};
		y_inner.x_begin = kernel_dim_b / 2;
		y_inner.x_end = src.width - kernel_dim_b / 2;
		y_inner.y_begin = kernel_dim_a / 2;
		y_inner.y_end = src.height - kernel_dim_a / 2;

		convolve(sub_view(src, x_inner), sub_view(x_dst, x_inner), x_kernel);
		convolve(sub_view(src, y_inner), sub_view(y_dst, y_inner), y_kernel);
	}
}





#ifdef SIMAGE_NO_SIMD



#else



#endif // SIMAGE_NO_SIMD