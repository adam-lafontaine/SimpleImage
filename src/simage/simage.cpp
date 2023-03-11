#include "simage.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"
//#define SIMAGE_NO_SIMD

#ifndef SIMAGE_NO_SIMD
#include "../util/simd.hpp"
#endif // !SIMAGE_NO_SIMD


#include <cmath>
#include <algorithm>

namespace cs = color_space;
namespace rng = std::ranges;



static void process_image_by_row(u32 n_rows, id_func_t const& row_func)
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
	class HSVp
	{
	public:
		T* H;
		T* S;
		T* V;
	};


	template <typename T>
	class LCHp
	{
	public:
		T* L;
		T* C;
		T* H;
	};


	using RGBf32p = RGBp<f32>;
	using RGBu16p = RGBp<u16>;

	using HSVf32p = HSVp<f32>;
	using HSVu16p = HSVp<u16>;

	using LCHf32p = LCHp<f32>;
	using LCHu16p = LCHp<u16>;
}


/* row begin */

namespace simage
{
	static RGBf32p rgb_row_begin(ViewRGBf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		RGBf32p rgb{};

		rgb.R = view.channel_data_[id_cast(RGB::R)] + offset;
		rgb.G = view.channel_data_[id_cast(RGB::G)] + offset;
		rgb.B = view.channel_data_[id_cast(RGB::B)] + offset;

		return rgb;
	}


	static HSVf32p hsv_row_begin(ViewHSVf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		HSVf32p hsv{};

		hsv.H = view.channel_data_[id_cast(HSV::H)] + offset;
		hsv.S = view.channel_data_[id_cast(HSV::S)] + offset;
		hsv.V = view.channel_data_[id_cast(HSV::V)] + offset;

		return hsv;
	}


	static LCHf32p lch_row_begin(ViewLCHf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.channel_width_ + view.x_begin;

		LCHf32p lch{};

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
	static void do_make_view(ChannelView2D<T, N>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
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


	View1f32 make_view_1(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View1f32 view;

		view.matrix_data_ = mb::push_elements(buffer, width * height);
		view.matrix_width = width;		
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

		assert(verify(view));

		return view;
	}


	View2f32 make_view_2(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2f32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View3f32 make_view_3(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3f32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}


	View4f32 make_view_4(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4f32 view;

		do_make_view(view, width, height, buffer);

		assert(verify(view));

		return view;
	}
}


/* map */

namespace simage
{
	void map_gray(ViewGray const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_f32(s[x]);
			}
		};

		process_image_by_row(src.height, row_func);
	}
	

	void map_gray(View1f32 const& src, ViewGray const& dst)
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

		process_image_by_row(src.height, row_func);
	}


	void map_gray(ViewYUV const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_f32(s[x].y);
			}
		};

		process_image_by_row(src.height, row_func);
	}
}


/* map_rgb */

namespace simage
{	
	static void map_rgba_no_simd(View const& src, ViewRGBAf32 const& dst)
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
				dr[x] = cs::to_channel_f32(s[x].channels[r]);
				dg[x] = cs::to_channel_f32(s[x].channels[g]);
				db[x] = cs::to_channel_f32(s[x].channels[b]);
				da[x] = cs::to_channel_f32(s[x].channels[a]);
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_rgb_no_simd(View const& src, ViewRGBf32 const& dst)
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
				dr[x] = cs::to_channel_f32(s[x].channels[r]);
				dg[x] = cs::to_channel_f32(s[x].channels[g]);
				db[x] = cs::to_channel_f32(s[x].channels[b]);
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_rgb_row_no_simd(View const& src, ViewRGBf32 const& dst, u32 y)
	{
		constexpr auto r = id_cast(RGB::R);
		constexpr auto g = id_cast(RGB::G);
		constexpr auto b = id_cast(RGB::B);

		auto s = row_begin(src, y);
		auto dr = channel_row_begin(dst, y, r);
		auto dg = channel_row_begin(dst, y, g);
		auto db = channel_row_begin(dst, y, b);

		for (u32 x = 0; x < src.width; ++x)
		{
			dr[x] = cs::to_channel_f32(s[x].channels[r]);
			dg[x] = cs::to_channel_f32(s[x].channels[g]);
			db[x] = cs::to_channel_f32(s[x].channels[b]);
		}
	}


#ifdef SIMAGE_NO_SIMD

	static void do_map_rgba(View const& src, ViewRGBAf32 const& dst)
	{
		map_rgba_no_simd(src, dst);
	}


	static void do_map_rgb(View const& src, ViewRGBf32 const& dst)
	{
		map_rgb_no_simd(src, dst);
	}

#else

	class Pixelf32Planar
	{
	public:
		f32 red[simd::VEC_LEN] = { 0 };
		f32 green[simd::VEC_LEN] = { 0 };
		f32 blue[simd::VEC_LEN] = { 0 };
		f32 alpha[simd::VEC_LEN] = { 0 };
	};


	Pixelf32Planar to_planar(Pixel* p_begin)
	{
		Pixelf32Planar planar;

		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			auto rgba = p_begin[i].rgba;

			planar.red[i] = cs::to_channel_f32(rgba.red);
			planar.green[i] = cs::to_channel_f32(rgba.green);
			planar.blue[i] = cs::to_channel_f32(rgba.blue);
			planar.alpha[i] = cs::to_channel_f32(rgba.alpha);
		}

		return planar;
	}


	static void map_rgba_row_simd(Pixel* src, f32* dr, f32* dg, f32* db, f32* da, u32 length)
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


	static void map_rgb_row_simd(Pixel* src, f32* dr, f32* dg, f32* db, u32 length)
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


	static void map_rgba_simd(View const& src, ViewRGBAf32 const& dst)
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

		process_image_by_row(src.height, row_func);
	}


	static void map_rgb_simd(View const& src, ViewRGBf32 const& dst)
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

		process_image_by_row(src.height, row_func);
	}


	static void do_map_rgba(View const& src, ViewRGBAf32 const& dst)
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


	static void do_map_rgb(View const& src, ViewRGBf32 const& dst)
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


	void map_rgba(View const& src, ViewRGBAf32 const& dst)
	{
		assert(verify(src, dst));

		do_map_rgba(src, dst);
	}


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
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

		process_image_by_row(src.height, row_func);
	}

	
	void map_rgb(View const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		do_map_rgb(src, dst);
	}


	void map_rgb(ViewRGBf32 const& src, View const& dst)
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

		process_image_by_row(src.height, row_func);
	}


	void map_rgb(View1f32 const& src, View const& dst)
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

		process_image_by_row(src.height, row_func);
	}
}


/* map_hsv */

namespace simage
{	
	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = hsv::f32_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_rgb_hsv(ViewRGBf32 const& src, ViewHSVf32 const& dst)
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

				auto hsv = hsv::f32_from_rgb_f32(r, g, b);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::f32_to_rgb_f32(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_by_row(src.height, row_func);
	}
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto lch = lch::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = lch::f32_to_rgb_u8(s.L[x], s.C[x], s.H[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_rgb_lch(ViewRGBf32 const& src, ViewLCHf32 const& dst)
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

				auto lch = lch::f32_from_rgb_f32(r, g, b);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void map_lch_rgb(ViewLCHf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = lch_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = lch::f32_to_rgb_f32(s.L[x], s.C[x], s.H[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_by_row(src.height, row_func);
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb_no_simd(ViewYUV const& src, ViewRGBf32 const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			auto d = rgb_row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgb = yuv::u8_to_rgb_f32(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::u8_to_rgb_f32(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_by_row(src.height, row_func);
	}


#ifdef SIMAGE_NO_SIMD

	static void do_map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{
		map_yuv_rgb_no_simd(src, dst);
	}

#else

	class YUV422u8f32Planar
	{
	public:
		f32 y1[simd::VEC_LEN] = { 0 };
		f32 y2[simd::VEC_LEN] = { 0 };
		f32 u[simd::VEC_LEN] = { 0 };
		f32 v[simd::VEC_LEN] = { 0 };
	};


	static YUV422u8f32Planar to_planar(YUV422u8* begin)
	{
		YUV422u8f32Planar planar;

		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			auto yuv = begin[i];

			planar.y1[i] = (f32)yuv.y1;
			planar.y2[i] = (f32)yuv.y2;
			planar.u[i] = (f32)yuv.u;
			planar.v[i] = (f32)yuv.v;
		}

		return planar;
	}


	static void map_yuv422_rgb_row(YUV422u8* yuv, f32* dr, f32* dg, f32* db, u32 length)
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


	static void do_map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
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


	

	void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		do_map_yuv_rgb(src, dst);
	}


	void mipmap_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{		
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d) 
		{
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
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

				auto rgb = yuv::u8_to_rgb_f32(y_avg, u_avg, v_avg);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_image_by_row(dst.height, row_func);
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
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
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

		process_image_by_row(dst.height, row_func);
	}
}


namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
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
				dr[x] = cs::to_channel_f32(s[x].red);
				dg[x] = cs::to_channel_f32(s[x].green);
				db[x] = cs::to_channel_f32(s[x].blue);
			}
		};

		process_image_by_row(src.height, row_func);
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
		assert(verify(view, range));

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


	View4f32 sub_view(View4f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3f32 sub_view(View3f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View2f32 sub_view(View2f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View1f32 sub_view(View1f32 const& view, Range2Du32 const& range)
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
	template <typename T, size_t N>
	static View1<T> select_channel(ChannelView2D<T, N> const& view, u32 ch)
	{
		View1<T> view1{};

		view1.matrix_width = view.channel_width_;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.matrix_data_ = view.channel_data_[ch];

		return view1;
	}


	View1f32 select_channel(ViewRGBAf32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewRGBf32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewHSVf32 const& view, HSV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, GA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, XY channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	ViewRGBf32 select_rgb(ViewRGBAf32 const& view)
	{
		assert(verify(view));

		ViewRGBf32 rgb;

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


/* fill */

namespace simage
{
	template <typename T>
	static void fill_no_simd(View1<T> const& view, T grayT)
	{		
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 i = 0; i < view.width; ++i)
			{
				d[i] = grayT;
			}
		};

		process_image_by_row(view.height, row_func);
	}


	template <size_t N>
	static void fill_n_channels_no_simd(ViewCHf32<N> const& view, Pixel color)
	{
		f32 channels[N] = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			channels[ch] = cs::to_channel_f32(color.channels[ch]);
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

		process_image_by_row(view.height, row_func);
	}

	template <class VIEW, typename COLOR>
	static void do_fill(VIEW const& view, COLOR color)
	{
		fill_no_simd(view, color);
	}


	template <size_t N>
	static void do_fill_n_channels(ViewCHf32<N> const& view, Pixel color)
	{
		fill_n_channels_no_simd(view, color);
	}


	void fill(View4f32 const& view, Pixel color)
	{
		assert(verify(view));

		do_fill_n_channels(view, color);
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		do_fill_n_channels(view, color);
	}


	void fill(View1f32 const& view, u8 gray)
	{
		assert(verify(view));		

		auto const gray32 = cs::to_channel_f32(gray);

		do_fill(view, gray32);
	}
}


/* transform */

namespace simage
{
	template <typename T>
	static void do_transform_view_1(View1<T> const& src, View1<T> const& dst, std::function<T(T)> const& func)
	{
		auto const row_func = [&](u32 y) 
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		};

		process_image_by_row(src.height, row_func);
	}


	template <typename T>
	static void do_transform_view_2_1(View2<T> const& src, View1<T> const& dst, std::function<T(T, T)> const& func)
	{
		auto const row_func = [&](u32 y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s0[x], s1[x]);
			}
		};

		process_image_by_row(src.height, row_func);
	}


	template <typename T>
	static void do_transform_view_3_1(View3<T> const& src, View1<T> const& dst, std::function<T(T, T, T)> const& func)
	{
		auto const row_func = [&](u32 y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto s2 = channel_row_begin(src, y, 2);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s0[x], s1[x], s2[x]);
			}
		};

		process_image_by_row(src.height, row_func);
	}


	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func)
	{
		assert(verify(src, dst));

		do_transform_view_1(src, dst, func);
	}


	void transform(View2f32 const& src, View1f32 const& dst, std::function<f32(f32, f32)> const& func)
	{
		assert(verify(src, dst));

		do_transform_view_2_1(src, dst, func);
	}


	void transform(View3f32 const& src, View1f32 const& dst, std::function<f32(f32, f32, f32)> const& func)
	{
		assert(verify(src, dst));

		do_transform_view_3_1(src, dst, func);
	}
}


/* shrink_view */
#if 0
namespace simage
{
	static f32 average(View1f32 const& view)
	{
		f32 total = 0.0f;

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


	static f32 average(ViewGray const& view)
	{
		f32 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += cs::to_channel_f32(s[x]);
			}
		}

		return total / (view.width * view.height);
	}


	template <size_t N>
	static std::array<f32, N> average(ViewCHf32<N> const& view)
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
	

	static cs::RGBf32 average(View const& view)
	{	
		f32 red = 0.0f;
		f32 green = 0.0f;
		f32 blue = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				auto p = s[x].rgba;
				red += cs::to_channel_f32(p.red);
				green += cs::to_channel_f32(p.green);
				blue += cs::to_channel_f32(p.blue);
			}
		}

		red /= (view.width * view.height);
		green /= (view.width * view.height);
		blue /= (view.width * view.height);

		return { red, green, blue };
	}


	template <class VIEW>
	static void do_shrink_1D(VIEW const& src, View1f32 const& dst)
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

		process_image_by_row(dst.height, row_func);
	}


	void shrink(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View3f32 const& src, View3f32 const& dst)
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

		process_image_by_row(dst.height, row_func);
	}


	void shrink(ViewGray const& src, View1f32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View const& src, ViewRGBf32 const& dst)
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

		process_image_by_row(dst.height, row_func);
	}
}
#endif

/* gradients */

namespace simage
{
	template <typename T>
	static void convolve(View1<T> const& src, View1<T> const& dst, Mat2Df32 const& kernel)
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

					d[x] = (T)total;
				}
			}
		};

		process_image_by_row(src.height, row_func);
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


	template <typename T>
	static void do_gradients(View1<T> const& src, View1<T> const& dst, XY ch)
	{
		constexpr auto grad_x = make_grad_x_11();
		constexpr auto grad_y = make_grad_y_11();
		constexpr u32 dim_max = 11;		

		/*constexpr auto grad_x = make_grad_x_5();
		constexpr auto grad_y = make_grad_y_5();
		constexpr u32 dim_max = 5;*/

		constexpr u32 dim_min = (u32)grad_x.size() / dim_max;

		Mat2Df32 kernel{};

		switch (ch)
		{
		case XY::X:
			kernel.data_ = (f32*)grad_x.data();
			kernel.width = dim_max;
			kernel.height = dim_min;
			break;
		case XY::Y:
			kernel.data_ = (f32*)grad_y.data();
			kernel.width = dim_min;
			kernel.height = dim_max;
			break;
		
		default:
			assert(false);
			return;
		}

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


	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst)
	{
		auto x_dst = select_channel(xy_dst, XY::X);
		auto y_dst = select_channel(xy_dst, XY::Y);

		assert(verify(src, x_dst));
		assert(verify(src, y_dst));

		do_gradients(src, x_dst, XY::X);
		do_gradients(src, y_dst, XY::Y);
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


	void blur(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		blur_1(src, dst);
	}


	void blur(View3f32 const& src, View3f32 const& dst)
	{
		assert(verify(src, dst));

		blur_n(src, dst);
	}
}


