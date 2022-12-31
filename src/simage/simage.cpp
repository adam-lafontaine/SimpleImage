#include "simage.hpp"
#include "../util/memory_buffer.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"

//#define SIMAGE_NO_SIMD

#ifndef SIMAGE_NO_SIMD
#include "../util/simd.hpp"
#endif // !SIMAGE_NO_SIMD



#include <cmath>
#include <algorithm>


namespace mb = memory_buffer;
namespace cs = color_space;



static void process_rows(u32 n_rows, id_func_t const& row_func)
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


/* row begin */

namespace simage
{
	// platform dependent, endianness
	class RGBAr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
		r32* A;
	};


	class RGBr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
	};


	class PixelRGBAr32
	{
	public:

		static constexpr u32 n_channels = 4;

		union 
		{
			RGBAr32p rgba;

			r32* channels[4] = {};
		};

		// for_each_xy
		/*r32& red() { return *rgba.R; }
		r32& green() { return *rgba.G; }
		r32& blue() { return *rgba.B; }
		r32& alpha() { return *rgba.A; }*/
	};


	class PixelRGBr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union 
		{
			RGBr32p rgb;

			r32* channels[3] = {};
		};

		// for_each_xy
		/*r32& red() { return *rgb.R; }
		r32& green() { return *rgb.G; }
		r32& blue() { return *rgb.B; }*/
	};


	class HSVr32p
	{
	public:
		r32* H;
		r32* S;
		r32* V;
	};


	class PixelHSVr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union
		{
			HSVr32p hsv;

			r32* channels[3] = {};
		};

		/*r32& hue() { return *hsv.H; }
		r32& sat() { return *hsv.S; }
		r32& val() { return *hsv.V; }*/
	};


	class Pixel3CHr32
	{
	public:
		static constexpr u32 n_channels = 3;

		union 
		{
			RGBr32p rgb;

			HSVr32p hsv;

			r32* channels[3] = {};
		};

	};



	template <typename T>
	static T* row_begin(Matrix2D<T> const& image, u32 y)
	{
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data_ + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <typename T>
	static T* row_begin(MatrixView<T> const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


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


	static PixelRGBr32 rgb_row_begin(ViewRGBr32 const& view, u32 y)
	{
		assert(verify(view));
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
		assert(verify(view));
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
		assert(verify(view));

		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


	template <typename T>
	static T* row_offset_begin(MatrixView<T> const& view, u32 y, int y_offset)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
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
	template <typename T>
	static T* xy_at(MatrixView<T> const& view, u32 x, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


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


/* platform */

namespace simage
{
	template <typename T>
	static bool do_create_image(Matrix2D<T>& image, u32 width, u32 height)
	{
		image.data_ = (T*)malloc(sizeof(T) * width * height);
		if(!image.data_)
		{
			return false;
		}

		image.width = width;
		image.height = height;

		return true;
	}


	template <typename T>
	static void do_destroy_image(Matrix2D<T>& image)
	{
		if (image.data_)
		{
			free(image.data_);
			image.data_ = nullptr;
		}

		image.width = 0;
		image.height = 0;
	}


	bool create_image(Image& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto result = do_create_image(image, width, height);

		assert(verify(image));

		return result;
	}


    bool create_image(ImageGray& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto result = do_create_image(image, width, height);

		assert(verify(image));

		return result;
	}


	bool create_image(ImageYUV& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto result = do_create_image(image, width, height);

		assert(verify(image));

		return result;
	}


	void destroy_image(Image& image)
	{
		do_destroy_image(image);
	}


	void destroy_image(ImageGray& image)
	{
		do_destroy_image(image);
	}


	void destroy_image(ImageYUV& image)
	{
		do_destroy_image(image);
	}
}


/* make view */

namespace simage
{
	template <typename T>
	static MatrixView<T> do_make_view(Matrix2D<T> const& image)
	{
		MatrixView<T> view;

		view.image_data = image.data_;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		return view;
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

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(ImageGray const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewYUV make_view(ImageYUV const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
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
	using u8_to_r32_f = std::function<r32(u8)>;
	using r32_to_u8_f = std::function<u8(r32)>;


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

		process_rows(src.height, row_func);
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

		process_rows(src.height, row_func);
	}
}


/* map_rgb */

namespace simage
{	
	void map_rgb(View const& src, ViewRGBAr32 const& dst)
	{
		assert(verify(src, dst));

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

		process_rows(src.height, row_func);
	}


	void map_rgb(ViewRGBAr32 const& src, View const& dst)
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

		process_rows(src.height, row_func);
	}

	
	void map_rgb(View const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));

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

		process_rows(src.height, row_func);
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

		process_rows(src.height, row_func);
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
			auto d = hsv_row_begin(dst, y).hsv;

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::r32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		};

		process_rows(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = hsv_row_begin(src, y).hsv;
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = hsv::r32_to_rgba_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_rows(src.height, row_func);
	}


	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgb_row_begin(src, y).rgb;
			auto d = hsv_row_begin(dst, y).hsv;

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

		process_rows(src.height, row_func);
	}


	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = hsv_row_begin(src, y).hsv;
			auto d = rgb_row_begin(dst, y).rgb;

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::r32_to_rgb_r32(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_rows(src.height, row_func);
	}
}


/* map_yuv_rgb */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2) == 2);

		auto const row_func = [&](u32 y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422*)s2;
			auto d = rgb_row_begin(dst, y).rgb;

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

		process_rows(src.height, row_func);
	}


	void map_yuv_rgb(ViewYUV const& src, View const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2) == 2);

		auto const row_func = [&](u32 y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422*)s2;
			auto d = row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgba = yuv::u8_to_rgba_u8(yuv.y1, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;

				++x;
				rgba = yuv::u8_to_rgba_u8(yuv.y2, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;
			}
		};

		process_rows(src.height, row_func);
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
			auto d = rgb_row_begin(dst, y).rgb;

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

		process_rows(dst.height, row_func);
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

				auto rgba = yuv::u8_to_rgba_u8(y_avg, u_avg, v_avg);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_rows(dst.height, row_func);
	}
}


/* sub_view */

namespace simage
{
	template <typename T>
	static MatrixView<T> do_sub_view(Matrix2D<T> const& image, Range2Du32 const& range)
	{
		MatrixView<T> sub_view;

		sub_view.image_data = image.data_;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
	}


	template <typename T>
	static MatrixView<T> do_sub_view(MatrixView<T> const& view, Range2Du32 const& range)
	{
		MatrixView<T> sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
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

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ImageGray const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View sub_view(View const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ViewGray const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& image_range)
	{
		auto width = image_range.x_end - image_range.x_begin;
		Range2Du32 camera_range = image_range;
		camera_range.x_end = camera_range.x_begin + width / 2;

		assert(verify(camera_src, camera_range));

		auto sub_view = do_sub_view(camera_src, camera_range);

		assert(verify(sub_view));

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

		auto sub_view = do_sub_view(view, range);

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

		process_rows(view.height, row_func);
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

		process_rows(view.height, row_func);
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

		process_rows(view.height, row_func);
	}


	static void fill_simd(View const& view, Pixel color)
	{
		static_assert(sizeof(Pixel) == sizeof(r32));

		auto ptr = (r32*)(&color);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd((r32*)d, *ptr, view.width);
		};

		process_rows(view.height, row_func);
	}


	static void fill_simd(ViewGray const& view, u8 gray)
	{
		static_assert(4 * sizeof(u8) == sizeof(r32));

		u8 bytes[4] = { gray, gray, gray, gray };
		auto ptr = (r32*)bytes;
		auto len32 = view.width / 4;

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd((r32*)d, *ptr, len32);

			for (u32 x = len32 * 4; x < view.width; ++x)
			{
				d[x] = gray;
			}
		};

		process_rows(view.height, row_func);
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

		process_rows(view.height, row_func);
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


	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		do_fill(view, color);
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		do_fill(view, gray);
	}


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

		process_rows(dst.height, row_func);
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

			Range2Du32 r;
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

		process_rows(dst.height, row_func);
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
			auto d = rgb_row_begin(dst, y).rgb;

			Range2Du32 r;
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

		process_rows(dst.height, row_func);
	}
}


/* make_histograms */

namespace simage
{
	/*template <size_t N>
	static std::array<View, N> split_view(View const& view)
	{
		std::array<View, N> sub_views;

		Range2Du32 r;
		r.x_begin = 0;
		r.x_end = view.width;

		for (u32 i = 0; i < N_THREADS; ++i)
		{
			r.y_begin = i * view.height / N_THREADS;
			r.y_end = r.y_begin + view.height / N_THREADS;
			sub_views[i] = sub_view(view, r);
		}
		sub_views.back().y_end = view.height;

		return sub_views;
	}*/


	inline constexpr u8 to_hist_bin_u8(u8 val, u32 n_bins)
	{
		return val * n_bins / 256;
	}


	static void make_histograms_from_rgb(View const& src, Histogram9r32& dst)
	{
		constexpr u32 PIXEL_STEP = 2;

		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_yuv = dst.yuv;
		auto n_bins = dst.n_bins;

		RGBAu8 rgba{};
		cs::HSVu8 hsv{};
		cs::YUVu8 yuv{};

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += PIXEL_STEP)
			{
				rgba = s[x].rgba;

				hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				yuv = yuv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

				h_rgb.R[to_hist_bin_u8(rgba.red, n_bins)]++;
				h_rgb.G[to_hist_bin_u8(rgba.green, n_bins)]++;
				h_rgb.B[to_hist_bin_u8(rgba.blue, n_bins)]++;

				h_hsv.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
				h_hsv.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
				h_hsv.V[to_hist_bin_u8(hsv.val, n_bins)]++;

				h_yuv.Y[to_hist_bin_u8(yuv.y, n_bins)]++;
				h_yuv.U[to_hist_bin_u8(yuv.u, n_bins)]++;
				h_yuv.V[to_hist_bin_u8(yuv.v, n_bins)]++;
			}
		}

		auto const total = (r32)src.width * src.height;

		for (u32 i = 0; i < 9; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_yuv(ViewYUV const& src, Histogram9r32& dst)
	{
		constexpr u32 PIXEL_STEP = 2;

		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_yuv = dst.yuv;
		auto n_bins = dst.n_bins;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v) 
		{
			auto rgba = yuv::u8_to_rgba_u8(yuv_y, yuv_u, yuv_v);
			auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			h_rgb.R[to_hist_bin_u8(rgba.red, n_bins)]++;
			h_rgb.G[to_hist_bin_u8(rgba.green, n_bins)]++;
			h_rgb.B[to_hist_bin_u8(rgba.blue, n_bins)]++;

			h_hsv.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			h_hsv.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			h_yuv.Y[to_hist_bin_u8(yuv_y, n_bins)]++;
			h_yuv.U[to_hist_bin_u8(yuv_u, n_bins)]++;
			h_yuv.V[to_hist_bin_u8(yuv_v, n_bins)]++;
		};

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422*)s2;
			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				update_bins(yuv.y1, yuv.u, yuv.v);
				update_bins(yuv.y2, yuv.u, yuv.v);
			}
		}

		auto const total = (r32)src.width * src.height;

		for (u32 i = 0; i < 9; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	void make_histograms(View const& src, Histogram9r32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(dst.n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % dst.n_bins == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_rgb(src, dst);
	}


	void make_histograms(ViewYUV const& src, Histogram9r32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(dst.n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % dst.n_bins == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_yuv(src, dst);
	}
}


#ifdef SIMAGE_NO_SIMD



#else



#endif