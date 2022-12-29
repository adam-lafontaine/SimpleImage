#include "simage.hpp"
#include "../util/memory_buffer.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"

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
				auto hsv = hsv::from_rgb(rgba.red, rgba.green, rgba.blue);
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

		constexpr auto ch_max = cs::to_channel_u8(1.0f);

		auto const row_func = [&](u32 y) 
		{
			auto s = hsv_row_begin(src, y).hsv;
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::to_rgb(s.H[x], s.S[x], s.V[x]);

				auto& rgba = d[x].rgba;				
				rgba.red = cs::to_channel_u8(rgb.red);
				rgba.green = cs::to_channel_u8(rgb.green);
				rgba.blue = cs::to_channel_u8(rgb.blue);
				rgba.alpha = ch_max;
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

				auto hsv = hsv::from_rgb(r, g, b);
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
				auto rgb = hsv::to_rgb(s.H[x], s.S[x], s.V[x]);
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
				auto rgb = yuv::to_rgb(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::to_rgb(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		};

		process_rows(src.height, row_func);
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

		constexpr auto r4 = id_cast(RGBA::R);
		constexpr auto g4 = id_cast(RGBA::G);
		constexpr auto b4 = id_cast(RGBA::B);

		constexpr auto r3 = id_cast(RGB::R);
		constexpr auto g3 = id_cast(RGB::G);
		constexpr auto b3 = id_cast(RGB::B);

		ViewRGBr32 rgb;

		rgb.image_width = view.image_width;
		rgb.width = view.width;
		rgb.height = view.height;
		rgb.range = view.range;

		rgb.image_channel_data[r3] = view.image_channel_data[r4];
		rgb.image_channel_data[g3] = view.image_channel_data[g4];
		rgb.image_channel_data[b3] = view.image_channel_data[b4];

		return rgb;
	}
}


/* fill */

namespace simage
{
	template <size_t N>
	static void fill_n_channels(ViewCHr32<N> const& view, Pixel color)
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


	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				d[x] = color;
			}
		};

		process_rows(view.height, row_func);
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				d[x] = gray;
			}
		};

		process_rows(view.height, row_func);
	}


	void fill(View4r32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3r32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1r32 const& view, u8 gray)
	{
		assert(verify(view));

		auto const gray32 = cs::to_channel_r32(gray);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				d[x] = gray32;
			}
		};

		process_rows(view.height, row_func);
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

			Range2Du32 r;
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


/* histogram */

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


	inline constexpr u8 to_hist_bin_u8(r32 val, u32 n_bins)
	{
		return cs::to_channel_u8(val) * n_bins / 256;
	}


	static void make_histograms(View const& src, HistRGBr32& h_rgb, HistHSVr32& h_hsv, HistYUVr32& h_yuv, u32 n_bins)
	{
		u32 rgb_r_counts[MAX_HIST_BINS] = { 0 };
		u32 rgb_g_counts[MAX_HIST_BINS] = { 0 };
		u32 rgb_b_counts[MAX_HIST_BINS] = { 0 };

		u32 hsv_h_counts[MAX_HIST_BINS] = { 0 };
		u32 hsv_s_counts[MAX_HIST_BINS] = { 0 };
		u32 hsv_v_counts[MAX_HIST_BINS] = { 0 };

		u32 yuv_y_counts[MAX_HIST_BINS] = { 0 };
		u32 yuv_u_counts[MAX_HIST_BINS] = { 0 };
		u32 yuv_v_counts[MAX_HIST_BINS] = { 0 };


		for (u32 y = 0; y < src.height; y += 2)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += 2)
			{
				auto rgba = s[x].rgba;

				rgb_r_counts[to_hist_bin_u8(rgba.red, n_bins)]++;
				rgb_g_counts[to_hist_bin_u8(rgba.green, n_bins)]++;
				rgb_b_counts[to_hist_bin_u8(rgba.blue, n_bins)]++;

				auto red = cs::to_channel_r32(rgba.red);
				auto green = cs::to_channel_r32(rgba.green);
				auto blue = cs::to_channel_r32(rgba.blue);

				auto hsv = hsv::from_rgb(red, green, blue);
				auto yuv = yuv::from_rgb(red, green, blue);				

				hsv_h_counts[to_hist_bin_u8(hsv.hue, n_bins)]++;
				hsv_s_counts[to_hist_bin_u8(hsv.sat, n_bins)]++;
				hsv_v_counts[to_hist_bin_u8(hsv.val, n_bins)]++;

				yuv_y_counts[to_hist_bin_u8(yuv.y, n_bins)]++;
				yuv_u_counts[to_hist_bin_u8(yuv.u, n_bins)]++;
				yuv_v_counts[to_hist_bin_u8(yuv.v, n_bins)]++;
			}
		}

		auto const total = (r32)src.width * src.height;

		for (u32 bin = 0; bin < MAX_HIST_BINS; ++bin)
		{
			h_rgb.R[bin] = rgb_r_counts[bin] / total;
			h_rgb.G[bin] = rgb_g_counts[bin] / total;
			h_rgb.B[bin] = rgb_b_counts[bin] / total;

			h_hsv.H[bin] = hsv_h_counts[bin] / total;
			h_hsv.S[bin] = hsv_s_counts[bin] / total;
			h_hsv.V[bin] = hsv_v_counts[bin] / total;

			h_yuv.Y[bin] = yuv_y_counts[bin] / total;
			h_yuv.U[bin] = yuv_u_counts[bin] / total;
			h_yuv.V[bin] = yuv_v_counts[bin] / total;
		}
	}


	void histograms(View const& src, Histogram9r32& hist)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(hist.n_bins <= MAX_HIST_BINS);

		hist.rgb = { 0 };
		hist.hsv = { 0 };
		hist.yuv = { 0 };

		make_histograms(src, hist.rgb, hist.hsv, hist.yuv, hist.n_bins);
	}
}

