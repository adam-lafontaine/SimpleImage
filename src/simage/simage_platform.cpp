#include "simage_platform.hpp"
#include "../util/execute.hpp"
#include "../util/color_space.hpp"


#ifndef SIMAGE_NO_SIMD
#include "../util/simd.hpp"
#endif // !SIMAGE_NO_SIMD



static void process_image_rows(u32 n_rows, id_func_t const& row_func)
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
			range.x_begin < range.x_end&&
			range.y_begin < range.y_end&&
			range.x_begin < image.width&&
			range.x_end <= image.width &&
			range.y_begin < image.height&&
			range.y_end <= image.height;
	}

#endif // !NDEBUG
}


/* platform */

namespace simage
{
	template <typename T>
	static bool do_create_image(Matrix2D<T>& image, u32 width, u32 height)
	{
		image.data_ = (T*)malloc(sizeof(T) * width * height);
		if (!image.data_)
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


/* row_offset_begin */

namespace simage
{
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
}


/* make_view */

namespace simage
{
	template <typename T>
	static MatrixView<T> do_make_view(Matrix2D<T> const& image)
	{
		MatrixView<T> view;

		view.matrix_data_ = image.data_;
		view.matrix_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		return view;
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


	ViewBGR make_view(ImageBGR const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewRGB make_view(ImageRGB const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}
}


/* sub_view */

namespace simage
{
	template <typename T>
	static MatrixView<T> do_sub_view(Matrix2D<T> const& image, Range2Du32 const& range)
	{
		MatrixView<T> sub_view;

		sub_view.matrix_data_ = image.data_;
		sub_view.matrix_width = image.width;
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

		sub_view.matrix_data_ = view.matrix_data_;
		sub_view.matrix_width = view.matrix_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

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


	ViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto width = range.x_end - range.x_begin;
		Range2Du32 camera_range = range;
		camera_range.x_end = camera_range.x_begin + width / 2;

		assert(verify(camera_src, camera_range));

		auto sub_view = do_sub_view(camera_src, camera_range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewBGR sub_view(ImageBGR const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewBGR sub_view(ViewBGR const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewRGB sub_view(ImageRGB const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewRGB sub_view(ViewRGB const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
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

#ifdef SIMAGE_NO_SIMD

	template <class VIEW, typename COLOR>
	static void do_fill(VIEW const& view, COLOR color)
	{
		fill_no_simd(view, color);
	}

#else

	static void fill_row_simd(f32* dst_begin, f32 value, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		f32* dst = 0;
		f32* val = &value;
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


	static void fill_simd(View const& view, Pixel color)
	{
		static_assert(sizeof(Pixel) == sizeof(f32));

		auto ptr = (f32*)(&color);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd((f32*)d, *ptr, view.width);
		};

		process_image_rows(view.height, row_func);
	}


	static void fill_simd(ViewGray const& view, u8 gray)
	{
		static_assert(4 * sizeof(u8) == sizeof(f32));

		u8 bytes[4] = { gray, gray, gray, gray };
		auto ptr = (f32*)bytes;
		auto len32 = view.width / 4;

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd((f32*)d, *ptr, len32);

			for (u32 x = len32 * 4; x < view.width; ++x)
			{
				d[x] = gray;
			}
		};

		process_image_rows(view.height, row_func);
	}


	template <class VIEW, typename COLOR>
	static void do_fill(VIEW const& view, COLOR color)
	{
		auto len32 = view.width * sizeof(COLOR) / sizeof(f32);
		if (len32 < simd::VEC_LEN)
		{
			fill_no_simd(view, color);
		}
		else
		{
			fill_simd(view, color);
		}
	}


#endif // SIMAGE_NO_SIMD


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
}


/* copy */

namespace simage
{
	template <class IMG_SRC, class IMG_DST>
	static void copy_no_simd(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x];
			}
		};

		process_image_rows(src.height, row_func);
	}


#ifdef SIMAGE_NO_SIMD

	template <class IMG_SRC, class IMG_DST>
	static void do_copy(IMG_SRC const& src, IMG_DST const& dst)
	{
		copy_no_simd(src, dst);
	}

#else

	static void simd_copy_row(Pixel* src_begin, Pixel* dst_begin, u32 length)
	{
		static_assert(sizeof(Pixel) == sizeof(f32));

		constexpr u32 STEP = simd::VEC_LEN;

		f32* src = 0;
		f32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i)
		{
			src = (f32*)(src_begin + i);
			dst = (f32*)(dst_begin + i);

			vec = simd::load(src);
			simd::store(dst, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void simd_copy_row(u8* src_begin, u8* dst_begin, u32 length)
	{
		static_assert(sizeof(u8) * 4 == sizeof(f32));

		constexpr u32 STEP = simd::VEC_LEN * sizeof(f32) / sizeof(u8);

		f32* src = 0;
		f32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i)
		{
			src = (f32*)(src_begin + i);
			dst = (f32*)(dst_begin + i);

			vec = simd::load(src);
			simd::store(dst, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	template <class IMG>
	static void copy_simd(IMG const& src, IMG const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			simd_copy_row(s, d, src.width);
		};

		process_image_rows(src.height, row_func);
	}


	template <class IMG>
	static void do_copy(IMG const& src, IMG const& dst)
	{
		if (src.width < simd::VEC_LEN)
		{
			copy_no_simd(src, dst);
		}
		else
		{
			copy_simd(src, dst);
		}
	}

#endif // SIMAGE_NO_SIMD


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}


	void copy(ViewGray const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}
}


/* map */

namespace simage
{
	void map_gray(View const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				d[x] = gray::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_gray(ViewGray const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				RGBAu8 gray = { s[x], s[x], s[x], 255 };

				d[x].rgba = gray;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_yuv(ViewYUV const& src, View const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		auto const row_func = [&](u32 y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			auto d = row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgba = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;

				++x;
				rgba = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_gray(ViewYUV const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x].y;
			}
		};

		process_image_rows(src.height, row_func);
	}


	void map_rgb(ViewBGR const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto& rgba = d[x].rgba;
				rgba.red = s[x].red;
				rgba.green = s[x].green;
				rgba.blue = s[x].blue;
				rgba.alpha = 255;
			}
		};

		process_image_rows(src.height, row_func);
	}	


	void map_rgb(ViewRGB const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto& rgba = d[x].rgba;
				rgba.red = s[x].red;
				rgba.green = s[x].green;
				rgba.blue = s[x].blue;
				rgba.alpha = 255;
			}
		};

		process_image_rows(src.height, row_func);
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


	static void for_each_rgb(View const& src, std::function<void(u8, u8, u8)> const& rgb_func)
	{
		constexpr u32 PIXEL_STEP = 4;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += PIXEL_STEP)
			{
				auto& rgba = s[x].rgba;

				rgb_func(rgba.red, rgba.green, rgba.blue);
			}
		}
	}


	static void for_each_yuv(ViewYUV const& src, std::function<void(u8, u8, u8)> const& yuv_func)
	{
		constexpr u32 PIXEL_STEP = 4;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				yuv_func(yuv.y1, yuv.u, yuv.v);
				yuv_func(yuv.y2, yuv.u, yuv.v);
			}
		}
	}


	static void for_each_bgr(ViewBGR const& src, std::function<void(u8, u8, u8)> const& rgb_func)
	{
		constexpr u32 PIXEL_STEP = 4;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += PIXEL_STEP)
			{
				auto& bgr = s[x];

				rgb_func(bgr.red, bgr.green, bgr.blue);
			}
		}
	}


	static void make_histograms_from_rgb(View const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;
		auto n_bins = dst.n_bins;

		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);
			auto lch = lch::u8_from_rgb_u8(red, green, blue);
			auto yuv = yuv::u8_from_rgb_u8(red, green, blue);

			h_rgb.R[to_hist_bin_u8(red, n_bins)]++;
			h_rgb.G[to_hist_bin_u8(green, n_bins)]++;
			h_rgb.B[to_hist_bin_u8(blue, n_bins)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			h_lch.L[to_hist_bin_u8(lch.light, n_bins)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			h_lch.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			h_yuv.Y[to_hist_bin_u8(yuv.y, n_bins)]++;
			h_yuv.U[to_hist_bin_u8(yuv.u, n_bins)]++;
			h_yuv.V[to_hist_bin_u8(yuv.v, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_yuv(ViewYUV const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;
		auto n_bins = dst.n_bins;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			h_rgb.R[to_hist_bin_u8(rgba.red, n_bins)]++;
			h_rgb.G[to_hist_bin_u8(rgba.green, n_bins)]++;
			h_rgb.B[to_hist_bin_u8(rgba.blue, n_bins)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			h_lch.L[to_hist_bin_u8(lch.light, n_bins)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			h_lch.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			h_yuv.Y[to_hist_bin_u8(yuv_y, n_bins)]++;
			h_yuv.U[to_hist_bin_u8(yuv_u, n_bins)]++;
			h_yuv.V[to_hist_bin_u8(yuv_v, n_bins)]++;
		};

		f32 total = 0.0f;

		for_each_yuv(src, update_bins);

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_bgr(ViewBGR const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;
		auto n_bins = dst.n_bins;

		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue)
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);
			auto lch = lch::u8_from_rgb_u8(red, green, blue);
			auto yuv = yuv::u8_from_rgb_u8(red, green, blue);

			h_rgb.R[to_hist_bin_u8(red, n_bins)]++;
			h_rgb.G[to_hist_bin_u8(green, n_bins)]++;
			h_rgb.B[to_hist_bin_u8(blue, n_bins)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			h_lch.L[to_hist_bin_u8(lch.light, n_bins)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			h_lch.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			h_yuv.Y[to_hist_bin_u8(yuv.y, n_bins)]++;
			h_yuv.U[to_hist_bin_u8(yuv.u, n_bins)]++;
			h_yuv.V[to_hist_bin_u8(yuv.v, n_bins)]++;

			total++;
		};

		for_each_bgr(src, update_bins);

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}
	
	
	void make_histograms(View const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(dst.n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % dst.n_bins == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_rgb(src, dst);
	}


	void make_histograms(ViewYUV const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(dst.n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % dst.n_bins == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_yuv(src, dst);
	}


	void make_histograms(ViewBGR const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(dst.n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % dst.n_bins == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_bgr(src, dst);
	}


	void make_histograms(View const& src, HistRGBf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			dst.R[to_hist_bin_u8(red, n_bins)]++;
			dst.G[to_hist_bin_u8(green, n_bins)]++;
			dst.B[to_hist_bin_u8(blue, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.R[bin] /= total;
			dst.G[bin] /= total;
			dst.B[bin] /= total;
		}
	}


	void make_histograms(View const& src, HistHSVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);

			if (hsv.sat)
			{
				dst.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			dst.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			dst.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.H[bin] /= total;
			dst.S[bin] /= total;
			dst.V[bin] /= total;
		}
	}


	void make_histograms(View const& src, HistLCHf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto lch = lch::u8_from_rgb_u8(red, green, blue);

			dst.L[to_hist_bin_u8(lch.light, n_bins)]++;
			dst.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			dst.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.L[bin] /= total;
			dst.C[bin] /= total;
			dst.H[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistYUVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			dst.Y[to_hist_bin_u8(yuv_y, n_bins)]++;
			dst.U[to_hist_bin_u8(yuv_u, n_bins)]++;
			dst.V[to_hist_bin_u8(yuv_v, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.Y[bin] /= total;
			dst.U[bin] /= total;
			dst.V[bin] /= total;
		}
	}
	
	
	void make_histograms(ViewYUV const& src, HistRGBf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);

			dst.R[to_hist_bin_u8(rgba.red, n_bins)]++;
			dst.G[to_hist_bin_u8(rgba.green, n_bins)]++;
			dst.B[to_hist_bin_u8(rgba.blue, n_bins)]++;

			total++;
		};		

		for_each_yuv(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.R[bin] /= total;
			dst.G[bin] /= total;
			dst.B[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistHSVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			if (hsv.sat)
			{
				dst.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			dst.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			dst.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.H[bin] /= total;
			dst.S[bin] /= total;
			dst.V[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistLCHf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			dst.L[to_hist_bin_u8(lch.light, n_bins)]++;
			dst.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			dst.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.L[bin] /= total;
			dst.C[bin] /= total;
			dst.H[bin] /= total;
		}
	}
}


