#include "simage_platform.hpp"
#include "../util/execute.hpp"


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
}

#endif // !NDEBUG


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


	static void fill_simd(View const& view, Pixel color)
	{
		static_assert(sizeof(Pixel) == sizeof(r32));

		auto ptr = (r32*)(&color);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_row_simd((r32*)d, *ptr, view.width);
		};

		process_image_rows(view.height, row_func);
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
		static_assert(sizeof(Pixel) == sizeof(r32));

		constexpr u32 STEP = simd::VEC_LEN;

		r32* src = 0;
		r32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i)
		{
			src = (r32*)(src_begin + i);
			dst = (r32*)(dst_begin + i);

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
		static_assert(sizeof(u8) * 4 == sizeof(r32));

		constexpr u32 STEP = simd::VEC_LEN * sizeof(r32) / sizeof(u8);

		r32* src = 0;
		r32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i)
		{
			src = (r32*)(src_begin + i);
			dst = (r32*)(dst_begin + i);

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
	void map(ViewGray const& src, View const& dst)
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
}