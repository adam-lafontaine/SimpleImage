#include "../simage.hpp"
#include "util/execute.hpp"
#include "util/color_space.hpp"

#include <cmath>

namespace cs = color_space;


static void process_by_row(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
}


/* channel pixels */

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
	class LCHp
	{
	public:
		T* L;
		T* C;
		T* H;
	};


	template <typename T>
	class YUVp
	{
	public:
		T* Y;
		T* U;
		T* V;
	};


	using RGBf32p = RGBp<f32>;
	using RGBAf32p = RGBAp<f32>;

	using HSVf32p = HSVp<f32>;
	using YUVf32p = YUVp<f32>;
	using LCHf32p = LCHp<f32>;
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


/* row begin */

namespace simage
{
	template <typename T>
	static inline T* row_begin(Matrix2D<T> const& image, u32 y)
	{
		return image.data_ + (u64)(y * image.width);
	}


	template <typename T>
	static inline T* row_begin(MatrixView<T> const& view, u32 y)
	{
		return view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
	}


	template <typename T>
	static inline T* xy_at(MatrixView<T> const& view, u32 x, u32 y)
	{
		return row_begin(view, y) + x;
	}


    template <typename T>
	static T* row_offset_begin(MatrixView<T> const& view, u32 y, int y_offset)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.matrix_width + view.x_begin;

		auto ptr = view.matrix_data_ + (u64)(offset);
		assert(ptr);

		return ptr;
	}


    template <typename T, size_t N>
	static inline u64 row_offset(ChannelView<T, N> const& view, u32 y)
	{
		return ((u64)view.y_begin + y) * view.channel_width_ + view.x_begin;
	}


	static RGBf32p rgb_row_begin(ViewRGBf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		RGBf32p rgb{};

		rgb.R = view.channel_data_[id_cast(RGB::R)] + offset;
		rgb.G = view.channel_data_[id_cast(RGB::G)] + offset;
		rgb.B = view.channel_data_[id_cast(RGB::B)] + offset;

		return rgb;
	}


	static RGBAf32p rgba_row_begin(ViewRGBAf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		RGBAf32p rgba{};

		rgba.R = view.channel_data_[id_cast(RGBA::R)] + offset;
		rgba.G = view.channel_data_[id_cast(RGBA::G)] + offset;
		rgba.B = view.channel_data_[id_cast(RGBA::B)] + offset;
		rgba.A = view.channel_data_[id_cast(RGBA::A)] + offset;

		return rgba;
	}


	static HSVf32p hsv_row_begin(ViewHSVf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		HSVf32p hsv{};

		hsv.H = view.channel_data_[id_cast(HSV::H)] + offset;
		hsv.S = view.channel_data_[id_cast(HSV::S)] + offset;
		hsv.V = view.channel_data_[id_cast(HSV::V)] + offset;

		return hsv;
	}


	static YUVf32p yuv_row_begin(ViewYUVf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		YUVf32p yuv{};

		yuv.Y = view.channel_data_[id_cast(YUV::Y)] + offset;
		yuv.U = view.channel_data_[id_cast(YUV::U)] + offset;
		yuv.V = view.channel_data_[id_cast(YUV::V)] + offset;

		return yuv;
	}


	static LCHf32p lch_row_begin(ViewLCHf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		LCHf32p lch{};

		lch.L = view.channel_data_[id_cast(LCH::L)] + offset;
		lch.C = view.channel_data_[id_cast(LCH::C)] + offset;
		lch.H = view.channel_data_[id_cast(LCH::H)] + offset;

		return lch;
	}


	template <typename T, size_t N>
	static std::array<T*, N> view_row_begin(ChannelView<T, N> const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		std::array<T*, N> rows = { 0 };

		for (u32 ch = 0; ch < N; ++ch)
		{
			rows[ch] = view.channel_data_[ch] + offset;
		}

		return rows;
	}


	template <typename T, size_t N>
	static T* channel_row_begin(ChannelView<T, N> const& view, u32 y, u32 ch)
	{
		assert(verify(view));

		assert(y < view.height);

		auto offset = row_offset(view, y);

		return view.channel_data_[ch] + offset;
	}
}


/* make_view static */

namespace simage
{
    template <typename T>
	static MatrixView<T> do_make_view(Matrix2D<T> const& image)
	{
		MatrixView<T> view;

		view.matrix_data_ = image.data_;
		view.matrix_width = image.width;

		view.width = image.width;
		view.height = image.height;

		view.range = make_range(image.width, image.height);

		return view;
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


    template <typename T, size_t N>
	static void do_make_view_n(ChannelView<T, N>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
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
}


/* sub_view static */

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


    template <typename T, size_t N>
	static ChannelView<T, N> do_sub_view(ChannelView<T, N> const& view, Range2Du32 const& range)
	{
		ChannelView<T, N> sub_view;

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
}


/* fill static */

namespace simage
{
    template <typename T>
	static void fill_channel_row(T* d, T value, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			d[i] = value;
		}
	}


	template <typename T>
	static void fill_channel(View1<T> const& view, T value)
	{		
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			fill_channel_row(d, value, view.width);
		};

		process_by_row(view.height, row_func);
	}


	template <size_t N>
	static void fill_n_channels(ChannelView<f32, N> const& view, Pixel color)
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
				fill_channel_row(d, channels[ch], view.width);
			}
		};

		process_by_row(view.height, row_func);
	}
}


/* convolution kernels */

namespace simage
{
    constexpr std::array<f32, 33> make_grad_x_3x11()
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


	constexpr std::array<f32, 33> make_grad_y_11x3()
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


    static constexpr std::array<f32, 9> make_gauss_3()
	{
		std::array<f32, 9> kernel = 
		{
			1.0f, 2.0f, 1.0f,
			2.0f, 4.0f, 2.0f,
			1.0f, 2.0f, 1.0f,
		};

		for (u32 i = 0; i < 9; ++i)
		{
			kernel[i] /= 16.0f;
		}

		return kernel;
	}


	static constexpr std::array<f32, 25> make_gauss_5()
	{
		std::array<f32, 25> kernel =
		{
			1.0f, 4.0f,  6.0f,  4.0f,  1.0f,
			4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
			6.0f, 24.0f, 36.0f, 24.0f, 6.0f,
			4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
			1.0f, 4.0f,  6.0f,  4.0f,  1.0f,
		};

		for (u32 i = 0; i < 25; ++i)
		{
			kernel[i] /= 256.0f;
		}

		return kernel;
	}


    static constexpr auto GRAD_X_3x11 = make_grad_x_3x11();
    static constexpr auto GRAD_Y_3x11 = make_grad_y_11x3();

    static constexpr std::array<f32, 15> GRAD_X_3x5
    {
        -0.08f, -0.12f, 0.0f, 0.12f, 0.08f
        -0.16f, -0.24f, 0.0f, 0.24f, 0.16f
        -0.08f, -0.12f, 0.0f, 0.12f, 0.08f
    };

    static constexpr std::array<f32, 15> GRAD_Y_3x5
    {
        -0.08f, -0.16f, -0.08f,
        -0.12f, -0.24f, -0.12f,
         0.00f,  0.00f,  0.00f,
         0.12f,  0.24f,  0.12f,
         0.08f,  0.16f,  0.08f,
    };

    static constexpr std::array<f32, 9> GRAD_X_3x3
	{
		-0.25f,  0.0f,  0.25f,
		-0.50f,  0.0f,  0.50f,
		-0.25f,  0.0f,  0.25f,
	};

	static constexpr std::array<f32, 9> GRAD_Y_3x3
	{
		-0.25f, -0.50f, -0.25f,
		 0.0f,   0.0f,   0.0f,
		 0.25f,  0.50f,  0.25f,
	};


	static constexpr auto GAUSS_3x3 = make_gauss_3();
	static constexpr auto GAUSS_5x5 = make_gauss_5();
}


/* convolve static */

namespace simage
{
	template <typename T>
    static f32 convolve_at_xy(View1<T> const& view, u32 x, u32 y, std::array<f32, 9> const& kernel_3x3)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 3; ++v)
        {
            auto s = row_begin(view, y - 1 + v);
            for (u32 u = 0; u < 3; ++u)
            {
                total += s[x - 1 + u] * kernel_3x3[w++];
            }
        }

        return total;
    }


	template <typename T>
    static f32 convolve_at_xy(View1<T> const& view, u32 x, u32 y, std::array<f32, 25> const& kernel_5x5)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 5; ++v)
        {
            auto s = row_begin(view, y - 2 + v);
            for (u32 u = 0; u < 5; ++u)
            {
                total += s[x - 2 + u] * kernel_5x5[w++];
            }
        }

        return total;
    }
}


/* gradients static */

namespace simage
{
    template <typename T>
    static f32 gradient_x_11(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 3; ++v)
        {
            auto s = row_begin(view, y - 1 + v);
            for (u32 u = 0; u < 11; ++u)
            {
                total += s[x - 5 + u] * GRAD_X_3x11[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_y_11(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 11; ++v)
        {
            auto s = row_begin(view, y - 5 + v);
            for (u32 u = 0; u < 3; ++u)
            {
                total += s[x - 1 + u] * GRAD_Y_3x11[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_x_5(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 3; ++v)
        {
            auto s = row_begin(view, y - 1 + v);
            for (u32 u = 0; u < 5; ++u)
            {
                total += s[x - 2 + u] * GRAD_X_3x5[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_y_5(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 5; ++v)
        {
            auto s = row_begin(view, y - 2 + v);
            for (u32 u = 0; u < 3; ++u)
            {
                total += s[x - 1 + u] * GRAD_Y_3x5[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_x_3(View1<T> const& view, u32 x, u32 y)
    {
		return convolve_at_xy(view, x, y, GRAD_X_3x3);
    }


    template <typename T>
    static f32 gradient_y_3(View1<T> const& view, u32 x, u32 y)
    {
		return convolve_at_xy(view, x, y, GRAD_X_3x3);
    }


    template <typename T>
    static T gradient_xy_11(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_11(view, x, y);
        auto grad_y = gradient_y_11(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static T gradient_xy_5(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_5(view, x, y);
        auto grad_y = gradient_y_5(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static T gradient_xy_3(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_3(view, x, y);
        auto grad_y = gradient_y_3(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static void gradients_row(View1<T> const& src, View1<T> const& dst, u32 y)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y >= 5 && y < height - 5)
		{
			d[0] = d[width - 1] = 0;

            d[1] = gradient_xy_3(src, 1, y);
            d[width - 2] = gradient_xy_3(src, width - 2, y);

            for (u32 x = 2; x < 5; ++x)
            {
                d[x] = gradient_xy_5(src, x, y);
                d[width - x - 1] = gradient_xy_5(src, width - x - 1, y);
            }

            for (u32 x = 5; x < width - 5; ++x)
            {
                d[x] = gradient_xy_11(src, x, y);
            }

			return;
		}
		
		if (y >= 2 && y < 5 || y >= height - 5 && y <= height - 3)
		{
			d[0] = d[width - 1] = 0;

            d[1] = gradient_xy_3(src, 1, y);
            d[width - 2] = gradient_xy_3(src, width - 2, y);

            for (u32 x = 2; x < width - 3; ++x)
            {
                d[x] = gradient_xy_5(src, x, y);
            }
            return;
		}

		if (y == 1 || y == height - 2)
		{
			 d[0] = d[width - 1] = 0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = gradient_xy_3(src, x, y);
            }
            return;
		}

		// y == 0 || y == height - 1
		for (u32 x = 0; x < width ; ++x)
		{
			d[x] = (T)0;
		}
    }


    template <typename T, class CONV>
    static void gradients_x_row(View1<T> const& src, View1<T> const& dst, u32 y, CONV const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y > 0 && y < height - 1)
		{
			d[0] = d[width - 1] = (T)0;

            d[1] = convert(gradient_x_3(src, 1, y));
            d[width - 2] = convert(gradient_x_3(src, width - 2, y));

            for (u32 x = 2; x < 5; ++x)
            {
                d[x] = convert(gradient_x_5(src, x, y));
                d[width - x - 1] = convert(gradient_x_5(src, width - x - 1, y));
            }

            for (u32 x = 5; x < width - 5; ++x)
            {
                d[x] = convert(gradient_x_11(src, x, y));
            }

			return;
		}

		// y == 0 || y == height - 1
		for (u32 x = 0; x < width ; ++x)
		{
			d[x] = (T)0;
		}
    }


    template <typename T, class CONV>
    static void gradients_y_row(View1<T> const& src, View1<T> const& dst, u32 y, CONV const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y >= 5 && y < height - 5)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_11(src, x, y));
            }

			return;
		}
		
		if (y >= 2 && y < 5 || y >= height - 5 && y <= height - 3)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_5(src, x, y));
            }
            return;
		}

		if (y == 1 || y == height - 2)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_3(src, x, y));
            }
            return;
		}

		if (y == 0 || y == height - 1)
		{
			for (u32 x = 0; x < width ; ++x)
            {
                d[x] = (T)0;
            }
            return;
		}
    }


    template <typename T>
    static void gradients_1(View1<T> const& src, View1<T> const& dst)
    {
        auto const row_func = [&](u32 y)
        {            
            gradients_row(src, dst, y);
        };

        process_by_row(src.height, row_func);
    }
	
	
	template <typename T>
    static void gradients_xy_1(View1<T> const& src, View1<T> const& x_dst, View1<T> const& y_dst)
    {
		auto const convert = [](f32 grad){ return (T)grad; };

        auto const row_func = [&](u32 y)
        {            
            gradients_x_row(src, x_dst, y, convert);
            gradients_y_row(src, y_dst, y, convert);
        };

        process_by_row(src.height, row_func);
    }


    template <typename T>
    static void gradients_unsigned_xy_1(View1<T> const& src, View1<T> const& x_dst, View1<T> const& y_dst)
    {
		auto const convert = [](f32 grad){ return (T)std::abs(grad); };

        auto const row_func = [&](u32 y)
        {            
            gradients_x_row(src, x_dst, y, convert);
            gradients_y_row(src, y_dst, y, convert);
        };

        process_by_row(src.height, row_func);
    }
}


/* blur static */

namespace simage
{  
	template <typename T>
	static void blur_row(View1<T> const& src, View1<T> const& dst, u32 y)
	{
		auto const width = src.width;
        auto const height = src.height;

		auto s = row_begin(src, y);
		auto d = row_begin(dst, y);

		if (y >= 2 && y < height - 2)
		{
			d[0] = s[0];
			d[width - 1] = s[width - 1];

			d[1] = (T)convolve_at_xy(src, 1, y, GAUSS_3x3);
			d[width - 2] = (T)convolve_at_xy(src, width - 2, y, GAUSS_3x3);

			for (u32 x = 2; x < width - 2; ++x)
			{
				d[x] = (T)convolve_at_xy(src, x, y, GAUSS_5x5);
			}

			return;
		}

		if (y == 1 || y == height - 2)
		{
			d[0] = s[0];
			d[width - 1] = s[width - 1];

			for (u32 x = 1; x < width - 1; ++x)
			{
				d[x] = (T)convolve_at_xy(src, x, y, GAUSS_3x3);
			}

			return;
		}

		// y == 0 || y == height - 1
		for (u32 x = 0; x < width; ++x)
		{
			d[x] = s[x];
		}
	}

	 
    template <typename T>
	static void blur_1(View1<T> const& src, View1<T> const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			blur_row(src, dst, y);
		};

		process_by_row(src.height, row_func);
	}


	template <typename T, size_t N>
	static void blur_n(ChannelView<T, N> const& src, ChannelView<T, N> const& dst)
	{
		for (u32 ch = 0; ch < N; ++ch)
		{
			blur_1(select_channel(src, ch), select_channel(dst, ch));
		}
	}
}


/* rotate static */

namespace simage
{
    static Point2Df32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, f32 theta_rotate)
	{
		auto const dx_dst = (f32)pt.x - (f32)origin.x;
		auto const dy_dst = (f32)pt.y - (f32)origin.y;

		auto const radius = std::hypotf(dx_dst, dy_dst);

		auto const theta_dst = atan2f(dy_dst, dx_dst);
		auto const theta_src = theta_dst - theta_rotate;

		auto const dx_src = radius * cosf(theta_src);
		auto const dy_src = radius * sinf(theta_src);

		Point2Df32 pt_src{};
		pt_src.x = (f32)origin.x + dx_src;
		pt_src.y = (f32)origin.y + dy_src;

		return pt_src;
	}


    template <typename T>
    static T get_pixel_value(MatrixView<T> const& src, Point2Df32 location)
    {
        constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return 0;
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
    }
	

	static Pixel get_pixel_value(View const& src, Point2Df32 location)
	{
		constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return to_pixel(0, 0, 0);
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
	}


    template <typename T>
    static void rotate_1(View1<T> const& src, View1<T> const& dst, Point2Du32 origin, f32 rad)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, rad);
				d[x] = get_pixel_value(src, src_pt);
			}
		};

		process_by_row(src.height, row_func);
	}


    template <typename T, size_t N>
	void rotate_channels(ChannelView<T, N> const& src, ChannelView<T, N> const& dst, Point2Du32 origin, f32 rad)
	{
		constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const row_func = [&](u32 y)
		{
			auto d = view_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, rad);
				auto is_out = src_pt.x < zero || src_pt.x >= width || src_pt.y < zero || src_pt.y >= height;

				if (src_pt.x < zero || src_pt.x >= width || src_pt.y < zero || src_pt.y >= height)
				{
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = 0;
					}
				}
				else
				{
					auto src_x = (u32)floorf(src_pt.x);
					auto src_y = (u32)floorf(src_pt.y);
					auto s = view_row_begin(src, src_y);
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = s[ch][src_x];
					}
				}
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* make_view */

namespace simage
{
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


	View make_view(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View view;

		do_make_view_1(view, width, height, buffer);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		ViewGray view;

		do_make_view_1(view, width, height, buffer);
		assert(verify(view));

		return view;
	}
}


/* sub_view */

namespace simage
{
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


/* split channels */

namespace simage
{
	void split_rgb(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = s[x].rgba;
				r[x] = rgba.red;
				g[x] = rgba.green;
				b[x] = rgba.blue;
			}
		};

		process_by_row(src.height, row_func);
	}


	void split_rgba(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue, ViewGray const& alpha)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));
		assert(verify(src, alpha));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);
			auto a = row_begin(alpha, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = s[x].rgba;
				r[x] = rgba.red;
				g[x] = rgba.green;
				b[x] = rgba.blue;
				a[x] = rgba.alpha;
			}
		};

		process_by_row(src.height, row_func);
	}


	void split_hsv(View const& src, ViewGray const& hue, ViewGray const& sat, ViewGray const& val)
	{
		assert(verify(src, hue));
		assert(verify(src, sat));
		assert(verify(src, val));

		auto const row_func = [&](u32 y)
		{
			auto p = row_begin(src, y);
			auto h = row_begin(hue, y);
			auto s = row_begin(sat, y);
			auto v = row_begin(val, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = p[x].rgba;
				auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				h[x] = hsv.hue;
				s[x] = hsv.sat;
				v[x] = hsv.val;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		fill_channel(view, color);
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}
}


/* copy */

namespace simage
{	
	template <typename PIXEL>
	static void copy_row(PIXEL* s, PIXEL* d, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			d[i] = s[i];
		}
	}


	template <class IMG_SRC, class IMG_DST>
	static void do_copy(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			copy_row(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
	}
}


/* alpha blend */

namespace simage
{
	static void alpha_blend_row(Pixel* src, Pixel* cur, Pixel* dst, u32 width)
	{
		auto const blend = [](u8 s, u8 c, f32 a)
		{
			auto blended = a * s + (1.0f - a) * c;
			return (u8)(blended + 0.5f);
		};

		for (u32 x = 0; x < width; ++x)
		{
			auto s = src[x].rgba;
			auto c = cur[x].rgba;
			auto& d = dst[x].rgba;

			auto a = cs::to_channel_f32(s.alpha);
			d.red = blend(s.red, c.red, a);
			d.green = blend(s.green, c.green, a);
			d.blue = blend(s.blue, c.blue, a);
		}
	}


	void alpha_blend(View const& src, View const& cur, View const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_row(s, c, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void alpha_blend(View const& src, View const& cur_dst)
	{
		assert(verify(src, cur_dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(cur_dst, y);

			alpha_blend_row(s, d, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
}


/* for_each_pixel */

namespace simage
{
	template <typename T>
	static void do_for_each_pixel(View1<T> const& view, std::function<void(T&)> const& func)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(s[x]);
			}
		};

		process_by_row(view.height, row_func);
	}


	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
	}


	void for_each_pixel(ViewGray const& view, std::function<void(u8&)> const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
	}
}


/* transform */

namespace simage
{
	template <class IMG_S, class IMG_D, class FUNC>	
	static void do_transform(IMG_S const& src, IMG_D const& dst, FUNC const& func)
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

		process_by_row(src.height, row_func);
	}


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void transform(ViewGray const& src, ViewGray const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void transform(View const& src, ViewGray const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min)
	{
		assert(verify(src, dst));

		return do_transform(src, dst, [&](u8 p){ return p >= min ? p : 0; });
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));

		auto mn = std::min(min, max);
		auto mx = std::max(min, max);

		return do_transform(src, dst, [&](u8 p){ return p >= mn && p <= mx ? p : 0; });
	}


	void binarize(View const& src, ViewGray const& dst, pixel_to_bool_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, [&](Pixel p){ return func(p) ? 255 : 0; });
	}


	void binarize(ViewGray const& src, ViewGray const& dst, u8_to_bool_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, [&](u8 p){ return func(p) ? 255 : 0; });
	}
}


/* blur */

namespace simage
{
    void blur(View const& src, View const& dst)
    {
        assert(verify(src, dst));

		//blur_1(src, dst); TODO
    }


    void blur(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

		blur_1(src, dst);
    }
}


/* gradients */

namespace simage
{
    void gradients(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        gradients_1(src, dst);
    }


    void gradients_xy(ViewGray const& src, ViewGray const& dst_x, ViewGray const& dst_y)
    {
        assert(verify(src, dst_x));
        assert(verify(src, dst_y));

        gradients_unsigned_xy_1(src, dst_x, dst_y);
    }
}


/* rotate */

namespace simage
{
	void rotate(View const& src, View const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}


	void rotate(ViewGray const& src, ViewGray const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}
}


/* centroid */

namespace simage
{
	Point2Du32 centroid(ViewGray const& src)
	{
		f64 total = 0.0;
		f64 x_total = 0.0;
		f64 y_total = 0.0;

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u64 val = s[x] ? 1 : 0;
				total += val;
				x_total += x * val;
				y_total += y * val;
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = (u32)(x_total / total);
			pt.y = (u32)(y_total / total);
		}

		return pt;
	}


	Point2Du32 centroid(ViewGray const& src, u8_to_bool_f const& func)
	{
		f64 total = 0.0;
		f64 x_total = 0.0;
		f64 y_total = 0.0;

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u64 val = func(s[x]) ? 1 : 0;
				total += val;
				x_total += x * val;
				y_total += y * val;
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = (u32)(x_total / total);
			pt.y = (u32)(y_total / total);
		}

		return pt;
	}


	
}


/* skeleton */

namespace simage
{
	template <class GRAY_IMG_T>
	static bool is_outside_edge(GRAY_IMG_T const& img, u32 x, u32 y)
	{
		assert(x >= 1);
		assert(x < img.width);
		assert(y >= 1);
		assert(y < img.height);

		constexpr std::array<int, 8> x_neighbors = { -1,  0,  1,  1,  1,  0, -1, -1 };
		constexpr std::array<int, 8> y_neighbors = { -1, -1, -1,  0,  1,  1,  1,  0 };

		constexpr auto n_neighbors = x_neighbors.size();
		u32 value_count = 0;
		u32 flip = 0;

		auto xi = (u32)(x + x_neighbors[n_neighbors - 1]);
		auto yi = (u32)(y + y_neighbors[n_neighbors - 1]);
		auto val = *xy_at(img, xi, yi);
		bool is_on = val != 0;

		for (u32 i = 0; i < n_neighbors; ++i)
		{
			xi = (u32)(x + x_neighbors[i]);
			yi = (u32)(y + y_neighbors[i]);

			val = *xy_at(img, xi, yi);
			flip += (val != 0) != is_on;

			is_on = val != 0;
			value_count += is_on;
		}

		return value_count > 1 && value_count < 7 && flip == 2;
	}


	template <class GRAY_IMG_T>
	static u32 skeleton_once(GRAY_IMG_T const& img)
	{
		u32 pixel_count = 0;

		auto width = img.width;
		auto height = img.height;

		auto const xy_func = [&](u32 x, u32 y)
		{
			auto& p = *xy_at(img, x, y);
			if (p == 0)
			{
				return;
			}

			if (is_outside_edge(img, x, y))
			{
				p = 0;
			}

			pixel_count += p > 0;
		};

		u32 x_begin = 1;
		u32 x_end = width - 1;
		u32 y_begin = 1;
		u32 y_end = height - 2;
		u32 x = 0;
		u32 y = 0;

		auto const done = [&]() { return !(x_begin < x_end && y_begin < y_end); };

		while (!done())
		{
			// iterate clockwise
			y = y_begin;
			x = x_begin;
			for (; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (++y; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			for (--y; y > y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;

			if (done())
			{
				break;
			}

			// iterate counter clockwise
			for (++x; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (++x; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (--y; y >= y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;
		}

		return pixel_count;
	}


	void skeleton(ViewGray const& src_dst)
	{
		assert(verify(src_dst));

		u32 current_count = 0;
		u32 pixel_count = skeleton_once(src_dst);
		u32 max_iter = 100; // src.width / 2;

		for (u32 i = 1; pixel_count != current_count && i < max_iter; ++i)
		{
			current_count = pixel_count;
			pixel_count = skeleton_once(src_dst);
		}
	}
}


/* make_histograms */

namespace simage
{
namespace hist
{	

	inline constexpr u8 to_hist_bin_u8(u8 val, u32 n_bins)
	{
		return val * n_bins / 256;
	}


	static void for_each_rgb(View const& src, std::function<void(u8, u8, u8)> const& rgb_func)
	{
		constexpr u32 PIXEL_STEP = 1;

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
		constexpr u32 PIXEL_STEP = 1;

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
		constexpr u32 PIXEL_STEP = 1;

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
}


/* make view */

namespace simage
{
	static MemoryBuffer<f32>& reinterpret_buffer(MemoryBuffer<Pixel>& buffer)
	{
		static_assert(sizeof(f32) == sizeof(Pixel));

		return *reinterpret_cast<MemoryBuffer<f32>*>(&buffer);
	}


	View1f32 make_view_1(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View1f32 view;

		do_make_view_1(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View2f32 make_view_2(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View3f32 make_view_3(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View4f32 make_view_4(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}
}


/* sub_view */

namespace simage
{
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
	template <typename T, size_t N, typename CH>
	static View1<T> select_channel(ChannelView<T, N> const& view, CH ch)
	{
		View1<T> view1{};

		view1.matrix_width = view.channel_width_;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.matrix_data_ = view.channel_data_[id_cast(ch)];

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


	View1f32 select_channel(ViewLCHf32 const& view, LCH channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewYUVf32 const& view, YUV channel)
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


/* map */

namespace simage
{
	static void map_channel_row_u8_to_f32(u8* src, f32* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_f32(src[x]);
		}
	}


	static void map_channel_row_f32_to_u8(f32* src, u8* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_u8(src[x]);
		}
	}


	void map_gray(View1u8 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_u8_to_f32(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
	

	void map_gray(View1f32 const& src, View1u8 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_f32_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
	}


	void map_gray(View1f32 const& src, View const& dst)
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


/* map_rgb */

namespace simage
{	
	static void map_rgba_row_u8_to_f32(Pixel* src, RGBAf32p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_f32(src[x].rgba.red);
			dst.G[x] = cs::to_channel_f32(src[x].rgba.green);
			dst.B[x] = cs::to_channel_f32(src[x].rgba.blue);
			dst.A[x] = cs::to_channel_f32(src[x].rgba.alpha);
		}
	}


	static void map_rgba_row_f32_to_u8(RGBAf32p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = cs::to_channel_u8(src.A[x]);
		}
	}


	static void map_rgb_row_u8_to_f32(Pixel* src, RGBf32p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_f32(src[x].rgba.red);
			dst.G[x] = cs::to_channel_f32(src[x].rgba.green);
			dst.B[x] = cs::to_channel_f32(src[x].rgba.blue);
		}
	}


	static void map_rgb_row_f32_to_u8(RGBf32p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = 255;
		}
	}


	void map_rgba(View const& src, ViewRGBAf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgba_row_begin(dst, y);

			map_rgba_row_u8_to_f32(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgba_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgba_row_f32_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}

	
	void map_rgb(View const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			map_rgb_row_u8_to_f32(s, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void map_rgb(ViewRGBf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgb_row_f32_to_u8(s, d, src.width);
		};

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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
				auto rgb = hsv::f32_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
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

		process_by_row(src.height, row_func);
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
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

		process_by_row(src.height, row_func);
	}

/*
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

		process_by_row(dst.height, row_func);
	}*/


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

		process_by_row(dst.height, row_func);
	}


	void map_yuv_rgb(ViewYUVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = yuv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = yuv::f32_to_rgb_u8(s.Y[x], s.U[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* map bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
	{		
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d.R[x] = cs::to_channel_f32(s[x].red);
				d.G[x] = cs::to_channel_f32(s[x].green);
				d.B[x] = cs::to_channel_f32(s[x].blue);
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* fill */

namespace simage
{

	void fill(View4f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1f32 const& view, f32 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}


	void fill(View1f32 const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, cs::to_channel_f32(gray));
	}
}


/* transform */

namespace simage
{
	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y) 
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s[x]);
			}
		};

		process_by_row(src.height, row_func);
	}


	void transform(View2f32 const& src, View1f32 const& dst, std::function<f32(f32, f32)> const& func32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s0[x], s1[x]);
			}
		};

		process_by_row(src.height, row_func);
	}


	void transform(View3f32 const& src, View1f32 const& dst, std::function<f32(f32, f32, f32)> const& func32)
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
				d[x] = func32(s0[x], s1[x], s2[x]);
			}
		};

		process_by_row(src.height, row_func);
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min32 ? s[x] : 0.0f;
			}
		};

		process_by_row(src.height, row_func);
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32, f32 max32)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min32 && s[x] <= max32 ? s[x] : 0.0f;
			}
		};

		process_by_row(src.height, row_func);
	}


	void binarize(View1f32 const& src, View1f32 const& dst, std::function<bool(f32)> func32)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s[x]) ? 1.0f : 0.0f;
			}
		};

		process_by_row(src.height, row_func);
	}
}


/* alpha blend */

namespace simage
{
	static void alpha_blend_row(RGBAf32p const& src, RGBf32p const& cur, RGBf32p const& dst, u32 width)
	{
		auto const blend = [](f32 s, f32 c, f32 a)
		{
			return a * s + (1.0f - a) * c;
		};

		for (u32 x = 0; x < width; ++x)
		{
			auto a = src.A[x];

			dst.R[x] = blend(src.R[x], cur.R[x], a);
			dst.G[x] = blend(src.G[x], cur.G[x], a);
			dst.B[x] = blend(src.B[x], cur.B[x], a);
		}
	}


	void alpha_blend(ViewRGBAf32 const& src, ViewRGBf32 const& cur, ViewRGBf32 const& dst)
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


/* rotate */

namespace simage
{
	void rotate(View4f32 const& src, View4f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View3f32 const& src, View3f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View2f32 const& src, View2f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View1f32 const& src, View1f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}
}


/* blur */

namespace simage
{
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


/* gradients */

namespace simage
{
    void gradients(View1f32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        gradients_1(src, dst);
    }


	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst)
	{
		auto dst_x = select_channel(xy_dst, XY::X);
		auto dst_y = select_channel(xy_dst, XY::Y);

		assert(verify(src, dst_x));
		assert(verify(src, dst_y));

		gradients_xy_1(src, dst_x, dst_y);
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

		process_by_row(dst.height, row_func);
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

		process_by_row(dst.height, row_func);
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

		process_by_row(dst.height, row_func);
	}
}
#endif


#ifndef SIMAGE_NO_CUDA


/* reinterperet */

namespace simage
{
	template <typename T>
	static MemoryBuffer<T>& reinterpret_device_buffer(cuda::DeviceBuffer<T>& buffer)
	{		
		static_assert(sizeof(MemoryBuffer<T>) == sizeof(cuda::DeviceBuffer<T>));

		static_assert(std::is_same<decltype(MemoryBuffer<T>::data_),      T*>::value, "MemoryBuffer.data_");
		static_assert(std::is_same<decltype(MemoryBuffer<T>::capacity_), u32>::value, "MemoryBuffer.capacity_");
		static_assert(std::is_same<decltype(MemoryBuffer<T>::size_),     u32>::value, "MemoryBuffer.size_");

		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::data_),      T*>::value, "DeviceBuffer.data_");
		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::capacity_), u32>::value, "DeviceBuffer.capacity_");
		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::size_),     u32>::value, "DeviceBuffer.size_");

		return *reinterpret_cast<MemoryBuffer<T>*>(&buffer);
	}

/*
	template <typename T>
	static MatrixView<T> reinterpret_device_view(DeviceMatrixView<T> const& device_view)
	{
		static_assert(sizeof(MatrixView<T>) == sizeof(DeviceMatrixView<T>));

		static_assert(std::is_same<decltype(MatrixView<T>::matrix_data_),  T*>::value, "MatrixView.matrix_data_");
		static_assert(std::is_same<decltype(MatrixView<T>::matrix_width), u32>::value, "MatrixView.matrix_width");
		static_assert(std::is_same<decltype(MatrixView<T>::width),        u32>::value, "MatrixView.width");
		static_assert(std::is_same<decltype(MatrixView<T>::height),       u32>::value, "MatrixView.height");
		static_assert(std::is_same<decltype(MatrixView<T>::range), Range2Du32>::value, "MatrixView.range");

		static_assert(std::is_same<decltype(DeviceMatrixView<T>::matrix_data_),  T*>::value, "DeviceMatrixView.matrix_data_");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::matrix_width), u32>::value, "DeviceMatrixView.matrix_width");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::width),        u32>::value, "DeviceMatrixView.width");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::height),       u32>::value, "DeviceMatrixView.height");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::range), Range2Du32>::value, "DeviceMatrixView.range");

		MatrixView<T> view;

		view.matrix_data_ = device_view.matrix_data_;
		view.matrix_width = device_view.matrix_width;
		view.width = device_view.width;
		view.height = device_view.height;
		view.range = device_view.range;

		return view;
	}
*/

	template <typename T>
	static DeviceMatrixView<T> convert_host_view(MatrixView<T> const& view)
	{
		DeviceMatrixView<T> device_view;

		device_view.matrix_data_ = view.matrix_data_;
		device_view.matrix_width = view.matrix_width;
		device_view.width = view.width;
		device_view.height = view.height;
		device_view.range = view.range;

		return device_view;

	}


	template <typename T>
	static MatrixView<T> convert_device_view(DeviceMatrixView<T> const& device_view)
	{
		MatrixView<T> view;

		view.matrix_data_ = device_view.matrix_data_;
		view.matrix_width = device_view.matrix_width;
		view.width = device_view.width;
		view.height = device_view.height;
		view.range = device_view.range;

		return view;
	}
}



/* make_view */

namespace simage
{
	DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view(width, height, reinterpret_device_buffer(buffer));

		auto device_view = convert_host_view(view);
		
		assert(verify(device_view));

		return device_view;
	}


	DeviceViewGray make_view(u32 width, u32 height, DeviceBuffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view(width, height, reinterpret_device_buffer(buffer));

		auto device_view = convert_host_view(view);
		
		assert(verify(device_view));

		return device_view;
	}
}


/* copy device */

namespace simage
{
	void copy_to_device(View const& host_src, DeviceView const& device_dst)
	{
		assert(verify(host_src, device_dst));

		auto dst = convert_device_view(device_dst);

        auto const bytes_per_row = sizeof(Pixel) * host_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(dst, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_by_row(host_src.height, row_func);
	}


    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst)
	{
		assert(verify(host_src, device_dst));

		auto dst = convert_device_view(device_dst);

        auto const bytes_per_row = sizeof(u8) * host_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(dst, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_by_row(host_src.height, row_func);
	}


    void copy_to_host(DeviceView const& device_src, View const& host_dst)
	{		
		assert(verify(device_src, host_dst));

		auto src = convert_device_view(device_src);

        auto const bytes_per_row = sizeof(Pixel) * device_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_dst, y);
            auto d = row_begin(src, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_by_row(device_src.height, row_func);
	}


    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst)
	{
		assert(verify(device_src, host_dst));

		auto src = convert_device_view(device_src);

        auto const bytes_per_row = sizeof(u8) * device_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_dst, y);
            auto d = row_begin(src, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_by_row(device_src.height, row_func);
	}
}


/* sub_view */

namespace simage
{
	DeviceView sub_view(DeviceView const& view, Range2Du32 const& range)
	{
		auto h_view = convert_device_view(view);

		auto h_sub_view = sub_view(h_view, range);

		auto sub_view = convert_host_view(h_sub_view);

		assert(verify(sub_view));

		return sub_view;
	}


	DeviceViewGray sub_view(DeviceViewGray const& view, Range2Du32 const& range)
	{
		auto h_view = convert_device_view(view);

		auto h_sub_view = sub_view(h_view, range);

		auto sub_view = convert_host_view(h_sub_view);

		assert(verify(sub_view));

		return sub_view;
	}
}


#endif // SIMAGE_NO_CUDA