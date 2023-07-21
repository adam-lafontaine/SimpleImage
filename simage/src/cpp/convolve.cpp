/* convolution kernels */

#include <algorithm>

namespace simage
{
	static constexpr f32 div16(int i) { return i / 16.0f; }

	static constexpr std::array<f32, 9> GAUSS_3x3
	{
		div16(1), div16(2), div16(1),
		div16(2), div16(4), div16(2),
		div16(1), div16(2), div16(1),
	};


	static constexpr f32 div256(int i) { return i / 256.0f; }

	static constexpr std::array<f32, 25> GAUSS_5x5
	{
		div256(1), div256(4),  div256(6),  div256(4),  div256(1),
		div256(4), div256(16), div256(24), div256(16), div256(4),
		div256(6), div256(24), div256(36), div256(24), div256(6),
		div256(4), div256(16), div256(24), div256(16), div256(4),
		div256(1), div256(4),  div256(6),  div256(4),  div256(1),
	};


	static constexpr f32 div140(int i) { return i / 140.0f; }

	static constexpr std::array<f32, 49> GAUSS_7x7
	{
		div140(1), div140(1), div140(2),  div140(2), div140(2), div140(1), div140(1),
		div140(1), div140(2), div140(2),  div140(4), div140(2), div140(2), div140(1),
		div140(2), div140(2), div140(4),  div140(8), div140(4), div140(2), div140(2),
		div140(2), div140(4), div140(8), div140(16), div140(8), div140(4), div140(2),
		div140(2), div140(2), div140(4),  div140(8), div140(4), div140(2), div140(2),
		div140(1), div140(2), div140(2),  div140(4), div140(2), div140(2), div140(1),
		div140(1), div140(1), div140(2),  div140(2), div140(2), div140(1), div140(1),
	};


	static constexpr f32 div548(int i) { return i / 548.0f; }

	static constexpr std::array<f32, 81> GAUSS_9x9
	{
		div548(1), div548(1),  div548(2),  div548(2),  div548(4),  div548(2),  div548(2), div548(1), div548(1),
		div548(1), div548(2),  div548(2),  div548(4),  div548(8),  div548(4),  div548(2), div548(2), div548(1),
		div548(2), div548(2),  div548(4),  div548(8), div548(16),  div548(8),  div548(4), div548(2), div548(2),
		div548(2), div548(4),  div548(8), div548(16), div548(32), div548(16),  div548(8), div548(4), div548(2),
		div548(4), div548(8), div548(16), div548(32), div548(64), div548(32), div548(16), div548(8), div548(4),
		div548(2), div548(4),  div548(8), div548(16), div548(32), div548(16),  div548(8), div548(4), div548(2),
		div548(2), div548(2),  div548(4),  div548(8), div548(16),  div548(8),  div548(4), div548(2), div548(2),
		div548(1), div548(2),  div548(2),  div548(4),  div548(8),  div548(4),  div548(2), div548(2), div548(1),
		div548(1), div548(1),  div548(2),  div548(2),  div548(4),  div548(2),  div548(2), div548(1), div548(1),
	};


	static constexpr f32 div465(int i) { return i / 465.0f; }

	static constexpr std::array<f32, 121> GAUSS_11x11
	{
		div465(1), div465(1), div465(2), div465(2), div465(3), div465(3), div465(3), div465(2), div465(2), div465(1), div465(1),
		div465(1), div465(2), div465(2), div465(3), div465(4), div465(4), div465(4), div465(3), div465(2), div465(2), div465(1),
		div465(2), div465(2), div465(3), div465(4), div465(5), div465(5), div465(5), div465(4), div465(3), div465(2), div465(2),
		div465(2), div465(3), div465(4), div465(5), div465(7), div465(7), div465(7), div465(5), div465(4), div465(3), div465(2),
		div465(3), div465(4), div465(5), div465(7), div465(9), div465(9), div465(9), div465(7), div465(5), div465(4), div465(3),
		div465(3), div465(4), div465(5), div465(7), div465(9), div465(9), div465(9), div465(7), div465(5), div465(4), div465(3),
		div465(3), div465(4), div465(5), div465(7), div465(9), div465(9), div465(9), div465(7), div465(5), div465(4), div465(3),
		div465(2), div465(3), div465(4), div465(5), div465(7), div465(7), div465(7), div465(5), div465(4), div465(3), div465(2),
		div465(2), div465(2), div465(3), div465(4), div465(5), div465(5), div465(5), div465(4), div465(3), div465(2), div465(2),
		div465(1), div465(2), div465(2), div465(3), div465(4), div465(4), div465(4), div465(3), div465(2), div465(2), div465(1),
		div465(1), div465(1), div465(2), div465(2), div465(3), div465(3), div465(3), div465(2), div465(2), div465(1), div465(1),
	};


	static constexpr f32 get_sum()
	{
		//auto arr = GAUSS_3x3;
		//auto arr = GAUSS_5x5;
		auto arr = GAUSS_7x7;
		//auto arr = GAUSS_9x9;
		//auto arr = GAUSS_11x11;

		f32 sum = 0.0f;
		for (size_t i = 0; i < arr.size(); ++i)
		{
			sum += arr[i];
		}

		return sum;
	}


	static constexpr auto SUM = get_sum();
	

    static constexpr std::array<f32, 9> GRAD_X_3x3
	{
		-0.25f,  0.0f,  0.25f,
		-0.50f,  0.0f,  0.50f,
		-0.25f,  0.0f,  0.25f,
	};

	static constexpr std::array<f32, 9> GRAD_Y_3x3
	{
		-0.2f, -0.6f, -0.2f,
		0.0f,  0.0f,  0.0f,
		0.2f,  0.6f,  0.2f,
	};


	static constexpr std::array<f32, 15> GRAD_X_3x5
	{
		-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
		-0.24f, -0.36f, 0.0f, 0.36f, 0.24f
		-0.08f, -0.12f, 0.0f, 0.12f, 0.08f
	};


	static constexpr std::array<f32, 15> GRAD_Y_3x5
	{
		-0.08f, -0.24f, -0.08f,
		-0.12f, -0.36f, -0.12f,
		0.00f,  0.00f,  0.00f,
		0.12f,  0.36f,  0.12f,
		0.08f,  0.24f,  0.08f,
	};


	static constexpr std::array<f32, 21> GRAD_X_3x7
	{
		-0.04f, -0.07f, -0.09f, 0.0f, 0.09f, 0.07f, 0.04f,
		-0.12f, -0.21f, -0.27f, 0.0f, 0.27f, 0.21f, 0.12f,
		-0.04f, -0.07f, -0.09f, 0.0f, 0.09f, 0.07f, 0.04f,
	};


	static constexpr std::array<f32, 21> GRAD_Y_3x7
	{
		-0.04f, -0.12f, -0.04f,
		-0.07f, -0.21f, -0.07f,
		-0.09f, -0.27f, -0.09f,
		0.00f,  0.00f,  0.00f,
		0.09f,  0.27f,  0.09f,
		0.07f,  0.21f,  0.07f,
		0.04f,  0.12f,  0.04f,
	};


	static constexpr std::array<f32, 27> GRAD_X_3x9
	{
		-0.02f, -0.04f, -0.06f, -0.08f, 0.0f, 0.08f, 0.06f, 0.04f, 0.02f,
		-0.06f, -0.12f, -0.18f, -0.24f, 0.0f, 0.24f, 0.18f, 0.12f, 0.06f,
		-0.02f, -0.04f, -0.06f, -0.08f, 0.0f, 0.08f, 0.06f, 0.04f, 0.02f,
	};


	static constexpr std::array<f32, 27> GRAD_Y_3x9
	{
		-0.02f, -0.09f, -0.02f,
		-0.04f, -0.12f, -0.04f,
		-0.06f, -0.15f, -0.06f,
		-0.08f, -0.18f, -0.08f,
		0.00f,  0.00f,  0.00f,
		0.08f,  0.18f,  0.08f,
		0.06f,  0.15f,  0.06f,
		0.04f,  0.12f,  0.04f,
		0.02f,  0.09f,  0.02f,
	};


    static constexpr std::array<f32, 33> GRAD_X_3x11
	{
		-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
		-0.06f, -0.09f, -0.12f, -0.15f, -0.18f, 0.0f, 0.18f, 0.15f, 0.12f, 0.09f, 0.06f,
		-0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
	};


    static constexpr std::array<f32, 33> GRAD_Y_3x11
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
	};
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


	template <typename T>
	static f32 convolve_at_xy(View1<T> const& view, u32 x, u32 y, f32* kernel, u32 k_width, u32 k_height)
    {
        f32 total = 0.0f;
        u32 w = 0;

        auto rx = x - (k_width / 2);
        auto ry = y - (k_height / 2);

        for (u32 v = 0; v < k_height; ++v)
        {
            auto s = row_begin(view, ry + v);
            for (u32 u = 0; u < k_width; ++u)
            {
                total += s[rx + u] * kernel[w++];
            }
        }

        return total;
    }


	template <typename T>
	static f32 convolve_at_xy_gauss_3x3(View1<T> const& view, u32 x, u32 y)
	{
		return convolve_at_xy(view, x, y, (f32*)GAUSS_3x3.data(), 3, 3);
	}


	template <typename T>
	static f32 convolve_at_xy_gauss_5x5(View1<T> const& view, u32 x, u32 y)
	{
		return convolve_at_xy(view, x, y, (f32*)GAUSS_5x5.data(), 5, 5);
	}


	template <typename T>
	static f32 convolve_at_xy_gauss_7x7(View1<T> const& view, u32 x, u32 y)
	{
		return convolve_at_xy(view, x, y, (f32*)GAUSS_7x7.data(), 7, 7);
	}


	template <typename T>
	static f32 convolve_at_xy_gauss_9x9(View1<T> const& view, u32 x, u32 y)
	{
		return convolve_at_xy(view, x, y, (f32*)GAUSS_9x9.data(), 9, 9);
	}


	template <typename T>
	static f32 convolve_at_xy_gauss_11x11(View1<T> const& view, u32 x, u32 y)
	{
		return convolve_at_xy(view, x, y, (f32*)GAUSS_11x11.data(), 11, 11);
	}


	template <typename T, class CONV_F>
	static void convolve_top_bottom(View1<T> const& dst, u32 row_col, CONV_F const& convolve_at_xy)
	{
		auto x_begin = row_col;
		auto x_end = dst.width - row_col;
		auto y_begin = row_col;
		auto y_last = dst.height - row_col - 1;
		
		auto d_top = row_begin(dst, y_begin);
		auto d_btm = row_begin(dst, y_last);

		for (u32 x = x_begin; x < x_end; ++x)
		{
			d_top[x] = convolve_at_xy(x, y_begin);
			d_btm[x] = convolve_at_xy(x, y_last);
		}
	}


	template <typename T, class CONV_F>
	static void convolve_left_right(View1<T> const& dst, u32 row_col, CONV_F const& convolve_at_xy)
	{
		auto y_begin = row_col + 1;
		auto y_end = dst.height - row_col;
		auto x_begin = row_col;
		auto x_last = dst.width - row_col - 1;

		for (u32 y = y_begin; y < y_end; ++y)
		{
			auto d = row_begin(dst, y);

			d[x_begin] = convolve_at_xy(x_begin, y);
			d[x_last] = convolve_at_xy(x_last, y);
		}
	}
}
