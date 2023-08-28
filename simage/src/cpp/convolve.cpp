/* convolution kernels */

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
	template <typename T, size_t KW, size_t KH>
	static inline f32 convolve_at_xy_f32(View1<T> const& view, u32 x, u32 y, f32* kernel)
    {
		constexpr u32 k_width = (u32)KW;
		constexpr u32 k_height = (u32)KH;

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


	template <size_t KW, size_t KH>
	static inline u8 convolve_at_xy(View1<u8> const& view, u32 x, u32 y, f32* kernel_array)
	{
		auto val32 = convolve_at_xy_f32<u8, KW, KH>(view, x, y, kernel_array);
		return abs_to_u8(val32);
	}


	template <size_t KW, size_t KH>
	static inline f32 convolve_at_xy(View1<f32> const& view, u32 x, u32 y, f32* kernel_array)
	{
		return convolve_at_xy_f32<f32, KW, KH>(view, x, y, kernel_array);
	}


	template <size_t KW, size_t KH>
	static Pixel convolve_at_xy(View const& view, u32 x, u32 y, f32* kernel_array)
    {
		constexpr u32 k_width = (u32)KW;
		constexpr u32 k_height = (u32)KH;

		auto kernel = kernel_array;

        f32 red = 0.0f;
        f32 green = 0.0f;
        f32 blue = 0.0f;

        u32 w = 0;

        auto rx = x - (k_width / 2);
        auto ry = y - (k_height / 2);

        for (u32 v = 0; v < k_height; ++v)
        {
            auto s = row_begin(view, ry + v);
            for (u32 u = 0; u < k_width; ++u)
            {
                auto rgba = s[rx + u].rgba;
                auto kw = kernel[w++];

                red += rgba.red * kw;
                green += rgba.green * kw;
                blue += rgba.blue * kw;
            }
        }

        auto p = *xy_at(view, x, y);
        p.rgba.red = abs_to_u8(red);
        p.rgba.green = abs_to_u8(green);
        p.rgba.blue = abs_to_u8(blue);

        return p;
    }


	template <typename T, size_t KW, size_t KH, class convert_to_T>
	static inline void do_convolve_span(View1<T> const& src, View1<T> const& dst, u32 x_begin, u32 x_end, u32 y, f32* kernel, convert_to_T const& convert)
	{
		constexpr u32 k_width = (u32)KW;
		constexpr u32 k_height = (u32)KH;
		constexpr u32 k_size = k_width * k_height;

		auto d = row_begin(dst, y);
		
		u32 ry = y - (k_height / 2);

		u32 w = 0;

		u32 v = 0;
		u32 u = 0;

		auto kw = kernel[w];
		auto s = row_begin(src, ry + v);

		u32 xs = x_begin - (k_width / 2) + u;
		for (u32 x = x_begin; x < x_end; ++x)
		{
			d[x] = convert(kw * s[xs]);
			++xs;
		}

		for (w = 1; w < k_size; ++w)
		{
			v = w / k_width;
			u = w - (v * k_width);

			kw = kernel[w];
			s = row_begin(src, ry + v);			
			
			xs = x_begin - (k_width / 2) + u;
			for (u32 x = x_begin; x < x_end; ++x)
			{
				d[x] += convert(kw * s[xs]);
				++xs;
			}
		}
	}


	template <size_t KW, size_t KH>
	static void convolve_span(View1<u8> const& src, View1<u8> const& dst, u32 x_begin, u32 x_end, u32 y, f32* kernel)
	{
		do_convolve_span<u8, KW, KH>(src, dst, x_begin, x_end, y, kernel, abs_to_u8);
	}


	template <size_t KW, size_t KH>
	static void convolve_span(View1<f32> const& src, View1<f32> const& dst, u32 x_begin, u32 x_end, u32 y, f32* kernel)
	{
		auto const f = [](f32 a){ return a; };
		do_convolve_span<f32, KW, KH>(src, dst, x_begin, x_end, y, kernel, f);
	}


	template <size_t KW, size_t KH>
	static void convolve_span(View1<Pixel> const& src, View1<Pixel> const& dst, u32 x_begin, u32 x_end, u32 y, f32* kernel)
	{
		constexpr u32 k_width = (u32)KW;
		constexpr u32 k_height = (u32)KH;
		constexpr u32 k_size = k_width * k_height;

		auto d = row_begin(dst, y);
		
		u32 ry = y - (k_height / 2);

		u32 w = 0;

		u32 v = w / k_width;
		u32 u = w - (v * k_width);

		auto kw = kernel[w];
		auto s = row_begin(src, ry + v);

		u32 xs = x_begin - (k_width / 2) + u;
		for (u32 x = x_begin; x < x_end; ++x)
		{
			d[x].rgba.red = abs_to_u8(kw * s[xs].rgba.red);
			d[x].rgba.green = abs_to_u8(kw * s[xs].rgba.green);
			d[x].rgba.blue = abs_to_u8(kw * s[xs].rgba.blue);
			++xs;
		}

		for (w = 1; w < k_size; ++w)
		{
			v = w / k_width;
			u = w - (v * k_width);

			kw = kernel[w];
			s = row_begin(src, ry + v);	

			xs = x_begin - (k_width / 2) + u;
			for (u32 x = x_begin; x < x_end; ++x)
			{
				d[x].rgba.red = abs_to_u8(kw * s[xs].rgba.red);
				d[x].rgba.green = abs_to_u8(kw * s[xs].rgba.green);
				d[x].rgba.blue = abs_to_u8(kw * s[xs].rgba.blue);
				++xs;
			}
		}
	}
}
