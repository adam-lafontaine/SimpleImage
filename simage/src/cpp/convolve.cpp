/* convolution kernels */

namespace simage
{
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
