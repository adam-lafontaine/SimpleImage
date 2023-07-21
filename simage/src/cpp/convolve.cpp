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


	static constexpr f32 div448(int i) { return i / 448.0f; }

	static constexpr std::array<f32, 81> GAUSS_9x9
	{
		div448(1), div448(1),  div448(2),  div448(2),  div448(4),  div448(2),  div448(2), div448(1), div448(1),
		div448(1), div448(2),  div448(2),  div448(4),  div448(8),  div448(4),  div448(2), div448(2), div448(1),
		div448(2), div448(2),  div448(4),  div448(8), div448(16),  div448(8),  div448(4), div448(2), div448(2),
		div448(2), div448(4),  div448(8), div448(16), div448(32), div448(16),  div448(8), div448(4), div448(2),
		div448(4), div448(8), div448(16), div448(32), div448(64), div448(32), div448(16), div448(8), div448(4),
		div448(2), div448(4),  div448(8), div448(16), div448(32), div448(16),  div448(8), div448(4), div448(2),
		div448(2), div448(2),  div448(4),  div448(8), div448(16),  div448(8),  div448(4), div448(2), div448(2),
		div448(1), div448(2),  div448(2),  div448(4),  div448(8),  div448(4),  div448(2), div448(2), div448(1),
		div448(1), div448(1),  div448(2),  div448(2),  div448(4),  div448(2),  div448(2), div448(1), div448(1),
	};


	static constexpr f32 div225(int i) { return i / 225.0f; }

	static constexpr std::array<f32, 121> GAUSS_11x11
	{
		div225(1), div225(1), div225(2), div225(2), div225(3), div225(3), div225(3), div225(2), div225(2), div225(1), div225(1),
		div225(1), div225(2), div225(2), div225(3), div225(4), div225(4), div225(4), div225(3), div225(2), div225(2), div225(1),
		div225(2), div225(2), div225(3), div225(4), div225(5), div225(5), div225(5), div225(4), div225(3), div225(2), div225(2),
		div225(2), div225(3), div225(4), div225(5), div225(7), div225(7), div225(7), div225(5), div225(4), div225(3), div225(2),
		div225(3), div225(4), div225(5), div225(7), div225(9), div225(9), div225(9), div225(7), div225(5), div225(4), div225(3),
		div225(3), div225(4), div225(5), div225(7), div225(9), div225(9), div225(9), div225(7), div225(5), div225(4), div225(3),
		div225(3), div225(4), div225(5), div225(7), div225(9), div225(9), div225(9), div225(7), div225(5), div225(4), div225(3),
		div225(2), div225(3), div225(4), div225(5), div225(7), div225(7), div225(7), div225(5), div225(4), div225(3), div225(2),
		div225(2), div225(2), div225(3), div225(4), div225(5), div225(5), div225(5), div225(4), div225(3), div225(2), div225(2),
		div225(1), div225(2), div225(2), div225(3), div225(4), div225(4), div225(4), div225(3), div225(2), div225(2), div225(1),
		div225(1), div225(1), div225(2), div225(2), div225(3), div225(3), div225(3), div225(2), div225(2), div225(1), div225(1),
	};
	

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
}
