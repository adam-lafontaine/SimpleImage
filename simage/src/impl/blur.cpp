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
		for (u32 y = 0; y < src.height; ++y)
		{
			blur_row(src, dst, y);
		}
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

