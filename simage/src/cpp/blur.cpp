/* blur static */

namespace simage
{  
	template <typename T>
	static void blur_at_xy(View1<T> const& src, View1<T> const& dst, u32 x, u32 y)
	{
		constexpr auto gauss_11x11 = GAUSS_11x11.data();

		auto& d = *xy_at(dst, x, y);
		d = convolve_at_xy<11, 11>(src, x, y, (f32*)gauss_11x11);
	}


	template <typename T>
	static void blur_outer_at_xy(View1<T> const& src, View1<T> const& dst, u32 x, u32 y)
	{
		u32 const w = src.width;
		u32 const h = src.height;

		auto& d = *xy_at(dst, x, y);

		auto rc = std::min({x, w - x - 1, y, h - y - 1});

		assert(rc < 5);

		constexpr auto gauss_3x3 = GAUSS_3x3.data();
		constexpr auto gauss_5x5 = GAUSS_5x5.data();
		constexpr auto gauss_7x7 = GAUSS_7x7.data();
		constexpr auto gauss_9x9 = GAUSS_9x9.data();

		switch (rc)
		{
		case 0:
            // copy
            d = *xy_at(src, x, y);
            return;

		case 1:
            d = convolve_at_xy<3, 3>(src, x, y, (f32*)gauss_3x3);
            return;

        case 2:
            d = convolve_at_xy<5, 5>(src, x, y, (f32*)gauss_5x5);
            return;
        
        case 3:
            d = convolve_at_xy<7, 7>(src, x, y, (f32*)gauss_7x7);
            return;

        case 4:
            d = convolve_at_xy<9, 9>(src, x, y, (f32*)gauss_9x9);
            return;

		default: return;
		}
	}


	template <typename T>
	static void copy_span(View1<T> const& src, View1<T> const& dst, u32 x_begin, u32 x_end, u32 y)
	{
		auto s = row_begin(src, y);
		auto d = row_begin(dst, y);

		for (u32 x = x_begin; x < x_end; ++x)
		{
			d[x] = s[x];
		}
	}


	template <typename T>
	static void blur_top_bottom(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

		u32 y1 = 0;
		u32 y2 = height - 1;
		
		for (u32 y1 = 0; y1 < 5; ++y1)
		{
			for (u32 x = 0; x < width; ++x)
			{
				blur_outer_at_xy(src, dst, x, y1);
				blur_outer_at_xy(src, dst, x, y2);
			}

			--y2;
		}
	}


	template <typename T>
	static void blur_left_right(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

		for (u32 y = 5; y < height - 5; ++y)
		{
			for (u32 x = 0; x < 5; ++x)
			{
				blur_outer_at_xy(src, dst, x, y);
				blur_outer_at_xy(src, dst, width - 1 - x, y);
			}
		}
	}


	template <typename T>
	static void blur_middle(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

		u32 x_begin = 5;
		u32 x_end = width - 5;

		for (u32 y = 5; y < height - 5; ++y)
		{
			for (u32 x = 5; x < width - 5; ++x)
			{
				blur_at_xy(src, dst, x, y);
			}
		}
	}
	
	
	static void blur_middle(View1<f32> const& src, View1<f32> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

		constexpr auto gauss_11x11 = GAUSS_11x11.data();

		u32 x_begin = 5;
		u32 x_end = width - 5;

		for (u32 y = 5; y < height - 5; ++y)
		{
			convolve_span<11, 11>(src, dst, x_begin, x_end, y, (f32*)gauss_11x11);
		}
	}
}


/* blur */

namespace simage
{
    void blur(View const& src, View const& dst)
    {
        assert(verify(src, dst));

		blur_top_bottom(src, dst);
		blur_left_right(src, dst);
		blur_middle(src, dst);
    }


    void blur(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

		blur_top_bottom(src, dst);
		blur_left_right(src, dst);
		blur_middle(src, dst);
    }
}


/* blur */

namespace simage
{
	void blur(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		blur_top_bottom(src, dst);
		blur_left_right(src, dst);
		blur_middle(src, dst);
	}


	void blur(View3f32 const& src, View3f32 const& dst)
	{
		assert(verify(src, dst));

		auto const channel_func = [&](u32 ch)
		{
			blur(select_channel(src, ch), select_channel(dst, ch));
		};

		std::array<std::function<void()>, 3> f_list
		{
			[&](){ channel_func(0); },
			[&](){ channel_func(1); },
			[&](){ channel_func(2); },
		};

    	execute(f_list);
	}
}

