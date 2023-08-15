/* blur static */

namespace simage
{  
	template <typename T>
	static void blur_at_xy(View1<T> const& src, View1<T> const& dst, u32 x, u32 y)
	{
		u32 const w = src.width;
		u32 const h = src.height;

		auto& d = *xy_at(dst, x, y);

		auto rc = std::min({x, w - x - 1, y, h - y - 1});

		switch (rc)
		{
		case 0:
            // copy
            d = *xy_at(src, x, y);
            return;
/*
        case 1:
            d = convolve_at_xy(src, x, y, (f32*)GAUSS_3x3.data(), 3, 3);
            return;

        case 2:
            d = convolve_at_xy(src, x, y, (f32*)GAUSS_5x5.data(), 5, 5);
            return;
        
        case 3:
            d = convolve_at_xy(src, x, y, (f32*)GAUSS_7x7.data(), 7, 7);
            return;

        case 4:
            d = convolve_at_xy(src, x, y, (f32*)GAUSS_9x9.data(), 9, 9);
            return;
        
        default:
            d = convolve_at_xy(src, x, y, (f32*)GAUSS_11x11.data(), 11, 11);
            return;
*/
		case 1:
            d = convolve_at_xy<3, 3>(src, x, y, (f32*)GAUSS_3x3.data());
            return;

        case 2:
            d = convolve_at_xy<5, 5>(src, x, y, (f32*)GAUSS_5x5.data());
            return;
        
        case 3:
            d = convolve_at_xy<7, 7>(src, x, y, (f32*)GAUSS_7x7.data());
            return;

        case 4:
            d = convolve_at_xy<9, 9>(src, x, y, (f32*)GAUSS_9x9.data());
            return;
        
        default:
            d = convolve_at_xy<11, 11>(src, x, y, (f32*)GAUSS_11x11.data());
            return;
		}
	}




    template <typename T>
	static void blur_1(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

		for (u32 y = 0; y < height; ++y)
		{
			for (u32 x = 0; x < width; ++x)
			{
				blur_at_xy(src, dst, x, y);
			}
		}
	}
}


/* blur */

namespace simage
{
    void blur(View const& src, View const& dst)
    {
        assert(verify(src, dst));

		blur_1(src, dst);
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

		auto const channel_func = [&](u32 ch)
		{
			blur_1(select_channel(src, ch), select_channel(dst, ch));
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

