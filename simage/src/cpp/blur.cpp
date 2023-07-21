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
            break;

        case 1:
            // gauss3
            d = (T)convolve_at_xy(src, x, y, (f32*)GAUSS_3x3.data(), 3, 3);
            break;

        case 2:
            // gauss5
            d = (T)convolve_at_xy(src, x, y, (f32*)GAUSS_5x5.data(), 5, 5);
            break;
        
        case 3:
		//default:
            // gauss7
            d = (T)convolve_at_xy(src, x, y, (f32*)GAUSS_7x7.data(), 7, 7);
            break;

        case 4:
		//default:
            // gauss9
            d = (T)convolve_at_xy(src, x, y, (f32*)GAUSS_9x9.data(), 9, 9);
            break;
        
        default:
            // gauss11
            d = (T)convolve_at_xy(src, x, y, (f32*)GAUSS_11x11.data(), 11, 11);
            break;
		}
	}




    template <typename T>
	static void blur_1(View1<T> const& src, View1<T> const& dst)
	{
		auto const width = src.width;
        auto const height = src.height;

#if 0

		auto const copy_xy = [&](u32 x, u32 y){ return *xy_at(src, x, y); };
		auto const gauss3_xy = [&](u32 x, u32 y){ return (T)convolve_at_xy_gauss_3x3(src, x, y); };
		auto const gauss5_xy = [&](u32 x, u32 y){ return (T)convolve_at_xy_gauss_5x5(src, x, y); };
		auto const gauss7_xy = [&](u32 x, u32 y){ return (T)convolve_at_xy_gauss_7x7(src, x, y); };
		auto const gauss9_xy = [&](u32 x, u32 y){ return (T)convolve_at_xy_gauss_9x9(src, x, y); };
		auto const gauss11_xy = [&](u32 x, u32 y){ return (T)convolve_at_xy_gauss_11x11(src, x, y); };

		convolve_top_bottom(dst, 0, copy_xy);
		convolve_left_right(dst, 0, copy_xy);

		convolve_top_bottom(dst, 1, gauss3_xy);
		convolve_left_right(dst, 1, gauss3_xy);

		convolve_top_bottom(dst, 2, gauss5_xy);
		convolve_left_right(dst, 2, gauss5_xy);

		/*convolve_top_bottom(dst, 3, gauss7_xy);
		convolve_left_right(dst, 3, gauss7_xy);

		/*convolve_top_bottom(dst, 4, gauss9_xy);
		convolve_left_right(dst, 4, gauss9_xy);*/

		u32 const rc = 3;

		for (u32 y = rc; y < height - rc; ++y)
		{
			auto d = row_begin(dst, y);
			for (u32 x = rc; x < width - rc; ++x)
			{
				//d[x] = gauss5_xy(x, y);
				d[x] = gauss7_xy(x, y);
				//d[x] = gauss9_xy(x, y);
				//d[x] = gauss11_xy(x, y);
			}
		}

#else

		for (u32 y = 0; y < height; ++y)
		{
			for (u32 x = 0; x < width; ++x)
			{
				blur_at_xy(src, dst, x, y);
			}
		}

#endif
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

