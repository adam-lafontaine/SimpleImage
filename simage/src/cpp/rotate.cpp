/* rotate static */

namespace simage
{
    static Point2Df32 find_rotation_src(u32 x, u32 y, Point2Du32 const& origin, f32 theta_rotate)
	{
		auto const dx_dst = (f32)x - (f32)origin.x;
		auto const dy_dst = (f32)y - (f32)origin.y;

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
    static T get_pixel_value(View1<T> const& src, Point2Df32 location)
    {
        constexpr auto zero = 0.0f;
		constexpr auto black = (T)0;

		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return black;
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
    }
	

	static Pixel get_pixel_value(View const& src, Point2Df32 location)
	{
		constexpr auto zero = 0.0f;
		constexpr auto black = to_pixel(0, 0, 0);

		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return black;
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
	}


    template <typename T>
    static void rotate_1(View1<T> const& src, View1<T> const& dst, Point2Du32 pivot, f32 rad, T default_color)
	{
		auto const cos = cosf(rad);
        auto const sin = sinf(rad);

        auto const sw = (i32)src.width;
        auto const sh = (i32)src.height;

        auto const spx = (i32)pivot.x;
        auto const spy = (i32)pivot.y;

        auto const dpx = (i32)pivot.x;
        auto const dpy = (i32)pivot.y;
        
        f32 dysin = -dpy * sin;
        f32 dycos = -dpy * cos;

        for (u32 y = 0; y < dst.height; y++)
        {
            auto d = row_begin(dst, y);            
            
            auto dxsin = -dpx * sin;
            auto dxcos = -dpx * cos;

            for (u32 x = 0; x < dst.width; x++)
            { 
                auto sx = (i32)(dxcos + dysin + 0.5f) + spx;
                auto sy = (i32)(dycos - dxsin + 0.5f) + spy;

                auto out = (sx < 0 || sx >= sw || sy < 0 || sy >= sh);

                d[x] = out ? default_color : *xy_at(src, (u32)sx, (u32)sy);

                dxsin += sin;
                dxcos += cos;
            }

            dysin += sin;
            dycos += cos;
        }    
	}


	template <typename T, size_t N>
	static void rotate_n(ChannelMatrix2D<T, N> const& src, ChannelMatrix2D<T, N> const& dst, Point2Du32 origin, f32 rad)
	{
		auto ch_src = split_channels(src);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = view_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src(x, y, origin, rad);

				for (u32 ch = 0; ch < (u32)N; ++ch)
				{
					d[ch][x] = get_pixel_value(ch_src[ch], src_pt);
				}
			}
		}
	}
}


/* rotate */

namespace simage
{
	void rotate(View const& src, View const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad, to_pixel(0, 0, 0));
	}


	void rotate(ViewGray const& src, ViewGray const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad, (u8)0);
	}
}


/* rotate */

namespace simage
{
	void rotate(View4f32 const& src, View4f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));
		
		rotate_n(src, dst, origin, rad);
	}


	void rotate(View3f32 const& src, View3f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));
		
		rotate_n(src, dst, origin, rad);
	}


	void rotate(View2f32 const& src, View2f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));
		
		rotate_n(src, dst, origin, rad);
	}


	void rotate(View1f32 const& src, View1f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad, 0.0f);
	}
}
