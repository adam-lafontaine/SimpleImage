/* rotate static */

namespace simage
{
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
                auto sx = (i32)roundf(dxcos + dysin) + spx;
                auto sy = (i32)roundf(dycos - dxsin) + spy;

                dxsin += sin;
                dxcos += cos;

                auto out = (sx < 0 || sx >= sw || sy < 0 || sy >= sh);

                d[x] = out ? default_color : *xy_at(src, (u32)sx, (u32)sy);
            }

            dysin += sin;
            dycos += cos;
        }    
	}


	template <typename T, size_t N>
	static void rotate_n(ChannelMatrix2D<T, N> const& src, ChannelMatrix2D<T, N> const& dst, Point2Du32 pivot, f32 rad, T default_color)
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

		auto ch_src = split_channels(src);
		auto ch_dst = split_channels(dst);

		for (u32 y = 0; y < dst.height; y++)
		{
			auto d = view_row_begin(dst, y);

			auto dxsin = -dpx * sin;
            auto dxcos = -dpx * cos;

			for (u32 x = 0; x < dst.width; x++)
			{
				auto sx = (i32)roundf(dxcos + dysin) + spx;
                auto sy = (i32)roundf(dycos - dxsin) + spy;

				dxsin += sin;
                dxcos += cos;

                auto out = (sx < 0 || sx >= sw || sy < 0 || sy >= sh);

				for (u32 ch = 0; ch < (u32)N; ++ch)
				{
					d[ch][x] = out ? default_color : *xy_at(ch_src[ch], (u32)sx, (u32)sy);
				}
			}

			dysin += sin;
            dycos += cos;
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
		
		rotate_n(src, dst, origin, rad, 0.0f);
	}


	void rotate(View3f32 const& src, View3f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));
		
		rotate_n(src, dst, origin, rad, 0.0f);
	}


	void rotate(View2f32 const& src, View2f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));
		
		rotate_n(src, dst, origin, rad, 0.0f);
	}


	void rotate(View1f32 const& src, View1f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad, 0.0f);
	}
}
