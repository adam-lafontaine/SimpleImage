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
    static T get_pixel_value(MatrixView2D<T> const& src, Point2Df32 location)
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
    static void rotate_1(View1<T> const& src, View1<T> const& dst, Point2Du32 origin, f32 rad)
	{
		u32 len = src.width * src.height;

		u32 y = 0;
		u32 x = 0;

		for (u32 i = 0; i < len; ++i)
		{
			y = i / src.width;
			x = i - y * src.width;

			auto src_pt = find_rotation_src(x, y, origin, rad);
			dst.data[i] = get_pixel_value(src, src_pt);
		}
	}


	template <typename T, size_t N>
    static void rotate_n(ChannelMatrix2D<T, N> const& src, ChannelMatrix2D<T, N> const& dst, Point2Du32 origin, f32 rad)
	{
		for (u32 ch = 0; ch < N; ++ch)
		{
			rotate_1(select_channel(src, ch), select_channel(dst, ch), origin, rad);
		}
	}
}


/* rotate */

namespace simage
{
	void rotate(View const& src, View const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}


	void rotate(ViewGray const& src, ViewGray const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
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

		rotate_1(src, dst, origin, rad);
	}
}
