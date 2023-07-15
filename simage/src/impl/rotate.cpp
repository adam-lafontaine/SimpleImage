/* rotate static */

namespace simage
{
    static Point2Df32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, f32 theta_rotate)
	{
		auto const dx_dst = (f32)pt.x - (f32)origin.x;
		auto const dy_dst = (f32)pt.y - (f32)origin.y;

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
    static T get_pixel_value(MatrixView<T> const& src, Point2Df32 location)
    {
        constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return 0;
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
    }
	

	static Pixel get_pixel_value(View const& src, Point2Df32 location)
	{
		constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const x = location.x;
		auto const y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return to_pixel(0, 0, 0);
		}

		return *xy_at(src, (u32)floorf(x), (u32)floorf(y));
	}


    template <typename T>
    static void rotate_1(View1<T> const& src, View1<T> const& dst, Point2Du32 origin, f32 rad)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, rad);
				d[x] = get_pixel_value(src, src_pt);
			}
		}
	}


    template <typename T, size_t N>
	static void rotate_channels(ChannelView<T, N> const& src, ChannelView<T, N> const& dst, Point2Du32 origin, f32 rad)
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

		auto const channel_func = [&](RGBA ch)
		{
			rotate_1(select_channel(src, ch), select_channel(dst, ch), origin, rad);
		};

		std::array<std::function<void()>, 4> f_list
		{
			[&](){ channel_func(RGBA::R); },
			[&](){ channel_func(RGBA::G); },
			[&](){ channel_func(RGBA::B); },
			[&](){ channel_func(RGBA::A); },
		};

    	execute(f_list);
	}


	void rotate(View3f32 const& src, View3f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		auto const channel_func = [&](RGB ch)
		{
			rotate_1(select_channel(src, ch), select_channel(dst, ch), origin, rad);
		};

		std::array<std::function<void()>, 3> f_list
		{
			[&](){ channel_func(RGB::R); },
			[&](){ channel_func(RGB::G); },
			[&](){ channel_func(RGB::B); },
		};

    	execute(f_list);
	}


	void rotate(View2f32 const& src, View2f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		auto const channel_func = [&](u32 ch)
		{
			rotate_1(select_channel(src, ch), select_channel(dst, ch), origin, rad);
		};

		std::array<std::function<void()>, 2> f_list
		{
			[&](){ channel_func(0); },
			[&](){ channel_func(1); },
		};

    	execute(f_list);
	}


	void rotate(View1f32 const& src, View1f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}
}
