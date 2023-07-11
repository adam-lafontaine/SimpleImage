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
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, rad);
				d[x] = get_pixel_value(src, src_pt);
			}
		};

		process_by_row(src.height, row_func);
	}


    template <typename T, size_t N>
	void rotate_channels(ChannelView<T, N> const& src, ChannelView<T, N> const& dst, Point2Du32 origin, f32 rad)
	{
		constexpr auto zero = 0.0f;
		auto const width = (f32)src.width;
		auto const height = (f32)src.height;

		auto const row_func = [&](u32 y)
		{
			auto d = view_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, rad);
				auto is_out = src_pt.x < zero || src_pt.x >= width || src_pt.y < zero || src_pt.y >= height;

				if (src_pt.x < zero || src_pt.x >= width || src_pt.y < zero || src_pt.y >= height)
				{
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = 0;
					}
				}
				else
				{
					auto src_x = (u32)floorf(src_pt.x);
					auto src_y = (u32)floorf(src_pt.y);
					auto s = view_row_begin(src, src_y);
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = s[ch][src_x];
					}
				}
			}
		};

		process_by_row(src.height, row_func);
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

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View3f32 const& src, View3f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View2f32 const& src, View2f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_channels(src, dst, origin, rad);
	}


	void rotate(View1f32 const& src, View1f32 const& dst, Point2Du32 origin, f32 rad)
	{
		assert(verify(src, dst));

		rotate_1(src, dst, origin, rad);
	}
}
