/* skeleton */

namespace simage
{
	template <class GRAY_IMG_T>
	static bool is_outside_edge(GRAY_IMG_T const& img, u32 x, u32 y)
	{
		assert(x >= 1);
		assert(x < img.width);
		assert(y >= 1);
		assert(y < img.height);

		constexpr std::array<int, 8> x_neighbors = { -1,  0,  1,  1,  1,  0, -1, -1 };
		constexpr std::array<int, 8> y_neighbors = { -1, -1, -1,  0,  1,  1,  1,  0 };

		constexpr auto n_neighbors = x_neighbors.size();
		u32 value_count = 0;
		u32 flip = 0;

		auto xi = (u32)(x + x_neighbors[n_neighbors - 1]);
		auto yi = (u32)(y + y_neighbors[n_neighbors - 1]);
		auto val = *xy_at(img, xi, yi);
		bool is_on = val != 0;

		for (u32 i = 0; i < n_neighbors; ++i)
		{
			xi = (u32)(x + x_neighbors[i]);
			yi = (u32)(y + y_neighbors[i]);

			val = *xy_at(img, xi, yi);
			flip += (val != 0) != is_on;

			is_on = val != 0;
			value_count += is_on;
		}

		return value_count > 1 && value_count < 7 && flip == 2;
	}


	template <class GRAY_IMG_T>
	static u32 skeleton_once(GRAY_IMG_T const& img)
	{
		u32 pixel_count = 0;

		auto width = img.width;
		auto height = img.height;

		auto const xy_func = [&](u32 x, u32 y)
		{
			auto& p = *xy_at(img, x, y);
			if (p == 0)
			{
				return;
			}

			if (is_outside_edge(img, x, y))
			{
				p = 0;
			}

			pixel_count += p > 0;
		};

		u32 x_begin = 1;
		u32 x_end = width - 1;
		u32 y_begin = 1;
		u32 y_end = height - 2;
		u32 x = 0;
		u32 y = 0;

		auto const done = [&]() { return !(x_begin < x_end && y_begin < y_end); };

		while (!done())
		{
			// iterate clockwise
			y = y_begin;
			x = x_begin;
			for (; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (++y; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			for (--y; y > y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;

			if (done())
			{
				break;
			}

			// iterate counter clockwise
			for (++x; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (++x; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (--y; y >= y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;
		}

		return pixel_count;
	}


	void skeleton(ViewGray const& src_dst)
	{
		PROFILE_BLOCK(PL::Skeleton)
		assert(verify(src_dst));

		u32 current_count = 0;
		u32 pixel_count = skeleton_once(src_dst);
		u32 max_iter = 100; // src.width / 2;

		for (u32 i = 1; pixel_count != current_count && i < max_iter; ++i)
		{
			current_count = pixel_count;
			pixel_count = skeleton_once(src_dst);
		}
	}
}
