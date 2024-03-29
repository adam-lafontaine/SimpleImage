/* for_each_pixel */

namespace simage
{
	template <typename T>
	static inline void for_each_pixel_view(View1<T> const& view, std::function<void(T&)> const& func)
	{
		auto len = view.width * view.height;
		auto s = row_begin(view, 0);

		for (u32 i = 0; i < len; ++i)
		{
			func(s[i]);
		}
	}


	template <typename T>
	static inline void for_each_pixel_sub_view(View1<T> const& view, std::function<void(T&)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(s[x]);
			}
		}
	}


	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_pixel_view(view, func);
		}
		else
		{
			for_each_pixel_sub_view(view, func);
		}
	}


	void for_each_pixel(ViewGray const& view, std::function<void(u8&)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_pixel_view(view, func);
		}
		else
		{
			for_each_pixel_sub_view(view, func);
		}
	}


	void for_each_pixel(View1f32 const& view, std::function<void(f32&)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_pixel_view(view, func);
		}
		else
		{
			for_each_pixel_sub_view(view, func);
		}
	}
}


/* for_each_xy */

namespace simage
{
	template <typename T, class FUNC>
	static inline void for_each_xy_view(View1<T> const& view, FUNC const& func)
	{
		u32 len = view.width * view.height;
		auto d = row_begin(view, 0);

		u32 x = 0;
		u32 y = 0;

		for (u32 i = 0; i < len; ++i)
		{
			y = i / view.width;
			x = i - (view.width * y);

			d[i] = func(x, y);
		}
	}


	template <typename T, class FUNC>
	static inline void for_each_xy_sub_view(View1<T> const& view, FUNC const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				d[x] = func(x, y);
			}
		}
	}


	void for_each_xy(View const& view, std::function<Pixel(u32 x, u32 y)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_xy_view(view, func);
		}
		else
		{
			for_each_xy_view(view, func);
		}
	}


	void for_each_xy(ViewGray const& view, std::function<u8(u32 x, u32 y)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_xy_view(view, func);
		}
		else
		{
			for_each_xy_view(view, func);
		}
	}


	void for_each_xy(View1f32 const& view, std::function<f32(u32 x, u32 y)> const& func)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			for_each_xy_view(view, func);
		}
		else
		{
			for_each_xy_view(view, func);
		}
	}
}
