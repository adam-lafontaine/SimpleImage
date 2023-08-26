/* transform */

namespace simage
{
	template <class IMG_S, class IMG_D, class FUNC>	
	static void do_transform_view(IMG_S const& src, IMG_D const& dst, FUNC const& func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		}
	}


	template <class IMG_S, class IMG_D, class FUNC>	
	static void do_transform_sub_view(IMG_S const& src, IMG_D const& dst, FUNC const& func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		}
	}


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void transform(ViewGray const& src, ViewGray const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void transform(View const& src, ViewGray const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, func);
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min)
	{
		assert(verify(src, dst));

		return do_transform(src, dst, [&](u8 p){ return p >= min ? p : 0; });
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));

		auto mn = std::min(min, max);
		auto mx = std::max(min, max);

		return do_transform(src, dst, [&](u8 p){ return p >= mn && p <= mx ? p : 0; });
	}


	void binarize(View const& src, ViewGray const& dst, pixel_to_bool_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, [&](Pixel p){ return func(p) ? 255 : 0; });
	}


	void binarize(ViewGray const& src, ViewGray const& dst, u8_to_bool_f const& func)
	{
		assert(verify(src, dst));

		do_transform(src, dst, [&](u8 p){ return func(p) ? 255 : 0; });
	}
}


/* transform */

namespace simage
{
	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func32)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s[x]);
			}
		}
	}


	void transform(View2f32 const& src, View1f32 const& dst, std::function<f32(f32, f32)> const& func32)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s0[x], s1[x]);
			}
		}
	}


	void transform(View3f32 const& src, View1f32 const& dst, std::function<f32(f32, f32, f32)> const& func32)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s0 = channel_row_begin(src, y, 0);
			auto s1 = channel_row_begin(src, y, 1);
			auto s2 = channel_row_begin(src, y, 2);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s0[x], s1[x], s2[x]);
			}
		}
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min32 ? s[x] : 0.0f;
			}
		}
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32, f32 max32)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min32 && s[x] <= max32 ? s[x] : 0.0f;
			}
		}
	}


	void binarize(View1f32 const& src, View1f32 const& dst, std::function<bool(f32)> func32)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func32(s[x]) ? 1.0f : 0.0f;
			}
		}
	}
}
