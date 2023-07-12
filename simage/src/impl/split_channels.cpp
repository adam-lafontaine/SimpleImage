/* split channels */

namespace simage
{
	void split_rgb(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = s[x].rgba;
				r[x] = rgba.red;
				g[x] = rgba.green;
				b[x] = rgba.blue;
			}
		};

		process_by_row(src.height, row_func);
	}


	void split_rgba(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue, ViewGray const& alpha)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));
		assert(verify(src, alpha));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);
			auto a = row_begin(alpha, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = s[x].rgba;
				r[x] = rgba.red;
				g[x] = rgba.green;
				b[x] = rgba.blue;
				a[x] = rgba.alpha;
			}
		};

		process_by_row(src.height, row_func);
	}


	void split_hsv(View const& src, ViewGray const& hue, ViewGray const& sat, ViewGray const& val)
	{
		assert(verify(src, hue));
		assert(verify(src, sat));
		assert(verify(src, val));

		auto const row_func = [&](u32 y)
		{
			auto p = row_begin(src, y);
			auto h = row_begin(hue, y);
			auto s = row_begin(sat, y);
			auto v = row_begin(val, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const rgba = p[x].rgba;
				auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				h[x] = hsv.hue;
				s[x] = hsv.sat;
				v[x] = hsv.val;
			}
		};

		process_by_row(src.height, row_func);
	}
}
