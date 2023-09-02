/* split channels */

namespace simage
{
	void split_rgb(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));

		for (u32 y = 0; y < src.height; ++y)
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
		}
	}


	void split_rgba(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue, ViewGray const& alpha)
	{
		assert(verify(src, red));
		assert(verify(src, green));
		assert(verify(src, blue));
		assert(verify(src, alpha));

		for (u32 y = 0; y < src.height; ++y)
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
		}
	}


	void split_hsv(View const& src, ViewGray const& hue, ViewGray const& sat, ViewGray const& val)
	{
		assert(verify(src, hue));
		assert(verify(src, sat));
		assert(verify(src, val));

		for (u32 y = 0; y < src.height; ++y)
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
		}
	}


	template <typename T, size_t N>
	static std::array<View1<T>, N> split_channels(ChannelMatrix2D<T, N> const& src)
	{
		std::array<View1<T>, N> views{};

		for (u32 i = 0; i < N; ++i)
		{
			views[i].matrix_data = src.channel_data[i];
			views[i].matrix_width = src.width;
			views[i].width = src.width;
			views[i].height = src.height;
			views[i].range = make_range(src.width, src.height);
			
		}

		return views;
	}
}
