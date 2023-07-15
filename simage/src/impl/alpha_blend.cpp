/* alpha blend */

namespace simage
{
	static void alpha_blend_row(Pixel* src, Pixel* cur, Pixel* dst, u32 width)
	{
		auto const blend = [](u8 s, u8 c, f32 a)
		{
			auto blended = a * s + (1.0f - a) * c;
			return (u8)(blended + 0.5f);
		};

		for (u32 x = 0; x < width; ++x)
		{
			auto s = src[x].rgba;
			auto c = cur[x].rgba;
			auto& d = dst[x].rgba;

			auto a = cs::to_channel_f32(s.alpha);
			d.red = blend(s.red, c.red, a);
			d.green = blend(s.green, c.green, a);
			d.blue = blend(s.blue, c.blue, a);
		}
	}


	void alpha_blend(View const& src, View const& cur, View const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_row(s, c, d, src.width);
		}
	}


	void alpha_blend(View const& src, View const& cur_dst)
	{
		assert(verify(src, cur_dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(cur_dst, y);

			alpha_blend_row(s, d, d, src.width);
		}
	}
}


/* alpha blend */

namespace simage
{
	static inline f32 blend(f32 s, f32 c, f32 a)
	{
		return a * s + (1.0f - a) * c;
	}


	static void alpha_blend_channel(View1f32 const& src, View1f32 const& cur, View1f32 const& alpha, View1f32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto a = row_begin(alpha, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = blend(s[x], c[x], a[x]);
			}
		}
	}


	void alpha_blend(ViewRGBAf32 const& src, ViewRGBf32 const& cur, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const alpha_ch = select_channel(src, RGBA::A);
		auto const src_rgb = select_rgb(src);

		auto const channel_func = [&](RGB ch)
		{
			auto s_ch = select_channel(src_rgb, ch);
			auto c_ch = select_channel(cur, ch);
			auto d_ch = select_channel(dst, ch);

			alpha_blend_channel(s_ch, c_ch, alpha_ch, d_ch);
		};

		std::array<std::function<void()>, 3> f_list
		{
			[&](){ channel_func(RGB::R); },
			[&](){ channel_func(RGB::G); },
			[&](){ channel_func(RGB::B); },
		};

    	execute(f_list);
	}
}
