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
		PROFILE_BLOCK(PL::AlphaBlendView)
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_row(s, c, d, src.width);
		};

		process_by_row(src.height, row_func);
	}


	void alpha_blend(View const& src, View const& cur_dst)
	{
		PROFILE_BLOCK(PL::AlphaBlendView)
		assert(verify(src, cur_dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(cur_dst, y);

			alpha_blend_row(s, d, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
}


/* alpha blend */

namespace simage
{
	static void alpha_blend_row(RGBAf32p const& src, RGBf32p const& cur, RGBf32p const& dst, u32 width)
	{
		auto const blend = [](f32 s, f32 c, f32 a)
		{
			return a * s + (1.0f - a) * c;
		};

		for (u32 x = 0; x < width; ++x)
		{
			auto a = src.A[x];

			dst.R[x] = blend(src.R[x], cur.R[x], a);
			dst.G[x] = blend(src.G[x], cur.G[x], a);
			dst.B[x] = blend(src.B[x], cur.B[x], a);
		}
	}


	void alpha_blend(ViewRGBAf32 const& src, ViewRGBf32 const& cur, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const row_func = [&](u32 y)
		{			
			auto s = rgba_row_begin(src, y);
			auto c = rgb_row_begin(cur, y);
			auto d = rgb_row_begin(dst, y);

			alpha_blend_row(s, c, d, src.width);
		};

		process_by_row(src.height, row_func);
	}
}
