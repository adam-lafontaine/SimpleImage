/* alpha_blend_span no_simd*/

namespace simage
{
	static inline u8 blend_linear(u8 s, u8 c, f32 a)
	{
		auto blended = a * s + (1.0f - a) * c;
		return round_to_u8(blended);
	}


	static inline f32 blend_linear(f32 s, f32 c, f32 a) 
	{ 
		return a * s + (1.0f - a) * c; 
	};


	static inline void alpha_blend_span_u8(Pixel* src, Pixel* cur, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto s = src[i].rgba;
			auto c = cur[i].rgba;
			auto& d = dst[i].rgba;

			auto a = cs::to_channel_f32(s.alpha);
			d.red = blend_linear(s.red, c.red, a);
			d.green = blend_linear(s.green, c.green, a);
			d.blue = blend_linear(s.blue, c.blue, a);
		}
	}


	static inline void alpha_blend_span_u8(Pixel* src, u8 alpha, Pixel* cur, Pixel* dst, u32 len)
	{
		auto a = cs::to_channel_f32(alpha);

		for (u32 i = 0; i < len; ++i)
		{
			auto s = src[i].rgba;
			auto c = cur[i].rgba;
			auto& d = dst[i].rgba;
			
			d.red = blend_linear(s.red, c.red, a);
			d.green = blend_linear(s.green, c.green, a);
			d.blue = blend_linear(s.blue, c.blue, a);
		}
	}


	static inline void alpha_blend_span_f32(f32* src, f32* cur, f32* alpha, f32* dst, u32 len)
	{		
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = blend_linear(src[i], cur[i], alpha[i]);
		}
	}


	static inline void alpha_blend_span_f32(f32* src, f32 alpha, f32* cur, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = blend_linear(src[i], cur[i], alpha);
		}
	}
}


/* alpha blend */

namespace simage
{
	static void alpha_blend_view(View const& src, View const& cur, View const& dst)
	{
		u32 len = src.width * src.height;

		auto s = row_begin(src, 0);
		auto c = row_begin(cur, 0);
		auto d = row_begin(dst, 0);

		alpha_blend_span_u8(s, c, d, len);
	}


	static void alpha_blend_sub_view(View const& src, View const& cur, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_span_u8(s, c, d, src.width);
		}
	}


	static void alpha_blend_view(View const& src, u8 alpha, View const& cur, View const& dst)
	{
		u32 len = src.width * src.height;

		auto s = row_begin(src, 0);
		auto c = row_begin(cur, 0);
		auto d = row_begin(dst, 0);

		alpha_blend_span_u8(s, alpha, c, d, len);
	}


	static void alpha_blend_sub_view(View const& src, u8 alpha, View const& cur, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_span_u8(s, alpha, c, d, src.width);
		}
	}



	void alpha_blend(View const& src, View const& cur, View const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		if (is_1d(src) && is_1d(cur) && is_1d(dst))
		{
			alpha_blend_view(src, cur, dst);
		}
		else
		{
			alpha_blend_sub_view(src, cur, dst);
		}
	}


	void alpha_blend(View const& src, View const& cur_dst)
	{
		assert(verify(src, cur_dst));

		if (is_1d(src) && is_1d(cur_dst))
		{
			alpha_blend_view(src, cur_dst, cur_dst);
		}
		else
		{
			alpha_blend_sub_view(src, cur_dst, cur_dst);
		}
	}


	void alpha_blend(View const& src, u8 alpha, View const& cur, View const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		if (is_1d(src) && is_1d(cur) && is_1d(dst))
		{
			alpha_blend_view(src, alpha, cur, dst);
		}
		else
		{
			alpha_blend_sub_view(src, alpha, cur, dst);
		}
	}


	void alpha_blend(View const& src, u8 alpha, View const& cur_dst)
	{
		assert(verify(src, cur_dst));

		if (is_1d(src) && is_1d(cur_dst))
		{
			alpha_blend_view(src, alpha, cur_dst, cur_dst);
		}
		else
		{
			alpha_blend_sub_view(src, alpha, cur_dst, cur_dst);
		}
	}
}


/* alpha blend */

namespace simage
{
	
	void alpha_blend(ViewRGBAf32 const& src, ViewRGBf32 const& cur, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto alpha = select_channel(src, RGBA::A);
		auto src_rgb = select_rgb(src);

		u32 len = src.width * src.height;

		auto a = row_begin(alpha, 0);

		for (u32 ch = 0; ch < 3; ++ch)
		{
			auto s = row_begin(select_channel(src_rgb, ch), 0);
			auto c = row_begin(select_channel(cur, ch), 0);
			auto d = row_begin(select_channel(dst, ch), 0);

			alpha_blend_span_f32(s, c, a, d, len);
		}
	}
}
