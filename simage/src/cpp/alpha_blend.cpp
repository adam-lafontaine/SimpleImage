/* alpha_blend_span no_simd*/

namespace simage
{
	static inline void alpha_blend_span_u8_no_simd(Pixel* src, Pixel* cur, Pixel* dst, u32 len)
	{
		constexpr auto blend = [](u8 s, u8 c, f32 a)
		{
			auto blended = a * s + (1.0f - a) * c;
			return round_to_u8(blended);
		};

		for (u32 i = 0; i < len; ++i)
		{
			auto s = src[i].rgba;
			auto c = cur[i].rgba;
			auto& d = dst[i].rgba;

			auto a = cs::to_channel_f32(s.alpha);
			d.red = blend(s.red, c.red, a);
			d.green = blend(s.green, c.green, a);
			d.blue = blend(s.blue, c.blue, a);
		}
	}


	static inline void alpha_blend_span_f32_no_simd(f32* src, f32* cur, f32* alpha, f32* dst, u32 len)
	{
		constexpr auto blend = [](f32 s, f32 c, f32 a){ return a * s + (1.0f - a) * c; };

		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = blend(src[i], cur[i], alpha[i]);
		}
	}
}


/* alpha_blend no_simd */

namespace simage
{
	static inline void alpha_blend_u8_no_simd(View const& src, View const& cur, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_span_u8_no_simd(s, c, d, src.width);
		}
	}


	static inline void alpha_blend_u8_no_simd(View const& src, View const& cur_dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(cur_dst, y);

			alpha_blend_span_u8_no_simd(s, d, d, src.width);
		}
	}


	static inline void alpha_blend_f32_no_simd(View1f32 const& src, View1f32 const& cur, View1f32 const& alpha, View1f32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto a = row_begin(alpha, y);
			auto d = row_begin(dst, y);
			
			alpha_blend_span_f32_no_simd(s, c, a, d, src.width);
		}
	}
}


/* alpha_blend simd */

namespace simage
{
#ifndef SIMAGE_NO_SIMD

	static inline void alpha_blend_span_u8(Pixel* src, Pixel* cur, Pixel* dst, u32 len)
	{
		alpha_blend_span_u8_no_simd(src, cur, dst, len);
	}


	static inline void alpha_blend_span_f32(f32* src, f32* cur, f32* alpha, f32* dst, u32 len)
	{
		constexpr auto step = (u32)simd::LEN;

		simd::vecf32 v_val;
		simd::vecf32 v_alpha;
		simd::vecf32 v_dst;

		simd::vecf32 v_one = simd::load_f32_broadcast(1.0f);		

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			v_val = simd::load_f32(src + i);
			v_alpha = simd::load_f32(alpha + i);
			v_dst = simd::mul(v_val, v_alpha);

			v_val = simd::load_f32(cur + i);
			v_alpha = simd::sub(v_one, v_alpha);
			v_dst = simd::fmadd(v_val, v_alpha, v_dst);

			simd::store_f32(v_dst, dst + i);
		}

		i = len - step;
		v_val = simd::load_f32(src + i);
		v_alpha = simd::load_f32(alpha + i);
		v_dst = simd::mul(v_val, v_alpha);

		v_val = simd::load_f32(cur + i);
		v_alpha = simd::sub(v_one, v_alpha);
		v_dst = simd::fmadd(v_val, v_alpha, v_dst);

		simd::store_f32(v_dst, dst + i);
	}

#endif // !SIMAGE_NO_SIMD
}


/* alpha_blend  simd option*/

namespace simage
{
#ifdef SIMAGE_NO_SIMD

	static inline void alpha_blend_u8(View const& src, View const& cur, View const& dst)
	{
		alpha_blend_u8_no_simd(src, cur, dst);
	}


	static inline void alpha_blend_u8(View const& src, View const& cur_dst)
	{
		alpha_blend_u8_no_simd(src, cur_dst);
	}


	static inline void alpha_blend_f32(View1f32 const& src, View1f32 const& cur, View1f32 const& alpha, View1f32 const& dst)
	{
		alpha_blend_f32_no_simd(src, cur, alpha, dst);
	}

#else

	static inline void alpha_blend_u8(View const& src, View const& cur, View const& dst)
	{
		if (src.width < simd::LEN)
		{
			alpha_blend_u8_no_simd(src, cur, dst);
			return;
		}

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto d = row_begin(dst, y);

			alpha_blend_span_u8(s, c, d, src.width);
		}
	}


	static inline void alpha_blend_u8(View const& src, View const& cur_dst)
	{
		if (src.width < simd::LEN)
		{
			alpha_blend_u8_no_simd(src, cur_dst);
			return;
		}

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(cur_dst, y);

			alpha_blend_span_u8(s, d, d, src.width);
		}
	}


	static inline void alpha_blend_f32(View1f32 const& src, View1f32 const& cur, View1f32 const& alpha, View1f32 const& dst)
	{
		if (src.width < simd::LEN)
		{
			alpha_blend_f32_no_simd(src, cur, alpha, dst);
			return;
		}
		
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto c = row_begin(cur, y);
			auto a = row_begin(alpha, y);
			auto d = row_begin(dst, y);
			
			alpha_blend_span_f32(s, c, a, d, src.width);
		}
	}

#endif // SIMAGE_NO_SIMD
}


/* alpha blend */

namespace simage
{
	void alpha_blend(View const& src, View const& cur, View const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		alpha_blend_u8(src, cur, dst);
	}


	void alpha_blend(View const& src, View const& cur_dst)
	{
		assert(verify(src, cur_dst));

		alpha_blend_u8(src, cur_dst);
	}
}


/* alpha blend */

namespace simage
{
	
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

			alpha_blend_f32(s_ch, c_ch, alpha_ch, d_ch);
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
