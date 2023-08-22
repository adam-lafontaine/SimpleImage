/* map_span no_simd */

namespace simage
{
	static inline void map_span_gray_no_simd(u8* src, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = cs::to_channel_f32(src[i]);
		}
	}


	static inline void map_span_gray_no_simd(f32* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = cs::to_channel_u8(src[i]);
		}
	}


	static inline void map_span_rgb_to_gray_no_simd(f32* r, f32* g, f32* b, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = gray::f32_from_rgb_f32(r[i], g[i], b[i]);
		}
	}


	
}


/* map no_simd */

namespace simage
{
	template <class SRC, class DST>
	static inline void map_channel_gray_no_simd(SRC const& src, DST const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_span_gray_no_simd(s, d, src.width);
		}
	}


	static inline void map_rgb_to_gray_no_simd(ViewRGBf32 const& src, View1f32 const& dst)
	{
		auto red = select_channel(src, RGB::R);
		auto green = select_channel(src, RGB::G);
		auto blue = select_channel(src, RGB::B);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);

			map_span_rgb_to_gray_no_simd(r, g, b, d, src.width);
		}
	}

}


/* map_span simd */

namespace simage
{
#ifndef SIMAGE_NO_SIMD

	static void map_span_gray(u8* src, f32* dst, u32 len)
	{		
		constexpr auto step = (u32)simd::LEN;
		constexpr f32 scalar = 1.0f / 255.0f;
		
		simd::vecf32 gray255;
		simd::vecf32 gray1;

		auto conv = simd::load_f32_broadcast(scalar);

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			gray255 = simd::load_gray(src + i);
			gray1 = simd::mul(gray255, conv);
			simd::store_gray(gray1, dst + i);
		}

		i = len - step;
		gray255 = simd::load_gray(src + i);
		gray1 = simd::mul(gray255, conv);
		simd::store_gray(gray1, dst + i);
	}


	static void map_span_gray(f32* src, u8* dst, u32 len)
    {
		constexpr auto step = (u32)simd::LEN;
		constexpr f32 scalar = 255.0f;
		
		simd::vecf32 gray255;
		simd::vecf32 gray1;

		auto conv = simd::load_f32_broadcast(scalar);

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			gray1 = simd::load_gray(src + i);
			gray255 = simd::mul(gray1, conv);
			simd::store_gray(gray255, dst + i);
		}

		i = len - step;
		gray1 = simd::load_gray(src + i);
		gray255 = simd::mul(gray1, conv);
		simd::store_gray(gray255, dst + i);
    }


	static void map_span_rgb_to_gray(f32* r, f32* g, f32* b, f32* dst, u32 len)
	{
		constexpr auto step = (u32)simd::LEN;

		simd::vecf32 v_c_red = simd::load_f32_broadcast(gray::COEFF_RED);
		simd::vecf32 v_c_green = simd::load_f32_broadcast(gray::COEFF_GREEN);
		simd::vecf32 v_c_blue = simd::load_f32_broadcast(gray::COEFF_BLUE);

		simd::vecf32 v_red;
		simd::vecf32 v_green;
		simd::vecf32 v_blue;
		simd::vecf32 v_gray;

		u32 i = 0;
		for (; i <= (len - step); i += step)
		{
			v_red = simd::load_f32(r + i);
			v_green = simd::load_f32(g + i);
			v_blue = simd::load_f32(b + i);
			
			v_gray = simd::fmadd(v_blue, v_c_blue, simd::fmadd(v_green, v_c_green, simd::mul(v_red, v_c_red)));

			simd::store_f32(v_gray, dst + i);
		}

		i = len - step;
		v_red = simd::load_f32(r + i);
		v_green = simd::load_f32(g + i);
		v_blue = simd::load_f32(b + i);

		v_gray = simd::fmadd(v_blue, v_c_blue, simd::fmadd(v_green, v_c_green, simd::mul(v_red, v_c_red)));

		simd::store_f32(v_gray, dst + i);
	}

	

#endif // !SIMAGE_NO_SIMD
}


/* map view simd option */

namespace simage
{

#ifdef SIMAGE_NO_SIMD

	template <class SRC, class DST>
	static inline void map_channel_gray(SRC const& src, DST const& dst)
	{
		map_channel_gray_no_simd(src, dst);
	}


	static inline void map_rgb_to_gray(ViewRGBf32 const& src, View1f32 const& dst)
	{
		map_rgb_to_gray_no_simd(src, dst);
	}

#else

	template <class SRC, class DST>
	static void map_channel_gray(SRC const& src, DST const& dst)
	{
		if (src.width < simd::LEN)
		{
			map_channel_gray_no_simd(src, dst);
			return;
		}

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_span_gray(s, d, src.width);
		}
	}


	static void map_rgb_to_gray(ViewRGBf32 const& src, View1f32 const& dst)
	{
		if (src.width < simd::LEN)
		{
			map_rgb_to_gray_no_simd(src, dst);
			return;
		}

		auto red = select_channel(src, RGB::R);
		auto green = select_channel(src, RGB::G);
		auto blue = select_channel(src, RGB::B);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto r = row_begin(red, y);
			auto g = row_begin(green, y);
			auto b = row_begin(blue, y);

			map_span_rgb_to_gray(r, g, b, d, src.width);
		}		
	}



#endif
}


/* map_gray u8 */

namespace simage
{
	static inline void map_span_rgb_to_gray(Pixel* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			dst[i] = gray::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
		}
	}


	static inline void map_span_yuv_gray(YUV2u8* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = src[i].y;
		}
	}	


	void map_gray(View const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			map_span_rgb_to_gray(s, d, src.width);
		}
	}


	void map_gray(ViewYUV const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_yuv_gray(s, d, src.width);
		}
	}
}


/* map_gray f32 */

namespace simage
{
	static inline void map_span_yuv_gray(YUV2u8* src, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = cs::to_channel_f32(src[i].y);
		}
	}


	static inline void map_span_rgb_to_gray(Pixel* src, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			dst[i] = gray::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
		}
	}


	void map_gray(View1u8 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		map_channel_gray(src, dst);
	}
	

	void map_gray(View1f32 const& src, View1u8 const& dst)
	{
		assert(verify(src, dst));

		map_channel_gray(src, dst);
	}


	void map_gray(ViewYUV const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_yuv_gray(s, d, src.width);
		}
	}


	void map_gray(View const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			map_span_rgb_to_gray(s, d, src.width);
		}
	}


	void map_gray(ViewRGBf32 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		// slower
		map_rgb_to_gray(src, dst);

		//map_rgb_to_gray_no_simd(src, dst);
	}
}


/* map_rgba u8 */

namespace simage
{
	static inline void map_span_gray_rgba(u8* src, Pixel* dst, u32 len)
	{
		RGBAu8 gray{};

		for (u32 i = 0; i < len; ++i)
		{
			gray = { src[i], src[i], src[i], 255 };

			dst[i].rgba = gray;
		}
	}


	static inline void map_span_yuv_rgba(YUV422u8* src, Pixel* dst, u32 len)
	{
		for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src[i422];

			auto i = 2 * i422;
			auto rgba = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
			dst[i].rgba.red = rgba.red;
			dst[i].rgba.green = rgba.green;
			dst[i].rgba.blue = rgba.blue;
			dst[i].rgba.red = 255;

			++i;
			rgba = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			dst[i].rgba.red = rgba.red;
			dst[i].rgba.green = rgba.green;
			dst[i].rgba.blue = rgba.blue;
			dst[i].rgba.red = 255;
		}
	}


	static inline void map_span_bgr_rgba(BGRu8* src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& rgba = dst[i].rgba;
			rgba.red = src[i].red;
			rgba.green = src[i].green;
			rgba.blue = src[i].blue;
			rgba.alpha = 255;
		}
	}


	static inline void map_span_rgb_rgba(RGBu8* src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& rgba = dst[i].rgba;
			rgba.red = src[i].red;
			rgba.green = src[i].green;
			rgba.blue = src[i].blue;
			rgba.alpha = 255;
		}
	}


	void map_rgba(ViewGray const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_gray_rgba(s, d, src.width);
		}
	}


	void map_rgba(ViewYUV const& src, View const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			auto d = row_begin(dst, y);

			map_span_yuv_rgba(s422, d, src.width);
		}
	}


	void map_rgba(ViewBGR const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_bgr_rgba(s, d, src.width);
		}
	}	


	void map_rgba(ViewRGB const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_rgb_rgba(s, d, src.width);
		}
	}
	
}


/* map_rgb */

namespace simage
{	
	static void map_span_rgba_u8_to_f32(Pixel* src, RGBAf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst.R[i] = cs::to_channel_f32(src[i].rgba.red);
			dst.G[i] = cs::to_channel_f32(src[i].rgba.green);
			dst.B[i] = cs::to_channel_f32(src[i].rgba.blue);
			dst.A[i] = cs::to_channel_f32(src[i].rgba.alpha);
		}
	}


	static void map_span_rgb_u8_to_f32(Pixel* src, RGBf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst.R[i] = cs::to_channel_f32(src[i].rgba.red);
			dst.G[i] = cs::to_channel_f32(src[i].rgba.green);
			dst.B[i] = cs::to_channel_f32(src[i].rgba.blue);
		}
	}


	static void map_span_rgba_f32_to_u8(RGBAf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& rgba = dst[i].rgba;
			rgba.red = cs::to_channel_u8(src.R[i]);
			rgba.green = cs::to_channel_u8(src.G[i]);
			rgba.blue = cs::to_channel_u8(src.B[i]);
			rgba.alpha = cs::to_channel_u8(src.A[i]);
		}
	}


	static void map_span_rgb_f32_to_u8(RGBf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& rgba = dst[i].rgba;
			rgba.red = cs::to_channel_u8(src.R[i]);
			rgba.green = cs::to_channel_u8(src.G[i]);
			rgba.blue = cs::to_channel_u8(src.B[i]);
			rgba.alpha = 255;
		}
	}


	static inline void map_span_rgba_f32_to_u8(f32* src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto const gray = cs::to_channel_u8(src[i]);
			auto& rgba = dst[i].rgba;

			rgba.red = gray;
			rgba.green = gray;
			rgba.blue = gray;
			rgba.alpha = 255;
		}
	}


	void map_rgba(View const& src, ViewRGBAf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgba_row_begin(dst, y);

			map_span_rgba_u8_to_f32(s, d, src.width);
		}
	}

	
	void map_rgb(View const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			map_span_rgb_u8_to_f32(s, d, src.width);
		}
	}


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgba_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_rgba_f32_to_u8(s, d, src.width);
		}
	}


	void map_rgba(ViewRGBf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_rgb_f32_to_u8(s, d, src.width);
		}
	}


	void map_rgba(View1f32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			map_span_rgba_f32_to_u8(s, d, src.width);
		}
	}

}


/* map_hsv */

namespace simage
{	
	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		}
	}


	void map_hsv_rgba(ViewHSVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y) 
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::f32_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_rgb_hsv(ViewRGBf32 const& src, ViewHSVf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto hsv = hsv::f32_from_rgb_f32(r, g, b);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		}
	}


	void map_hsv_rgb(ViewHSVf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::f32_to_rgb_f32(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto lch = lch::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		}
	}


	void map_lch_rgba(ViewLCHf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = lch::f32_to_rgb_u8(s.L[x], s.C[x], s.H[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_rgb_lch(ViewRGBf32 const& src, ViewLCHf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto lch = lch::f32_from_rgb_f32(r, g, b);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		}
	}


	void map_lch_rgb(ViewLCHf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = lch_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = lch::f32_to_rgb_f32(s.L[x], s.C[x], s.H[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s422 = (YUV422u8*)row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgb = yuv::u8_to_rgb_f32(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::u8_to_rgb_f32(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}

/*
	void mipmap_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{		
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d) 
		{
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		for (u32 y = 0; y < src.height; ++y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgb = yuv::u8_to_rgb_f32(y_avg, u_avg, v_avg);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}*/


	void map_yuv_rgb2(ViewYUV const& src, View const& dst)
	{
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d)
		{
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		for (u32 y = 0; y < src.height; ++y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgba = yuv::u8_to_rgb_u8(y_avg, u_avg, v_avg);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_yuv_rgba(ViewYUVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y) 
		{
			auto s = yuv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = yuv::f32_to_rgb_u8(s.Y[x], s.U[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}
}


/* map bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
	{		
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d.R[x] = cs::to_channel_f32(s[x].red);
				d.G[x] = cs::to_channel_f32(s[x].green);
				d.B[x] = cs::to_channel_f32(s[x].blue);
			}
		}
	}
}
