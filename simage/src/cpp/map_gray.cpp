/* gray to gray */

/* no_simd */

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
}

/*
namespace simage
{
#ifdef SIMAGE_NO_SIMD

    template <typename TSRC, typename TDST>
	static inline void map_view_gray_1(View1<TSRC> const& src, View1<TDST> const& dst)
    {
        map_view_gray_1_no_simd(src, dst);
    }


    template <class ViewSRC, class ViewDST>
	static inline void map_sub_view_gray_1(ViewSRC const& src, ViewDST const& dst)
    {
        map_sub_view_gray_1_no_simd(src, dst);
    }

#else

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


    template <typename TSRC, typename TDST>
	static inline void map_view_gray_1(View1<TSRC> const& src, View1<TDST> const& dst)
    {
        auto len = src.width * src.height;

        if (len < simd::LEN)
        {
            map_view_gray_1_no_simd(src, dst);
            return;
        }

        map_span_gray(src.data, dst.data, len);
    }


    template <class ViewSRC, class ViewDST>
	static inline void map_sub_view_gray_1(ViewSRC const& src, ViewDST const& dst)
    {
        if (src.width < simd::LEN)
        {
            map_sub_view_gray_1_no_simd(src, dst);
            return;
        }

        for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_span_gray(s, d, src.width);
		}
    }

#endif // SIMAGE_NO_SIMD
}
*/

/* yuv to gray */

namespace simage
{
	template <typename YUVu8T>
    static inline void map_span_yuv_to_gray(YUVu8T* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = src[i].y;
		}
	}


	template <typename YUVu8T>
    static inline void map_span_yuv_to_gray(YUVu8T* src, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = cs::to_channel_f32(src[i].y);
		}
	}
}


/* rgb to gray */

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


	static inline void map_span_rgb_to_gray(BGRu8* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto bgr = src[i];
			dst[i] = gray::u8_from_rgb_u8(bgr.red, bgr.green, bgr.blue);
		}
	}


	static inline void map_span_rgb_to_gray(RGBu8* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = src[i];
			dst[i] = gray::u8_from_rgb_u8(rgb.red, rgb.green, rgb.blue);
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


    static inline void map_span_rgb_to_gray(f32* r, f32* g, f32* b, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = gray::f32_from_rgb_f32(r[i], g[i], b[i]);
		}
	}


    static inline void map_span_rgb_to_gray(f32* r, f32* g, f32* b, u8* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = gray::u8_from_rgb_f32(r[i], g[i], b[i]);
        }
    }
    

    static inline void map_span_rgb_to_gray(RGBf32p const& src, f32* dst, u32 len)
	{
		map_span_rgb_to_gray(src.R, src.G, src.B, dst, len);
	}


	static inline void map_span_rgb_to_gray(RGBf32p const& src, u8* dst, u32 len)
	{
		map_span_rgb_to_gray(src.R, src.G, src.B, dst, len);
	}
}


namespace simage
{
	template <class RGBT, class GrayT>
    static inline void map_view_rgb_to_gray(View1<RGBT> const& src, View1<GrayT> const& dst)
    {
        u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

        map_span_rgb_to_gray(s, d, len);
    }


	template <class RGBT, class GrayT>
    static inline void map_sub_view_rgb_to_gray(View1<RGBT> const& src, View1<GrayT> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_rgb_to_gray(s, d, src.width);
		}
	}


	void map_gray(View const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
	}


	void map_gray(ViewBGR const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
	}


	void map_gray(ViewRGB const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
	}
}


/* yuv to gray */

namespace simage
{
	template <class YUVT, class GrayT>
    static inline void map_view_yuv_to_gray(View1<YUVT> const& src, View1<GrayT> const& dst)
    {
        u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

        map_span_yuv_to_gray(s, d, len);
    }


	template <class YUVT, class GrayT>
    static inline void map_sub_view_yuv_to_gray(View1<YUVT> const& src, View1<GrayT> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_yuv_to_gray(s, d, src.width);
		}
	}


	void map_gray(ViewYUV const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			map_view_yuv_to_gray(src, dst);
		}
		else
		{
			map_sub_view_yuv_to_gray(src, dst);
		}
	}


	void map_gray(ViewUVY const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
		{
			map_view_yuv_to_gray(src, dst);
		}
		else
		{
			map_sub_view_yuv_to_gray(src, dst);
		}
	}
}


/* gray to gray */

namespace simage
{
    template <typename TSRC, typename TDST>
	static inline void map_view_gray_1(View1<TSRC> const& src, View1<TDST> const& dst)
	{
		auto len = src.width * src.height;

		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

		map_span_gray_no_simd(s, d, len);
	}


	template <typename TSRC, typename TDST>
	static inline void map_sub_view_gray_1(View1<TSRC> const& src, View1<TDST> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_gray_no_simd(s, d, src.width);
		}
	}


    void map_gray(View1u8 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(src))
		{
			map_view_gray_1(src, dst);
		}
		else
		{
			map_sub_view_gray_1(src, dst);
		}
    }


	void map_gray(View1f32 const& src, View1u8 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
		{
			map_view_gray_1(src, dst);
		}
		else
		{
			map_sub_view_gray_1(src, dst);
		}
    }    
}


/* yuv to gray */

namespace simage
{
	void map_gray(ViewYUV const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
		{
			map_view_yuv_to_gray(src, dst);
		}
		else
		{
			map_sub_view_yuv_to_gray(src, dst);
		}
    }


	void map_gray(ViewUVY const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
		{
			map_view_yuv_to_gray(src, dst);
		}
		else
		{
			map_sub_view_yuv_to_gray(src, dst);
		}
    }
}


/* rgb to gray */

namespace simage
{
	template <class GrayT>
    static inline void map_view_rgb_to_gray(ViewRGBf32 const& src, View1<GrayT> const& dst)
    {
        u32 len = src.width * src.height;
		auto d = row_begin(dst, 0);
		auto s = rgb_row_begin(src, 0);

        map_span_rgb_to_gray(s, d, len);
    }


	template <class GrayT>
	static inline void map_sub_view_rgb_to_gray(ViewRGBf32 const& src, View1<GrayT> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = rgb_row_begin(src, y);

			map_span_rgb_to_gray(s, d, src.width);
		}
	}


    void map_gray(View const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
    }


	void map_gray(ViewRGBf32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
    }


	void map_gray(ViewRGBf32 const& src, View1u8 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
		{
			map_view_rgb_to_gray(src, dst);
		}
		else
		{
			map_sub_view_rgb_to_gray(src, dst);
		}
    }
}