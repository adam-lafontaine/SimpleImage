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


    template <typename TSRC, typename TDST>
	static inline void map_view_gray_1(View1<TSRC> const& src, View1<TDST> const& dst)
	{
		auto len = src.width * src.height;

		map_span_gray_no_simd(src.data, dst.data, len);
	}


	template <class ViewSRC, class ViewDST>
	static inline void map_sub_view_gray_1(ViewSRC const& src, ViewDST const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_span_gray_no_simd(s, d, src.width);
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
    static inline void map_span_yuv_to_gray(YUV2u8* src, u8* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = src[i].y;
		}
	}


    static inline void map_span_yuv_to_gray(YUV2u8* src, f32* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = cs::to_channel_f32(src[i].y);
		}
	}


    template <class ViewSRC, class ViewDST>
    static inline void map_view_yuv_to_gray(ViewSRC const& src, ViewDST const& dst)
    {
        auto len = src.width * src.height;

        map_span_yuv_to_gray(src.data, dst.data, len);
    }


    template <class ViewSRC, class ViewDST>
    static inline void map_sub_view_yuv_to_gray(ViewSRC const& src, ViewDST const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			map_span_yuv_to_gray(s, d, src.width);
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


    static inline void map_span_rgb_to_gray(Pixel* src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			auto gray = gray::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
            dst[i] = to_pixel(gray, gray, gray);
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


    static inline void map_span_rgb_to_gray(RGBAf32p const& src, f32* dst, u32 len)
	{
		map_span_rgb_to_gray(src.R, src.G, src.B, dst, len);
	}
}


namespace simage
{
    template <class TSRC, class TDST>
    static inline void map_view_rgb_to_gray(View1<TSRC> const& src, View1<TDST> const& dst)
    {
        u32 len = src.width * src.height;

        map_span_rgb_to_gray(src.data, dst.data, len);
    }


    template <size_t N, class TDST>
    static inline void map_view_rgb_to_gray(ChannelMatrix2D<f32, N> const& src, View1<TDST> const& dst)
    {
        u32 len = src.width * src.height;

        map_span_rgb_to_gray(row_begin(src, 0), dst.data, len);
    }



    template <class ViewSRC, class ViewDST>
    static inline void map_sub_view_rgb_to_gray(ViewSRC const& src, ViewDST const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			map_span_rgb_to_gray(s, d, src.width);
		}
    }
}


namespace simage
{
	void map_gray(View const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

        map_view_rgb_to_gray(src, dst);
	}


    void map_gray(SubView const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb_to_gray(src, dst);
    }


	void map_gray(SubView const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb_to_gray(src, dst);
    }


	void map_gray(ViewYUV const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

        map_view_yuv_to_gray(src, dst);
	}
}



namespace simage
{
    void map_gray(View1u8 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_view_gray_1(src, dst);
    }


    void map_gray(SubView1u8 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_1(src, dst);
    }


	void map_gray(View1f32 const& src, View1u8 const& dst)
    {
        assert(verify(src, dst));

        map_view_gray_1(src, dst);
    }


	void map_gray(View1f32 const& src, SubView1u8 const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_1(src, dst);
    }


    void map_gray(ViewYUV const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_view_yuv_to_gray(src, dst);
    }
}


namespace simage
{
    void map_gray(View const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_view_rgb_to_gray(src, dst);
    }


	void map_gray(SubView const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb_to_gray(src, dst);
    }


	void map_gray(ViewRGBf32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        map_view_rgb_to_gray(src, dst);
    }


	void map_gray(ViewRGBf32 const& src, View1u8 const& dst)
    {
        assert(verify(src, dst));

        map_view_rgb_to_gray(src, dst);
    }

	void map_gray(ViewRGBf32 const& src, SubView1u8 const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb_to_gray(src, dst);
    }
}