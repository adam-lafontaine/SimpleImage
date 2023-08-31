/* yuv */

namespace simage
{
    static inline void map_span_yuv_rgb(YUV2u8* src, Pixel* dst, u32 len)
    {
        auto src422 = (YUYVu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
			auto& d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;

			++i;
			rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;
		}
    }


	static inline void map_span_yuv_rgb(UVY2u8* src, Pixel* dst, u32 len)
    {
        auto src422 = (UYVYu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
			auto& d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;

			++i;
			rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;
		}
    }


    static inline void map_span_yuv_rgb(YUV2u8* src, RGBf32p const& dst, u32 len)
    {
        auto src422 = (YUYVu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
            dst.R[i] =  cs::to_channel_f32(rgb.red);
            dst.G[i] =  cs::to_channel_f32(rgb.green);
            dst.B[i] =  cs::to_channel_f32(rgb.blue);

			++i;
			rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			dst.R[i] =  cs::to_channel_f32(rgb.red);
            dst.G[i] =  cs::to_channel_f32(rgb.green);
            dst.B[i] =  cs::to_channel_f32(rgb.blue);
		}
    }


	static inline void map_span_yuv_rgb(UVY2u8* src, RGBf32p const& dst, u32 len)
    {
        auto src422 = (UYVYu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
            dst.R[i] =  cs::to_channel_f32(rgb.red);
            dst.G[i] =  cs::to_channel_f32(rgb.green);
            dst.B[i] =  cs::to_channel_f32(rgb.blue);

			++i;
			rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			dst.R[i] =  cs::to_channel_f32(rgb.red);
            dst.G[i] =  cs::to_channel_f32(rgb.green);
            dst.B[i] =  cs::to_channel_f32(rgb.blue);
		}
    }
}


/* hsv */

namespace simage
{
    static inline void map_span_rgb_hsv(Pixel* src, HSVf32p const& dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			auto hsv = hsv::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			dst.H[i] = hsv.hue;
			dst.S[i] = hsv.sat;
			dst.V[i] = hsv.val;
		}
    }


    static inline void map_span_hsv_rgba(HSVf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = hsv::f32_to_rgb_u8(src.H[i], src.S[i], src.V[i]);
			auto& d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;
		}
	}


    static inline void map_span_rgb_hsv(RGBf32p const& src, HSVf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto hsv = hsv::f32_from_rgb_f32(src.R[i], src.G[i], src.B[i]);
			dst.H[i] = hsv.hue;
			dst.S[i] = hsv.sat;
			dst.V[i] = hsv.val;
		}
	}


	static inline void map_span_hsv_rgb(HSVf32p const& src, RGBf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = hsv::f32_to_rgb_f32(src.H[i], src.S[i], src.V[i]);
			dst.R[i] = rgb.red;
			dst.G[i] = rgb.green;
			dst.B[i] = rgb.blue;
		}
	}
}


/* lch */

namespace simage
{
    static inline void map_span_rgb_lch(Pixel* src, LCHf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			auto lch = lch::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			dst.L[i] = lch.light;
			dst.C[i] = lch.chroma;
			dst.H[i] = lch.hue;
		}
	}


	static inline void map_span_lch_rgba(LCHf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = lch::f32_to_rgb_u8(src.L[i], src.C[i], src.H[i]);
			auto& d = dst[i].rgba;
			d.red = rgb.red;
			d.green = rgb.green;
			d.blue = rgb.blue;
			d.alpha = 255;
		}
	}


	static inline void map_span_rgb_lch(RGBf32p const& src, LCHf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto lch = lch::f32_from_rgb_f32(src.R[i], src.G[i], src.B[i]);
			dst.L[i] = lch.light;
			dst.C[i] = lch.chroma;
			dst.H[i] = lch.hue;
		}
	}


	static inline void map_span_lch_rgb(LCHf32p const& src, RGBf32p const& dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = lch::f32_to_rgb_f32(src.L[i], src.C[i], src.H[i]);
			dst.R[i] = rgb.red;
			dst.G[i] = rgb.green;
			dst.B[i] = rgb.blue;
		}
	}
}


/* map_yuv */

namespace simage
{
    void map_yuv_rgba(ViewYUV const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, dst.data, len);
    }


	void map_yuv_rgba(ViewUVY const& src, View const& dst)
	{
		assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, dst.data, len);
	}


    void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, rgb_row_begin(dst, 0), len);
    }


	void map_yuv_rgb(ViewUVY const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, rgb_row_begin(dst, 0), len);
    }
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb_hsv(src.data, hsv_row_begin(dst, 0), len);
    }


	void map_hsv_rgba(ViewHSVf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_hsv_rgba(hsv_row_begin(src, 0), dst.data, len);
    }


	void map_rgb_hsv(ViewRGBf32 const& src, ViewHSVf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb_hsv(rgb_row_begin(src, 0), hsv_row_begin(dst, 0), len);
    }


	void map_hsv_rgb(ViewHSVf32 const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_hsv_rgb(hsv_row_begin(src, 0), rgb_row_begin(dst, 0), len);
    }
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb_lch(src.data, lch_row_begin(dst, 0), len);
    }


	void map_lch_rgba(ViewLCHf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_lch_rgba(lch_row_begin(src, 0), dst.data, len);
    }


	void map_rgb_lch(ViewRGBf32 const& src, ViewLCHf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb_lch(rgb_row_begin(src, 0), lch_row_begin(dst, 0), len);
    }


	void map_lch_rgb(ViewLCHf32 const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_lch_rgb(lch_row_begin(src, 0), rgb_row_begin(dst, 0), len);
    }
}