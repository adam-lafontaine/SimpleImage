/* map_span yuv */

namespace simage
{
    static inline void map_span_yuv_rgba(YUV2u8* src, Pixel* dst, u32 len)
    {
        auto src422 = (YUYVu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto& d1 = dst[i].rgba;
			yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v, &d1.red, &d1.green, &d1.blue);

			++i;
			auto& d2 = dst[i].rgba;
			yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v, &d2.red, &d2.green, &d2.blue);
		}
    }


	static inline void map_span_yuv_rgba(UVY2u8* src, Pixel* dst, u32 len)
    {
        auto src422 = (UYVYu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto& d1 = dst[i].rgba;
			yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v, &d1.red, &d1.green, &d1.blue);

			++i;
			auto& d2 = dst[i].rgba;
			yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v, &d2.red, &d2.green, &d2.blue);
		}
    }


    static inline void map_span_yuv_rgb(YUV2u8* src, RGBf32p const& dst, u32 len)
    {
        auto src422 = (YUYVu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			yuv::u8_to_rgb_f32(yuv.y1, yuv.u, yuv.v, dst.R + i, dst.G + i, dst.B + i);

			++i;
			yuv::u8_to_rgb_f32(yuv.y2, yuv.u, yuv.v, dst.R + i, dst.G + i, dst.B + i);
		}
    }


	static inline void map_span_yuv_rgb(UVY2u8* src, RGBf32p const& dst, u32 len)
    {
        auto src422 = (UYVYu8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			yuv::u8_to_rgb_f32(yuv.y1, yuv.u, yuv.v, dst.R + i, dst.G + i, dst.B + i);

			++i;
			yuv::u8_to_rgb_f32(yuv.y2, yuv.u, yuv.v, dst.R + i, dst.G + i, dst.B + i);
		}
    }


	static inline void map_span_yuv_rgba(YUVf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& d = dst[i].rgba;
			yuv::f32_to_rgb_u8(src.Y[i], src.U[i], src.V[i], &d.red, &d.green, &d.blue);
		}
	}
}


/* map_span hsv */

namespace simage
{
    static inline void map_span_rgb_hsv(Pixel* src, HSVf32p const& dst, u32 len)
    {
		auto ph = dst.H;
		auto ps = dst.S;
		auto pv = dst.V;

        for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			hsv::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue, (ph + i), (ps + i), (pv + i));
		}
    }


    static inline void map_span_hsv_rgba(HSVf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& d = dst[i].rgba;
			hsv::f32_to_rgb_u8(src.H[i], src.S[i], src.V[i], &d.red, &d.green, &d.blue);
		}
	}


    static inline void map_span_rgb_hsv(RGBf32p const& src, HSVf32p const& dst, u32 len)
	{
		auto ph = dst.H;
		auto ps = dst.S;
		auto pv = dst.V;

		for (u32 i = 0; i < len; ++i)
		{
			hsv::f32_from_rgb_f32(src.R[i], src.G[i], src.B[i], (ph + i), (ps + i), (pv + i));
		}
	}


	static inline void map_span_hsv_rgb(HSVf32p const& src, RGBf32p const& dst, u32 len)
	{
		auto pr = dst.R;
		auto pg = dst.G;
		auto pb = dst.B;

		for (u32 i = 0; i < len; ++i)
		{
			hsv::f32_to_rgb_f32(src.H[i], src.S[i], src.V[i], (pr + i), (pg + i), (pb + i));
		}
	}
}


/* map_span lch */

namespace simage
{
    static inline void map_span_rgb_lch(Pixel* src, LCHf32p const& dst, u32 len)
	{
		auto pl = dst.L;
		auto pc = dst.C;
		auto ph = dst.H;

		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = src[i].rgba;
			lch::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue, (pl + i), (pc + i), (ph + i));
		}
	}


	static inline void map_span_lch_rgba(LCHf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto& d = dst[i].rgba;
			lch::f32_to_rgb_u8(src.L[i], src.C[i], src.H[i], &d.red, &d.green, &d.blue);
		}
	}


	static inline void map_span_rgb_lch(RGBf32p const& src, LCHf32p const& dst, u32 len)
	{
		auto pl = dst.L;
		auto pc = dst.C;
		auto ph = dst.H;

		for (u32 i = 0; i < len; ++i)
		{
			lch::f32_from_rgb_f32(src.R[i], src.G[i], src.B[i], (pl + i), (pc + i), (ph + i));
		}
	}


	static inline void map_span_lch_rgb(LCHf32p const& src, RGBf32p const& dst, u32 len)
	{
		auto pr = dst.R;
		auto pg = dst.G;
		auto pb = dst.B;

		for (u32 i = 0; i < len; ++i)
		{
			lch::f32_to_rgb_f32(src.L[i], src.C[i], src.H[i], (pr + i), (pg + i), (pb + i));
		}
	}
}


/* map_yuv */

namespace simage
{
	template <typename YUV>
	static inline void map_view_yuv_rgba(View1<YUV> const& src, View const& dst)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

		map_span_yuv_rgba(s, d, len);
	}


	template <typename YUV>
	static inline void map_sub_view_yuv_rgba(View1<YUV> const& src, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_yuv_rgba(s, d, src.width);
		}
	}


	template <typename YUV>
	static inline void map_view_yuv_rgb(View1<YUV> const& src, ViewRGBf32 const& dst)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = rgb_row_begin(dst, 0);

		map_span_yuv_rgb(s, d, len);
	}


	template <typename YUV>
	static inline void map_sub_view_yuv_rgb(View1<YUV> const& src, ViewRGBf32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			map_span_yuv_rgb(s, d, src.width);
		}
	}


	static inline void map_view_yuv_rgba(ViewYUVf32 const& src, View const& dst)
	{
		u32 len = src.width * src.height;
		auto s = yuv_row_begin(src, 0);
		auto d = row_begin(dst, 0);

		map_span_yuv_rgba(s, d, len);
	}


	static inline void map_sub_view_yuv_rgba(ViewYUVf32 const& src, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = yuv_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_yuv_rgba(s, d, src.width);
		}
	}


    void map_yuv_rgba(ViewYUV const& src, View const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			map_view_yuv_rgba(src, dst);
		}
		else
		{
			map_sub_view_yuv_rgba(src, dst);
		}
    }


	void map_yuv_rgba(ViewUVY const& src, View const& dst)
	{
		assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
		{
			map_view_yuv_rgba(src, dst);
		}
		else
		{
			map_sub_view_yuv_rgba(src, dst);
		}
	}


    void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
		{
			map_view_yuv_rgb(src, dst);
		}
		else
		{
			map_sub_view_yuv_rgb(src, dst);
		}
    }


	void map_yuv_rgb(ViewUVY const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
		{
			map_view_yuv_rgb(src, dst);
		}
		else
		{
			map_sub_view_yuv_rgb(src, dst);
		}
    }


	void map_yuv_rgba(ViewYUVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		if (is_1d(dst))
		{
			map_view_yuv_rgba(src, dst);
		}
		else
		{
			map_sub_view_yuv_rgba(src, dst);
		}
	}
}


/* map_hsv */

namespace simage
{
	static inline void map_view_rgba_hsv(View const& src, ViewHSVf32 const& dst)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = hsv_row_begin(dst, 0);

		map_span_rgb_hsv(s, d, len);
	}


	static inline void map_sub_view_rgba_hsv(View const& src, ViewHSVf32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			map_span_rgb_hsv(s, d, src.width);
		}
	}


	static inline void map_view_hsv_rgba(ViewHSVf32 const& src, View const& dst)
	{
		u32 len = src.width * src.height;
		auto s = hsv_row_begin(src, 0);
		auto d = row_begin(dst, 0);

		map_span_hsv_rgba(s, d, len);
	}


	static inline void map_sub_view_hsv_rgba(ViewHSVf32 const& src, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_hsv_rgba(s, d, src.width);
		}
	}


	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(src))
		{
			map_view_rgba_hsv(src, dst);
		}
		else
		{
			map_sub_view_rgba_hsv(src, dst);
		}
    }


	void map_hsv_rgba(ViewHSVf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(dst))
		{
			map_view_hsv_rgba(src, dst);
		}
		else
		{
			map_sub_view_hsv_rgba(src, dst);
		}
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
	static inline void map_view_rgba_lch(View const& src, ViewHSVf32 const& dst)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = lch_row_begin(dst, 0);

		map_span_rgb_lch(s, d, len);
	}


	static inline void map_sub_view_rgba_lch(View const& src, ViewHSVf32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			map_span_rgb_lch(s, d, src.width);
		}
	}


	static inline void map_view_lch_rgba(ViewHSVf32 const& src, View const& dst)
	{
		u32 len = src.width * src.height;
		auto s = lch_row_begin(src, 0);
		auto d = row_begin(dst, 0);

		map_span_lch_rgba(s, d, len);
	}


	static inline void map_sub_view_lch_rgba(ViewHSVf32 const& src, View const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_span_lch_rgba(s, d, src.width);
		}
	}


	void map_rgb_lch(View const& src, ViewLCHf32 const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(src))
		{
			map_view_rgba_lch(src, dst);
		}
		else
		{
			map_sub_view_rgba_lch(src, dst);
		}
    }


	void map_lch_rgba(ViewLCHf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

		if (is_1d(dst))
		{
			map_view_lch_rgba(src, dst);
		}
		else
		{
			map_sub_view_lch_rgba(src, dst);
		}
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