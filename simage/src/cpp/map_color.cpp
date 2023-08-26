namespace simage
{
    static inline void map_span_yuv_rgb(YUV2u8* src, Pixel* dst, u32 len)
    {
        auto src422 = (YUV422u8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
			auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
            dst[i] = to_pixel(rgb.red, rgb.green, rgb.blue);

			++i;
			rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
			dst[i] = to_pixel(rgb.red, rgb.green, rgb.blue);
		}
    }


    static inline void map_span_yuv(YUV2u8* src, YUVf32p const& dst, u32 len)
    {
        auto src422 = (YUV422u8*)src;

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			auto i = 2 * i422;
            dst.Y[i] = cs::to_channel_f32(yuv.y1);
            dst.U[i] = cs::to_channel_f32(yuv.u);
            dst.V[i] = cs::to_channel_f32(yuv.v);

            ++i;
            dst.Y[i] = cs::to_channel_f32(yuv.y2);
            dst.U[i] = cs::to_channel_f32(yuv.u);
            dst.V[i] = cs::to_channel_f32(yuv.v);
        }
    }


    static inline void map_span_yuv_rgb(YUV2u8* src, RGBf32p const& dst, u32 len)
    {
        auto src422 = (YUV422u8*)src;

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


    static inline void map_span_yuv_rgba(YUVf32p const& src, Pixel* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			auto rgb = yuv::f32_to_rgb_u8(src.Y[i], src.U[i], src.V[i]);

            dst[i] = to_pixel(rgb.red, rgb.green, rgb.blue);
		}
	}
}


namespace simage
{
    void map_yuv_rgba(ViewYUV const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, dst.data, len);
    }


    void map_yuv(ViewYUV const& src, ViewYUVf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv(src.data, yuv_row_begin(dst, 0), len);
    }


    void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgb(src.data, rgb_row_begin(dst, 0), len);
    }


	void map_yuv_rgba(ViewYUVf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_yuv_rgba(yuv_row_begin(src, 0), dst.data, len);
    }
}