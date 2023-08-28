/* gray to rgb */

namespace simage
{
    static inline void map_span_gray_rgb(u8* src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = to_pixel(src[i], src[i], src[i]);
        }
    }


    static inline void map_span_gray_rgb(f32* src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto gray = cs::to_channel_u8(src[i]);
            dst[i] = to_pixel(gray, gray, gray);
        }
    }


    static inline void map_span_rgb(Pixel* src, RGBf32p const& dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto rgba = src[i].rgba;
            dst.R[i] =  cs::to_channel_f32(rgba.red);
            dst.G[i] =  cs::to_channel_f32(rgba.green);
            dst.B[i] =  cs::to_channel_f32(rgba.blue);
        }
    }


    static inline void map_span_rgb(Pixel* src, RGBAf32p const& dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto rgba = src[i].rgba;
            dst.R[i] =  cs::to_channel_f32(rgba.red);
            dst.G[i] =  cs::to_channel_f32(rgba.green);
            dst.B[i] =  cs::to_channel_f32(rgba.blue);
            dst.A[i] = cs::to_channel_f32(rgba.alpha);
        }
    }


    template <typename RGBT>
    static inline void map_span_rgb(RGBT* src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto rgb = src[i];
            dst[i] = to_pixel(rgb.red, rgb.green, rgb.blue);
        }
    }


    template <typename RGBT>
    static inline void map_span_rgb(RGBT* src, RGBf32p dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto rgb = src[i];
            dst.R[i] =  cs::to_channel_f32(rgb.red);
            dst.G[i] =  cs::to_channel_f32(rgb.green);
            dst.B[i] =  cs::to_channel_f32(rgb.blue);
        }
    }


    static inline void map_span_rgb(RGBf32p const& src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto red = cs::to_channel_u8(src.R[i]);
            auto green = cs::to_channel_u8(src.G[i]);
            auto blue = cs::to_channel_u8(src.B[i]);

            dst[i] = to_pixel(red, green, blue);
        }
    }


    static inline void map_span_rgb(RGBAf32p const& src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto red = cs::to_channel_u8(src.R[i]);
            auto green = cs::to_channel_u8(src.G[i]);
            auto blue = cs::to_channel_u8(src.B[i]);
            auto alpha = cs::to_channel_u8(src.A[i]);

            dst[i] = to_pixel(red, green, blue, alpha);
        }
    }
}


/* map_rgba */

namespace simage
{
    template <class ViewSRC, class ViewDST>
    static inline void map_view_rgb(ViewSRC const& src, ViewDST const& dst)
    {
        u32 len = src.width * src.height;

        map_span_rgb(src.data, dst.data, len);
    }


    template <class ViewSRC, class ViewDST>
    static inline void map_sub_view_rgb(ViewSRC const& src, ViewDST const& dst)
    {
        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


    template <typename T>
    static inline void map_view_gray_rgb(View1<T> const& src, View const& dst)
    {
        u32 len = src.width * src.height;

        map_span_gray_rgb(src.data, dst.data, len);
    }


    template <class ViewSRC, class ViewDST>
    static inline void map_sub_view_gray_rgb(ViewSRC const& src, ViewDST const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_gray_rgb(s, d, src.width);
        }
    }


	void map_rgba(ViewGray const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_view_gray_rgb(src, dst);
    }


    void map_rgba(ViewGray const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_rgb(src, dst);
    }


    void map_rgba(SubViewGray const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_rgb(src, dst);
    }
    

	void map_rgba(ViewBGR const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_view_rgb(src, dst);
    }


    void map_rgba(SubViewBGR const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb(src, dst);
    }


    void map_rgba(SubViewBGR const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb(src, dst);
    }


	void map_rgba(ViewRGB const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_view_rgb(src, dst);
    }


    void map_rgba(SubViewRGB const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb(src, dst);
    }


    void map_rgba(SubViewRGB const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_rgb(src, dst);
    }
}


namespace simage
{
    void map_rgba(View const& src, ViewRGBAf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(src.data, rgba_row_begin(dst, 0), len);
    }


	void map_rgba(SubView const& src, ViewRGBAf32 const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = row_begin(src, y);
            auto d = rgba_row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


	void map_rgb(View const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(src.data, rgb_row_begin(dst, 0), len);
    }


	void map_rgb(SubView const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = row_begin(src, y);
            auto d = rgb_row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(rgba_row_begin(src, 0), dst.data, len);
    }


	void map_rgba(ViewRGBAf32 const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = rgba_row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


	void map_rgba(ViewRGBf32 const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.width; ++y)
        {
            auto s = rgb_row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


	void map_rgba(ViewRGBf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(rgb_row_begin(src, 0), dst.data, len);
    }


	void map_rgba(View1f32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_view_gray_rgb(src, dst);
    }


	void map_rgba(SubView1f32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_rgb(src, dst);
    }


	void map_rgba(View1f32 const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_rgb(src, dst);
    }


	void map_rgba(SubView1f32 const& src, SubView const& dst)
    {
        assert(verify(src, dst));

        map_sub_view_gray_rgb(src, dst);
    }


    void map_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(src.data, rgb_row_begin(dst, 0), len);
    }


    void map_rgb(ViewRGB const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        map_span_rgb(src.data, rgb_row_begin(dst, 0), len);
    }
}