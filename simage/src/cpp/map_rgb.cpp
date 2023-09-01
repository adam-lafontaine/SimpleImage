/* gray to rgb */

namespace simage
{
    static inline void map_span_gray_rgb(u8* src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto& rgba = dst[i].rgba;
            auto gray = src[i];
            rgba.red = gray;
            rgba.green = gray;
            rgba.blue = gray;
            rgba.alpha = 255;
        }
    }


    static inline void map_span_gray_rgb(f32* src, Pixel* dst, u32 len)
    {
        for (u32 i = 0; i < len; ++i)
        {
            auto gray = cs::to_channel_u8(src[i]);
            auto& rgba = dst[i].rgba;
            rgba.red = gray;
            rgba.green = gray;
            rgba.blue = gray;
            rgba.alpha = 255;
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
            auto& rgba = dst[i].rgba;
            rgba.red = rgb.red;
            rgba.green = rgb.green;
            rgba.blue = rgb.blue;
            rgba.alpha = 255;
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
            auto& rgba = dst[i].rgba;
            rgba.red = cs::to_channel_u8(src.R[i]);
            rgba.green = cs::to_channel_u8(src.G[i]);
            rgba.blue = cs::to_channel_u8(src.B[i]);
            rgba.alpha = 255;
        }
    }


    static inline void map_span_rgb(RGBAf32p const& src, Pixel* dst, u32 len)
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
}


/* map_rgba */

namespace simage
{
    template <class ViewSRC, class ViewDST>
    static inline void map_view_rgb(ViewSRC const& src, ViewDST const& dst)
    {
        u32 len = src.width * src.height;
        auto s = row_begin(src, 0);
        auto d = row_begin(dst, 0);

        map_span_rgb(s, d, src.width);
    }


    template <class ViewSRC, class ViewDST>
    static inline void map_sub_view_rgb(ViewSRC const& src, ViewDST const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
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
        auto s = row_begin(src, 0);
        auto d = row_begin(dst, 0);

        map_span_gray_rgb(s, d, len);
    }


    template <typename T>
    static inline void map_sub_view_gray_rgb(View1<T> const& src, View const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
        {
            auto s = row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_gray_rgb(s, d, src.width);
        }
    }


	void map_rgba(ViewGray const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
        {
            map_view_gray_rgb(src, dst);
        }
        else
        {
            map_sub_view_gray_rgb(src, dst);
        }
    }
    

	void map_rgba(ViewBGR const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


	void map_rgba(ViewRGB const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }
}


namespace simage
{
    template <typename T>
    static inline void map_view_rgb(View1<T> const& src, ViewRGBAf32 const& dst)
    {
        u32 len = src.width * src.height;
        auto s = row_begin(src, 0);
        auto d = rgba_row_begin(dst, 0);

        map_span_rgb(s, d, len);
    }


    template <typename T>
    static inline void map_sub_view_rgb(View1<T> const& src, ViewRGBAf32 const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
        {
            auto s = row_begin(src, y);
            auto d = rgba_row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


    template <typename T>
    static inline void map_view_rgb(View1<T> const& src, ViewRGBf32 const& dst)
    {
        u32 len = src.width * src.height;
        auto s = row_begin(src, 0);
        auto d = rgb_row_begin(dst, 0);

        map_span_rgb(s, d, len);
    }


    template <typename T>
    static inline void map_sub_view_rgb(View1<T> const& src, ViewRGBf32 const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
        {
            auto s = row_begin(src, y);
            auto d = rgb_row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


    template <typename T>
    static inline void map_view_rgb(ViewRGBAf32 const& src, View1<T> const& dst)
    {
        u32 len = src.width * src.height;
        auto s = rgba_row_begin(src, 0);
        auto d = row_begin(dst, 0);

        map_span_rgb(s, d, len);
    }


    template <typename T>
    static inline void map_sub_view_rgb(ViewRGBAf32 const& src, View1<T> const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
        {
            auto s = rgba_row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


    template <typename T>
    static inline void map_view_rgb(ViewRGBf32 const& src, View1<T> const& dst)
    {
        u32 len = src.width * src.height;
        auto s = rgb_row_begin(src, 0);
        auto d = row_begin(dst, 0);

        map_span_rgb(s, d, len);
    }


    template <typename T>
    static inline void map_sub_view_rgb(ViewRGBf32 const& src, View1<T> const& dst)
    {
        for (u32 y = 0; y < src.height; ++y)
        {
            auto s = rgb_row_begin(src, y);
            auto d = row_begin(dst, y);

            map_span_rgb(s, d, src.width);
        }
    }


    void map_rgba(View const& src, ViewRGBAf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


	void map_rgb(View const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


	void map_rgba(ViewRGBf32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


	void map_rgba(View1f32 const& src, View const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(dst))
        {
            map_view_gray_rgb(src, dst);
        }
        else
        {
            map_sub_view_gray_rgb(src, dst);
        }
    }


    void map_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }


    void map_rgb(ViewRGB const& src, ViewRGBf32 const& dst)
    {
        assert(verify(src, dst));

        if (is_1d(src))
        {
            map_view_rgb(src, dst);
        }
        else
        {
            map_sub_view_rgb(src, dst);
        }
    }
}