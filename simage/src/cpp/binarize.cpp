namespace simage
{
    template <typename T, class FUNC>
    static inline void binarize_span_1(T* src, T* dst, u32 len, FUNC const& func32)
    {
        constexpr T zero = (T)(0);
        constexpr T one = (T)(1);

        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = func32(src[i]) ? one : zero;
        }
    }


    template <typename T, class FUNC>
    static inline void binarize_span_2(T* src_a, T* src_b, T* dst, u32 len, FUNC const& func32)
    {
        constexpr T zero = (T)(0);
        constexpr T one = (T)(1);

        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = func32(src_a[i], src_b[i]) ? one : zero;
        }
    }


    template <typename T, class FUNC>
    static inline void binarize_span_3(T* src_a, T* src_b, T* src_c, T* dst, u32 len, FUNC const& func32)
    {
        constexpr T zero = (T)(0);
        constexpr T one = (T)(1);

        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = func32(src_a[i], src_b[i], src_c[i]) ? one : zero;
        }
    }


    template <typename T, class FUNC>
    static inline void binarize_span_4(T* src_a, T* src_b, T* src_c, T* src_d, T* dst, u32 len, FUNC const& func32)
    {
        constexpr T zero = (T)(0);
        constexpr T one = (T)(1);

        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = func32(src_a[i], src_b[i], src_c[i], src_d[i]) ? one : zero;
        }
    }
}


/* binarize */

namespace simage
{
	void binarize(View1f32 const& src, View1f32 const& dst, std::function<bool(f32)> const& func32)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        binarize_span_1(src.data, dst.data, len, func32);
    }


	void binarize(View2f32 const& src, View1f32 const& dst, std::function<bool(f32, f32)> const& func32)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        auto begin = view_row_begin(src, 0);

        binarize_span_2(begin[0], begin[1], dst.data, len, func32);
    }


	void binarize(View3f32 const& src, View1f32 const& dst, std::function<bool(f32, f32, f32)> const& func32)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        auto begin = view_row_begin(src, 0);

        binarize_span_3(begin[0], begin[1], begin[2], dst.data, len, func32);
    }


	void binarize(View4f32 const& src, View1f32 const& dst, std::function<bool(f32, f32, f32, f32)> const& func32)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        auto begin = view_row_begin(src, 0);

        binarize_span_3(begin[0], begin[1], begin[2], begin[3], dst.data, len, func32);
    }
}