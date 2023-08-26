/* transform */

namespace simage
{
	template <typename TSRC, typename TDST, class FUNC>	
	static void transform_view(View1<TSRC> const& src, View1<TDST> const& dst, FUNC const& func)
	{
		u32 len = src.width * src.height;

		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = func(src[i]);
		}
	}


	template <typename TSRC, typename TDST, class FUNC>	
	static void transform_sub_view(View1<TSRC> const& src, View1<TDST> const& dst, FUNC const& func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		}
	}
}


/* binarize span */

namespace simage
{
	template <typename TSRC, class FUNC>
    static inline void binarize_span_1(TSRC* src, u8* dst, u32 len, FUNC const& bool_func32)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = bool_func32(src[i]) * 255;
        }
    }


    template <typename TSRC, class FUNC>
    static inline void binarize_span_1(TSRC* src, f32* dst, u32 len, FUNC const& bool_func32)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = bool_func32(src[i]) * 1.0f;
        }
    }
}


/* threshold span*/

namespace simage
{
	template <typename T>
	static inline void threshold_span(T* src, T* dst, u32 len, T min)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = (src[i] >= min) * src[i];
		}
	}


	template <typename T>
	static inline void threshold_span(T* src, T* dst, u32 len, T min, T max)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = (src[i] >= min && src[i] <= max) * src[i];
		}
	}
}


/* transform */

namespace simage
{
	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));

		transform_view(src, dst, func);
	}


	void transform(ViewGray const& src, ViewGray const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		transform_view(src, dst, func);
	}


	void transform(View const& src, ViewGray const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		transform_view(src, dst, func);
	}


	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func32)
	{
		assert(verify(src, dst));

		transform_view(src, dst, func32);
	}
	
}


/* binarize */

namespace simage
{
	void binarize(View const& src, ViewGray const& dst, pixel_to_bool_f const& func)
	{
		assert(verify(src, dst));

		u32 len = src.width * src.height;

		binarize_span_1(src.data, dst.data, len, func);
	}


	void binarize(ViewGray const& src, ViewGray const& dst, u8_to_bool_f const& func)
	{
		assert(verify(src, dst));

		u32 len = src.width * src.height;

		binarize_span_1(src.data, dst.data, len, func);
	}


	void binarize(View1f32 const& src, View1f32 const& dst, std::function<bool(f32)> const& func32)
    {
        assert(verify(src, dst));

        u32 len = src.width * src.height;

        binarize_span_1(src.data, dst.data, len, func32);
    }
}


/* threshold */

namespace simage
{
	void threshold(ViewGray const& src, ViewGray const& dst, u8 min)
	{
		assert(verify(src, dst));

		u32 len = src.width * src.height;

		threshold_span(src.data, dst.data, len, min);
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));

		auto [mn, mx] = std::minmax(min, max);

		u32 len = src.width * src.height;

		threshold_span(src.data, dst.data, len, mn, mx);
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32)
	{
		assert(verify(src, dst));

        u32 len = src.width * src.height;

		threshold_span(src.data, dst.data, len, min32);
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32, f32 max32)
	{
		assert(verify(src, dst));

        u32 len = src.width * src.height;

		threshold_span(src.data, dst.data, len, min32, max32);
	}
}