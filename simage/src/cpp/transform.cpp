


/* binarize span */

namespace simage
{
	
}


/* threshold span*/

namespace simage
{
	
}


/* transform */

namespace simage
{
	template <typename TSRC, typename TDST, class FUNC>	
	static void transform_view(View1<TSRC> const& src, View1<TDST> const& dst, FUNC const& func)
	{
		u32 len = src.width * src.height;

		for (u32 i = 0; i < len; ++i)
		{
			dst.data[i] = func(src.data[i]);
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


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			transform_view(src, dst, func);
		}
		else
		{
			transform_sub_view(src, dst, func);
		}
	}


	void transform(ViewGray const& src, ViewGray const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			transform_view(src, dst, func);
		}
		else
		{
			transform_sub_view(src, dst, func);
		}
	}


	void transform(View const& src, ViewGray const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			transform_view(src, dst, func);
		}
		else
		{
			transform_sub_view(src, dst, func);
		}
	}


	void transform(View1f32 const& src, View1f32 const& dst, std::function<f32(f32)> const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			transform_view(src, dst, func);
		}
		else
		{
			transform_sub_view(src, dst, func);
		}
	}
	
}


/* binarize */

namespace simage
{
	template <typename TSRC, class FUNC>
    static inline void binarize_span(TSRC* src, u8* dst, u32 len, FUNC const& bool_func)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = bool_func(src[i]) * 255;
        }
    }


    template <typename TSRC, class FUNC>
    static inline void binarize_span(TSRC* src, f32* dst, u32 len, FUNC const& bool_func)
    {
        for (u32 i = 0; i < len; ++i)
        {
            dst[i] = bool_func(src[i]) * 1.0f;
        }
    }


	template <typename TSRC, typename TDST, class FUNC>
	static inline void binarize_view(View1<TSRC> const& src, View1<TDST> const& dst, FUNC const& bool_func)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

		binarize_span(s, d, len, bool_func);
	}


	template <typename TSRC, typename TDST, class FUNC>
	static inline void binarize_sub_view(View1<TSRC> const& src, View1<TDST> const& dst, FUNC const& bool_func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			binarize_span(s, d, src.width, bool_func);
		}
	}


	void binarize(View const& src, ViewGray const& dst, pixel_to_bool_f const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			binarize_view(src, dst, func);
		}
		else
		{
			binarize_sub_view(src, dst, func);
		}
	}


	void binarize(ViewGray const& src, ViewGray const& dst, u8_to_bool_f const& func)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			binarize_view(src, dst, func);
		}
		else
		{
			binarize_sub_view(src, dst, func);
		}
	}


	void binarize(View1f32 const& src, View1f32 const& dst, std::function<bool(f32)> const& func)
    {
        assert(verify(src, dst));

        if (is_1d(src) && is_1d(dst))
		{
			binarize_view(src, dst, func);
		}
		else
		{
			binarize_sub_view(src, dst, func);
		}
    }
}


/* threshold */

namespace simage
{
	template <typename T>
	static inline void threshold_view(View1<T> const& src, View1<T> const& dst, T min, T max)
	{
		u32 len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

		for (u32 i = 0; i < len; ++i)
		{
			d[i] = (s[i] >= min && s[i] <= max) * s[i];
		}
	}


	template <typename T>
	static inline void threshold_sub_view(View1<T> const& src, View1<T> const& dst, T min, T max)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 i = 0; i < src.width; ++i)
			{
				d[i] = (s[i] >= min && s[i] <= max) * s[i];
			}
		}
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			threshold_view(src, dst, min, (u8)255);
		}
		else
		{
			threshold_sub_view(src, dst, min, (u8)255);
		}
	}


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));

		auto [mn, mx] = std::minmax(min, max);

		if (is_1d(src) && is_1d(dst))
		{
			threshold_view(src, dst, mn, mx);
		}
		else
		{
			threshold_sub_view(src, dst, mn, mx);
		}
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32)
	{
		assert(verify(src, dst));

        threshold_view(src, dst, min32, 1.0f);
	}


	void threshold(View1f32 const& src, View1f32 const& dst, f32 min32, f32 max32)
	{
		assert(verify(src, dst));

        auto [mn, mx] = std::minmax(min32, max32);

		threshold_view(src, dst, mn, mx);
	}
}