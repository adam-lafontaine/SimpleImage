/* copy span */

namespace simage
{	
	template <typename T>
	static void copy_span_no_simd(T* src, T* dst, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = src[i];
		}
	}


#ifdef SIMAGE_NO_SIMD

	template <typename T>
	static inline void copy_view_1(View1<T> const& src, View1<T> const& dst)
	{
		auto len = src.width * src.height;
		auto s = row_begin(src, 0);
		auto d = row_begin(dst, 0);

		copy_span(s, d, len);
	}


	template <typename T>
	static inline void copy_sub_view_1(View1<T> const& src, View1<T> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			copy_span(s, d, src.width);
		}
	}

#else

	template <typename T>
	static void copy_span(T* src, T* dst, u32 len)
	{
		constexpr auto step = (u32)simd::LEN * sizeof(f32) / sizeof(T);

		simd::vecf32 v_bytes{};

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			v_bytes = simd::load_bytes(src + i);
			simd::store_bytes(v_bytes, dst + i);
		}

		i = len - step;
		v_bytes = simd::load_bytes(src + i);
		simd::store_bytes(v_bytes, dst + i);
	}


	template <typename T>
	static inline void copy_view_1(View1<T> const& src, View1<T> const& dst)
	{
		auto len = src.width * src.height;

		if (len < simd::LEN)
		{
			copy_view_1_no_simd(src, dst);
			return;
		}

		copy_span(src.data, dst.data, len);		
	}


	template <class ViewSRC, class ViewDST>
	static inline void copy_sub_view_1(ViewSRC const& src, ViewDST const& dst)
	{
		if (src.width < simd::LEN)
		{
			copy_sub_view_1_no_simd(src, dst);
			return;
		}
		
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			copy_span(s, d, src.width);
		}
	}

#endif // SIMAGE_NO_SIMD
}


namespace simage
{
	template <typename T, size_t N>
	static inline void copy_view_n(ChannelMatrix2D<T, N> const& src, ChannelMatrix2D<T, N> const& dst)
	{
		for (u32 ch = 0; ch < (u32)N; ++ch)
		{
			copy_view_1(select_channel(src, ch), select_channel(dst, ch));
		}
	}
}


/* copy */

namespace simage
{
	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			copy_view_1(src, dst);
		}
		else
		{
			copy_sub_view_1(src, dst);
		}
	}


	void copy(ViewGray const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		if (is_1d(src) && is_1d(dst))
		{
			copy_view_1(src, dst);
		}
		else
		{
			copy_sub_view_1(src, dst);
		}
	}
}


/* copy */

namespace simage
{
	void copy(View4f32 const& src, View4f32 const& dst)
	{
		assert(verify(src, dst));

		copy_view_n(src, dst);
	}


	void copy(View3f32 const& src, View3f32 const& dst)
	{
		assert(verify(src, dst));

		copy_view_n(src, dst);
	}


	void copy(View2f32 const& src, View2f32 const& dst)
	{
		assert(verify(src, dst));

		copy_view_n(src, dst);
	}


	void copy(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		copy_view_1(src, dst);
	}
}
