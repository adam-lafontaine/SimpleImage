/* copy row */

namespace simage
{
#ifdef SIMAGE_NO_SIMD

	template <typename T>
	static void copy_row(T* src, T* dst, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			dst[i] = src[i];
		}
	}

#else

	template <typename T>
	static void copy_row(T* src, T* dst, u32 width)
	{
		constexpr auto step = (u32)simd::LEN * sizeof(f32) / sizeof(T);

		simd::vecf32 v_bytes;

		u32 x = 0;
        for (; x < width; x += step)
		{
			simd::load_bytes(src + x, v_bytes);
			simd::store_bytes(v_bytes, dst + x);
		}

		x = width - step;
		simd::load_bytes(src + x, v_bytes);
		simd::store_bytes(v_bytes, dst + x);
	}

#endif // SIMAGE_NO_SIMD
}


/* copy */

namespace simage
{		
	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			copy_row(s, d, src.width);
		}
	}


	void copy(ViewGray const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			copy_row(s, d, src.width);
		}
	}
}
