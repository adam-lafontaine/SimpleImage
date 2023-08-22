/* copy span */

namespace simage
{
	template <typename T>
	static inline void copy_1_no_simd(View1<T> const& src, View1<T> const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 i = 0; i < src.width; ++i)
			{
				d[i] = s[i];
			}
		}
	}


#ifdef SIMAGE_NO_SIMD

	template <typename T>
	static inline void copy_1(View1<T> const& src, View1<T> const& dst)
	{
		copy_1_no_simd(src, dst);
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
	static void copy_1(View1<T> const& src, View1<T> const& dst)
	{
		if (src.width < simd::LEN)
		{
			copy_1_no_simd(src, dst);
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


/* copy */

namespace simage
{
	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		copy_1(src, dst);
	}


	void copy(ViewGray const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		copy_1(src, dst);
	}
}


/* copy */

namespace simage
{
	void copy(View4f32 const& src, View4f32 const& dst)
	{
		assert(verify(src, dst));

		std::array<std::function<void()>, 4> f_list = 
		{ 
			[&]() { copy_1(select_channel(src, 0), select_channel(dst, 0)); },
			[&]() { copy_1(select_channel(src, 1), select_channel(dst, 1)); },
			[&]() { copy_1(select_channel(src, 2), select_channel(dst, 2)); },
			[&]() { copy_1(select_channel(src, 3), select_channel(dst, 3)); }
		};

		execute(f_list);
	}


	void copy(View3f32 const& src, View3f32 const& dst)
	{
		assert(verify(src, dst));

		std::array<std::function<void()>, 3> f_list =
		{
			[&]() { copy_1(select_channel(src, 0), select_channel(dst, 0)); },
			[&]() { copy_1(select_channel(src, 1), select_channel(dst, 1)); },
			[&]() { copy_1(select_channel(src, 2), select_channel(dst, 2)); },
		};

		execute(f_list);
	}


	void copy(View2f32 const& src, View2f32 const& dst)
	{
		assert(verify(src, dst));

		std::array<std::function<void()>, 2> f_list =
		{
			[&]() { copy_1(select_channel(src, 0), select_channel(dst, 0)); },
			[&]() { copy_1(select_channel(src, 1), select_channel(dst, 1)); },
		};

		execute(f_list);
	}


	void copy(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		copy_1(src, dst);
	}
}
