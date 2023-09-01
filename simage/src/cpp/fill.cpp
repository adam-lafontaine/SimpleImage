/* fill span */

namespace simage
{
	template <typename T>
	static void fill_view_channel_no_simd(View1<T> const& view, T value)
	{
		auto len = view.width * view.height;
		auto d = view.data;

		for (u32 i = 0; i < len; ++i)
		{
			d[i] = value;
		}
	}


	template <typename T>
	static inline void fill_span_no_simd(T* dst, T value, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = value;
		}
	}


#ifdef SIMAGE_NO_SIMD

	template <typename T>
	static inline void fill_view_1(View1<T> const& view, T value)
	{
		auto len = src.width * src.height;
		auto d = row_begin(view, 0);

		fill_span_no_simd(d, value, len);
	}


	template <typename T>
	static inline void fill_sub_view_1(View1<T> const& view, T value)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			fill_span_no_simd(d, value, view.width);
		}
	}

#else


	static inline void fill_span(f32* dst, simd::vecf32 const& v_val, u32 len)
	{
		constexpr auto step = simd::LEN;		

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			simd::store_f32(v_val, dst + i);
		}

		i = len - step;
		simd::store_f32(v_val, dst + i);
	}


	template <typename T>
	static void fill_view_channel(View1<T> const& view, T value)
	{
		fill_view_channel_no_simd(view, value);
	}


	static void fill_view_channel(View1<f32> const& view, f32 value)
	{
		auto len = view.width * view.height;

		if (len < simd::LEN)
		{
			fill_view_channel_no_simd(view, value);
			return;
		}
		
		auto v_val = simd::load_f32_broadcast(value);	

		fill_span(view.data, v_val, len);
	}


	template <typename T>
	static void fill_sub_view_channel(SubView1<T> const& view, T value)
	{
		fill_sub_view_channel_no_simd(view, value);
	}


	static void fill_sub_view_channel(SubView1<f32> const& view, f32 value)
	{
		if (view.width < simd::LEN)
		{
			fill_sub_view_channel_no_simd(view, value);
			return;
		}
		
		auto v_val = simd::load_f32_broadcast(value);

		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			fill_span(d, v_val, view.width);
		}
	}

#endif // SIMAGE_NO_SIMD
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			fill_view_1(view, color);
		}
		else
		{
			fill_sub_view_1(view, color);
		}
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			fill_view_1(view, gray);
		}
		else
		{
			fill_sub_view_1(view, gray);
		}
	}
}


/* fill */

namespace simage
{

	void fill(View4f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_view_1(select_channel(view, RGBA::R), cs::to_channel_f32(color.rgba.red));
		fill_view_1(select_channel(view, RGBA::G), cs::to_channel_f32(color.rgba.green));
		fill_view_1(select_channel(view, RGBA::B), cs::to_channel_f32(color.rgba.blue));
		fill_view_1(select_channel(view, RGBA::A), cs::to_channel_f32(color.rgba.alpha));
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_view_1(select_channel(view, RGB::R), cs::to_channel_f32(color.rgba.red));
		fill_view_1(select_channel(view, RGB::G), cs::to_channel_f32(color.rgba.green));
		fill_view_1(select_channel(view, RGB::B), cs::to_channel_f32(color.rgba.blue));
	}


	void fill(View1f32 const& view, f32 gray)
	{
		assert(verify(view));

		if (is_1d(view))
		{
			fill_view_1(view, gray);
		}
		else
		{
			fill_sub_view_1(view, gray);
		}
	}


	void fill(View1f32 const& view, u8 gray)
	{
		assert(verify(view));

		auto gray32 = cs::to_channel_f32(gray);

		if (is_1d(view))
		{
			fill_view_1(view, gray32);
		}
		else
		{
			fill_sub_view_1(view, gray32);
		}
	}
}
