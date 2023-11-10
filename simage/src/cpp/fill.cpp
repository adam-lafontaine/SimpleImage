/* fill span */

namespace simage
{
	template <typename T>
	static void fill_view_channel(View1<T> const& view, T value)
	{
		auto len = view.width * view.height;
		auto d = view.data;

		for (u32 i = 0; i < len; ++i)
		{
			d[i] = value;
		}
	}


	template <typename T>
	static inline void fill_span(T* dst, T value, u32 len)
	{
		for (u32 i = 0; i < len; ++i)
		{
			dst[i] = value;
		}
	}
	

	template <typename T>
	static inline void fill_view_1(View1<T> const& view, T value)
	{
		auto len = view.width * view.height;
		auto d = row_begin(view, 0);

		fill_span(d, value, len);
	}


	template <typename T>
	static inline void fill_sub_view_1(View1<T> const& view, T value)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			fill_span(d, value, view.width);
		}
	}
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

		fill_view_1(select_channel(view, RGBA::R), cs::u8_to_channel_f32(color.rgba.red));
		fill_view_1(select_channel(view, RGBA::G), cs::u8_to_channel_f32(color.rgba.green));
		fill_view_1(select_channel(view, RGBA::B), cs::u8_to_channel_f32(color.rgba.blue));
		fill_view_1(select_channel(view, RGBA::A), cs::u8_to_channel_f32(color.rgba.alpha));
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_view_1(select_channel(view, RGB::R), cs::u8_to_channel_f32(color.rgba.red));
		fill_view_1(select_channel(view, RGB::G), cs::u8_to_channel_f32(color.rgba.green));
		fill_view_1(select_channel(view, RGB::B), cs::u8_to_channel_f32(color.rgba.blue));
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

		auto gray32 = cs::u8_to_channel_f32(gray);

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
