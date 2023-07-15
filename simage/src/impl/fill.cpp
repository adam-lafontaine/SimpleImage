/* fill static */

namespace simage
{
    template <typename T>
	static void fill_channel_row(T* d, T value, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			d[i] = value;
		}
	}


	template <typename T>
	static void fill_channel(View1<T> const& view, T value)
	{		
		assert(verify(view));

		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			fill_channel_row(d, value, view.width);
		}
	}


	template <size_t N>
	static void fill_n_channels(ChannelView<f32, N> const& view, Pixel color)
	{		
		
		for (u32 ch = 0; ch < N; ++ch)
		{
			auto const c = cs::to_channel_f32(color.channels[ch]);

			auto channel = select_channel(view, ch);
			fill_channel(channel, c);

			/*for (u32 y = 0; y < view.height; ++y)
			{
				auto d = channel_row_begin(view, y, ch);
				fill_channel_row(d, c, view.width);
			}*/
		}
	}
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		fill_channel(view, color);
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}
}


/* fill */

namespace simage
{

	void fill(View4f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1f32 const& view, f32 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}


	void fill(View1f32 const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, cs::to_channel_f32(gray));
	}
}
