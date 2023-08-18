/* select_channel */

namespace simage
{
	template <typename T, size_t N, typename CH>
	static View1<T> select_channel(ChannelView<T, N> const& view, CH ch)
	{
		View1<T> view1{};

		view1.matrix_width = view.channel_width_;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.matrix_data_ = view.channel_data_[id_cast(ch)];

		return view1;
	}


	View1f32 select_channel(ViewRGBAf32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewRGBf32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewHSVf32 const& view, HSV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewLCHf32 const& view, LCH channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewYUVf32 const& view, YUV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, GA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, XY channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	ViewRGBf32 select_rgb(ViewRGBAf32 const& view)
	{
		assert(verify(view));

		ViewRGBf32 rgb;

		rgb.channel_width_ = view.channel_width_;
		rgb.width = view.width;
		rgb.height = view.height;
		rgb.range = view.range;

		rgb.channel_data_[id_cast(RGB::R)] = view.channel_data_[id_cast(RGB::R)];
		rgb.channel_data_[id_cast(RGB::G)] = view.channel_data_[id_cast(RGB::G)];
		rgb.channel_data_[id_cast(RGB::B)] = view.channel_data_[id_cast(RGB::B)];

		return rgb;
	}


	template <typename T, size_t N>
	static std::array<View1<T>, N> split_channels(ChannelView<T, N> const& view)
	{
		std::array<View1<T>, N> arr = { 0 };

		for (u32 ch = 0; ch < (u32)N; ++ch)
		{
			arr[ch] = select_channel(view, ch);
		}

		return arr;
	}
}
