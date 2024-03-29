/* select_channel */

namespace simage
{
	template <typename T, size_t N, typename CH>
	static View1<T> select_channel(ChannelMatrix2D<T, N> const& view, CH ch)
	{
		View1<T> view1{};	

		view1.matrix_data = view.channel_data[id_cast(ch)];
		view1.matrix_width = view.width;

		view1.width = view.width;
		view1.height = view.height;

		view1.x_begin = 0;
		view1.x_end = view.width;
		view1.y_begin = 0;
		view1.y_end = view.height;

		return view1;
	}


	View1f32 select_channel(ViewRGBAf32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewRGBf32 const& view, RGB channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewHSVf32 const& view, HSV channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewLCHf32 const& view, LCH channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(ViewYUVf32 const& view, YUV channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, GA channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	View1f32 select_channel(View2f32 const& view, XY channel)
	{
		assert(verify(view));

		auto view1 = select_channel(view, id_cast(channel));

		assert(verify(view1));

		return view1;
	}


	ViewRGBf32 select_rgb(ViewRGBAf32 const& view)
	{
		assert(verify(view));

		ViewRGBf32 rgb;
		
		rgb.width = view.width;
		rgb.height = view.height;

		rgb.channel_data[id_cast(RGB::R)] = view.channel_data[id_cast(RGBA::R)];
		rgb.channel_data[id_cast(RGB::G)] = view.channel_data[id_cast(RGBA::G)];
		rgb.channel_data[id_cast(RGB::B)] = view.channel_data[id_cast(RGBA::B)];

		return rgb;
	}
}
