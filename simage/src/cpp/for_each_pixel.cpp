/* for_each_pixel */

namespace simage
{
	template <typename T>
	static void for_each_pixel_view(View1<T> const& view, std::function<void(T&)> const& func)
	{
		auto len = view.width * view.height;

		for (u32 i = 0; i < len; ++i)
		{
			func(view.data[i]);
		}
	}


	template <typename T>
	static void for_each_pixel_sub_view(SubView1<T> const& view, std::function<void(T&)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(s[x]);
			}
		}
	}	
}


namespace simage
{
	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func)
	{
		assert(verify(view));

		for_each_pixel_view(view, func);
	}


	void for_each_pixel(ViewGray const& view, std::function<void(u8&)> const& func)
	{
		assert(verify(view));

		for_each_pixel_view(view, func);
	}


	void for_each_pixel(SubView const& view, std::function<void(Pixel&)> const& func)
	{
		assert(verify(view));

		for_each_pixel_sub_view(view, func);
	}


	void for_each_pixel(SubViewGray const& view, std::function<void(u8&)> const& func)
	{
		assert(verify(view));

		for_each_pixel_sub_view(view, func);
	}


	void for_each_pixel(View1f32 const& view, std::function<void(f32&)> const& func)
	{
		assert(verify(view));

		for_each_pixel_view(view, func);
	}
}
