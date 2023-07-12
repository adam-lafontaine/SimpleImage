/* for_each_pixel */

namespace simage
{
	template <typename T>
	static void do_for_each_pixel(View1<T> const& view, std::function<void(T&)> const& func)
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


	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
	}


	void for_each_pixel(ViewGray const& view, std::function<void(u8&)> const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
	}
}
