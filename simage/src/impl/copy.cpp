/* copy */

namespace simage
{	
	template <typename PIXEL>
	static void copy_row(PIXEL* s, PIXEL* d, u32 width)
	{
		for (u32 i = 0; i < width; ++i)
		{
			d[i] = s[i];
		}
	}


	template <class IMG_SRC, class IMG_DST>
	static void do_copy(IMG_SRC const& src, IMG_DST const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			copy_row(s, d, src.width);
		}
	}


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}


	void copy(ViewGray const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}
}
