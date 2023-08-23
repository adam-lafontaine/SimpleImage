/* sub_view static */

namespace simage
{
    template <typename T>
	static SubMatrix2D<T> do_sub_view(Matrix2D<T> const& image, Range2Du32 const& range)
	{
		SubMatrix2D<T> sub_view;

		sub_view.matrix_data_ = image.data_;
		sub_view.matrix_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
	}


	template <typename T>
	static SubMatrix2D<T> do_sub_view(SubMatrix2D<T> const& view, Range2Du32 const& range)
	{
		SubMatrix2D<T> sub_view;

		sub_view.matrix_data_ = view.matrix_data_;
		sub_view.matrix_width = view.matrix_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
	}


    template <typename T, size_t N>
	static ChannelView<T, N> do_sub_view(ChannelView<T, N> const& view, Range2Du32 const& range)
	{
		ChannelView<T, N> sub_view;

		sub_view.channel_width_ = view.channel_width_;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			sub_view.channel_data_[ch] = view.channel_data_[ch];
		}

		return sub_view;
	}
}


/* sub_view */

namespace simage
{
	View sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ImageGray const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View sub_view(View const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewGray sub_view(ViewGray const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto width = range.x_end - range.x_begin;
		Range2Du32 camera_range = range;
		camera_range.x_end = camera_range.x_begin + width / 2;

		assert(verify(camera_src, camera_range));

		auto sub_view = do_sub_view(camera_src, camera_range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewBGR sub_view(ImageBGR const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewBGR sub_view(ViewBGR const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewRGB sub_view(ImageRGB const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	ViewRGB sub_view(ViewRGB const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}
}


/* sub_view */

namespace simage
{
	View4f32 sub_view(View4f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3f32 sub_view(View3f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View2f32 sub_view(View2f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View1f32 sub_view(View1f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}
}


