/* sub_view static */

namespace simage
{
    template <typename T>
	static SubMatrixView2D<T> do_sub_view(Matrix2D<T> const& image, Range2Du32 const& range)
	{
		SubMatrixView2D<T> sub_view;

		sub_view.matrix_data_ = image.data_;
		sub_view.matrix_width = image.width;
		
		sub_view.range = range;

		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
	}


	template <typename T>
	static SubMatrixView2D<T> do_sub_view(MatrixView2D<T> const& view, Range2Du32 const& range)
	{
		SubMatrixView2D<T> sub_view;

		sub_view.matrix_data_ = view.data;
		sub_view.matrix_width = view.width;
		
		sub_view.range = range;

		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		return sub_view;
	}


	template <typename T>
	static SubMatrixView2D<T> do_sub_view(SubMatrixView2D<T> const& view, Range2Du32 const& range)
	{
		SubMatrixView2D<T> sub_view;

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
	static ChannelSubMatrix2D<T, N> do_sub_view(ChannelMatrix2D<T, N> const& view, Range2Du32 const& range)
	{
		ChannelSubMatrix2D<T, N> sub_view;

		sub_view.channel_width_ = view.channel_width_;

		sub_view.range = range;
		
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			sub_view.channel_data_[ch] = view.channel_data_[ch];
		}

		return sub_view;
	}


    template <typename T, size_t N>
	static ChannelSubMatrix2D<T, N> do_sub_view(ChannelSubMatrix2D<T, N> const& view, Range2Du32 const& range)
	{
		ChannelSubMatrix2D<T, N> sub_view;

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
	SubView sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView sub_view(View const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView sub_view(SubView const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewGray sub_view(ImageGray const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto sub_view = do_sub_view(image, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewGray sub_view(ViewGray const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewGray sub_view(SubViewGray const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewYUV sub_view(ViewYUV const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewYUV sub_view(SubViewYUV const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewBGR sub_view(ImageBGR const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewBGR sub_view(ViewBGR const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewBGR sub_view(SubViewBGR const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewRGB sub_view(ImageRGB const& camera_src, Range2Du32 const& range)
	{
		assert(verify(camera_src, range));

		auto sub_view = do_sub_view(camera_src, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewRGB sub_view(ViewRGB const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubViewRGB sub_view(SubViewRGB const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}
}


/* sub_view */

namespace simage
{
	SubView4f32 sub_view(View4f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView3f32 sub_view(View3f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView2f32 sub_view(View2f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView1f32 sub_view(View1f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView4f32 sub_view(SubView4f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView3f32 sub_view(SubView3f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView2f32 sub_view(SubView2f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	SubView1f32 sub_view(SubView1f32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}
}


