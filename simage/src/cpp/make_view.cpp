/* make_view static */

namespace simage
{
    template <typename T>
	static SubMatrix2D<T> do_make_view(Matrix2D<T> const& image)
	{
		SubMatrix2D<T> view;

		view.matrix_data_ = image.data_;
		view.matrix_width = image.width;

		view.width = image.width;
		view.height = image.height;

		view.range = make_range(image.width, image.height);

		return view;
	}


	template <typename T>
	static void do_make_view_1(View1<T>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		view.matrix_data_ = mb::push_elements(buffer, width * height);
		view.matrix_width = width;		
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);
	}


    template <typename T, size_t N>
	static void do_make_view_n(ChannelView<T, N>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		view.channel_width_ = width;
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.channel_data_[ch] = mb::push_elements(buffer, width * height);
		}
	}
}


/* make_view */

namespace simage
{
	View make_view(Image const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(ImageGray const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewYUV make_view(ImageYUV const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewBGR make_view(ImageBGR const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	ViewRGB make_view(ImageRGB const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);
		assert(verify(view));

		return view;
	}


	View make_view(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View view;

		do_make_view_1(view, width, height, buffer);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		ViewGray view;

		do_make_view_1(view, width, height, buffer);
		assert(verify(view));

		return view;
	}
}


/* make view */

namespace simage
{
	static MemoryBuffer<f32>& reinterpret_buffer(MemoryBuffer<Pixel>& buffer)
	{
		static_assert(sizeof(f32) == sizeof(Pixel));

		return *reinterpret_cast<MemoryBuffer<f32>*>(&buffer);
	}


	View1f32 make_view_1(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		View1f32 view;

		do_make_view_1(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View2f32 make_view_2(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View3f32 make_view_3(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View4f32 make_view_4(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4f32 view;

		do_make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}
}

