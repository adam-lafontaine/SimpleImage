/* make_view static */

namespace simage
{
    template <typename T>
	static View1<T> do_make_view_1(Matrix2D<T> const& image)
	{
		View1<T> view{};

		view.matrix_data = image.data;
		view.matrix_width = image.width;
		view.width = image.width;
		view.height = image.height;

		view.x_begin = 0;
		view.x_end = image.width;
		view.y_begin = 0;
		view.y_end = image.height;
	}


	template <typename T>
	static View1<T> do_make_view_1(u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		View1<T> view{};

		view.matrix_data = mb::push_elements(buffer, width * height);
		view.width = width;
		view.height = height;

		view.x_begin = 0;
		view.x_end = image.width;
		view.y_begin = 0;
		view.y_end = image.height;
	}


    template <typename T, size_t N>
	static inline void make_view_n(ChannelMatrix2D<T, N>& view, u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		view.width = width;
		view.height = height;

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.channel_data[ch] = mb::push_elements(buffer, width * height);
		}
	}
}


/* make_view */

namespace simage
{
	View make_view(Image const& image)
	{
		assert(verify(image));

		auto view = do_make_view_1(image);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(ImageGray const& image)
	{
		assert(verify(image));

		auto view = do_make_view_1(image);
		assert(verify(view));

		return view;
	}


	ViewYUV make_view(ImageYUV const& image)
	{
		assert(verify(image));

		auto view = do_make_view_1(image);
		assert(verify(view));

		return view;
	}


	ViewBGR make_view(ImageBGR const& image)
	{
		assert(verify(image));

		auto view = do_make_view_1(image);
		assert(verify(view));

		return view;
	}


	ViewRGB make_view(ImageRGB const& image)
	{
		assert(verify(image));

		auto view = do_make_view_1(image);
		assert(verify(view));

		return view;
	}


	View make_view(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		auto view  = do_make_view_1(width, height, buffer);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view  = do_make_view_1(width, height, buffer);
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

		auto view = do_make_view_1(width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View2f32 make_view_2(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 2));

		View2f32 view;

		make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View3f32 make_view_3(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 3));

		View3f32 view;

		make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}


	View4f32 make_view_4(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height * 4));

		View4f32 view;

		make_view_n(view, width, height, reinterpret_buffer(buffer));

		assert(verify(view));

		return view;
	}
}

