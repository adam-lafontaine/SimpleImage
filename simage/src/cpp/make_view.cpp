/* make_view static */

namespace simage
{
    template <typename T>
	static View1<T> make_view_from_image(Matrix2D<T> const& image)
	{
		View1<T> view{};

		view.matrix_data = image.data_;
		view.matrix_width = image.width;
		view.width = image.width;
		view.height = image.height;

		view.range = make_range(image.width, image.height);

		return view;
	}


	template <typename T>
	static View1<T> make_view_from_buffer(u32 width, u32 height, MemoryBuffer<T>& buffer)
	{
		View1<T> view{};

		view.matrix_data = mb::push_elements(buffer, width * height);
		view.matrix_width = width;
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

		return view;
	}


	template <typename T>
	View1<T> make_view_from_buffer8(u32 width, u32 height, Buffer8& buffer)
	{
		View1<T> view;

		view.matrix_data = (T*)mb::push_elements(buffer, width * height * (u32)sizeof(T));
		view.matrix_width = width;
		view.width = width;
		view.height = height;

		view.range = make_range(width, height);

		return view;
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

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(ImageGray const& image)
	{
		assert(verify(image));

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	ViewRGB make_view(ImageRGB const& image)
	{
		assert(verify(image));

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	ViewBGR make_view(ImageBGR const& image)
	{
		assert(verify(image));

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	ViewYUV make_view(ImageYUV const& image)
	{
		assert(verify(image));

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	ViewUVY make_view(ImageUVY const& image)
	{
		assert(verify(image));

		auto view = make_view_from_image(image);
		assert(verify(view));

		return view;
	}


	View make_view(u32 width, u32 height, Buffer32& buffer)
	{
		assert(verify(buffer, width * height));

		auto view  = make_view_from_buffer(width, height, buffer);
		assert(verify(view));

		return view;
	}


	ViewGray make_view(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view  = make_view_from_buffer(width, height, buffer);
		assert(verify(view));

		return view;
	}


	ViewRGB make_view_rgb(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view_from_buffer8<RGBu8>(width, height, buffer);

		assert(verify(view));

		return view;
	}


	ViewBGR make_view_bgr(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view_from_buffer8<BGRu8>(width, height, buffer);

		assert(verify(view));

		return view;
	}


	ViewYUV make_view_yuv(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view_from_buffer8<YUV2u8>(width, height, buffer);

		assert(verify(view));

		return view;
	}


	ViewUVY make_view_uvy(u32 width, u32 height, Buffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view_from_buffer8<UVY2u8>(width, height, buffer);

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

		auto view = make_view_from_buffer(width, height, reinterpret_buffer(buffer));

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

