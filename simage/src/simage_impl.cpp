

#ifndef SIMAGE_NO_CUDA


/* reinterperet */

namespace simage
{
	template <typename T>
	static MemoryBuffer<T>& reinterpret_device_buffer(cuda::DeviceBuffer<T>& buffer)
	{		
		static_assert(sizeof(MemoryBuffer<T>) == sizeof(cuda::DeviceBuffer<T>));

		static_assert(std::is_same<decltype(MemoryBuffer<T>::data_),      T*>::value, "MemoryBuffer.data_");
		static_assert(std::is_same<decltype(MemoryBuffer<T>::capacity_), u32>::value, "MemoryBuffer.capacity_");
		static_assert(std::is_same<decltype(MemoryBuffer<T>::size_),     u32>::value, "MemoryBuffer.size_");

		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::data_),      T*>::value, "DeviceBuffer.data_");
		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::capacity_), u32>::value, "DeviceBuffer.capacity_");
		static_assert(std::is_same<decltype(cuda::DeviceBuffer<T>::size_),     u32>::value, "DeviceBuffer.size_");

		return *reinterpret_cast<MemoryBuffer<T>*>(&buffer);
	}

/*
	template <typename T>
	static MatrixView<T> reinterpret_device_view(DeviceMatrixView<T> const& device_view)
	{
		static_assert(sizeof(MatrixView<T>) == sizeof(DeviceMatrixView<T>));

		static_assert(std::is_same<decltype(MatrixView<T>::matrix_data_),  T*>::value, "MatrixView.matrix_data_");
		static_assert(std::is_same<decltype(MatrixView<T>::matrix_width), u32>::value, "MatrixView.matrix_width");
		static_assert(std::is_same<decltype(MatrixView<T>::width),        u32>::value, "MatrixView.width");
		static_assert(std::is_same<decltype(MatrixView<T>::height),       u32>::value, "MatrixView.height");
		static_assert(std::is_same<decltype(MatrixView<T>::range), Range2Du32>::value, "MatrixView.range");

		static_assert(std::is_same<decltype(DeviceMatrixView<T>::matrix_data_),  T*>::value, "DeviceMatrixView.matrix_data_");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::matrix_width), u32>::value, "DeviceMatrixView.matrix_width");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::width),        u32>::value, "DeviceMatrixView.width");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::height),       u32>::value, "DeviceMatrixView.height");
		static_assert(std::is_same<decltype(DeviceMatrixView<T>::range), Range2Du32>::value, "DeviceMatrixView.range");

		MatrixView<T> view;

		view.matrix_data_ = device_view.matrix_data_;
		view.matrix_width = device_view.matrix_width;
		view.width = device_view.width;
		view.height = device_view.height;
		view.range = device_view.range;

		return view;
	}
*/

	template <typename T>
	static DeviceMatrixView<T> convert_host_view(MatrixView<T> const& view)
	{
		DeviceMatrixView<T> device_view;

		device_view.matrix_data_ = view.matrix_data_;
		device_view.matrix_width = view.matrix_width;
		device_view.width = view.width;
		device_view.height = view.height;
		device_view.range = view.range;

		return device_view;

	}


	template <typename T>
	static MatrixView<T> convert_device_view(DeviceMatrixView<T> const& device_view)
	{
		MatrixView<T> view;

		view.matrix_data_ = device_view.matrix_data_;
		view.matrix_width = device_view.matrix_width;
		view.width = device_view.width;
		view.height = device_view.height;
		view.range = device_view.range;

		return view;
	}
}



/* make_view */

namespace simage
{
	DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view(width, height, reinterpret_device_buffer(buffer));

		auto device_view = convert_host_view(view);
		
		assert(verify(device_view));

		return device_view;
	}


	DeviceViewGray make_view(u32 width, u32 height, DeviceBuffer8& buffer)
	{
		assert(verify(buffer, width * height));

		auto view = make_view(width, height, reinterpret_device_buffer(buffer));

		auto device_view = convert_host_view(view);
		
		assert(verify(device_view));

		return device_view;
	}
}


/* copy device */

namespace simage
{
	void copy_to_device(View const& host_src, DeviceView const& device_dst)
	{
		assert(verify(host_src, device_dst));

		auto dst = convert_device_view(device_dst);

        auto const bytes_per_row = sizeof(Pixel) * host_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(dst, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_by_row(host_src.height, row_func);
	}


    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst)
	{
		assert(verify(host_src, device_dst));

		auto dst = convert_device_view(device_dst);

        auto const bytes_per_row = sizeof(u8) * host_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(dst, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        };

        process_by_row(host_src.height, row_func);
	}


    void copy_to_host(DeviceView const& device_src, View const& host_dst)
	{		
		assert(verify(device_src, host_dst));

		auto src = convert_device_view(device_src);

        auto const bytes_per_row = sizeof(Pixel) * device_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_dst, y);
            auto d = row_begin(src, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_by_row(device_src.height, row_func);
	}


    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst)
	{
		assert(verify(device_src, host_dst));

		auto src = convert_device_view(device_src);

        auto const bytes_per_row = sizeof(u8) * device_src.width;

        auto const row_func = [&](u32 y)
        {
            auto h = row_begin(host_dst, y);
            auto d = row_begin(src, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        };

        process_by_row(device_src.height, row_func);
	}
}


/* sub_view */

namespace simage
{
	DeviceView sub_view(DeviceView const& view, Range2Du32 const& range)
	{
		auto h_view = convert_device_view(view);

		auto h_sub_view = sub_view(h_view, range);

		auto sub_view = convert_host_view(h_sub_view);

		assert(verify(sub_view));

		return sub_view;
	}


	DeviceViewGray sub_view(DeviceViewGray const& view, Range2Du32 const& range)
	{
		auto h_view = convert_device_view(view);

		auto h_sub_view = sub_view(h_view, range);

		auto sub_view = convert_host_view(h_sub_view);

		assert(verify(sub_view));

		return sub_view;
	}
}


#endif // SIMAGE_NO_CUDA