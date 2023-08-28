/* verify */

namespace simage
{
#ifndef NDEBUG

	template <typename T>
	static bool verify(Matrix2D<T> const& image)
	{
		return image.width && image.height && image.data_;
	}


	template <typename T>
	static bool verify(MatrixView2D<T> const& image)
	{
		return image.width && image.height && image.data;
	}


	template <typename T>
	static bool verify(SubMatrixView2D<T> const& view)
	{
		return 
			view.matrix_width && 
			view.width && 
			view.height && 
			view.matrix_data_ &&
			(view.x_end - view.x_begin) == view.width &&
			(view.y_end - view.y_begin) == view.height;
	}


	template <typename T, size_t N>
	static bool verify(ChannelMatrix2D<T,N> const& view)
	{
		return view.width && view.height && view.channel_data[0];
	}


	/*template <typename T, size_t N>
	static bool verify(ChannelSubMatrix2D<T,N> const& view)
	{
		return 
			view.channel_width_ && 
			view.width && view.height && 
			view.channel_data_[0] &&
			(view.x_end - view.x_begin) == view.width &&
			(view.y_end - view.y_begin) == view.height;
	}*/


	template <typename T>
	static bool verify(MemoryBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}


	template <class IMG>
	static bool verify(IMG const& image, Range2Du32 const& range)
	{
		return
			verify(image) &&
			range.x_begin < range.x_end &&
			range.y_begin < range.y_end &&
			range.x_begin < image.width &&
			range.x_end <= image.width &&
			range.y_begin < image.height &&
			range.y_end <= image.height;
	}


#ifndef SIMAGE_NO_CUDA

	template <typename T>
	static bool verify(cuda::DeviceBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}


	template <typename T>
	static bool verify(DeviceMatrix2D<T> const& view)
	{
		return view.width && view.height && view.data_;
	}


#endif // SIMAGE_NO_CUDA

#ifndef SIMAGE_NO_USB_CAMERA

	static bool verify(CameraUSB const& camera)
	{
		return camera.frame_width && camera.frame_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView2D<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.frame_width == view.width &&
			camera.frame_height == view.height;
	}


#endif // SIMAGE_NO_USB_CAMERA


	template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}


#endif // !NDEBUG
}
