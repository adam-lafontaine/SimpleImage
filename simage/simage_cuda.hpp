#pragma once

#include "./libs/cuda/device.hpp"


/* device buffer */

namespace simage
{
	using DeviceBuffer32 = cuda::DeviceBuffer<Pixel>;
	using DeviceBuffer8 = cuda::DeviceBuffer<u8>;


	template <typename T>
	cuda::DeviceBuffer<T> create_device_buffer(u32 n_pixels)
	{
		cuda::DeviceBuffer<T> buffer;
		if (!cuda::create_device_buffer(buffer, n_pixels)) { assert(false); }
		return buffer;
	}


	template <typename T>
	cuda::DeviceBuffer<T> create_unified_buffer(u32 n_pixels)
	{
		cuda::DeviceBuffer<T> buffer;
		if (!cuda::create_unified_buffer(buffer, n_pixels)) { assert(false); }
		return buffer;
	}


	inline DeviceBuffer32 create_device_buffer32(u32 n_pixels)
	{
		return create_device_buffer<Pixel>(n_pixels);
	}


	inline DeviceBuffer8 create_device_buffer8(u32 n_pixels)
	{
		return create_device_buffer<u8>(n_pixels);
	}


	inline DeviceBuffer32 create_unified_buffer32(u32 n_pixels)
	{
		return create_unified_buffer<Pixel>(n_pixels);
	}


	inline DeviceBuffer8 create_unified_buffer8(u32 n_pixels)
	{
		return create_unified_buffer<u8>(n_pixels);
	}


	template <typename T>
	inline void destroy_buffer(cuda::DeviceBuffer<T>& buffer)
	{
		cuda::destroy_buffer(buffer);
	}
}


/* make_device_view */

namespace simage
{
	template <typename T>
	DeviceMatrix2D<T> make_device_view(u32 width, u32 height, cuda::DeviceBuffer<T>& buffer)
    {
        DeviceMatrix2D<T> view{};

        view.data_ = cuda::push_elements(buffer, width * height);
        view.width = width;
        view.height = height;

        return view;
    }
}


/* copy device */

namespace simage
{
	void copy_to_device(View const& host_src, DeviceView const& device_dst);

    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst);

	void copy_to_device(ViewYUV const& host_src, DeviceViewYUV const& device_dst);


    void copy_to_host(DeviceView const& device_src, View const& host_dst);

    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst);

	void copy_to_host(DeviceViewYUV const& device_src, ViewYUV const& host_dst);
}


/* color space conversion */

namespace simage
{
	void map_rgb_gray(DeviceView const& src, DeviceViewGray const& dst);

	void map_yuv_rgba(DeviceViewYUV const& src, DeviceView const& dst);

	void map_bgr_rgba(DeviceViewBGR const& src, DeviceView const& dst);
}


/* alpha blend */

namespace simage
{
    void alpha_blend(DeviceView const& src, DeviceView const& cur, DeviceView const& dst);
}


/* threshold */

namespace simage
{
    void threshold(DeviceViewGray const& src, DeviceViewGray const& dst, u8 min, u8 max);

    inline void threshold(DeviceViewGray const& src, DeviceViewGray const& dst, u8 min)
    {
        threshold(src, dst, min, 255);
    }
}


/* blur */

namespace simage
{
    void blur(DeviceViewGray const& src, DeviceViewGray const& dst);

    void blur(DeviceView const& src, DeviceView const& dst);
}


/* gradients */

namespace simage
{
    void gradients(DeviceViewGray const& src, DeviceViewGray const& dst);

    void gradients_xy(DeviceViewGray const& src, DeviceViewGray const& dst_x, DeviceViewGray const& dst_y);
}


/* rotate */

namespace simage
{
	void rotate(DeviceView const& src, DeviceView const& dst, Point2Du32 origin, f32 rad);

	void rotate(DeviceViewGray const& src, DeviceViewGray const& dst, Point2Du32 origin, f32 rad);
}