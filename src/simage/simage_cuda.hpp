#pragma once

#include "simage_platform.hpp"
#include "../cuda/device.hpp"


/* cuda image */

namespace simage
{
    template <typename T, size_t N>
	class DeviceChannelMatrix2D
	{
	public:

		T* channel_data_[N] = {};		

		u32 width = 0;
		u32 height = 0;
	};


    template <typename T>
    class DeviceMatrix2D
    {
    public:
        T* data_ = nullptr;

        u32 width = 0;
		u32 height = 0;
    }; 

    using DeviceView1r16 = DeviceMatrix2D<cuda::r16>;

    using DeviceView4r16 = DeviceChannelMatrix2D<cuda::r16, 4>;
	using DeviceView3r16 = DeviceChannelMatrix2D<cuda::r16, 3>;
	using DeviceView2r16 = DeviceChannelMatrix2D<cuda::r16, 2>;

	using DeviceView = DeviceMatrix2D<Pixel>;
}


/* make_image */

namespace simage
{
    using DeviceBuffer32 = DeviceBuffer<Pixel>;

    
    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer);
}


/* copy_to_device */

namespace simage
{
    void copy_to_device(Image const& src, DeviceView const& dst);

    void copy_to_device(View const& src, DeviceView const& dst);
}


/* copy_to_host */

namespace simage
{
    void copy_to_host(DeviceView const& src, Image const& dst);

    void copy_to_host(DeviceView const& src, View const& dst);
}