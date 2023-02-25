#pragma once

#include "simage_platform.hpp"
#include "../cuda/device.hpp"


/* cuda image */

namespace simage
{
    template <typename T, size_t N>
	class DeviceChannelMatrix
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


	using DeviceImage = DeviceMatrix2D<Pixel>;
}


/* make_image */

namespace simage
{
    using DeviceBuffer32 = DeviceBuffer<Pixel>;

    
    DeviceImage make_image(u32 width, u32 height, DeviceBuffer32& buffer);
}


/* copy_to_device */

namespace simage
{
    void copy_to_device(Image const& src, DeviceImage const& dst);

    void copy_to_device(View const& src, DeviceImage const& dst);
}


/* copy_to_host */

namespace simage
{
    void copy_to_host(DeviceImage const& src, Image const& dst);

    void copy_to_host(DeviceImage const& src, View const& dst);
}