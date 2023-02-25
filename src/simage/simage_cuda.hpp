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

		T* channel_data[N] = {};		

		u32 width = 0;
		u32 height = 0;
	};


    template <typename T>
    class DeviceMatrix2D
    {
    public:
        T* data = nullptr;

        u32 width = 0;
		u32 height = 0;
    };


	using DeviceImage = DeviceMatrix2D<Pixel>;
}


namespace simage
{
    
}