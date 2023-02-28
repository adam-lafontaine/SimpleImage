#pragma once

#include "simage_platform.hpp"
#include "../cuda/device.hpp"


/* cuda image */

namespace simage
{
    template <typename T, size_t N>
	class DeviceChannelView2D
	{
	public:

        u32 channel_width_ = 0;

		T* channel_data_[N] = {};		

		u32 width = 0;
		u32 height = 0;

        union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
	};


    template <typename T>
    class DeviceView2D
    {
    public:
        T* matrix_data_ = nullptr;
        u32 matrix_width = 0;

        u32 width = 0;
		u32 height = 0;

        union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
    }; 

    using DeviceView1r16 = DeviceView2D<cuda::r16>;

    using DeviceView4r16 = DeviceChannelView2D<cuda::r16, 4>;
	using DeviceView3r16 = DeviceChannelView2D<cuda::r16, 3>;
	using DeviceView2r16 = DeviceChannelView2D<cuda::r16, 2>;

	using DeviceView = DeviceView2D<Pixel>;    
}


/* make_view */

namespace simage
{
    using DeviceBuffer32 = DeviceBuffer<Pixel>;
    using DeviceBuffer16 = DeviceBuffer<cuda::r16>;

    
    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer);


    DeviceView1r16 make_view_1(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView2r16 make_view_2(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView3r16 make_view_3(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView4r16 make_view_4(u32 width, u32 height, DeviceBuffer16& buffer);
}


/* copy_to_device */

namespace simage
{
    void copy_to_device(View const& src, DeviceView const& dst);
}


/* copy_to_host */

namespace simage
{
    void copy_to_host(DeviceView const& src, View const& dst);
}