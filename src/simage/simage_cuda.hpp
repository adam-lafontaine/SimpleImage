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

    template <size_t N>
	using DeviceViewCHr16 = DeviceChannelView2D<u16, N>;

    using DeviceView1r16 = DeviceView2D<u16>;

    using DeviceView4r16 = DeviceViewCHr16<4>;
	using DeviceView3r16 = DeviceViewCHr16<3>;
	using DeviceView2r16 = DeviceViewCHr16<2>;

	using DeviceView = DeviceView2D<Pixel>;
    using DeviceViewGray = DeviceView2D<u8>;

    using DeviceViewRGBAr16 = DeviceView4r16;
    using DeviceViewRGBr16 = DeviceView3r16;
}


/* make_view */

namespace simage
{
    using DeviceBuffer32 = DeviceBuffer<Pixel>;
    using DeviceBuffer16 = DeviceBuffer<u16>;
    using DeviceBuffer8 = DeviceBuffer<u8>;

    
    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer);

    DeviceViewGray make_view(u32 width, u32 height, DeviceBuffer8& buffer);


    DeviceView1r16 make_view_1(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView2r16 make_view_2(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView3r16 make_view_3(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView4r16 make_view_4(u32 width, u32 height, DeviceBuffer16& buffer);
}


/* device copy */

namespace simage
{
    void copy_to_device(View const& src, DeviceView const& dst);

    void copy_to_device(ViewGray const& src, DeviceViewGray const& dst);

    void copy_to_host(DeviceView const& src, View const& dst);

    void copy_to_host(DeviceViewGray const& src, ViewGray const& dst);
}


/* sub_view */

namespace simage
{
	DeviceView4r16 sub_view(DeviceView4r16 const& view, Range2Du32 const& range);

	DeviceView3r16 sub_view(DeviceView3r16 const& view, Range2Du32 const& range);

	DeviceView2r16 sub_view(DeviceView2r16 const& view, Range2Du32 const& range);

	DeviceView4r16 sub_view(DeviceView4r16 const& view, Range2Du32 const& range);
}


/* map gray */

namespace simage
{
    void map_gray(DeviceViewGray const& src, DeviceView1r16 const& dst);

    void map_gray(DeviceView1r16 const& src, DeviceViewGray const& dst);
}


/* map rgb */

namespace simage
{
    void map_rgba(DeviceView const& src, DeviceViewRGBAr16 const& dst);

	void map_rgba(DeviceViewRGBAr16 const& src, DeviceView const& dst);

    void map_rgb(DeviceView const& src, DeviceViewRGBr16 const& dst);

	void map_rgb(DeviceViewRGBr16 const& src, DeviceView const& dst);
}