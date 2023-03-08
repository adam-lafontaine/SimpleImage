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
	using DeviceViewCHu16 = DeviceChannelView2D<u16, N>;

    using DeviceView1u16 = DeviceView2D<u16>;

    using DeviceView4u16 = DeviceViewCHu16<4>;
	using DeviceView3u16 = DeviceViewCHu16<3>;
	using DeviceView2u16 = DeviceViewCHu16<2>;

	using DeviceView = DeviceView2D<Pixel>;
    using DeviceViewGray = DeviceView2D<u8>;

    using DeviceViewRGBAu16 = DeviceView4u16;
    using DeviceViewRGBu16 = DeviceView3u16;
	using DeviceViewHSVu16 = DeviceView3u16;
}


/* make_view */

namespace simage
{
    using DeviceBuffer32 = DeviceBuffer<Pixel>;
    using DeviceBuffer16 = DeviceBuffer<u16>;
    using DeviceBuffer8 = DeviceBuffer<u8>;

    
    DeviceView make_view(u32 width, u32 height, DeviceBuffer32& buffer);

    DeviceViewGray make_view(u32 width, u32 height, DeviceBuffer8& buffer);


    DeviceView1u16 make_view_1(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView2u16 make_view_2(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView3u16 make_view_3(u32 width, u32 height, DeviceBuffer16& buffer);

    DeviceView4u16 make_view_4(u32 width, u32 height, DeviceBuffer16& buffer);
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
	DeviceView4u16 sub_view(DeviceView4u16 const& view, Range2Du32 const& range);

	DeviceView3u16 sub_view(DeviceView3u16 const& view, Range2Du32 const& range);

	DeviceView2u16 sub_view(DeviceView2u16 const& view, Range2Du32 const& range);

	DeviceView1u16 sub_view(DeviceView1u16 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	DeviceView1u16 select_channel(DeviceViewRGBAu16 const& view, RGBA channel);

	DeviceView1u16 select_channel(DeviceViewRGBu16 const& view, RGB channel);

	//DeviceView1u16 select_channel(DeviceViewHSVu16 const& view, HSV channel);

	//DeviceView1u16 select_channel(DeviceView2u16 const& view, GA channel);

	DeviceView1u16 select_channel(DeviceView2u16 const& view, XY channel);


	DeviceViewRGBu16 select_rgb(DeviceViewRGBAu16 const& view);
}


/* map gray */

namespace simage
{
    void map_gray(DeviceViewGray const& src, DeviceView1u16 const& dst);

    void map_gray(DeviceView1u16 const& src, DeviceViewGray const& dst);
}


/* map rgb */

namespace simage
{
    void map_rgba(DeviceView const& src, DeviceViewRGBAu16 const& dst);

	void map_rgba(DeviceViewRGBAu16 const& src, DeviceView const& dst);

    void map_rgb(DeviceView const& src, DeviceViewRGBu16 const& dst);

	void map_rgb(DeviceViewRGBu16 const& src, DeviceView const& dst);
}


/* map hsv */

namespace simage
{
	void map_rgb_hsv(DeviceView const& src, DeviceViewHSVu16 const& dst);

	void map_rgb_hsv(DeviceViewRGBu16 const& src, DeviceViewHSVu16 const& dst);

	void map_hsv_rgb(DeviceViewHSVu16 const& src, DeviceView const& dst);

	void map_hsv_rgb(DeviceViewHSVu16 const& src, DeviceViewRGBu16 const& dst);
}