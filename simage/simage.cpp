#include "simage.hpp"
#include "src/util/color_space.hpp"

#include <cmath>
#include <algorithm>
#include <functional>
#include <array>
#include <vector>

namespace cs = color_space;


static constexpr inline u8 round_to_u8(f32 val)
{
    return (u8)(val + 0.5f);
}


static inline u8 abs_to_u8(f32 val)
{
    return (u8)(std::abs(val) + 0.5f);
}


static inline u8 hypot_to_u8(f32 a, f32 b)
{
    return (u8)(std::hypotf(a, b) + 0.5f);
}


void process_range(u32 id_begin, u32 id_end, std::function<void(u32)> const& id_func)
{
    assert(id_begin <= id_end);

    for (u32 i = id_begin; i < id_end; ++i)
    {
        id_func(i);
    }
}

#include "src/cpp/verify.cpp"
#include "src/cpp/channel_pixels.cpp"
#include "src/cpp/platform_image.cpp"

#include "src/cpp/simd.cpp"

#include "src/cpp/row_begin.cpp"
#include "src/cpp/select_channel.cpp"
#include "src/cpp/make_view.cpp"
#include "src/cpp/sub_view.cpp"
#include "src/cpp/fill.cpp"
#include "src/cpp/convolve.cpp"
#include "src/cpp/gradients.cpp"
#include "src/cpp/blur.cpp"
#include "src/cpp/rotate.cpp"
#include "src/cpp/split_channels.cpp"
#include "src/cpp/copy.cpp"
#include "src/cpp/map_gray.cpp"
#include "src/cpp/map_rgb.cpp"
#include "src/cpp/map_color.cpp"
#include "src/cpp/alpha_blend.cpp"
#include "src/cpp/for_each_pixel.cpp"
#include "src/cpp/transform.cpp"
#include "src/cpp/centroid.cpp"
#include "src/cpp/skeleton.cpp"
#include "src/cpp/histogram.cpp"

#include "libs/stb/stb_simage.cpp"

#ifndef SIMAGE_NO_USB_CAMERA

#ifdef _WIN32

#include "libs/opencv/opencv_simage.cpp"

#else

#include "libs/uvc/uvc_simage.cpp"

#endif // _WIN32

#endif // !SIMAGE_NO_USB_CAMERA


#ifndef SIMAGE_NO_CUDA


/* row_begin */

namespace simage
{
    template <typename T>
    static T* row_begin(DeviceMatrix2D<T> const& view, u32 y)
    {
        return view.data + (u64)(y * view.width);
    }
}


/* copy device */

namespace simage
{
    template <typename T>
    static void copy_view_to_device(View1<T> const& src, DeviceView1<T> const& dst)
    {
        auto const bytes = sizeof(T) * src.width * src.height;

        auto h_src = src.data;
        auto d_dst = dst.data;
        if(!cuda::memcpy_to_device(h_src, d_dst, bytes)) { assert(false); }
    }


    template <typename T>
    static void copy_sub_view_to_device(SubView1<T> const& src, DeviceView1<T> const& dst)
    {
        auto const bytes_per_row = sizeof(T) * src.width;

        for (u32 y = 0; y < src.height; ++y)
        {
            auto h_src = row_begin(src, y);
            auto d_dst = row_begin(dst, y);
            if(!cuda::memcpy_to_device(h_src, d_dst, bytes_per_row)) { assert(false); }
        }
    }


    template <typename T>
    static void copy_device_to_view(DeviceView1<T> const& src, View1<T> const& dst)
    {
        auto const bytes = sizeof(T) * src.width * src.height;

        auto h_dst = dst.data;
        auto d_src = src.data;
        if(!cuda::memcpy_to_host(d_src, h_dst, bytes)) { assert(false); }
    }


    template <typename T>
    static void copy_device_to_sub_view(DeviceView1<T> const& src, SubView1<T> const& dst)
    {
        auto const bytes_per_row = sizeof(T) * src.width;

        for (u32 y = 0; y < src.height; ++y)
        {
            auto h_dst = row_begin(dst, y);
            auto d_src = row_begin(src, y);
            if(!cuda::memcpy_to_host(d_src, h_dst, bytes_per_row)) { assert(false); }
        }
    }


	void copy_to_device(View const& host_src, DeviceView const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_view_to_device(host_src, device_dst);
    }


    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_view_to_device(host_src, device_dst);
    }


    void copy_to_device(ViewYUV const& host_src, DeviceViewYUV const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_view_to_device(host_src, device_dst);
    }

    void copy_to_device(SubView const& host_src, DeviceView const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_sub_view_to_device(host_src, device_dst);
    }


    void copy_to_device(SubViewGray const& host_src, DeviceViewGray const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_sub_view_to_device(host_src, device_dst);
    }


    void copy_to_device(SubViewYUV const& host_src, DeviceViewYUV const& device_dst)
    {
        assert(verify(host_src, device_dst));

        copy_sub_view_to_device(host_src, device_dst);
    }


    void copy_to_host(DeviceView const& device_src, View const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_view(device_src, host_dst);
    }


    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_view(device_src, host_dst);
    }


    void copy_to_host(DeviceViewYUV const& device_src, ViewYUV const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_view(device_src, host_dst);
    }


    void copy_to_host(DeviceView const& device_src, SubView const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_sub_view(device_src, host_dst);
    }


    void copy_to_host(DeviceViewGray const& device_src, SubViewGray const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_sub_view(device_src, host_dst);
    }


    void copy_to_host(DeviceViewYUV const& device_src, SubViewYUV const& host_dst)
    {
        assert(verify(device_src, host_dst));

        copy_device_to_sub_view(device_src, host_dst);
    }
}

#endif // SIMAGE_NO_CUDA