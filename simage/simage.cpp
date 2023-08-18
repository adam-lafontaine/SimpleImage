#include "simage.hpp"
#include "src/util/execute.cpp"
#include "src/util/color_space.hpp"

#include <cmath>

namespace cs = color_space;


static inline u8 round_to_u8(f32 val)
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
#include "src/cpp/map_channels.cpp"
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
        return view.data_ + (u64)(y * view.width);
    }
}


/* copy device */

namespace simage
{
    template <typename T>
    static void do_copy_to_device(MatrixView<T> const& host_src, DeviceMatrix2D<T> const& device_dst)
    {
        if (host_src.width == host_src.matrix_width)
        {
            auto bytes = sizeof(T) * host_src.width * host_src.height;
            auto h = row_begin(host_src, 0);
            auto d = device_dst.data_;
            if(!cuda::memcpy_to_device(h, d, bytes)) { assert(false); }

            return;
        }

        auto const bytes_per_row = sizeof(T) * host_src.width;

        for (u32 y = 0; y < host_src.height; ++y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(device_dst, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        }
    }


    template <typename T>
    static void do_copy_to_host(DeviceMatrix2D<T> const& device_src, MatrixView<T> const& host_dst)
    {
        if (host_dst.width == host_dst.matrix_width)
        {
            auto bytes = sizeof(T) * host_dst.width * host_dst.height;
            auto d = device_src.data_;
            auto h = row_begin(host_dst, 0);
            if(!cuda::memcpy_to_host(d, h, bytes)) { assert(false); }

            return;
        }
        
        auto const bytes_per_row = sizeof(T) * device_src.width;

        for (u32 y = 0; y < host_dst.height; ++y)
        {
            auto d = row_begin(device_src, y);
            auto h = row_begin(host_dst, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        }
    }


	void copy_to_device(View const& host_src, DeviceView const& device_dst)
    {
        assert(verify(host_src, device_dst));

        do_copy_to_device(host_src, device_dst);
    }


    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst)
    {
        assert(verify(host_src, device_dst));

        do_copy_to_device(host_src, device_dst);
    }


    void copy_to_device(ViewYUV const& host_src, DeviceViewYUV const& device_dst)
    {
        assert(verify(host_src, device_dst));

        do_copy_to_device(host_src, device_dst);
    }


    void copy_to_host(DeviceView const& device_src, View const& host_dst)
    {
        assert(verify(device_src, host_dst));

        do_copy_to_host(device_src, host_dst);
    }


    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst)
    {
        assert(verify(device_src, host_dst));

        do_copy_to_host(device_src, host_dst);
    }


    void copy_to_host(DeviceViewYUV const& device_src, ViewYUV const& host_dst)
    {
        assert(verify(device_src, host_dst));

        do_copy_to_host(device_src, host_dst);
    }
}

#endif // SIMAGE_NO_CUDA