#include "simage_cuda.cpp"
#include "../cuda/cuda_def.cuh"

namespace color_space
{
    template <typename T>
    class RGB
    {
    public:
        T red;
        T green;
        T blue;
    };


    class HSVu16
    {
    public:
        u16 hue;
        u16 sat;
        u16 val;
    };


    template <typename T>
    class YUV
    {
    public:
        T y;
        T u;
        T v;
    };


    using RGBf32 = RGB<f32>;
    using RGBu8 = RGB<u8>;
    using RGBu16 = RGB<u16>;

    using YUVf32 = YUV<f32>;
    using YUVu8 = YUV<u8>;
    using YUVu16 = YUV<u16>;
}


namespace cs = color_space;

namespace gpuf{ namespace color_space {}}


namespace gpucs = gpuf::color_space;


using RGBA = simage::RGBA;
using RGB = simage::RGB;
using HSV = simage::HSV;
using YUV = simage::YUV;


template <typename T>
using DeviceView2D = simage::DeviceView2D<T>;

template <typename T, size_t N>
using DeviceChannelView2D = simage::DeviceChannelView2D<T, N>;

using DeviceView1u16 = simage::DeviceView1u16;

using DeviceView4u16 = simage::DeviceView4u16;
using DeviceView3u16 = simage::DeviceView3u16;
using DeviceView2u16 = simage::DeviceView2u16;

using DeviceView = simage::DeviceView;
using DeviceViewGray = simage::DeviceViewGray;

using DeviceViewRGBAu16 = simage::DeviceView4u16;
using DeviceViewRGBu16 = simage::DeviceView3u16;
using DeviceViewHSVu16 = simage::DeviceView3u16;
using DeviceViewYUVu16 = simage::DeviceView3u16;


class ChannelXY
{
public:
	u32 ch;
	u32 x;
	u32 y;
};


constexpr u8 CH_U8_MAX = 255;
constexpr u16 CH_U16_MAX = CH_U8_MAX * 256;


constexpr int THREADS_PER_BLOCK = 512;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


namespace gpuf
{
    template <typename T>
	GPU_CONSTEXPR_FUNCTION
	inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}
}


/* conversion */

namespace gpuf
{
    
namespace color_space
{

    GPU_CONSTEXPR_FUNCTION
    inline f32 clamp(f32 value)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
        else if (value > 1.0f)
        {
            value = 1.0f;
        }

        return value;
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    inline T round_to_unsigned(f32 value)
    {
        return (T)(value + 0.5f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u32 round_to_u32(f32 value)
    {
        return gpucs::round_to_unsigned<u32>(value);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u16 round_to_u16(f32 value)
    {
        return gpucs::round_to_unsigned<u16>(value);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 round_to_u8(f32 value)
    {
        return gpucs::round_to_unsigned<u8>(value);
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 to_channel_f32(f32 value)
    {
        return clamp(value);
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 to_channel_f32(u8 value)
    {
        return (f32)value / CH_U8_MAX;
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 to_channel_f32(u16 value)
    {
        return (f32)value / CH_U16_MAX;
    }
    
    
    GPU_CONSTEXPR_FUNCTION
    inline u8 to_channel_u8(f32 value)
    {
        return gpucs::round_to_u8(gpucs::clamp(value) * CH_U8_MAX);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 to_channel_u8(u16 value)
    {
        return u8(value / 256);
    }
    
    
    GPU_CONSTEXPR_FUNCTION
    inline u16 to_channel_u16(f32 value)
    {
        return gpucs::round_to_u16(gpucs::clamp(value) * CH_U16_MAX);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u16 to_channel_u16(u8 value)
    {
        return (u16)(value * 256);
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    inline f32 to_grayscale_standard(T r, T g, T b)
    {
        constexpr f32 COEFF_R = 0.299f;
        constexpr f32 COEFF_G = 0.587f;
        constexpr f32 COEFF_B = 0.114f;

        return COEFF_R * r + COEFF_G * g + COEFF_B * b;
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    inline T rgb_unsigned_to_gray(T r, T g, T b)
    {
        auto grayT32 = gpucs::to_grayscale_standard(r, g, b);
        return gpucs::round_to_unsigned<T>(grayT32);
    }


    GPU_FUNCTION
    static cs::HSVu16 rgb_u16_to_hsv_u16(u16 r, u16 g, u16 b)
    {
        auto max = (u16)umax(r, umax(g, b));
        auto min = (u16)umin(r, umin(g, b));

        u16 h = 0;
        u16 s = 0;
        u16 v = max;

        if (max == min)
        {
            return { h, s, v };
        }

        s = gpucs::to_channel_u16((f32)(max - min) / max);

        auto const r_is_max = r == max;
        auto const r_is_min = r == min;
        auto const g_is_max = g == max;
        auto const g_is_min = g == min;
        auto const b_is_max = b == max;
        auto const b_is_min = b == min;

        constexpr u16 delta_h = CH_U16_MAX / 6;
        u16 delta_c = 0;        
        u16 h_id = 0;

        if (r_is_max && b_is_min)
        {
            h_id = 0;
            delta_c = g - min;
        }
        else if (g_is_max && b_is_min)
        {
            h_id = 1;
            delta_c = max - r;
        }
        else if (g_is_max && r_is_min)
        {
            h_id = 2;
            delta_c = b - min;
        }
        else if (b_is_max && r_is_min)
        {
            h_id = 3;
            delta_c = max - g;
        }
        else if (b_is_max && g_is_min)
        {
            h_id = 4;
            delta_c = r - min;
        }
        else
        {
            h_id = 5;
            delta_c = max - b;
        }

        h = (u16)(delta_h * (h_id + (f32)delta_c / (max - min)));

        return { h, s, v };
    }


    GPU_FUNCTION
    static cs::RGBu16 hsv_u16_to_rgb_u16(u16 h, u16 s, u16 v)
    {
        if (v == 0 || s == 0)
        {
            return { v, v, v };
        }

        auto max = v;
        auto range = (f32)s / CH_U16_MAX * v;
        auto min = gpucs::round_to_u16(max - range);

        constexpr u16 delta_h = CH_U16_MAX / 6;

        auto d = (f32)h / delta_h;
        auto h_id = (int)d;
        auto ratio = d - h_id;

        auto rise = gpucs::round_to_u16(min + ratio * range);
        auto fall = gpucs::round_to_u16(max - ratio * range);

        u16 r = 0;
        u16 g = 0;
        u16 b = 0;

        switch (h_id)
        {
        case 0:
            r = max;
            g = rise;
            b = min;
            break;
        case 1:
            r = fall;
            g = max;
            b = min;
            break;
        case 2:
            r = min;
            g = max;
            b = rise;
            break;
        case 3:
            r = min;
            g = fall;
            b = max;
            break;
        case 4:
            r = rise;
            g = min;
            b = max;
            break;
        default:
            r = max;
            g = min;
            b = fall;
            break;
        }

        return { r, g, b };
    }


    GPU_FUNCTION
    static cs::HSVu16 rgb_u8_to_hsv_u16(u8 r, u8 g, u8 b)
    {
        auto R = gpucs::to_channel_u16(r);
        auto G = gpucs::to_channel_u16(g);
        auto B = gpucs::to_channel_u16(b);

        return gpucs::rgb_u16_to_hsv_u16(R, G, B);
    }


    GPU_FUNCTION
    static simage::RGBAu8 hsv_u16_to_rgba_u8(u16 h, u16 s, u16 v)
    {
        auto rgb = gpucs::hsv_u16_to_rgb_u16(h, s, v);

        return {
            gpucs::to_channel_u8(rgb.red),
            gpucs::to_channel_u8(rgb.green),
            gpucs::to_channel_u8(rgb.blue),
            255
        };
    }


    GPU_CONSTEXPR_FUNCTION
    inline cs::YUVf32 rgb_f32_to_yuv_f32(f32 r, f32 g, f32 b)
    {
        constexpr f32 ry = 0.299f;
        constexpr f32 gy = 0.587f;
        constexpr f32 by = 0.114f;

        constexpr f32 ru = -0.14713f;
        constexpr f32 gu = -0.28886f;
        constexpr f32 bu = 0.436f;

        constexpr f32 rv = 0.615f;
        constexpr f32 gv = -0.51499f;
        constexpr f32 bv = -0.10001f;

        f32 y = (ry * r) + (gy * g) + (by * b);
        f32 u = (ru * r) + (gu * g) + (bu * b) + 0.5f;
        f32 v = (rv * r) + (gv * g) + (bv * b) + 0.5f;

        return { y, u, v };
    }


    GPU_CONSTEXPR_FUNCTION
    inline cs::RGBf32 yuv_f32_to_rgb_f32(f32 y, f32 u, f32 v)
    {
        constexpr f32 yr = 1.0f;
        constexpr f32 ur = 0.0f;
        constexpr f32 vr = 1.13983f;

        constexpr f32 yg = 1.0f;
        constexpr f32 ug = -0.39465f;
        constexpr f32 vg = -0.5806f;

        constexpr f32 yb = 1.0f;
        constexpr f32 ub = 2.03211f;
        constexpr f32 vb = 0.0f;

        u -= 0.5f;
        v -= 0.5f;

        auto R = (yr * y) + (ur * u) + (vr * v);
        auto G = (yg * y) + (ug * u) + (vg * v);
        auto B = (yb * y) + (ub * u) + (vb * v);

        return { R, G, B };
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    cs::YUVu16 rgb_to_yuv_u8(T r, T g, T b)
    {
        auto R = gpucs::to_channel_f32(r);
        auto G = gpucs::to_channel_f32(g);
        auto B = gpucs::to_channel_f32(b);

        auto yuv = gpucs::rgb_f32_to_yuv_f32(R, G, B);

        return {
            gpucs::to_channel_u8(yuv.y),
            gpucs::to_channel_u8(yuv.u),
            gpucs::to_channel_u8(yuv.v),
        };
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    cs::YUVu16 rgb_to_yuv_u16(T r, T g, T b)
    {
        auto R = gpucs::to_channel_f32(r);
        auto G = gpucs::to_channel_f32(g);
        auto B = gpucs::to_channel_f32(b);

        auto yuv = gpucs::rgb_f32_to_yuv_f32(R, G, B);

        return {
            gpucs::to_channel_u16(yuv.y),
            gpucs::to_channel_u16(yuv.u),
            gpucs::to_channel_u16(yuv.v),
        };
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    cs::RGBu8 yuv_to_rgb_u8(T y, T u, T v)
    {
        auto Y = gpucs::to_channel_f32(y);
        auto U = gpucs::to_channel_f32(u);
        auto V = gpucs::to_channel_f32(v);

        auto rgb = gpucs::yuv_f32_to_rgb_f32(Y, U, V);

        return {
            gpucs::to_channel_u8(rgb.red),
            gpucs::to_channel_u8(rgb.green),
            gpucs::to_channel_u8(rgb.blue),
        };
    }


    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    cs::RGBu16 yuv_to_rgb_u16(T y, T u, T v)
    {
        auto Y = gpucs::to_channel_f32(y);
        auto U = gpucs::to_channel_f32(u);
        auto V = gpucs::to_channel_f32(v);

        auto rgb = gpucs::yuv_f32_to_rgb_f32(Y, U, V);

        return {
            gpucs::to_channel_u16(rgb.red),
            gpucs::to_channel_u16(rgb.green),
            gpucs::to_channel_u16(rgb.blue),
        };
    }

} // color_space
} // gpuf




/* row begin */

namespace gpuf
{
    template <typename T>
    GPU_FUNCTION
	inline T* row_begin(DeviceView2D<T> const& view, u32 y)
	{
		return view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
	}


    template <typename T, size_t N>
    GPU_FUNCTION
	inline T* channel_row_begin(DeviceChannelView2D<T, N> const& view, u32 y, u32 ch)
	{
		auto offset = (size_t)((view.y_begin + y) * view.channel_width_ + view.x_begin);

		return view.channel_data_[ch] + offset;
	}
}


/* xy_at */

namespace gpuf
{
    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceView2D<T> const& view, u32 x, u32 y)
    {
        return gpuf::row_begin(view, y) + x;
    }


    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceView2D<T> const& view, Point2Du32 const& pt)
    {
        return gpuf::row_begin(view, pt.y) + pt.x;
    }


    template <typename T, size_t N, typename CH>
    GPU_FUNCTION
    inline T* channel_xy_at(DeviceChannelView2D<T, N> const& view, u32 x, u32 y, CH ch)
    {
        return channel_row_begin(view, y, gpuf::id_cast(ch)) + x;
    }


    template <typename T, size_t N, typename CH>
    GPU_FUNCTION
    inline T* channel_xy_at(DeviceChannelView2D<T, N> const& view, Point2Du32 const& pt, CH ch)
    {
        return channel_row_begin(view, pt.y, gpuf::id_cast(ch)) + pt.x;
    }


    template <typename T, size_t N>
    GPU_FUNCTION
    inline T* channel_xy_at(DeviceChannelView2D<T, N> const& view, ChannelXY const& cxy)
    {
        return channel_row_begin(view, cxy.y, cxy.ch) + cxy.x;
    }
}


/* get_thread_xy */

namespace gpuf
{
    template <class VIEW>
	GPU_FUNCTION
	static Point2Du32 get_thread_xy(VIEW const& view, u32 thread_id)
	{
		// n_threads = width * height
		Point2Du32 p{};

		p.y = thread_id / view.width;
		p.x = thread_id - p.y * view.width;

		return p;
	}


    template <class VIEW>
	GPU_FUNCTION
	static ChannelXY get_thread_channel_xy(VIEW const& view, u32 thread_id)
	{
		auto width = view.width;
		auto height = view.height;

		ChannelXY cxy{};

		cxy.ch = thread_id / (width * height);
		cxy.y = (thread_id - width * height * cxy.ch) / width;
		cxy.x = (thread_id - width * height * cxy.ch) - cxy.y * width;

		return cxy;
	}
}


/* map rgb */

namespace gpu
{  
    GPU_KERNAL    
    static void rgba_u8_to_rgba_u16(DeviceView src, DeviceViewRGBAu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto rgba = gpuf::xy_at(src, cxy.x, cxy.y)->rgba;
        u8 s = 0;
        auto& d = *gpuf::channel_xy_at(dst, cxy);

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGBA::R):
            s = rgba.red;
            break;
        case gpuf::id_cast(RGBA::G):
            s = rgba.green;
            break;
        case gpuf::id_cast(RGBA::B):
            s = rgba.blue;
            break;
        case gpuf::id_cast(RGBA::A):
            s = rgba.alpha;
            break;
        default:
            return;
        }

        d = gpucs::to_channel_u16(s);
    }


    GPU_KERNAL    
    static void rgb_u8_to_rgb_u16(DeviceView src, DeviceViewRGBu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto rgba = gpuf::xy_at(src, cxy.x, cxy.y)->rgba;
        u8 s = 0;
        auto& d = *gpuf::channel_xy_at(dst, cxy);

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGB::R):
            s = rgba.red;
            break;
        case gpuf::id_cast(RGB::G):
            s = rgba.green;
            break;
        case gpuf::id_cast(RGB::B):
            s = rgba.blue;
            break;
        default:
            return;
        }

        d = gpucs::to_channel_u16(s);
    }
    

    GPU_KERNAL
    static void rgba_u16_to_rgba_u8(DeviceViewRGBAu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto s = gpucs::to_channel_u8(*gpuf::channel_xy_at(src, cxy));
        auto& d = gpuf::xy_at(dst, cxy.x, cxy.y)->rgba;

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGBA::R):
            d.red = s;
            break;
        case gpuf::id_cast(RGBA::G):
            d.green = s;
            break;
        case gpuf::id_cast(RGBA::B):
            d.blue = s;
            break;
        case gpuf::id_cast(RGBA::A):
            d.alpha = s;
            break;
        default:
            return;
        }
    }


    GPU_KERNAL
    static void rgb_u16_to_rgb_u8(DeviceViewRGBu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto s = gpucs::to_channel_u8(*gpuf::channel_xy_at(src, cxy));
        auto& d = gpuf::xy_at(dst, cxy.x, cxy.y)->rgba;

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGB::R):
            d.red = s;
            break;
        case gpuf::id_cast(RGB::G):
            d.green = s;
            break;
        case gpuf::id_cast(RGB::B):
            d.blue = s;
            break;
        default:
            return;
        }
    }
}


namespace simage
{
    void map_rgba(DeviceView const& src, DeviceViewRGBAu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgba_u8_to_rgba_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgba_u8_to_rgba_u16");
		assert(result);
    }


	void map_rgba(DeviceViewRGBAu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgba_u16_to_rgba_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgba_u16_to_rgba_u8");
		assert(result);
    }


    void map_rgb(DeviceView const& src, DeviceViewRGBu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u8_to_rgb_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u8_to_rgb_u16");
		assert(result);
    }


	void map_rgb(DeviceViewRGBu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u16_to_rgb_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u16_to_rgb_u8");
		assert(result);
    }
}


/* map gray */

namespace gpu
{
    GPU_KERNAL
    static void gray_u8_to_gray_u16(DeviceViewGray src, DeviceView1u16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = *gpuf::xy_at(src, xy);
        auto& d = *gpuf::xy_at(dst, xy);

        d = gpucs::to_channel_u16(s);
    }


    GPU_KERNAL
    static void gray_u16_to_gray_u8(DeviceView1u16 src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = *gpuf::xy_at(src, xy);
        auto& d = *gpuf::xy_at(dst, xy);

        d = gpucs::to_channel_u8(s);
    }
}


namespace simage
{
    void map_gray(DeviceViewGray const& src, DeviceView1u16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::gray_u8_to_gray_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::gray_u8_to_gray_u16");
		assert(result);
    }


    void map_gray(DeviceView1u16 const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::gray_u16_to_gray_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::gray_u16_to_gray_u8");
		assert(result);
    }    
}


/* map rgb gray */

namespace gpu
{
    GPU_KERNAL
    static void rgba_u8_to_gray_u8(DeviceView src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy)->rgba;
        auto& d = *gpuf::xy_at(dst, xy);        

        auto gray8 = gpucs::rgb_unsigned_to_gray(s.red, s.green, s.blue);

        d = gray8;
    }


    GPU_KERNAL
    static void rgba_u8_to_gray_u16(DeviceView src, DeviceView1u16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy)->rgba;
        auto& d = *gpuf::xy_at(dst, xy);

        auto gray8 = gpucs::rgb_unsigned_to_gray(s.red, s.green, s.blue);

        d = gpucs::to_channel_u16(gray8);
    }


    GPU_KERNAL
    static void rgb_u16_to_gray_u16(DeviceViewRGBu16 src, DeviceView1u16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto r = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::R);
        auto g = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::G);
        auto b = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::B);

        auto& d = *gpuf::xy_at(dst, xy);

        auto gray16 = gpucs::rgb_unsigned_to_gray(r, g, b);

        d = gray16;
    }


    GPU_KERNAL
    static void rgb_u16_to_gray_u8(DeviceViewRGBu16 src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto r = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::R);
        auto g = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::G);
        auto b = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::B);

        auto& d = *gpuf::xy_at(dst, xy);

        auto gray16 = gpucs::rgb_unsigned_to_gray(r, g, b);

        d = gpucs::to_channel_u8(gray16);
    }


    GPU_KERNAL
    static void gray_u16_to_rgba_u8(DeviceView1u16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto gray8 = gpucs::to_channel_u8(*gpuf::xy_at(src, xy));
        auto& d = gpuf::xy_at(dst, xy)->rgba;

        d = { gray8, gray8, gray8, 255 };
    }

}


namespace simage
{
    void map_rgb_gray(DeviceView const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgba_u8_to_gray_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgba_u8_to_gray_u8");
		assert(result);
    }


    void map_rgb_gray(DeviceView const& src, DeviceView1u16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgba_u8_to_gray_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgba_u8_to_gray_u16");
		assert(result);
    }


    void map_rgb_gray(DeviceViewRGBu16 const& src, DeviceView1u16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u16_to_gray_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u16_to_gray_u16");
		assert(result);
    }


    void map_rgb_gray(DeviceViewRGBu16 const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u16_to_gray_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u16_to_gray_u8");
		assert(result);
    }


    void map_gray_rgb(DeviceView1u16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::gray_u16_to_rgba_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::gray_u16_to_rgba_u8");
		assert(result);
    }
}


/* map hsv */

namespace gpu
{
    GPU_KERNAL
    static void rgb_u8_to_hsv_u16(DeviceView src, DeviceViewHSVu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy)->rgba;

        auto hsv = gpucs::rgb_u8_to_hsv_u16(s.red, s.green, s.blue);

        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::H) = hsv.hue;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::S) = hsv.sat;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::V) = hsv.val;
    }


    GPU_KERNAL
    static void rgb_u16_to_hsv_u16(DeviceViewRGBu16 src, DeviceViewHSVu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto r = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::R);
        auto g = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::G);
        auto b = *gpuf::channel_xy_at(src, xy.x, xy.y, RGB::B);

        auto hsv = gpucs::rgb_u16_to_hsv_u16(r, g, b);

        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::H) = hsv.hue;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::S) = hsv.sat;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, HSV::V) = hsv.val;
    }


    GPU_KERNAL
    static void hsv_u16_to_rgba_u8(DeviceViewHSVu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto h = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::H);
        auto s = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::S);
        auto v = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::V);

        auto& d = gpuf::xy_at(dst, xy.x, xy.y)->rgba;

        d = gpucs::hsv_u16_to_rgba_u8(h, s, v);
    }


    GPU_KERNAL
    static void hsv_u16_to_rgb_u16(DeviceViewHSVu16 src, DeviceViewRGBu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto h = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::H);
        auto s = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::S);
        auto v = *gpuf::channel_xy_at(src, xy.x, xy.y, HSV::V);

        auto rgb = gpucs::hsv_u16_to_rgb_u16(h, s, v);

        *gpuf::channel_xy_at(dst, xy.x, xy.y, RGB::R) = rgb.red;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, RGB::G) = rgb.green;
        *gpuf::channel_xy_at(dst, xy.x, xy.y, RGB::B) = rgb.blue;
    }
}


namespace simage
{
    void map_rgb_hsv(DeviceView const& src, DeviceViewHSVu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u8_to_hsv_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u8_to_hsv_u16");
		assert(result);
    }


    void map_rgb_hsv(DeviceViewRGBu16 const& src, DeviceViewHSVu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u16_to_hsv_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u16_to_hsv_u16");
		assert(result);
    }


    void map_hsv_rgb(DeviceViewHSVu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::hsv_u16_to_rgba_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::hsv_u16_to_rgba_u8");
		assert(result);
    }


    void map_hsv_rgb(DeviceViewHSVu16 const& src, DeviceViewRGBu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::hsv_u16_to_rgb_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::hsv_u16_to_rgb_u16");
		assert(result);
    }
}


/* map yuv */

namespace gpu
{
    GPU_KERNAL
    static void rgb_u8_to_yuv_u16(DeviceView src, DeviceViewYUVu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy)->rgba;

        auto& dy = *gpuf::channel_xy_at(dst, xy, YUV::Y);
        auto& du = *gpuf::channel_xy_at(dst, xy, YUV::U);
        auto& dv = *gpuf::channel_xy_at(dst, xy, YUV::V);
        
        auto yuv = gpucs::rgb_to_yuv_u16(s.red, s.green, s.blue);

        dy = yuv.y;
        du = yuv.u;
        dv = yuv.v;
    }


    GPU_KERNAL
    static void rgb_u16_to_yuv_u16(DeviceViewRGBu16 src, DeviceViewYUVu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto r16 = *gpuf::channel_xy_at(src, xy, RGB::R);
        auto g16 = *gpuf::channel_xy_at(src, xy, RGB::G);
        auto b16 = *gpuf::channel_xy_at(src, xy, RGB::B);

        auto& y16 = *gpuf::channel_xy_at(dst, xy, YUV::Y);
        auto& u16 = *gpuf::channel_xy_at(dst, xy, YUV::U);
        auto& v16 = *gpuf::channel_xy_at(dst, xy, YUV::V);

        auto yuv = gpucs::rgb_to_yuv_u16(r16, g16, b16);

        y16 = yuv.y;
        u16 = yuv.u;
        v16 = yuv.v;
    }


    GPU_KERNAL
    static void yuv_u16_to_rgb_u8(DeviceViewYUVu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto y16 = *gpuf::channel_xy_at(src, xy, YUV::Y);
        auto u16 = *gpuf::channel_xy_at(src, xy, YUV::U);
        auto v16 = *gpuf::channel_xy_at(src, xy, YUV::V);

        auto& rgba = gpuf::xy_at(dst, xy)->rgba;

        auto rgb = gpucs::yuv_to_rgb_u8(y16, u16, v16);

        rgba.red = rgb.red;
        rgba.green = rgb.green;
        rgba.blue = rgb.blue;
        rgba.alpha = 255;
    }


    GPU_KERNAL
    static void yuv_u16_to_rgb_u16(DeviceViewYUVu16 src, DeviceViewRGBu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto y16 = *gpuf::channel_xy_at(src, xy, YUV::Y);
        auto u16 = *gpuf::channel_xy_at(src, xy, YUV::U);
        auto v16 = *gpuf::channel_xy_at(src, xy, YUV::V);

        auto& r16 = *gpuf::channel_xy_at(dst, xy, RGB::R);
        auto& g16 = *gpuf::channel_xy_at(dst, xy, RGB::G);
        auto& b16 = *gpuf::channel_xy_at(dst, xy, RGB::B);

        auto rgb = gpucs::yuv_to_rgb_u16(y16, u16, v16);

        r16 = rgb.red;
        g16 = rgb.green;
        b16 = rgb.blue;
    }
}


namespace simage
{
	void map_rgb_yuv(DeviceView const& src, DeviceViewYUVu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u8_to_yuv_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u8_to_yuv_u16");
		assert(result);
    }


	void map_rgb_yuv(DeviceViewRGBu16 const& src, DeviceViewYUVu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::rgb_u16_to_yuv_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::rgb_u16_to_yuv_u16");
		assert(result);
    }


	void map_yuv_rgb(DeviceViewYUVu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::yuv_u16_to_rgb_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::yuv_u16_to_rgb_u8");
		assert(result);
    }


	void map_yuv_rgb(DeviceViewYUVu16 const& src, DeviceViewRGBu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::yuv_u16_to_rgb_u16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::yuv_u16_to_rgb_u16");
		assert(result);
    }
}