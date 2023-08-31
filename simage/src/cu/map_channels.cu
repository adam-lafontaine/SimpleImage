namespace gpuf
{    
    GPU_CONSTEXPR_FUNCTION
    inline f32 rgb_to_yuv_y(u8 r, u8 g, u8 b)
    {
        constexpr f32 COEFF_R = 0.299f;
        constexpr f32 COEFF_G = 0.587f;
        constexpr f32 COEFF_B = 0.114f;

        return COEFF_R * r + COEFF_G * g + COEFF_B * b;
    }
    

    GPU_CONSTEXPR_FUNCTION
    inline f32 rgb_to_yuv_u(u8 r, u8 g, u8 b)
    {
        constexpr f32 COEFF_R = -0.14713f;
        constexpr f32 COEFF_G = -0.28886f;
        constexpr f32 COEFF_B =  0.436f;

        return COEFF_R * r + COEFF_G * g + COEFF_B * b + 127.0f;
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 rgb_to_yuv_v(u8 r, u8 g, u8 b)
    {
        constexpr f32 COEFF_R = 0.615f;
        constexpr f32 COEFF_G = -0.51499f;
        constexpr f32 COEFF_B =  -0.10001f;

        return COEFF_R * r + COEFF_G * g + COEFF_B * b + 127.0f;
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 yuv_to_rgb_r(u8 y, u8 u, u8 v)
    {
        constexpr f32 COEFF_Y = 1.0f;
        constexpr f32 COEFF_U = 0.0f;
        constexpr f32 COEFF_V =  1.13983f;

        return COEFF_Y * y + COEFF_U * (u - 127.0f) + COEFF_V * (v - 127.0f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 yuv_to_rgb_g(u8 y, u8 u, u8 v)
    {
        constexpr f32 COEFF_Y = 1.0f;
        constexpr f32 COEFF_U = -0.39465f;
        constexpr f32 COEFF_V =  -0.5806f;

        return COEFF_Y * y + COEFF_U * (u - 127.0f) + COEFF_V * (v - 127.0f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline f32 yuv_to_rgb_b(u8 y, u8 u, u8 v)
    {
        constexpr f32 COEFF_Y = 1.0f;
        constexpr f32 COEFF_U = 2.03211f;
        constexpr f32 COEFF_V =  0.0f;

        return COEFF_Y * y + COEFF_U * (u - 127.0f) + COEFF_V * (v - 127.0f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 pixel_to_gray(Pixel const& p)
    {
        auto gray32 = gpuf::rgb_to_yuv_y(p.rgba.red, p.rgba.green, p.rgba.blue);
        return gpuf::round_to_u8(gray32);
    }



}


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

        auto s = *gpuf::xy_at(src, xy);
        auto& d = *gpuf::xy_at(dst, xy);

        d = gpuf::pixel_to_gray(s);
    }


    GPU_KERNAL
    static void yuv_u8_to_rgba_u8(DeviceViewYUV src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == dst.width * dst.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(dst, t);
        auto x = cxy.x;
        auto y = cxy.y;
        auto dst_ch = (RGBA)cxy.ch;

        auto& dst_rgba = gpuf::xy_at(dst, x, y)->rgba;

        if (dst_ch == RGBA::A)
        {
            dst_rgba.alpha = 255;
            return;
        }
        
        auto src_yuyv_x = (x >> 1) << 1;        

        auto src_yuyv = *(img::YUYVu8*)gpuf::xy_at(src, src_yuyv_x, y);

        auto yuv_u = src_yuyv.u;
        auto yuv_v = src_yuyv.v;
        
        auto yuv_y = gpuf::xy_at(src, x, y)->y;

        switch(dst_ch)
        {
        case RGBA::R:
            dst_rgba.red = gpuf::round_to_u8(gpuf::yuv_to_rgb_r(yuv_y, yuv_u, yuv_v));
            break;
        case RGBA::G:
            dst_rgba.green = gpuf::round_to_u8(gpuf::yuv_to_rgb_g(yuv_y, yuv_u, yuv_v));
            break;
        case RGBA::B:
            dst_rgba.blue = gpuf::round_to_u8(gpuf::yuv_to_rgb_b(yuv_y, yuv_u, yuv_v));
            break;
        default:
            break;
        }
    }


    GPU_KERNAL
    static void bgr_u8_to_rgba_u8(DeviceViewBGR src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == dst.width * dst.height * 4);

        auto dst_cxy = gpuf::get_thread_channel_xy(dst, t);
        auto dst_x = dst_cxy.x;
        auto dst_y = dst_cxy.y;
        auto dst_ch = (RGBA)dst_cxy.ch;

        auto& dst_rgba = gpuf::xy_at(dst, dst_x, dst_y)->rgba;

        if (dst_ch == RGBA::A)
        {
            dst_rgba.alpha = 255;
            return;
        }

        auto src_x = dst_cxy.x;
        auto src_y = dst_cxy.y;

        auto src_bgr = *gpuf::xy_at(src, src_x, src_y);

        switch(dst_ch)
        {
        case RGBA::R:
            dst_rgba.red = src_bgr.red;
            break;
        case RGBA::G:
            dst_rgba.green = src_bgr.green;
            break;
        case RGBA::B:
            dst_rgba.blue = src_bgr.blue;
            break;
        default:
            break;
        }
    }
}



/* color space conversion */

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


    void map_yuv_rgba(DeviceViewYUV const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::yuv_u8_to_rgba_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::yuv_u8_to_rgba_u8");
		assert(result);
    }


    void map_bgr_rgba(DeviceViewBGR const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::bgr_u8_to_rgba_u8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::bgr_u8_to_rgba_u8");
		assert(result);
    }
}