namespace gpuf
{
    GPU_FUNCTION
    static u8 blend_linear_u8(u8 s, u8 c, u8 a)
    {
        auto const a32 = a / 255.0f;
        
        return gpuf::round_to_u8(a32 * s + (1.0f - a32) * c);
    }
}


namespace gpu
{
    GPU_KERNAL
    static void alpha_blend(DeviceView src, DeviceView cur, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height * 3);

		auto cxy = gpuf::get_thread_channel_xy(dst, t);
        auto x = cxy.x;
        auto y = cxy.y;
        auto dst_ch = (RGBA)cxy.ch;

        assert(dst_ch != RGBA::A);

        auto& d = gpuf::xy_at(dst, x, y)->rgba;
        auto s = gpuf::xy_at(src, x, y)->rgba;
        auto c = gpuf::xy_at(cur, x, y)->rgba;

        switch(dst_ch)
        {
        case RGBA::R:
            d.red = gpuf::blend_linear_u8(s.red, c.red, s.alpha);
            break;
        case RGBA::G:
            d.green = gpuf::blend_linear_u8(s.green, c.green, s.alpha);
            break;
        case RGBA::B:
            d.blue = gpuf::blend_linear_u8(s.blue, c.blue, s.alpha);
            break;
        }
    }
}

namespace simage
{
    void alpha_blend(DeviceView const& src, DeviceView const& cur, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::alpha_blend, n_blocks, block_size, src, cur, dst, n_threads);

        auto result = cuda::launch_success("gpu::alpha_blend");
		assert(result);
    }
}