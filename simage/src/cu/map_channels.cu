namespace gpuf
{
    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    inline f32 to_grayscale_standard(T r, T g, T b)
    {
        constexpr f32 COEFF_R = 0.299f;
        constexpr f32 COEFF_G = 0.587f;
        constexpr f32 COEFF_B = 0.114f;

        return COEFF_R * r + COEFF_G * g + COEFF_B * b;
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 pixel_to_gray(Pixel const& p)
    {
        auto gray32 = gpuf::to_grayscale_standard(p.rgba.red, p.rgba.green, p.rgba.blue);
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
}