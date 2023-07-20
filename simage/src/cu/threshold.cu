namespace gpu
{
    GPU_KERNAL
	static void threshold(DeviceViewGray src, DeviceViewGray dst, u8 min, u8 max, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = *gpuf::xy_at(src, xy.x, xy.y);
		auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

		d = s >= min && s <= max ? s : 0;
	}
}


namespace simage
{
    void threshold(DeviceViewGray const& src, DeviceViewGray const& dst, u8 min, u8 max)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::threshold, n_blocks, block_size, src, dst, min, max, n_threads);

		auto result = cuda::launch_success("gpu::threshold");
		assert(result);
    }
}