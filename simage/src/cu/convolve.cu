GPU_CONSTEXPR_FUNCTION f32 div16(int i) { return i / 16.0f; }

GPU_GLOBAL_CONSTANT f32 GAUSS_3x3[]
{
    div16(1), div16(2), div16(1),
    div16(2), div16(4), div16(2),
    div16(1), div16(2), div16(1),
};


GPU_CONSTEXPR_FUNCTION f32 div256(int i) { return i / 256.0f; }

GPU_GLOBAL_CONSTANT f32 GAUSS_5x5[]
{
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(6), div256(24), div256(36), div256(24), div256(6),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
};


GPU_CONSTEXPR_FUNCTION f32 div140(int i) { return i / 140.0f; }

GPU_GLOBAL_CONSTANT f32 GAUSS_7x7[]
{
    div140(1), div140(1), div140(2),  div140(2), div140(2), div140(1), div140(1),
    div140(1), div140(2), div140(2),  div140(4), div140(2), div140(2), div140(1),
    div140(2), div140(2), div140(4),  div140(8), div140(4), div140(2), div140(2),
    div140(2), div140(4), div140(8), div140(16), div140(8), div140(4), div140(2),
    div140(2), div140(2), div140(4),  div140(8), div140(4), div140(2), div140(2),
    div140(1), div140(2), div140(2),  div140(4), div140(2), div140(2), div140(1),
    div140(1), div140(1), div140(2),  div140(2), div140(2), div140(1), div140(1),
};


GPU_CONSTEXPR_FUNCTION f32 div448(int i) { return i / 448.0f; }

GPU_GLOBAL_CONSTANT f32 GAUSS_9x9[]
{
    div448(1), div448(1),  div448(2),  div448(2),  div448(4),  div448(2),  div448(2), div448(1), div448(1),
    div448(1), div448(2),  div448(2),  div448(4),  div448(8),  div448(4),  div448(2), div448(2), div448(1),
    div448(2), div448(2),  div448(4),  div448(8), div448(16),  div448(8),  div448(4), div448(2), div448(2),
    div448(2), div448(4),  div448(8), div448(16), div448(32), div448(16),  div448(8), div448(4), div448(2),
    div448(4), div448(8), div448(16), div448(32), div448(64), div448(32), div448(16), div448(8), div448(4),
    div448(2), div448(4),  div448(8), div448(16), div448(32), div448(16),  div448(8), div448(4), div448(2),
    div448(2), div448(2),  div448(4),  div448(8), div448(16),  div448(8),  div448(4), div448(2), div448(2),
    div448(1), div448(2),  div448(2),  div448(4),  div448(8),  div448(4),  div448(2), div448(2), div448(1),
    div448(1), div448(1),  div448(2),  div448(2),  div448(4),  div448(2),  div448(2), div448(1), div448(1),
};


GPU_CONSTEXPR_FUNCTION f32 div548(int i) { return i / 548.0f; }

GPU_GLOBAL_CONSTANT f32 GAUSS_11x11[]
{
    div548(1), div548(1),  div548(2),  div548(2),  div548(4),  div548(2),  div548(2), div548(1), div548(1),
    div548(1), div548(2),  div548(2),  div548(4),  div548(8),  div548(4),  div548(2), div548(2), div548(1),
    div548(2), div548(2),  div548(4),  div548(8), div548(16),  div548(8),  div548(4), div548(2), div548(2),
    div548(2), div548(4),  div548(8), div548(16), div548(32), div548(16),  div548(8), div548(4), div548(2),
    div548(4), div548(8), div548(16), div548(32), div548(64), div548(32), div548(16), div548(8), div548(4),
    div548(2), div548(4),  div548(8), div548(16), div548(32), div548(16),  div548(8), div548(4), div548(2),
    div548(2), div548(2),  div548(4),  div548(8), div548(16),  div548(8),  div548(4), div548(2), div548(2),
    div548(1), div548(2),  div548(2),  div548(4),  div548(8),  div548(4),  div548(2), div548(2), div548(1),
    div548(1), div548(1),  div548(2),  div548(2),  div548(4),  div548(2),  div548(2), div548(1), div548(1),
};


GPU_GLOBAL_CONSTANT f32 GRAD_X_3x3[]
{
    -0.2f,  0.0f,  0.2f,
    -0.6f,  0.0f,  0.6f,
    -0.2f,  0.0f,  0.2f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_Y_3x3[]
{
    -0.2f, -0.6f, -0.2f,
     0.0f,  0.0f,  0.0f,
     0.2f,  0.6f,  0.2f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_X_3x5[]
{
    -0.08f, -0.12f, 0.0f, 0.12f, 0.08f
    -0.24f, -0.36f, 0.0f, 0.36f, 0.24f
    -0.08f, -0.12f, 0.0f, 0.12f, 0.08f
};


GPU_GLOBAL_CONSTANT f32 GRAD_Y_3x5[]
{
    -0.08f, -0.24f, -0.08f,
    -0.12f, -0.36f, -0.12f,
    0.00f,  0.00f,  0.00f,
    0.12f,  0.36f,  0.12f,
    0.08f,  0.24f,  0.08f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_X_3x7[]
{
    -0.04f, -0.07f, -0.09f, 0.0f, 0.09f, 0.07f, 0.04f,
    -0.12f, -0.21f, -0.27f, 0.0f, 0.27f, 0.21f, 0.12f,
    -0.04f, -0.07f, -0.09f, 0.0f, 0.09f, 0.07f, 0.04f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_Y_3x7[]
{
    -0.04f, -0.12f, -0.04f,
    -0.07f, -0.21f, -0.07f,
    -0.09f, -0.27f, -0.09f,
    0.00f,  0.00f,  0.00f,
    0.09f,  0.27f,  0.09f,
    0.07f,  0.21f,  0.07f,
    0.04f,  0.12f,  0.04f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_X_3x9[]
{
    -0.02f, -0.04f, -0.06f, -0.08f, 0.0f, 0.08f, 0.06f, 0.04f, 0.02f,
    -0.06f, -0.12f, -0.18f, -0.24f, 0.0f, 0.24f, 0.18f, 0.12f, 0.06f,
    -0.02f, -0.04f, -0.06f, -0.08f, 0.0f, 0.08f, 0.06f, 0.04f, 0.02f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_Y_3x9[]
{
    -0.02f, -0.09f, -0.02f,
    -0.04f, -0.12f, -0.04f,
    -0.06f, -0.15f, -0.06f,
    -0.08f, -0.18f, -0.08f,
    0.00f,  0.00f,  0.00f,
    0.08f,  0.18f,  0.08f,
    0.06f,  0.15f,  0.06f,
    0.04f,  0.12f,  0.04f,
    0.02f,  0.09f,  0.02f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_X_3x11[]
{
    -0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
    -0.06f, -0.09f, -0.12f, -0.15f, -0.18f, 0.0f, 0.18f, 0.15f, 0.12f, 0.09f, 0.06f,
    -0.02f, -0.03f, -0.04f, -0.05f, -0.06f, 0.0f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f,
};


GPU_GLOBAL_CONSTANT f32 GRAD_Y_3x11[]
{
    -0.02f, -0.06f, -0.02f,
    -0.03f, -0.09f, -0.03f,
    -0.04f, -0.12f, -0.04f,
    -0.05f, -0.15f, -0.05f,
    -0.06f, -0.18f, -0.06f,
    0.00f,  0.00f,  0.00f,
    0.06f,  0.18f,  0.06f,
    0.05f,  0.15f,  0.05f,
    0.04f,  0.12f,  0.04f,
    0.03f,  0.09f,  0.03f,
    0.02f,  0.06f,  0.02f,
};


namespace gpuf
{
    GPU_FUNCTION
    static u8 convolve_at_xy(DeviceViewGray const& view, u32 x, u32 y, f32* kernel, u32 k_width, u32 k_height)
    {
        f32 total = 0.0f;
        u32 w = 0;

        auto rx = x - (k_width / 2);
        auto ry = y - (k_height / 2);

        for (u32 v = 0; v < k_height; ++v)
        {
            auto s = gpuf::row_begin(view, ry + v);
            for (u32 u = 0; u < k_width; ++u)
            {
                total += s[rx + u] * kernel[w++];
            }
        }

        return gpuf::round_to_u8(total);
    }
}