// per-batch trellis lambda scale, set from Python per image-bucket
// (R-lambda closed loop); TRE_LAM env still overrides (experiments)
static double g_lam_scale = 1.0;

// Exact quantization tables generated from gpuwebp/vp8_tables.py
// and setup_quant logic ported from gpuwebp/vp8_encode.py

#include "gen_tables.inc"

static void fillQt(long long* q_, long long* iq_, long long* bias,
                      long long* zt, long long* sh_,
                      int q0, int q1, int bias_dc, int bias_ac, int is_luma_ac) {
    q_[0] = q0; q_[1] = q1;
    iq_[0] = (1LL << QFIX) / q0;
    iq_[1] = (1LL << QFIX) / q1;
    bias[0] = (long long)bias_dc << (QFIX - 8);
    bias[1] = (long long)bias_ac << (QFIX - 8);
    zt[0] = ((1LL << QFIX) - 1 - bias[0]) / iq_[0];
    zt[1] = ((1LL << QFIX) - 1 - bias[1]) / iq_[1];
    int i;
    for (i = 2; i < 16; i++) {
        q_[i] = q_[1]; iq_[i] = iq_[1];
        bias[i] = bias[1]; zt[i] = zt[1];
    }
    if (is_luma_ac) {
        for (i = 0; i < 16; i++)
            sh_[i] = ((long long)FREQ_SHARPENING[i] * q_[i]) >> 11;
    } else {
        for (i = 0; i < 16; i++) sh_[i] = 0;
    }
}

static void setup_quant_exact(int quality, QuantTables* qt) {
    int q = QUALITY_TO_Q[quality > 100 ? 100 : quality];
    if (q < 0) q = 0;
    if (q > 127) q = 127;

    int dc_q = DC_TABLE[q];
    int ac_q = AC_TABLE[q];
    int y2_dc = dc_q * 2;
    int y2_ac = AC_TABLE2[q];
    int uv_dc = DC_TABLE[q > 117 ? 117 : (q > 2 ? q - 2 : 0)];

    fillQt(qt->y1q, qt->y1iq, qt->y1b, qt->y1z, qt->y1s, dc_q, ac_q, 96, 110, 1);
    fillQt(qt->y2q, qt->y2iq, qt->y2b, qt->y2z, qt->y2s, y2_dc, y2_ac, 96, 108, 0);
    fillQt(qt->uvq, qt->uviq, qt->uvb, qt->uvz, qt->uvs, uv_dc, ac_q, 110, 115, 0);

    qt->y1deq[0] = dc_q;
    for (int i = 1; i < 16; i++) qt->y1deq[i] = ac_q;
    qt->y2deq[0] = y2_dc;
    for (int i = 1; i < 16; i++) qt->y2deq[i] = y2_ac;
    qt->uvdeq[0] = uv_dc;
    for (int i = 1; i < 16; i++) qt->uvdeq[i] = ac_q;

    int qstep = ac_q >> 2; if (qstep > 63) qstep = 63;
    int level = FILTER_LEVEL_SHARP0[qstep] * 300 / 356;
    qt->filter_level = (level < 2) ? 0 : (level > 63 ? 63 : level);

    long long sum = 0;
    for (int i = 0; i < 16; i++) sum += qt->y1q[i];
    qt->y1_qavg = (sum + 8) >> 4;
    long long sum2 = 0;
    for (int i = 0; i < 16; i++) sum2 += qt->y2q[i];
    long long q2avg = (sum2 + 8) >> 4;
    // libwebp SetupMatrices lambdas (quant_enc.c) for trellis RDO;
    // TRE_LAM scales both for RD-curve experiments / per-content tuning
    // (optimal lambda = |dD/dR| at the operating point, Shoham-Gersho /
    // HEVC R-lambda model — the q^2 form is a corpus-average fit)
    double _ls = g_lam_scale;
    const char* _le = getenv("TRE_LAM");
    if (_le) _ls = atof(_le);
    if (_ls <= 0) _ls = 1.0;
    qt->lam_i4 = (long long)((7.0 * qt->y1_qavg * qt->y1_qavg / 8.0) * _ls);
    if (qt->lam_i4 < 1) qt->lam_i4 = 1;
    qt->lam_i16 = (long long)((double)(q2avg * q2avg >> 2) * _ls);
    if (qt->lam_i16 < 1) qt->lam_i16 = 1;
}
