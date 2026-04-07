#include "cec2022_hybrid6.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace optimsolution {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

static const double kShift10[] = {
    -14.386804758751207, 70.92451068336354, 54.035479973086844, -50.249947519157985,
    59.854126935756824, 55.33809077336364, 69.75603555410916, -8.932298016237233,
    38.99757717875651, 41.91030028923895
};

static const double kShift20[] = {
    -14.386804758751207, 70.92451068336354, 54.035479973086844, -50.249947519157985,
    59.854126935756824, 55.33809077336364, 69.75603555410916, -8.932298016237233,
    38.99757717875651, 41.91030028923895, -25.5032936825898, 38.68126404596551,
    14.585345438954116, 5.229760299078078, -42.49253291380244, -31.909567445474476,
    24.43399165218416, -60.0379321929244, 45.06179969620975, -11.038922471478081
};

static const double kRotation10[] = {
    0.9521380763887576, 0.0, 0.0, 0.0,
    -0.30566825725075925, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.2429747791990189,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.4991329886096274, -0.8317628967168527, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, -1.2455855136313554, 0.0, 0.15120587573993605,
    0.0, 0.0, 0.0, 0.0,
    0.30566825725075936, 0.0, 0.0, 0.0,
    0.9521380763887576, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 1.1259317304217875, 0.0, 1.468989873961979,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.014663067582229905, 0.0,
    0.0, 1.7654792950264548, 0.0, 0.16185198181436128,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.8245783778601099, 0.5421018675012594, 0.0,
    0.0, 0.9564346253957725, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.2663395547927711,
    0.11956608589745188, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    -1.1263275184716273, 0.0, 0.0, 0.7837436917867452
};

static const double kRotation20[] = {
    0.8696110659125621, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.4937373735523928, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, -0.09123946344345418, 0.014915640022975475, 0.0,
    0.0, -0.10404615716454335, 0.0, 0.0,
    0.769098209198652, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.5578020072261625,
    -0.27923493068460725, 0.0, 0.0, 0.0,
    0.0, -0.758456758290417, 0.21755392158862025, 0.0,
    0.0, -0.5460277072263909, 0.0, 0.0,
    -0.2287445685848527, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.001498631229198244,
    -0.16413730957178507, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.5559075833266394,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    -0.6070541954972394, 0.0, 0.5528705768135658, 0.0,
    0.0, 0.0, 0.12956113546496154, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.24830371774882443, 0.0, 0.0, 0.7169276347043751,
    0.0, 0.0, 0.0, 0.0,
    0.0, -0.723264250846796, 0.0, 0.0,
    0.0, 1.4392384656583994, 0.0, 0.0,
    0.0, 0.08254451232376461, 0.08217218595188461, 0.0,
    0.0, -0.3808339659427453, 0.0, 0.0,
    0.5469741185854117, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.7009071203487228,
    0.22571696428849225, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.1436993639885749, 0.0,
    0.0, 0.06482339335515919, 0.0, 0.6432903763205894,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.3283233894431303,
    0.0, 0.0, 0.0, 0.0,
    -0.15783818734044014, 0.0, 0.0, 0.901412303480214,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.6888476701404846, 0.0, 0.0,
    0.0, 0.37277466952923627, 0.0, 0.0,
    0.0, -0.1259369573449792, -0.7470165494009303, 0.0,
    0.0, 0.04403272188006685, 0.0, 0.0,
    0.01428103200418615, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.3011069666728273,
    -0.577319602859312, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.4213531381758436, 0.0,
    0.0, -0.2369101039114026, 0.0, 0.22621735716389782,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.8525385589121888,
    -0.49373737355239516, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.869611065912564, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.26146061148056265, 0.0,
    0.0, -0.2644654862461743, 0.0, -1.7090522613692674,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.40084813432973165,
    0.0, 0.0, 0.0, -0.37102895572746364,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    -0.634793999399263, 0.0, -0.44395372727743987, 0.0,
    0.0, 0.0, 0.5121319950705527, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.4170222520846724, 0.0, 0.0, -0.9539265299941126,
    0.0, 0.0, 0.0, 0.0,
    0.0, 1.2994448648769832, 0.0, 0.0,
    0.0, 1.067713104433666, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.7243019917603731,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    -0.23773050152204572, 0.0, 0.5479592566925348, 0.0,
    0.0, 0.0, -0.3444001834828161, 0.0,
    0.0, -0.6249116775025584, -0.1595622368065382, 0.0,
    0.0, 0.5769293768983865, 0.0, 0.0,
    0.22858943616081798, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.08648086462190159,
    0.43754509677038, 0.0, 0.0, 0.0,
    0.0, -0.05683649967231136, 0.6018319651251742, 0.0,
    0.0, 0.4595728929997663, 0.0, 0.0,
    0.06727631342934012, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.31534035288668855,
    -0.5651504344606406, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    -1.319406587787655, 0.0, 0.0, -0.18843032582367136,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.29698260957676853, 0.0, 0.0,
    0.0, 0.1653450141150066, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.16938387628012164,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.4147360494362221, 0.0, 0.4438241389731618, 0.0,
    0.0, 0.0, 0.7760948688266709, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.14002933961196878, 0.0,
    0.0, 1.1162875880237086, 0.0, 0.0031913636730888234,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.26381995069219755
};

static const int kShuffle10[] = {
    3, 4, 6, 5, 1, 10, 7, 8, 2, 9
};

static const int kShuffle20[] = {
    11, 1, 5, 14, 8, 18, 4, 19, 13, 15,
    20, 7, 10, 12, 16, 17, 3, 6, 9, 2
};

inline const double* shift_data_for_dim(int dim) {
    switch (dim) {
    case 10: return kShift10;
    case 20: return kShift20;
    default: return nullptr;
    }
}

inline const double* rotation_data_for_dim(int dim) {
    switch (dim) {
    case 10: return kRotation10;
    case 20: return kRotation20;
    default: return nullptr;
    }
}

inline const int* shuffle_data_for_dim(int dim) {
    switch (dim) {
    case 10: return kShuffle10;
    case 20: return kShuffle20;
    default: return nullptr;
    }
}

inline Vec apply_shift_rotate(const Vec& x, const Vec& shift, const std::vector<double>& rotation) {
    const int D = static_cast<int>(x.size());
    Vec z(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double acc = 0.0;
        for (int j = 0; j < D; ++j) {
            acc += rotation[i * D + j] * (x[j] - shift[j]);
        }
        z[i] = acc;
    }
    return z;
}

inline Vec apply_shuffle(const Vec& z, const std::vector<int>& shuffle) {
    const int D = static_cast<int>(z.size());
    Vec y(D, 0.0);
    for (int i = 0; i < D; ++i) {
        y[i] = z[shuffle[i] - 1];
    }
    return y;
}

inline double katsuura_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    double f = 1.0;
    const double tmp3 = std::pow(static_cast<double>(n), 1.2);
    for (int i = 0; i < n; ++i) {
        const double zi = (5.0 / 100.0) * x[i];
        double temp = 0.0;
        for (int j = 1; j <= 32; ++j) {
            const double twop = std::ldexp(1.0, j);
            temp += std::fabs(twop * zi - std::floor(twop * zi + 0.5)) / twop;
        }
        f *= std::pow(1.0 + (i + 1) * temp, 10.0 / tmp3);
    }
    const double scale = 10.0 / (static_cast<double>(n) * static_cast<double>(n));
    return f * scale - scale;
}

inline double happycat_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    const double alpha = 1.0 / 8.0;
    double r2 = 0.0;
    double sumz = 0.0;
    for (int i = 0; i < n; ++i) {
        const double zi = (5.0 / 100.0) * x[i] - 1.0;
        r2 += zi * zi;
        sumz += zi;
    }
    return std::pow(std::fabs(r2 - n), 2.0 * alpha) + (0.5 * r2 + sumz) / n + 0.5;
}

inline double griewank_rosenbrock_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    Vec z(n, 0.0);
    for (int i = 0; i < n; ++i) {
        z[i] = (5.0 / 100.0) * x[i] + 1.0;
    }
    double f = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const double tmp1 = z[i] * z[i] - z[i + 1];
        const double tmp2 = z[i] - 1.0;
        const double temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        f += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
    }
    const double tmp1 = z[n - 1] * z[n - 1] - z[0];
    const double tmp2 = z[n - 1] - 1.0;
    const double temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    f += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
    return f;
}

inline double schwefel_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    double f = 0.0;
    for (int i = 0; i < n; ++i) {
        double zi = (1000.0 / 100.0) * x[i] + 4.209687462275036e+002;
        if (zi > 500.0) {
            const double m = std::fmod(zi, 500.0);
            f -= (500.0 - m) * std::sin(std::sqrt(500.0 - m));
            const double tmp = (zi - 500.0) / 100.0;
            f += (tmp * tmp) / n;
        } else if (zi < -500.0) {
            const double m = std::fmod(std::fabs(zi), 500.0);
            f -= (-500.0 + m) * std::sin(std::sqrt(500.0 - m));
            const double tmp = (zi + 500.0) / 100.0;
            f += (tmp * tmp) / n;
        } else {
            f -= zi * std::sin(std::sqrt(std::fabs(zi)));
        }
    }
    return f + 4.189828872724338e+002 * n;
}

inline double ackley_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double zi = x[i];
        sum1 += zi * zi;
        sum2 += std::cos(2.0 * kPi * zi);
    }
    sum1 = -0.2 * std::sqrt(sum1 / n);
    sum2 /= n;
    return std::exp(1.0) - 20.0 * std::exp(sum1) - std::exp(sum2) + 20.0;
}

inline double hybrid6_value(const Vec& x, const Vec& shift, const std::vector<double>& rotation, const std::vector<int>& shuffle) {
    const int D = static_cast<int>(x.size());

    Vec z = apply_shift_rotate(x, shift, rotation);
    Vec y = apply_shuffle(z, shuffle);

    const int g0 = static_cast<int>(std::ceil(0.3 * D));
    const int g1 = static_cast<int>(std::ceil(0.2 * D));
    const int g2 = static_cast<int>(std::ceil(0.2 * D));
    const int g3 = static_cast<int>(std::ceil(0.1 * D));
    const int g4 = D - g0 - g1 - g2 - g3;

    double f = 0.0;
    f += katsuura_basic(y.data(), g0);
    f += happycat_basic(y.data() + g0, g1);
    f += griewank_rosenbrock_basic(y.data() + g0 + g1, g2);
    f += schwefel_basic(y.data() + g0 + g1 + g2, g3);
    f += ackley_basic(y.data() + g0 + g1 + g2 + g3, g4);
    return f + 2200.0;
}

} // namespace

CEC2022Hybrid6::CEC2022Hybrid6()
{
    setName("cec2022hybrid6");
    setFullName("CEC 2022 F8 (reference hf06) - Hybrid Function 6");
    setModality("hybrid");
    setSeparability("non-separable");
    setCategory("CEC 2022 synthetic benchmark");
}

void CEC2022Hybrid6::init(int dim)
{
    if (!(dim == 10 || dim == 20)) {
        throw std::invalid_argument("CEC2022Hybrid6 supports only D = 10 or 20.");
    }

    Problem::init(dim);

    Vec lo(dim, -100.0), hi(dim, 100.0);
    setBounds(lo, hi);

    load_embedded_data(dim);
}

void CEC2022Hybrid6::load_embedded_data(int dim)
{
    const double* shift_ptr = shift_data_for_dim(dim);
    const double* rot_ptr   = rotation_data_for_dim(dim);
    const int* shuf_ptr     = shuffle_data_for_dim(dim);

    if (!shift_ptr || !rot_ptr || !shuf_ptr) {
        throw std::runtime_error("Embedded CEC2022Hybrid6 data are unavailable for the requested dimension.");
    }

    shift_.assign(shift_ptr, shift_ptr + dim);
    rotation_.assign(rot_ptr, rot_ptr + dim * dim);
    shuffle_.assign(shuf_ptr, shuf_ptr + dim);
}

double CEC2022Hybrid6::evaluate_core(const Vec& x)
{
    return hybrid6_value(x, shift_, rotation_, shuffle_);
}

void CEC2022Hybrid6::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    constexpr double eps = 1e-6;
    Vec xp = x, xm = x;
    for (int i = 0; i < D; ++i) {
        const double h = eps * std::max(1.0, std::fabs(x[i]));
        xp[i] = x[i] + h;
        xm[i] = x[i] - h;
        g[i] = (hybrid6_value(xp, shift_, rotation_, shuffle_) - hybrid6_value(xm, shift_, rotation_, shuffle_)) / (2.0 * h);
        xp[i] = x[i];
        xm[i] = x[i];
    }
}

} // namespace optimsolution
