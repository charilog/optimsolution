#include "cec2022_hybrid10.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace optimsolution {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

static const double kShift10[] = {
    -30.590574643849664, -55.788039181282763, -29.574519819381891, 61.04539564859374,
    69.964665914869244, 56.140031207418076, 25.504503075734007, 14.183193861592187,
    -23.060768220136296, -62.331475830791277,
};

static const double kShift20[] = {
    -30.590574643849664, -55.788039181282763, -29.574519819381891, 61.04539564859374,
    69.964665914869244, 56.140031207418076, 25.504503075734007, 14.183193861592187,
    -23.060768220136296, -62.331475830791277, 74.885711891057781, 44.159753007888668,
    -47.031196414973209, -53.318146454784156, 41.91253444268952, -56.033621717260047,
    2.9197980104572139, -46.954838245620408, -49.96639414223209, 54.813651505305273,
};

static const double kRotation10[] = {
    0.94554012217453653, 0, 0, 0.32550557193105328,
    0, 0, 0, 0,
    0, 0, 0, 0.98244212085553095,
    0.18656762626159434, 0, 0, 0,
    0, 0, 0, 0,
    0, 0.18656762626159493, -0.98244212085553118, 0,
    0, 0, 0, 0,
    0, 0, 0.32550557193105317, 0,
    0, -0.94554012217453676, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    -0.0083460326018288122, 0, 0, 0.99996517126338413,
    0, 0, 0, 0,
    0, 0, 0, -0.078190858817790199,
    -0.99693840812626766, 0, 0, 0,
    0, 0, 0, 0,
    0, -0.99693840812626744, 0.078190858817787312, 0,
    0, 0, 0, 0,
    0, 0, 0.99996517126338413, 0,
    0, 0.0083460326018288677, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1,
};

static const double kRotation20[] = {
    0.094357820205532641, 0, 0, 0,
    0, 0, 0, 0.43135906177387617,
    0, 0, 0, -0.67651111406887776,
    0, 0, -0.58937142290147349, 0,
    0, 0, 0, 0,
    0, -0.14714806864980176, 0, 0,
    0, -0.51999694842708621, 0, 0,
    0, 0, 0.78331413493294177, 0,
    0, 0, 0, 0,
    -0.30719633059886564, 0, 0, 0,
    0, 0, -0.1173466075475636, 0,
    0, 0, -0.0066052923384407602, 0,
    0, 0.3677044421765181, 0, 0,
    0, -0.92248554840379926, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, -0.25181453770612561,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, -0.96777551043609811, 0,
    0, 0, 0, 0,
    -0.037903363351909691, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, -0.99928140933703602, 0, 0,
    0, 0.13291757899410217, 0, 0,
    0, -0.78351683723990651, 0, 0,
    0, 0, -0.57332972852863795, 0,
    0, 0, 0, 0,
    -0.19931709746330861, 0, 0, 0,
    0, 0, -0.10136478375774655, 0,
    0, 0, 0.80267492906368987, 0,
    0, 0.54347255084203272, 0, 0,
    0, 0.22377606071159639, 0, 0,
    0, 0, 0, 0,
    -0.57124887096866095, 0, 0, 0,
    0, 0, 0, 0.60451085265868321,
    0, 0, 0, 0.5055993766738458,
    0, 0, -0.2293700650519182, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0.50733740958253981, 0, 0, 0,
    -0.57823264097724669, 0, 0, -0.63040893949970944,
    0, 0, 0, -0.10411692823652614,
    0, 0, 0.083717367558268987, 0,
    0, 0, -0.58173694434366507, 0,
    0, 0.75376190308263535, 0, 0,
    0, 0.29396687469177996, 0, 0,
    0, 0, 0, 0,
    0, -0.9372770527595663, 0, 0,
    0, -0.12610631023365568, 0, 0,
    0, 0, -0.1459053598097651, 0,
    0, 0, 0, 0,
    0.29038001113777945, 0, 0, 0,
    -0.42220389214480075, 0, 0, 0,
    0, 0, 0, 0.28200165477706707,
    0, 0, 0, -0.49398931396384976,
    0, 0, 0.7058282353733818, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    -0.80335867438493835, 0, 0, 0,
    -0.26712269846936348, 0, 0, -0.33293014361889633,
    0, 0, 0, -0.41523225274877429,
    0, 0, 0.98435072849058092, 0,
    0, 0, 0.1313447979610137, 0,
    0, 0.035693460831835541, 0, 0,
    0, -0.11192928223878409, 0, 0,
    0, 0, 0, 0,
    0.69750641763418375, 0, 0, 0,
    0, 0, 0, 0.6074298770610721,
    0, 0, 0, 0.20658301125172288,
    0, 0, 0.31911941538375471, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    -0.07647486061889508, 0, 0, 0,
    -0.76805146740518648, 0, 0, 0.61884729519307013,
    0, 0, 0, 0.14586488385584573,
    0, 0.28669162781169777, 0, 0,
    0, -0.31591413059143725, 0, 0,
    0, 0, 0.19084971673627876, 0,
    0, 0, 0, 0,
    0.88407157982655415, 0, 0, 0,
    0, 0, 0, 0,
    0.99928140933703613, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, -0.037903363351910135, 0, 0,
    0, 0, 0, -0.96777551043609811,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0.25181453770612561, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0.30228329233001572, 0, 0, 0,
    0.066252700229221306, 0, 0, 0.32980314386156834,
    0, 0, 0, -0.89188860133398296,
};

static const int kShuffle10[] = {
    10, 9, 7, 6, 3, 2, 8, 5, 4, 1,
};

static const int kShuffle20[] = {
    19, 4, 5, 18, 9, 20, 16, 13, 1, 12,
    15, 8, 6, 2, 17, 11, 14, 3, 10, 7,
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

inline double hgbat_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    double r2 = 0.0;
    double sumz = 0.0;
    for (int i = 0; i < n; ++i) {
        const double zi = (5.0 / 100.0) * x[i] - 1.0;
        r2 += zi * zi;
        sumz += zi;
    }
    return std::pow(std::fabs(r2 * r2 - sumz * sumz), 0.5) + (0.5 * r2 + sumz) / n + 0.5;
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

inline double rastrigin_basic(const double* x, int n) {
    double f = 0.0;
    for (int i = 0; i < n; ++i) {
        const double zi = (5.12 / 100.0) * x[i];
        f += zi * zi - 10.0 * std::cos(2.0 * kPi * zi) + 10.0;
    }
    return f;
}

inline double schwefel_basic(const double* x, int n) {
    if (n <= 0) return 0.0;
    double f = 0.0;
    for (int i = 0; i < n; ++i) {
        double zi = (1000.0 / 100.0) * x[i] + 4.209687462275036e+002;
        if (zi > 500.0) {
            f -= (500.0 - std::fmod(zi, 500.0)) * std::sin(std::sqrt(500.0 - std::fmod(zi, 500.0)));
            const double tmp = (zi - 500.0) / 100.0;
            f += (tmp * tmp) / n;
        } else if (zi < -500.0) {
            f -= (-500.0 + std::fmod(std::fabs(zi), 500.0)) * std::sin(std::sqrt(500.0 - std::fmod(std::fabs(zi), 500.0)));
            const double tmp = (zi + 500.0) / 100.0;
            f += (tmp * tmp) / n;
        } else {
            f -= zi * std::sin(std::sqrt(std::fabs(zi)));
        }
    }
    return f + 4.189828872724338e+002 * n;
}

inline double schaffer_f7_basic(const double* x, int n) {
    if (n <= 1) return 0.0;
    double f = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const double s = std::sqrt(x[i] * x[i] + x[i + 1] * x[i + 1]);
        const double t = std::sin(50.0 * std::pow(s, 0.2));
        f += std::sqrt(s) + std::sqrt(s) * t * t;
    }
    return (f * f) / (static_cast<double>(n - 1) * static_cast<double>(n - 1));
}

inline double hybrid10_value(const Vec& x, const Vec& shift, const std::vector<double>& rotation, const std::vector<int>& shuffle) {
    const int D = static_cast<int>(x.size());

    Vec z = apply_shift_rotate(x, shift, rotation);
    Vec y = apply_shuffle(z, shuffle);

    const int g0 = static_cast<int>(std::ceil(0.1 * D));
    const int g1 = static_cast<int>(std::ceil(0.2 * D));
    const int g2 = static_cast<int>(std::ceil(0.2 * D));
    const int g3 = static_cast<int>(std::ceil(0.2 * D));
    const int g4 = static_cast<int>(std::ceil(0.1 * D));
    const int g5 = D - g0 - g1 - g2 - g3 - g4;

    double f = 0.0;
    f += hgbat_basic(y.data(), g0);
    f += katsuura_basic(y.data() + g0, g1);
    f += ackley_basic(y.data() + g0 + g1, g2);
    f += rastrigin_basic(y.data() + g0 + g1 + g2, g3);
    f += schwefel_basic(y.data() + g0 + g1 + g2 + g3, g4);
    f += schaffer_f7_basic(y.data() + g0 + g1 + g2 + g3 + g4, g5);
    return f + 2000.0;
}

} // namespace

CEC2022Hybrid10::CEC2022Hybrid10()
{
    setName("cec2022hybrid10");
    setFullName("CEC 2022 F7 (reference hf10) - Hybrid Function 10");
    setModality("hybrid");
    setSeparability("non-separable");
    setCategory("CEC 2022 synthetic benchmark");
}

void CEC2022Hybrid10::init(int dim)
{
    if (!(dim == 10 || dim == 20)) {
        throw std::invalid_argument("CEC2022Hybrid10 supports only D = 10 or 20.");
    }

    Problem::init(dim);

    Vec lo(dim, -100.0), hi(dim, 100.0);
    setBounds(lo, hi);

    load_embedded_data(dim);
}

void CEC2022Hybrid10::load_embedded_data(int dim)
{
    const double* shift_ptr = shift_data_for_dim(dim);
    const double* rot_ptr   = rotation_data_for_dim(dim);
    const int* shuf_ptr     = shuffle_data_for_dim(dim);

    if (!shift_ptr || !rot_ptr || !shuf_ptr) {
        throw std::runtime_error("Embedded CEC2022Hybrid10 data are unavailable for the requested dimension.");
    }

    shift_.assign(shift_ptr, shift_ptr + dim);
    rotation_.assign(rot_ptr, rot_ptr + dim * dim);
    shuffle_.assign(shuf_ptr, shuf_ptr + dim);
}

double CEC2022Hybrid10::evaluate_core(const Vec& x)
{
    return hybrid10_value(x, shift_, rotation_, shuffle_);
}

void CEC2022Hybrid10::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    constexpr double eps = 1e-6;
    Vec xp = x, xm = x;
    for (int i = 0; i < D; ++i) {
        const double h = eps * std::max(1.0, std::fabs(x[i]));
        xp[i] = x[i] + h;
        xm[i] = x[i] - h;
        g[i] = (hybrid10_value(xp, shift_, rotation_, shuffle_) - hybrid10_value(xm, shift_, rotation_, shuffle_)) / (2.0 * h);
        xp[i] = x[i];
        xm[i] = x[i];
    }
}

} // namespace optimsolution
