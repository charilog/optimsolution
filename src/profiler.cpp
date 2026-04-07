#include "profiler.h"

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
  #include <sys/resource.h>
  #include <sys/time.h>
#else
  #include <unistd.h>
  #include <sys/resource.h>
  #include <cstdio>
#endif

namespace optimsolution {

long long Profiler::getPeakRSSKB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // bytes -> KB
        return static_cast<long long>(pmc.PeakWorkingSetSize / 1024);
    }
    return 0;
#elif defined(__APPLE__)
    // ru_maxrss in bytes to macOS -> KB
    struct rusage r{};
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        return static_cast<long long>(r.ru_maxrss / 1024);
    }
    return 0;
#else
    // Linux: ru_maxrss in KB 
    struct rusage r{};
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        return static_cast<long long>(r.ru_maxrss);
    }
    return 0;
#endif
}

} // namespace optimsolution
