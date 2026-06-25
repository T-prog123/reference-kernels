#include "common_profile.cuh"

enum class ProfileScenario {
    StressMixed512,
    TallDense4096,
};

#ifndef PROFILE_BLOCKED_CASE
#define PROFILE_BLOCKED_CASE 512
#endif

#if PROFILE_BLOCKED_CASE == 4096
constexpr ProfileScenario kScenario = ProfileScenario::TallDense4096;
#else
constexpr ProfileScenario kScenario = ProfileScenario::StressMixed512;
#endif

ProfileCase selected_case() {
    if constexpr (kScenario == ProfileScenario::StressMixed512) {
        return {
            "stress_mixed_512",
            640,
            512,
            2,
            ProfileCaseKind::Mixed,
            3,
            10,
        };
    } else {
        return {
            "tall_dense_4096",
            2,
            4096,
            1,
            ProfileCaseKind::Dense,
            1,
            3,
        };
    }
}

int main() {
    return profile_main(selected_case());
}
