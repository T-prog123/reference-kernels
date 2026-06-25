#include "common_profile.cuh"

int main() {
    ProfileCase c{
        "small_dense_176",
        40,
        176,
        1,
        ProfileCaseKind::Dense,
        5,
        50,
    };
    return profile_main(c);
}
