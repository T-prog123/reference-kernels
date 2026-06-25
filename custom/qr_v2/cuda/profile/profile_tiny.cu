#include "common_profile.cuh"

int main() {
    ProfileCase c{
        "tiny_dense_32",
        20,
        32,
        1,
        ProfileCaseKind::Dense,
        10,
        100,
    };
    return profile_main(c);
}
