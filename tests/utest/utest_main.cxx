#include "utest.h"
#include <vector>

// Initialize the utest state
UTEST_STATE();

int main(int argc, const char* const argv[]) {
    // Create a new vector for arguments
    std::vector<const char*> modified_argv(argv, argv + argc);

    // Always append the mixed-units flag
    modified_argv.push_back("--enable-mixed-units");

    // Update the argument count
    int modified_argc = static_cast<int>(modified_argv.size());

    // Call utest_main with the updated arguments
    return utest_main(modified_argc, modified_argv.data());
}