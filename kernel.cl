/*
 * a kernel that add the elements of two vectors pairwise
 */
__kernel void vector_add(__global const int* A,
                         __global int* B,
                         const unsigned int numberOfElements) {
    size_t i = get_global_id(0);
    if (i + 1 == numberOfElements) {
        B[i] = A[i] + 0;
    } else {
        B[i] = A[i] + A[i + 1];
    }
}
