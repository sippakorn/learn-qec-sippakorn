#include <m4ri/m4ri.h>
#include <stdlib.h>
#include <stdint.h>

/* Wrapper to write a single bit — exposes mzd_write_bit (inline in header) */
void helper_write_bit(mzd_t *M, int row, int col, int val) {
    mzd_write_bit(M, (rci_t)row, (rci_t)col, (BIT)val);
}

/* Wrapper to read a single bit */
int helper_read_bit(const mzd_t *M, int row, int col) {
    return (int)mzd_read_bit(M, (rci_t)row, (rci_t)col);
}

/* Wrapper to write an entire row from a uint8 array */
void helper_write_row(mzd_t *M, int row, const uint8_t *data, int ncols) {
    for (int col = 0; col < ncols; col++) {
        if (data[col]) {
            mzd_write_bit(M, (rci_t)row, (rci_t)col, 1);
        }
    }
}

/* Wrapper to read an entire row into a uint8 array */
void helper_read_row(const mzd_t *M, int row, uint8_t *data, int ncols) {
    for (int col = 0; col < ncols; col++) {
        data[col] = (uint8_t)mzd_read_bit(M, (rci_t)row, (rci_t)col);
    }
}

/* Wrapper: get nrows */
int helper_nrows(const mzd_t *M) { return (int)M->nrows; }

/* Wrapper: get ncols */
int helper_ncols(const mzd_t *M) { return (int)M->ncols; }