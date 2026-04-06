#pragma once
#include <Kokkos_Core.hpp>
#include <type_traits>

#include "common.hpp"

template <size_t StencilSize>
KOKKOS_INLINE_FUNCTION
static constexpr Kokkos::Array<Kokkos::Array<Real, StencilSize>, 6>  fdw()
{
    static_assert(StencilSize == 5 || StencilSize == 7 || StencilSize == 9,
                  "Unsupported stencil size - only 5, 7 or 9 are allowed.");

    if constexpr (StencilSize == 5) {
        // 5-point stencil, O(h^4)
        return  {{
          {0.0, 0.0, 1.0, 0.0, 0.0},
          {1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12}, // Accuracy 4
          {-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12}, // Accuracy 4
          {-0.5, 1, 0, -1, 0.5},                     // Accuracy 2
          {1, -4, 6, -4, 1},                          // Accuracy 2
          {0.0, 0.0, 0.0, 0.0, 0.0}
        }};
    }
    else if constexpr (StencilSize == 7) {
        // 7-point stencil, O(h^6)
        return {{
          {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
          {-1.0 / 60, 3.0 / 20, -3.0 / 4, 0, 3.0 / 4, -3.0 / 20, 1.0 / 60}, // Accuracy 6
          {1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90}, // Accuracy 6
          {1.0 / 8, -1, 13.0 / 8, 0, -13.0 / 8, 1, -1.0 / 8}, // Accuracy 4
          {-1.0 / 6, 2.0, -13.0 / 2, 28.0 / 3, -13.0 / 2, 2.0, -1.0 / 6},  // Accuracy 4
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        }};
    }
    else /* if constexpr (StencilSize == 9) */ {
        // 9-point stencil, O(h^8)

        return {{
          {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
          {1.0 / 280, -4.0 / 105, 1.0 / 5, -4.0 / 5, 0, 4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280}, // Accuracy 8
          {-1.0 / 560, 8.0 / 315, -1.0 / 5, 8.0 / 5, -205.0 / 72, 8.0 / 5, -1.0 / 5, 8.0 / 315, -1.0 / 560}, // Accuracy 8
          {-7.0 / 240, 3.0 / 10, -169.0 / 120, 61.0 / 30, 0, -61.0 / 30, 169.0 / 120, -3.0 / 10, 7.0 / 240}, // Accuracy 6
          {7.0 / 240, -2.0 / 5, 169.0 / 60, -122.0 / 15, 91.0 / 8, -122.0 / 15, 169.0 / 60, -2.0 / 5, 7.0 / 240}, // Accuracy 6
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        }};
    }
}
