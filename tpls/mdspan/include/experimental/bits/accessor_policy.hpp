//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Kokkos is licensed under 3-clause BSD terms of use:
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

// [mdspan.accessor.basic]
template <class ElementType>
class accessor_basic;

template <class ElementType>
class accessor_basic {
 public:
  using element_type  = ElementType;
  using pointer       = ElementType*;
  using offset_policy = accessor_basic;
  using reference     = ElementType&;

  constexpr accessor_basic() noexcept = default;

  constexpr accessor_basic(accessor_basic&&) noexcept = default;

  constexpr accessor_basic(const accessor_basic&) noexcept = default;

  accessor_basic& operator=(accessor_basic&&) noexcept = default;

  accessor_basic& operator=(const accessor_basic&) noexcept = default;

  constexpr typename offset_policy::pointer offset(pointer p, ptrdiff_t i) const
      noexcept {
    return typename offset_policy::pointer(p + i);
  }

  constexpr reference access(pointer p, ptrdiff_t i) const noexcept {
    // printf("basic accessor: %d, %08x \n", i, p);
    // fflush(stdout);
    return p[i];
  }

  constexpr ElementType* decay(pointer p) const noexcept { return p; }
};

}  // namespace fundamentals_v3
}  // namespace experimental
}  // namespace std
