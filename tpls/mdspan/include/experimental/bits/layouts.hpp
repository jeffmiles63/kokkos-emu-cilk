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

// [mdspan.layout]
class layout_right;
class layout_left;
class layout_stride;

}  // namespace fundamentals_v3
}  // namespace experimental
}  // namespace std

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

class layout_right {
 public:
  template <class Extents>
  class mapping {
   private:
    Extents m_extents;

   public:
    using index_type   = ptrdiff_t;
    using extents_type = Extents;

    constexpr mapping() noexcept = default;

    constexpr mapping(mapping&&) noexcept = default;

    constexpr mapping(const mapping&) noexcept = default;

    mapping& operator=(mapping&&) noexcept = default;

    mapping& operator=(const mapping&) noexcept = default;

    constexpr mapping(const Extents& ext) noexcept : m_extents(ext) {}

    constexpr const Extents& extents() const noexcept { return m_extents; }

   private:
    // ( ( ( ( i0 ) * N1 + i1 ) * N2 + i2 ) * N3 + i3 ) ...

    static constexpr index_type offset(const size_t, const ptrdiff_t sum) {
      //		  printf("offset returning %d \n", sum);
      //		  fflush(stdout);
      return sum;
    }

    template <class... Indices>
    inline constexpr index_type offset(const size_t r, ptrdiff_t sum,
                                       const index_type i,
                                       Indices... indices) const noexcept {
      return offset(r + 1, sum * m_extents.extent(r) + i, indices...);
    }

   public:
    constexpr index_type required_span_size() const noexcept {
      index_type size = 1;
      for (size_t r = 0; r < m_extents.rank(); r++) size *= m_extents.extent(r);
      return size;
    }

    template <class... Indices>
    constexpr typename enable_if<sizeof...(Indices) == Extents::rank(),
                                 index_type>::type
    operator()(Indices... indices) const noexcept {
      return offset(0, 0, indices...);
    }

    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return true; }
    static constexpr bool is_always_strided() noexcept { return true; }

    constexpr bool is_unique() const noexcept { return true; }
    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_strided() const noexcept { return true; }

    constexpr index_type stride(const size_t R) const noexcept {
      ptrdiff_t stride_ = 1;
      for (size_t r = m_extents.rank() - 1; r > R; r--)
        stride_ *= m_extents.extent(r);
      return stride_;
    }

  };  // class mapping

};  // class layout_right

}  // namespace fundamentals_v3
}  // namespace experimental
}  // namespace std

//----------------------------------------------------------------------------

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

class layout_left {
 public:
  template <class Extents>
  class mapping {
   private:
    Extents m_extents;

   public:
    using index_type   = ptrdiff_t;
    using extents_type = Extents;

    constexpr mapping() noexcept = default;

    constexpr mapping(mapping&&) noexcept = default;

    constexpr mapping(const mapping&) noexcept = default;

    mapping& operator=(mapping&&) noexcept = default;

    mapping& operator=(const mapping&) noexcept = default;

    constexpr mapping(const Extents& ext) noexcept : m_extents(ext) {}

    constexpr const Extents& extents() const noexcept { return m_extents; }

   private:
    // ( i0 + N0 * ( i1 + N1 * ( i2 + N2 * ( ... ) ) ) )

    static constexpr index_type offset(size_t) noexcept { return 0; }

    template <class... IndexType>
    constexpr index_type offset(const size_t r, index_type i,
                                IndexType... indices) const noexcept {
      return i + m_extents.extent(r) * offset(r + 1, indices...);
    }

   public:
    constexpr index_type required_span_size() const noexcept {
      ptrdiff_t size = 1;
      for (size_t r = 0; r < m_extents.rank(); r++) size *= m_extents.extent(r);
      return size;
    }

    template <class... Indices>
    constexpr typename enable_if<sizeof...(Indices) == Extents::rank(),
                                 index_type>::type
    operator()(Indices... indices) const noexcept {
      return offset(0, indices...);
    }

    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return true; }
    static constexpr bool is_always_strided() noexcept { return true; }

    constexpr bool is_unique() const noexcept { return true; }
    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_strided() const noexcept { return true; }

    constexpr index_type stride(const size_t R) const noexcept {
      ptrdiff_t stride_ = 1;
      for (size_t r = 0; r < R; r++) stride_ *= m_extents.extent(r);
      return stride_;
    }

  };  // class mapping

};  // class layout_left

}  // namespace fundamentals_v3
}  // namespace experimental
}  // namespace std

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

class layout_stride {
 public:
  template <class Extents>
  class mapping {
   private:
    using stride_t = array<ptrdiff_t, Extents::rank()>;

    Extents m_extents;
    stride_t m_stride;
    int m_contig;

   public:
    using index_type   = ptrdiff_t;
    using extents_type = Extents;

    constexpr mapping() noexcept = default;

    constexpr mapping(mapping&&) noexcept = default;

    constexpr mapping(const mapping&) noexcept = default;

    mapping& operator=(mapping&&) noexcept = default;

    mapping& operator=(const mapping&) noexcept = default;

    mapping(const Extents& ext, const stride_t& str) noexcept
        : m_extents(ext), m_stride(str), m_contig(1) {
      int p[Extents::rank() ? Extents::rank() : 1];

      // Fill permutation such that
      //   m_stride[ p[i] ] <= m_stride[ p[i+1] ]
      //
      for (size_t i = 0; i < Extents::rank(); ++i) {
        int j = i;

        while (j && m_stride[i] < m_stride[p[j - 1]]) {
          p[j] = p[j - 1];
          --j;
        }

        p[j] = i;
      }

      for (size_t i = 1; i < Extents::rank(); ++i) {
        const int j           = p[i - 1];
        const int k           = p[i];
        const index_type prev = m_stride[j] * m_extents.extent(j);
        if (m_stride[k] != prev) {
          m_contig = 0;
        }
      }
    }

    constexpr const Extents& extents() const noexcept { return m_extents; }

   private:
    // i0 * N0 + i1 * N1 + i2 * N2 + ...

    constexpr index_type offset(size_t) const noexcept { return 0; }

    template <class... IndexType>
    constexpr index_type offset(const size_t K, const index_type i,
                                IndexType... indices) const noexcept {
      return i * m_stride[K] + offset(K + 1, indices...);
    }

   public:
    index_type required_span_size() const noexcept {
      index_type max = 0;
      for (size_t i = 0; i < Extents::rank(); ++i)
        max += m_stride[i] * (m_extents.extent(i) - 1);
      return max;
    }

    template <class... Indices>
    constexpr typename enable_if<sizeof...(Indices) == Extents::rank(),
                                 index_type>::type
    operator()(Indices... indices) const noexcept {
      return offset(0, indices...);
    }

    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return false; }
    static constexpr bool is_always_strided() noexcept { return true; }

    constexpr bool is_unique() const noexcept { return true; }
    constexpr bool is_contiguous() const noexcept { return m_contig; }
    constexpr bool is_strided() const noexcept { return true; }

    constexpr index_type stride(size_t r) const noexcept { return m_stride[r]; }

  };  // class mapping

};  // class layout_stride

}  // namespace fundamentals_v3
}  // namespace experimental
}  // namespace std
