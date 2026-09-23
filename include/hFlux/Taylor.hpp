#pragma once
#include "common.hpp"

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylor(Real& val, Real x, Real y, CoeffView pcofs) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);

  val = 0.0;
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
        val += mon * pcofs(i, j);
        mon *= y;
    }
    sclx *= x;
  }
}

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorScalar(Real& val, Real x, Real y, CoeffView pcofs, int component = 0) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);

  val = 0.0;
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
        val += mon * pcofs(i, j, component);
        mon *= y;
    }
    sclx *= x;
  }
}

template<int ncomp, class Out, class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorN(Out& vals, Real x, Real y, CoeffView pcofs, int component0 = 0) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  KOKKOS_ASSERT(pcofs.extent_int(2) >= component0 + ncomp);

  vals = {};
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
      for (int d = 0; d < ncomp; ++d) {
         vals[d] += mon * pcofs(i, j, component0 + d);
       }
       mon *= y;
    }
    sclx *= x;
  }
}

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylor3(Dim3& vals, Real x, Real y, CoeffView pcofs, int component0 = 0) {

  evalTaylorN<3, Dim3, CoeffView>(vals, x, y, pcofs, component0);
}


template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorFourier(Real& val, Real x, Real y, Real phi, CoeffView pcofs) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  const int Nchannels = pcofs.extent_int(2);

  const int ncos = Nchannels / 2;

  val = 0.0;
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
        val += mon * pcofs(i, j, 0);
        mon *= y;
    }
    sclx *= x;
  }
	for (int channel = 1; channel < ncos + 1; ++channel) {
    Real sclx = cos(channel * phi);
    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
          val += mon * pcofs(i, j, channel);
          mon *= y;
      }
      sclx *= x;
    }
  }

	for (int channel = ncos+1; channel < Nchannels; ++channel) {
    Real sclx = sin((channel - ncos) * phi);
    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
          val += mon * pcofs(i, j, channel);
          mon *= y;
      }
      sclx *= x;
    }
  }
}


template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorFourierDerivative(Real& val, Real x, Real y, Real phi, CoeffView pcofs) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  const int Nchannels = pcofs.extent_int(2);

  const int ncos = Nchannels / 2;

  val = 0.0;
	for (int channel = 1; channel < ncos + 1; ++channel) {
    Real sclx = Kokkos::sin(channel * phi);
    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
          val += mon * pcofs(i, j, 0);
          mon *= y;
      }
      sclx *= x;
    }
  }

	for (int channel = ncos+1; channel < Nchannels; ++channel) {
    Real sclx = Kokkos::cos((channel - ncos) * phi);
    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
          val += mon * pcofs(i, j, channel);
          mon *= y;
      }
      sclx *= x;
    }
  }
}

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorFourier4(Dim3& vals, Real& correction, Real x, Real y, Real phi, CoeffView pcofs) {
  constexpr int channel_stride = 4;

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  const int Nchannels = pcofs.extent_int(2) / channel_stride;

  const int ncos = Nchannels / 2;

  correction = 0.0;
  vals = {};

  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
      for (int d = 0; d < 3; ++d) {
         vals[d] += mon * pcofs(i, j, d);
       }
       mon *= y;
    }
    sclx *= x;
  }

	for (int channel = 1; channel < ncos + 1; ++channel) {
    Real c = Kokkos::cos(channel * phi);
    Real s = Kokkos::sin(channel * phi);
    Real sclx = 1.0;

    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
      	for (int d = 0; d < 3; ++d) {
      	  vals[d] += mon * c * pcofs(i, j, channel * channel_stride  + d);
      	}

        correction += mon * s * pcofs(i, j, channel * channel_stride + 3);
        mon *= y;
      }
      sclx *= x;
    }
  }

	for (int channel = ncos+1; channel < Nchannels; ++channel) {
    Real c = Kokkos::cos((channel - ncos) * phi);
    Real s = Kokkos::sin((channel - ncos) * phi);
    Real sclx = 1.0;

    for (int i = 0; i < Px; ++i) {
      Real mon = sclx;
      for (int j = 0; j < Py; ++j) {
      	for (int d = 0; d < 3; ++d) {
      	  vals[d] += mon * s *pcofs(i, j, channel * channel_stride  + d);
      	}

        correction += mon * c * pcofs(i, j, channel * channel_stride + 3);
        mon *= y;
      }
      sclx *= x;
    }
  }
}
