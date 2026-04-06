module hFlux
using Libdl
const _libname = joinpath(@__DIR__, "..", "build", "src", "libhflux.so")

# Optionally allow users to point at a specific path (returns previous path)
const _lib_ref = Ref{String}(_libname)
function set_library_path(path::AbstractString)
    old = _lib_ref[]
    _lib_ref[] = String(path)
    return old
end

# Handy accessor for ccall
_lib() = _lib_ref[]

# -- Handle type with finalizer -------------------------------------------------
"""
    hFluxHandle(ptr)

Opaque handle for an hflux field interpolator. Automatically freed by the GC
(via `hflux_destroy`) when no longer referenced.
"""
struct hFluxHandle
    ptr::Ptr{Cvoid}
end


# Attach a finalizer to free native resources
function _attach_finalizer!(h::hFluxHandle)
    finalizer(h) do hh
        try
            if hh.ptr != C_NULL
                ccall((:hflux_destroy, _lib()), Cvoid, (Ptr{Cvoid},), hh.ptr)
            end
        catch err
            # Swallow errors in finalizer to avoid GC issues
        end
    end
    return h
end

# -- Kokkos lifecycle -----------------------------------------------------------
"""
    kokkos_init()

Initialize Kokkos inside the hflux library.
"""
function kokkos_init()
    ccall((:hflux_kokkos_init, _lib()), Cvoid, ())
    return nothing
end

"""
    kokkos_finalize()

Finalize Kokkos inside the hflux library.
"""
function kokkos_finalize()
    ccall((:hflux_kokkos_finalize, _lib()), Cvoid, ())
    return nothing
end

# -- Constructor / destructor ---------------------------------------------------
"""
    init(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ) -> hFluxHandle

Create and initialize an hflux field interpolator. Returns an `hFluxHandle`.
Corresponds to `hflux_init(..., void **fi)` which fills in the native pointer.
"""
function init(nR_data::Integer, nZ_data::Integer, nfields::Integer,
              nphi_data::Integer, nt::Integer,
              R0::Real, Z0::Real, dR::Real, dZ::Real)::hFluxHandle
    fi_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:hflux_init, _lib()), Cvoid,
          (Cint, Cint, Cint, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Ptr{Cvoid}}),
          Cint(nR_data), Cint(nZ_data), Cint(nfields),
          Cint(nphi_data), Cint(nt),
          Cdouble(R0), Cdouble(Z0), Cdouble(dR), Cdouble(dZ), fi_ref)
    h = hFluxHandle(fi_ref[])
    return h
end

# -- Core operations ------------------------------------------------------------
"""
    interpolate!(h, raw_field_data)

Populate/refresh internal interpolation data from `raw_field_data` (Vector{Float64}).
This is a thin wrapper around `hflux_interpolate`. Size/layout must match the
expectations of the C side.
"""
function interpolate!(h::hFluxHandle, raw_field_data::AbstractVector{<:Real})
    length(raw_field_data) > 0 || throw(ArgumentError("raw_field_data must be non-empty"))
    arr = Vector{Cdouble}(raw_field_data)
    ccall((:hflux_interpolate, _lib()), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}),
          h.ptr, pointer(arr))
    return nothing
end

"""
    get_corners!(h, corners)

Write domain corners into `corners` (Vector{Float64}).
The required length is defined by the C library (commonly 4 or 8 doubles).
"""
function get_corners!(h::hFluxHandle, corners::AbstractVector{<:Real})
    corners_vec = Vector{Cdouble}(corners)
    ccall((:hflux_getcorners, _lib()), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}),
          h.ptr, pointer(corners_vec))
    # copy back if the input wasn't already Cdouble
    if !(corners isa Vector{Cdouble})
        @inbounds for i in eachindex(corners)
            corners[i] = Float64(corners_vec[i])
        end
    end
    return corners
end

"""
    compute_poincare!(h, r0, dr, n_r, n_theta, n_turn, poincare_data)

Compute a Poincaré map. Results are written into `poincare_data` (Vector{Float64}).
You must size `poincare_data` according to your C implementation's contract
(e.g. `2 * n_r * n_theta * n_turn` if storing (R,Z) pairs).
"""
function compute_poincare!(h::hFluxHandle,
                           r0::Real, dr::Real,
                           n_r::Integer, n_theta::Integer, n_turn::Integer,
                           poincare_data::AbstractVector{<:Real})
    arr = Vector{Cdouble}(poincare_data)
    ccall((:hflux_compute_poincare, _lib()), Cvoid,
          (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Cint, Cint, Ptr{Cdouble}),
          h.ptr, Cdouble(r0), Cdouble(dr),
          Cint(n_r), Cint(n_theta), Cint(n_turn),
          pointer(arr))
    if !(poincare_data isa Vector{Cdouble})
        @inbounds for i in eachindex(poincare_data)
            poincare_data[i] = Float64(arr[i])
        end
    end
    return poincare_data
end

"""
    field_eval!(h, R, phi, Z, t, out)

Evaluate the field at the given mesh points. All input arrays must have the same
length `N = length(R)`, and `out` must have length `N`.
"""
function field_eval!(h::hFluxHandle,
                     R::AbstractVector{<:Real},
                     phi::AbstractVector{<:Real},
                     Z::AbstractVector{<:Real},
                     t::AbstractVector{<:Real},
                     out::AbstractVector{<:Real})
    N = length(R)
    (length(phi)==N && length(Z)==N && length(t)==N && length(out)==N) ||
        throw(ArgumentError("R, phi, Z, t, out must all have equal length"))
    Rv   = Vector{Cdouble}(R)
    phiv = Vector{Cdouble}(phi)
    Zv   = Vector{Cdouble}(Z)
    tv   = Vector{Cdouble}(t)
    outv = Vector{Cdouble}(out)
    ccall((:hflux_field_eval, _lib()), Cvoid,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          h.ptr, Cint(N), pointer(Rv), pointer(phiv), pointer(Zv), pointer(tv), pointer(outv))
    if !(out isa Vector{Cdouble})
        @inbounds for i in 1:N
            out[i] = Float64(outv[i])
        end
    end
    return out
end

"""
    psi_eval!(h, R, phi, Z, t, out; center_R=nothing, center_Z=nothing)

Evaluate ψ at mesh points. `out` must have length `N`.
If you pass `center_R` and `center_Z`, they must be length-1 vectors (Refs) to receive values.
"""
function psi_eval!(h::hFluxHandle,
                   R::AbstractVector{<:Real},
                   phi::AbstractVector{<:Real},
                   Z::AbstractVector{<:Real},
                   t::AbstractVector{<:Real},
                   out::AbstractVector{<:Real};
                   center_R::Union{Nothing,Ref{Cdouble}}=nothing,
                   center_Z::Union{Nothing,Ref{Cdouble}}=nothing)
    N = length(R)
    (length(phi)==N && length(Z)==N && length(t)==N && length(out)==N) ||
        throw(ArgumentError("R, phi, Z, t, out must all have equal length"))
    Rv   = Vector{Cdouble}(R)
    phiv = Vector{Cdouble}(phi)
    Zv   = Vector{Cdouble}(Z)
    tv   = Vector{Cdouble}(t)
    outv = Vector{Cdouble}(out)

    # If centers are not provided, pass dummy refs
    cR = center_R === nothing ? Ref{Cdouble}(0) : center_R
    cZ = center_Z === nothing ? Ref{Cdouble}(0) : center_Z

    ccall((:hflux_psi_eval, _lib()), Cvoid,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
          h.ptr, Cint(N), pointer(Rv), pointer(phiv), pointer(Zv), pointer(tv),
          pointer(outv), cR, cZ)

    if !(out isa Vector{Cdouble})
        @inbounds for i in 1:N
            out[i] = Float64(outv[i])
        end
    end
    return out, cR[], cZ[]
end

# -- Low-level raw pointer access (if ever needed) ------------------------------
rawptr(h::hFluxHandle) = h.ptr

end # module

# -- Simple usage example -------------------------------------------------------
#=
using .hFlux

HFlux.kokkos_init()
h = HFlux.init(nR, nZ, nfields, nphi, nt, R0, Z0, dR, dZ)

# Suppose you have raw field data to interpolate
HFlux.interpolate!(h, raw_field_data)

# Evaluate field on a mesh
N = length(R); out = zeros(Float64, N)
HFlux.field_eval!(h, R, phi, Z, t, out)

# Compute psi and get magnetic axis estimate
centerR = Ref{Cdouble}(0); centerZ = Ref{Cdouble}(0)
psi, Rax, Zax = HFlux.psi_eval!(h, R, phi, Z, t, out; center_R=centerR, center_Z=centerZ)

# Poincaré (ensure you size poincare_data per your C implementation)
pdata = zeros(Float64, 2*n_r*n_theta*n_turn)  # example size if storing (R,Z) pairs
HFlux.compute_poincare!(h, r0, dr, n_r, n_theta, n_turn, pdata)

# When done
HFlux.kokkos_finalize()
=#


