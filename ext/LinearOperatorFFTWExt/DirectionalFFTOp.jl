export FFTOpImpl

mutable struct DirFFTOpImpl{T, vecT, P <: AbstractFFTs.Plan{T}, IP <: AbstractFFTs.Plan{T}} <: DirFFTOp{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: vecT
  Mtu5 :: vecT
  plan :: P
  iplan :: IP
  shift::Bool
  unitary::Bool
end

LinearOperators.storage_type(op::DirFFTOpImpl) = typeof(op.Mv5)

"""
  FFTOp(T::Type; shape::Tuple, shift=true, unitary=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* `dims::Tuple`   - dimensions along which to perform the FFT
* (`shift=true`)  - if true, fftshifts are performed
* (`unitary=true`)  - if true, FFT is normalized such that it is unitary
* (`S = Vector{T}`) - type of temporary vector, change to use on GPU
* (`kwargs...`) - keyword arguments given to fft plan
"""
function LinearOperatorCollection.DirFFTOp(T::Type; shape::NTuple{D,Int64}, dims::NTuple{D,Int64}=ntuple(i->i, Val(D)), shift::Bool=true, unitary::Bool=true, S = Array{Complex{real(T)}}, kwargs...) where D
  
  tmpVec = similar(S(undef, 0), shape...)
  plan = plan_fft!(tmpVec, dims; kwargs...)
  iplan = plan_bfft!(tmpVec, dims; kwargs...)

  if unitary
    facF = T(1.0/sqrt(prod(shape)))
    facB = T(1.0/sqrt(prod(shape)))
  else
    facF = T(1.0)
    facB = T(1.0)
  end

  let shape_ = shape, plan_ = plan, iplan_ = iplan, tmpVec_ = tmpVec, facF_ = facF, facB_ = facB, dim_ = dims

    fun! = dir_fft_multiply!
    if shift
      fun! = dir_fft_multiply_shift!
    end

    return DirFFTOpImpl(prod(shape), prod(shape), false, false, (res, x) -> fun!(res, plan_, x, shape_, dim_, facF_, tmpVec_),
        nothing, (res, x) -> fun!(res, iplan_, x, shape_, facB_, tmpVec_),
        0, 0, 0, true, false, true, similar(tmpVec, 0), similar(tmpVec, 0), plan, iplan, shift, unitary)
  end
end

function dir_fft_multiply!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, ::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  plan * copyto!(tmpVec, x)
  res .= factor .* vec(tmpVec)
end

function dir_fft_multiply_shift!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, shape::NTuple{D}, dim::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  ifftshift!(tmpVec, reshape(x,shape), dim)
  plan * tmpVec
  fftshift!(reshape(res,shape), tmpVec, dim)
  res .*= factor
end


function Base.copy(S::FFTOpImpl)
  return FFTOp(eltype(S); shape=size(S.plan), shift=S.shift, unitary=S.unitary, S = LinearOperators.storage_type(S)) # TODO loses kwargs...
end
