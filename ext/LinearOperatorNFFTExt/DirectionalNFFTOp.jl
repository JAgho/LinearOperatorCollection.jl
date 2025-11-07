"""
    DirNFFTOpImpl(shape::Tuple, tr::Trajectory; kargs...)
    DirNFFTOpImpl(shape::Tuple, tr::AbstractMatrix; kargs...)

generates a `DirNFFTOpImpl` which evaluates the MRI Fourier signal encoding operator using the NFFT.

# Arguments:
* `shape::NTuple{D,Int64}`                  - size of image to encode/reconstruct
* `tr`                                      - Either a `Trajectory` object, or a `ND x Nsamples` matrix for an ND-dimenensional (e.g. 2D or 3D) NFFT with `Nsamples` k-space samples
* `dims::Union{UnitRange{Int64}, Integer}`  - dimensions along which to perform the NFFT. Dimensions must be contigious, e.g. `1:2` or `2:3` for a 2D NFFT on 3D data
* (`nodes=nothing`)                         - Array containg the trajectory nodes (redundant)
* (`kargs`)                                 - additional keyword arguments
"""
function LinearOperatorCollection.DirNFFTOp(::Type{T};
    shape::Tuple, nodes::AbstractMatrix{U}, dims::Union{UnitRange{Int64}, Integer}, kshape, toeplitz=false, oversamplingFactor=1.25, 
   kernelSize=3, kargs...) where {U <: Number, T <: Number}
  return DirNFFTOpImpl(shape, nodes, dims, kshape; toeplitz, oversamplingFactor, kernelSize, kargs... )
end

mutable struct DirNFFTOpImpl{T, vecT, P <: AbstractNFFTPlan} <: NFFTOp{T}
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
  toeplitz :: Bool
  klen::Tuple
end

LinearOperators.storage_type(op::DirNFFTOpImpl) = typeof(op.Mv5)

function DirNFFTOpImpl(shape::Tuple, tr::AbstractMatrix{T}, dims::Union{UnitRange{Int64}, Integer}, kshape; toeplitz=false, oversamplingFactor=1.25, kernelSize=3, S = Vector{Complex{T}}, kargs...) where {T}

  baseArrayType = Base.typename(S).wrapper # https://github.com/JuliaLang/julia/issues/35543
  plan = plan_nfft(baseArrayType, tr, shape, m=kernelSize, σ=oversamplingFactor, precompute=NFFT.TENSOR,
		                          fftflags=FFTW.ESTIMATE, blocking=true, dims=dims)
                          
  k_produ! = build_produ(kshape)
  k_ctprodu! = build_ctprodu(kshape)

  return DirNFFTOpImpl{eltype(S), S, typeof(plan)}(prod(kshape), prod(shape), false, false
            , (res,x) -> k_produ!(res,plan,x)
            , nothing
            , (res,y) -> k_ctprodu!(res,plan,y)
            , 0, 0, 0, false, false, false, S(undef, 0), S(undef, 0)
            , plan, toeplitz, kshape)
end



function Base.copy(S::DirNFFTOpImpl{T, vecT, P}) where {T, vecT, P}
  plan = copy(S.plan)
  k_produ! = build_produ(S.klen)#S.prod!
  k_ctprodu! = build_ctprodu(S.klen)#S.ctprod!
  return DirNFFTOpImpl{T, vecT, P}(prod(S.klen), prod(plan.N), false, false
              , (res, x) -> k_produ!(res,plan,x)
              , nothing
              , (res, y) -> k_ctprodu!(res,plan,y)
              , 0, 0, 0, false, false, false, vecT(undef, 0), vecT(undef, 0)
              , plan, S.toeplitz, S.klen)
end

function build_produ(kshape)

    function produ!(y::AbstractArray, plan::AbstractNFFTPlan, x::AbstractVector) 
        a = vec(mul!(reshape(y, kshape), plan, reshape(x,plan.N)))
        a .*= 1/(sqrt(plan.J))
        
      end
    return (res, plan, x) -> produ!(res, plan, x)
end

function build_ctprodu(kshape)

  function ctprodu!(x::AbstractVector, plan::AbstractNFFTPlan, y::AbstractArray) 
      a = vec(mul!(reshape(x, plan.N), adjoint(plan), reshape(y, kshape)))
      a .*= 1/(sqrt(prod(plan.N[plan.dims])))
    end
  return (res, plan, y) -> ctprodu!(res, plan, y)
end



function produ!(y::AbstractArray, plan::AbstractNFFTPlan, x::AbstractVector) 
    mul!(y, plan, reshape(x,plan.N))
  end
  
  function ctprodu!(x::AbstractVector, plan::AbstractNFFTPlan, y::AbstractArray)
    mul!(reshape(x, plan.N), adjoint(plan), y)
  end

