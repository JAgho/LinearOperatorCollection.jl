module LinearOperatorNFFTExt

using LinearOperatorCollection, NFFT, NFFT.AbstractNFFTs, FFTW, FFTW.AbstractFFTs

include("NFFTOp.jl")
include("DirectionalNFFTOp.jl")

end