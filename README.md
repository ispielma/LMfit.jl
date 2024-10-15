# LMfit

I have recently being curve fitting in Julia using the very nice [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl) package.  The main issue is that it requires alot of boilier plate coding to take a general function into a form that can be accepted.

Inspired by the python [lmfit](https://lmfit.github.io/lmfit-py/intro.html) package, this Julia package aims to provide an intuative wrapper around `LsqFit.jl`. 
