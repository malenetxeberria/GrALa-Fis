!
! Author: Malen Etxeberria Etxaniz (GitHub: malenetxeberria)
!
! Last update: 18/02/2020
!
! Description:  Phi 4 model implementation, in a 3D cubic lattice with helical boundary conditions. It 
!               includes a simple measurement of the order parameter Q in a temperature range denoted 
!               by parameters Tf and Ti.
!
! Restrictions: cubic lattice (z=6), zero external field, "single-spin-flip".


 program Phi4

  ! Parameters
    integer, parameter :: L=10, sps=5E6, therm=1E5     ! Number of sites in one dimension, Steps per site, Therm. steps
    real, parameter :: d=0.45, C=2.0, E0=1.0           ! Width param, Coupling const, Energy barrier for local wells
    real, parameter :: Ti=0.0, Tf=5.5, dT=0.05         ! Initial temp, final temp, Temp. step
  ! Variables
    integer :: N, Nt                                   ! Last site's number: N=L*L*L - 1, Number of temp loops
    real :: beta, Qavrg                                ! Inverse temp, Average Q value
    real, dimension(0:L*L*L-1) :: X                    ! L*L*L array containing all xi values (HELICAL B.C.)
    real, dimension(sps) :: Qarr                       ! Array containing Q values
  ! Dummy variables 
    integer :: k, tk

    N = L*L*L - 1 
    Nt = (Tf-Ti)/dT

    open(unit=11, file="Temp.txt", status="replace", action="write", position="append")
    Temp_sweep: do tk = 1, Nt

       call random_seed()
     ! Initialize X
       X = 1.0
       beta = 1.0/(Ti + (dT*tk))

       Thermalization: do k = 1, therm
          call sweep(L, N, beta, d, C, E0, X)
       enddo Thermalization

       MonteCarlo_loop: do k = 1, sps
          call sweep(L, N, beta, d, C, E0, X)
          Qarr(k) = sum(X) ! Qi*(N+1)
       enddo MonteCarlo_loop

       call average(sps, Qarr, Qavrg)
       write(unit=11, fmt=*) Ti + (dT*tk), Qavrg/(N+1)
    enddo Temp_sweep

    close(unit=11)
    
! -------------------------------------------------------------------------------------------------------------- !

 contains

  subroutine random(random_result, low, high)
! Returns a real random value in the range [low, high]. From "Fortran 95 Using F".

     real, intent(in) :: low, high
     real, intent(out) :: random_result
     real :: random_real

     call random_number(random_real)
     random_result = (high-low)*random_real + low
  endsubroutine random


  subroutine nn_values(i, X, L, N, nn_val)
! Returns an array containing the x values of the nearest neighbours of a given i site. It assumes that the 
! lattice is cubic, and thus the number of nearest neighbours (z) is 6. Helical boundary conditions are
! implemented here. 

     integer, intent(in) :: i, L, N
     real, dimension(0:N), intent(in) :: X 
     real, dimension(6), intent(out) :: nn_val
     integer :: L2
     
     L2 = L*L
     nn_val = (/ X(modulo(i+1, N+1)), X(modulo(i-1, N+1)),&
                &X(modulo(i+L, N+1)), X(modulo(i-L, N+1)),& 
                &X(modulo(i+L2, N+1)), X(modulo(i-L2, N+1)) /)
  endsubroutine nn_values


  subroutine energy_diff(E0, C, xi, dx, i, X, L, N, dE)
! This function returns the energy difference between states X_k and X_k+1. Although system's energy can be calculated
! by computing the whole Phi 4 hamiltonian, and thus the energy difference can be calculated by computing H_k+1 - H_k,
! it is not necessary to do so since we only change one xi value in each Monte Carlo step. Therefore, in order to 
! reduce the computation time, the energy difference of the whole system will be calculated by taking the energy 
! difference of that randomly selected i site, i.e, by computing the substraction h_k+1 - h_k, where h (in lower case)
! denotes the value of the hamiltonian in that randomly selected i site. The exact expression of said substraction is
!
!   dE = E0*dx*( 2*xi*(2*xi**2 + 2*dx**2 + 3*xi*dx - 2) + dx*(dx**2 - 2)) + 3*C*dx**2 + C*dx*sum_{j}^{nn}(xi-xj),
!
! where xi is the value in that i site, dx is the random value that is summed in order to create the new X_k+1 state
! and sum_{j}^{nn} denotes the summation among the 6 nearest neighbours of that i site.

     integer, intent(in) :: i, L, N
     real, intent(in) :: E0, C, xi, dx
     real, dimension(0:N), intent(in) :: X
     real, intent(out) :: dE
     integer :: j
     real :: summation
     real, dimension(6) :: nn_val

     call nn_values(i, X, L, N, nn_val)

     summation = 0.0
     NearestNeigh_sum: do j = 1, 6
        summation = summation + (xi-nn_val(j))
     enddo NearestNeigh_sum

     dE = E0*dx*( 2.0*xi*(2.0*xi**2 + 2.0*dx**2 + 3.0*xi*dx - 2.0) + dx*(dx**2 - 2.0) ) + 3.0*C*dx**2 + C*dx*summation 

  endsubroutine energy_diff
        

  subroutine sweep(L, N, beta, d, C, E0, X)
! Performs one 'sweep' of the lattice, i.e., one Monte Carlo step/spin, using Metropolis algorithm.

     integer, intent(in) :: L, N
     real, intent(in) :: beta, d, C, E0
     real, dimension(0:N), intent(in out) :: X
     integer :: k, rand_int
     real :: rand, xi, dx, dE, r
     real, dimension(6) :: nn_val

     Lattice_sweep: do k = 0, N
      ! Generate random site i
        call random(rand, 0.0, real(N))
        rand_int = nint(rand)
        xi = X(rand_int)

      ! Random dx 
        call random(dx, -d, d)

      ! Decide wether to change value from xi to xi + dx: is dE positive or negative?
        call energy_diff(E0, C, xi, dx, rand_int, X, L, N, dE)
        call random_number(r)

        if (dE <= 0.0) then  
           X(rand_int) = xi + dx

        elseif (r < exp(-dE*beta) ) then 
           X(rand_int) = xi + dx

        endif
     enddo Lattice_sweep
  endsubroutine sweep


  subroutine average(sps, arr, avrg)
! It calculates the average value over all the elements contained in a given array.
  
     integer, intent(in) :: sps
     real, dimension(sps), intent(in) :: arr
     real, intent(out) :: avrg
     integer :: h

     avrg = 0 
     Summation: do h = 0, sps
        avrg = avrg + arr(h)
     enddo Summation
     avrg = avrg/sps 
  endsubroutine average

endprogram Phi4 

