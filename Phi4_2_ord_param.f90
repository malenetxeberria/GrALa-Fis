!
! Author: Malen Etxeberria Etxaniz (GitHub: malenetxeberria)
! 
! Last update: 31/03/2020 
!
! Description:  Generalization of a Phi 4 model with two order parameters (primary and secondary) and a bilinear
!               coupling between them. The program considers a 3D cubic lattice with helical boudary conditions, 
!               and it includes a simple measurement of both Q1 and Q2 order parameters in a temperature range
!               denoted by parameters Tf and Ti.
!
! Restrictions: cubic lattice (z=6), zero external field, "single-spin-flip" per order parameter.


 program Phi4

  ! Parameters
    integer, parameter :: L=10, sps=5E6, therm=5E5    ! Nº of sites in one dimension, Steps per site, Therm. steps
    real, parameter :: d=0.45, E01=1.0, E02=1.0       ! Width param, Energy barriers for local wells
    real, parameter :: C1=2.0, C2=2.0, g=0.5          ! Coupling constants, Bilinear local couling constant
    real, parameter :: Ti=0.0, Tf=6.0, dT=0.05        ! Initial temp, final temp, Temp. step
  ! Variables
    integer :: N, Nt                                  ! Last site's number: N=L*L*L - 1, Number of temp. loops
    real :: beta, Q1avrg, Q2avrg                      ! Inverse temp, Average Q values
    real, dimension(0:L*L*L-1) :: X1, X2              ! L*L*L array containing spins (HELICAL B.C.)
    real, dimension(sps) :: Q1arr, Q2arr              ! Array containning Q values 
  ! Dummy variables
    integer :: k, tk

    N = L*L*L - 1
    Nt = (Tf-Ti)/dT
    
    open(unit=11, file="Temp.txt", status="replace", action="write", position="append")
    Temp_sweep: do tk = 1, Nt

       call random_seed()
     ! Initialize X
       X1 = 1.0
       X2 = 0.0
       beta = 1.0/(Ti + (dT*tk))

       Thermalization: do k = 1, therm
          call sweep(L, N, beta, d, C1, E01, C2, E02, g, X1, X2)
       enddo Thermalization

       MonteCarlo_loop: do k = 1, sps
          call sweep(L, N, beta, d, C1, E01, C2, E02, g, X1, X2)
          Q1arr(k) = sum(X1) ! Qi*(N+1)
          Q2arr(k) = sum(X2)
       enddo MonteCarlo_loop

     ! Average over the last measures
       call average(sps, Q1arr, Q1avrg)
       call average(sps, Q2arr, Q2avrg)
       write(unit=11, fmt=*) Ti + (dT*tk), Q1avrg/(N+1), Q2avrg/(N+1)
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
! Returns an array containing the x values of the nearest neighbours. It assumes that the 
! lattice is cubic, and thus the number of nearest neighbours (z) is 6. Helical boundary 
! conditions are implemented here. 

     integer, intent(in) :: i, L, N
     real, dimension(0:N), intent(in) :: X 
     real, dimension(6), intent(out) :: nn_val
     integer :: L2
     
     L2 = L*L
     nn_val = (/ X(modulo(i+1, N+1)), X(modulo(i-1, N+1)),&
                &X(modulo(i+L, N+1)), X(modulo(i-L, N+1)),& 
                &X(modulo(i+L2, N+1)), X(modulo(i-L2, N+1)) /)
  endsubroutine nn_values
    

  subroutine energy_diff(i, L, N, x1i, dx1, x2i, dx2, C1, E01, C2, E02, g, X1, X2, dE)
! This function returns the energy difference between global states X_k and X_k+1. Although system's energy can 
! be calculated by computing the whole Phi 4 hamiltonian, and thus the energy difference can be calculated by 
! computing H_k+1 - H_k, it is not necessary to do so since we only change one xi value in each Monte Carlo step. 
! Therefore, in order to reduce the computation time, the energy difference of the whole system will be calculated
! by taking the energy difference of that randomly selected i site, i.e, by computing the substraction h_k+1 - h_k, 
! where h (in lower case) denotes the value of the hamiltonian in that randomly selected i site. The exact expression 
! of said substraction is
!
! dE = E01*dx1*( 2*x1i*(2*x1i**2 + 2*dx1**2 + 3*dx1*x1i - 2) + dx1*(dx1**2 - 2)) + 3*C1*dx1**2 + 
!      C1*dx1*sum_{j}^{nn}(x1i-x1j) + E02*dx2*(2*x2i + dx2) + 3*C2*dx2**2 + C2*dx2*sum_{j}^{nn}(x2i-x2j) +
!      g*( dx1*x2i + dx2*x1i + dx1*dx2),
!
! where x1i and x2i are the values corresponding to both order parameters in that i site, dx1 and dx2 are the random 
! values that are summed in order to create the new X1_k+1 and X2_k+1 states and sum_{j}^{nn} denotes the summation
! among the 6 nearest neighbours of that i site.

     integer, intent(in) :: i, L, N
     real, intent(in) :: x1i, dx1, x2i, dx2, C1, E01, C2, E02, g
     real, dimension(0:N), intent(in out) :: X1, X2
     real, intent(out) :: dE
     integer :: j
     real :: sum1, sum2
     real, dimension(6) :: nn_val1, nn_val2

     call nn_values(i, X1, L, N, nn_val1)
     call nn_values(i, X2, L, N, nn_val2)

     sum1 = 0.0
     sum2 = 0.0
     NearestNeigh_sum: do j = 1,6
        sum1 = sum1 + (x1i-nn_val1(j))
        sum2 = sum2 + (x2i-nn_val2(j))
     enddo NearestNeigh_sum

     dE = E01*dx1*( 2.0*x1i*(2.0*x1i**2 + 2.0*dx1**2 + 3.0*x1i*dx1 - 2.0) + dx1*(dx1**2 - 2.0) ) + 3.0*C1*dx1**2 +&
          &C1*dx1*sum1 + E02*dx2*(2.0*x2i + dx2) + 3.0*C2*dx2**2 + C2*dx2*sum2 + g*( dx1*x2i + dx2*x1i + dx1*dx2 )

  endsubroutine energy_diff
        

  subroutine sweep(L, N, beta, d, C1, E01, C2, E02, g, X1, X2)
! "Performs one 'sweep' of the lattice, i.e., one Monte Carlo step/spin, using Metropolis algorithm."

     integer, intent(in) :: L, N
     real, intent(in) :: beta, d, C1, E01, C2, E02, g
     real, dimension(0:N), intent(in out) :: X1, X2
     integer :: k, rand_int
     real :: rand, x1i, x2i, dx1, dx2, dE, r
     real, dimension(6) :: nn_val

     Lattice_sweep: do k = 0, N
      ! Generate random site i
        call random(rand, 0.0, real(N))
        rand_int = nint(rand)
        x1i = X1(rand_int)
        x2i = X2(rand_int)

      ! Random dx1 and dx2
        call random(dx1, -d, d)
        call random(dx2, -d, d)

      ! Decide wether to change value from xi to xi + dx: dE positive or negative?
        call energy_diff(rand_int, L, N, x1i, dx1, x2i, dx2, C1, E01, C2, E02, g, X1, X2, dE)
        call random_number(r)

        if (dE <= 0.0) then  
           X1(rand_int) = x1i + dx1
           X2(rand_int) = x2i + dx2

        elseif (r < exp(-dE*beta) ) then
           X1(rand_int) = x1i + dx1
           X2(rand_int) = x2i + dx2

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

