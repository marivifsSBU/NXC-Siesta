program test_1d_fx_fc
  use precision, only: dp
  use xc_nn_siesta,  only: NNXC_INIT
  use xc_nn,         only: fx_forward_and_grad3, fc_forward_and_grad3
  implicit none

  integer, parameter :: n = 2001
  real(dp), parameter :: L = 5.0_dp, rho0 = 0.8_dp
  real(dp), parameter :: eps = 1.0d-5
  real(dp), parameter :: tiny = 1.0d-14

  integer :: i
  real(dp) :: x, dx, rho, drdx
  real(dp) :: Fx_u, Fx_d, Fc
  real(dp) :: d0, d1, d2
  real(dp) :: pi, kF, kF2, s
  real(dp) :: x0, x1, x2
  real(dp) :: ru, rd, rho2
  real(dp) :: gtot, gU, gD

  open(unit=10, file='fortran_1d_fx_fc.csv', status='replace', action='write')
  write(10,'(A)') 'i,x,rho,drdx,Fx_u,Fx_d,Fc'

  pi = 4.0_dp * atan(1.0_dp)

  call NNXC_INIT('nn_params/xc_nn_spin')

  dx = 2.0_dp * L / real(n-1, dp)

  do i = 1, n
     x = -L + dx * real(i-1, dp)

     rho  = rho0 * (1.0_dp - (x/L)**2)
     if (rho < 0.0_dp) rho = 0.0_dp

     if (rho > 0.0_dp) then
        drdx = -2.0_dp * rho0 * x / (L*L)
     else
        drdx = 0.0_dp
     end if

     ru = 0.5_dp * rho
     rd = ru

     gU   = 0.5_dp * abs(drdx)
     gD   = gU
     gtot = abs(drdx)

     ! Fx up
     rho2 = max(tiny, 2.0_dp*ru)
     kF2  = (3.0_dp*pi*pi*rho2)**(1.0_dp/3.0_dp)
     s    = gU / (kF2 * rho2)
     x0   = log(rho2**(1.0_dp/3.0_dp) + eps)
     x1   = log(1.0_dp + eps)
     x2   = (1.0_dp - exp(-(s*s))) * log(1.0_dp + s)
     call fx_forward_and_grad3(x0, x1, x2, Fx_u, d0, d1, d2)

     ! Fx down
     rho2 = max(tiny, 2.0_dp*rd)
     kF2  = (3.0_dp*pi*pi*rho2)**(1.0_dp/3.0_dp)
     s    = gD / (kF2 * rho2)
     x0   = log(rho2**(1.0_dp/3.0_dp) + eps)
     x1   = log(1.0_dp + eps)
     x2   = (1.0_dp - exp(-(s*s))) * log(1.0_dp + s)
     call fx_forward_and_grad3(x0, x1, x2, Fx_d, d0, d1, d2)

     ! Fc total (zeta=0)
     if (rho > tiny) then
        x0 = log(rho**(1.0_dp/3.0_dp) + eps)
        x1 = log(1.0_dp + eps)
        kF = (3.0_dp*pi*pi*rho)**(1.0_dp/3.0_dp)
        s  = gtot / (2.0_dp*kF*rho)
        x2 = (1.0_dp - exp(-(s*s))) * log(1.0_dp + s)
        call fc_forward_and_grad3(x0, x1, x2, Fc, d0, d1, d2)
     else
        Fc = 1.0_dp
     end if

     write(10,'(I6,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') &
          i, x, rho, drdx, Fx_u, Fx_d, Fc
  end do

  close(10)
  print *, "Wrote fortran_1d_fx_fc.csv"
end program test_1d_fx_fc