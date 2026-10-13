program test_1d_nngga
! Standalone 1D tester for SIESTA's xc_nn_siesta::NNGGAXC.
!
! Compile/link (example):
! gfortran -O2 \
!   -I${SIESTA_OBJ} \
!   -I${SIESTA_OBJ}/SiestaXC \
!   -I${SIESTA_OBJ}/fdf \
!   test_1d_nngga.f90 \
!   ${SIESTA_OBJ}/io.o \
!   -L${SIESTA_OBJ} \
!   -lSiestaXC -lfdf -o test_1d_nngga
!
! Run:
!   ./test_1d_nngga
! Notes:
!   This program calls NNXC_INIT(...) to load weights without requiring FDF.

  use precision,     only: dp
  use xc_nn_siesta,  only: NNGGAXC, NNXC_INIT
  implicit none

  integer, parameter :: n = 2001
  real(dp), parameter :: L = 5.0_dp
  real(dp), parameter :: rho0 = 0.8_dp
  real(dp), parameter :: dx = 2.0_dp*L / real(n-1, dp)
  real(dp), parameter :: tiny = 1.0d-14

  integer :: i, IREL, nSpin
  real(dp) :: x, rho, drdx
  real(dp) :: Dens(1), GDens(3,1)
  real(dp) :: EX, EC
  real(dp) :: dEXdD(1), dECdD(1)
  real(dp) :: dEXdGD(3,1), dECdGD(3,1)
  real(dp) :: edens, v_rho, v_gradx
  real(dp) :: Eint

  IREL  = 0
  nSpin = 1

  ! Load NN weights without FDF. Choose the desired model prefix:
  call NNXC_INIT('nn_params/xc_nn_spin')
  ! For an unpolarized model instead, use:
  ! call NNXC_INIT('nn_params/xc_nn')

  open(unit=10, file="fortran_1d_nngga.csv", status="replace", action="write")
  write(10,'(A)') "i,x,rho,drdx,EX,EC,edens,vrho,vgradx"

  Eint = 0.0_dp

  do i = 1, n
     x = -L + dx*real(i-1, dp)

     ! Parabola rho(x) = rho0*(1 - (x/L)^2), clipped at 0
     rho  = rho0 * (1.0_dp - (x/L)**2)
     if (rho < 0.0_dp) rho = 0.0_dp

     ! drho/dx = -2*rho0*x/L^2 (set to 0 where clipped)
     if (rho > 0.0_dp) then
        drdx = -2.0_dp*rho0*x/(L*L)
     else
        drdx = 0.0_dp
     end if

     Dens(1) = rho
     GDens(:,1) = 0.0_dp
     GDens(1,1) = drdx

     call NNGGAXC(IREL, nSpin, Dens, GDens, EX, EC, dEXdD, dECdD, dEXdGD, dECdGD)

     edens   = rho * (EX + EC)
     v_rho   = dEXdD(1) + dECdD(1)
     v_gradx = dEXdGD(1,1) + dECdGD(1,1)

     write(10,'(I6,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') &
          i, x, rho, drdx, EX, EC, edens, v_rho, v_gradx

     if (i == 1 .or. i == n) then
        Eint = Eint + 0.5_dp * edens * dx
     else
        Eint = Eint + edens * dx
     end if
  end do

  close(10)
  write(*,'("Wrote fortran_1d_nngga.csv; 1D integral E_xc=",ES24.16)') Eint

end program test_1d_nngga