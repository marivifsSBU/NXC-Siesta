!======================================================================
! test_xc_nn.f90
!----------------------------------------------------------------------
! Standalone Fortran test program for the NN-GGA XC functional.
!
! PURPOSE
!   This code validates the Fortran implementation of the neural-network
!   GGA exchange--correlation functional used in SIESTA. It:
!
!     1) Loads NN parameters exported from JAX/Equinox models
!        (ASCII files xc_nn_*.dat).
!     2) Reads a spin-polarized real-space grid file with columns:
!          rho_up, rho_dn, grad_up_x, grad_up_y, grad_up_z, grad_dn_x, grad_dn_y, grad_dn_z, weight
!     3) Evaluates the NN-enhanced LDA exchange and correlation energies
!        and their analytic derivatives at each grid point.
!     4) Integrates the XC energy:
!          E_xc = sum_i w_i * e_xc(i)
!
! INPUT FILES
!   - NN parameter files (prefix controls directory/name):
!       <prefix>_fx_W*.dat, <prefix>_fx_b*.dat
!       <prefix>_fc_W*.dat, <prefix>_fc_b*.dat
!
!   - Spin grid file containing:
!       rho_up, rho_dn, gx_up, gy_up, gz_up, gx_dn, gy_dn, gz_dn, w
!
! OUTPUT
!   - Prints sample per-point values (first 10 points)
!   - Writes a full CSV for all points
!   - Prints the integrated XC energy
!
!======================================================================
program test_xc_nn
  use precision,     only: dp
  use xc_nn_siesta,  only: NNGGAXC, NNXC_INIT
  implicit none

  character(len=*), parameter :: prefix    = "nn_params/xc_nn_spin"
  character(len=*), parameter :: grid_file = "nn_rho_grad_w.dat"
  character(len=*), parameter :: out_csv   = "fortran_grid.csv"

  integer :: IREL, nSpin
  real(dp) :: EX, EC
  real(dp) :: dEXdD(2), dECdD(2)
  real(dp) :: dEXdGD(3,2), dECdGD(3,2)
  real(dp) :: vrhou, vrhod
  real(dp) :: Pu(3), Pd(3)
  real(dp) :: vsuu, vsud, vsdd
  real(dp) :: A, B, C, det
  real(dp) :: puu, pud, pdu, pdd

  real(dp) :: rho_u, rho_d, gxu, gyu, gzu, gxd, gyd, gzd, w
  real(dp) :: rho, gx, gy, gz
  real(dp) :: grad_rho
  real(dp) :: exc, edens, exc_int

  integer :: ios, linecount
  integer :: j
  real(dp) :: vals(18)
  character(len=256) :: line

  ! Initialize NN weights without requiring FDF
  call NNXC_INIT(prefix)

  IREL  = 0
  nSpin = 2

  exc_int = 0.0_dp
  linecount = 0

  open(unit=10, file=grid_file, status='old', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,*) "Error opening grid file ", trim(grid_file), " iostat=", ios
     stop
  end if

  open(unit=20, file=out_csv, status='replace', action='write', iostat=ios)
  if (ios /= 0) then
     write(*,*) "Error opening output csv ", trim(out_csv), " iostat=", ios
     stop
  end if
  write(20,'(A)') "i,rho_u,rho_d,gxu,gyu,gzu,gxd,gyd,gzd,w,rho,grad_rho,exc,edens,vrhou,vrhod,vsuu,vsud,vsdd"

  do
     read(10, '(A)', iostat=ios) line
     if (ios /= 0) exit
     if (len_trim(line) == 0) cycle
     if (line(1:1) == '#') cycle

     read(line, *, iostat=ios) rho_u, rho_d, gxu, gyu, gzu, gxd, gyd, gzd, w
     if (ios /= 0) cycle

     linecount = linecount + 1

     rho = rho_u + rho_d
     if (rho <= 1.0d-14) cycle

     gx = gxu + gxd
     gy = gyu + gyd
     gz = gzu + gzd
     grad_rho = sqrt(gx*gx + gy*gy + gz*gz)

     ! Call SIESTA-style interface (returns EX/EC per electron and derivatives of energy density)
     call NNGGAXC(IREL, nSpin, (/ rho_u, rho_d /), reshape((/ gxu, gyu, gzu, gxd, gyd, gzd /), (/3,2/)), &
                  EX, EC, dEXdD, dECdD, dEXdGD, dECdGD)

     exc   = EX + EC
     edens = rho * exc

     vrhou = dEXdD(1) + dECdD(1)
     vrhod = dEXdD(2) + dECdD(2)

     ! Convert gradient-vector derivatives into PySCF-style vsigma coefficients
     Pu = dEXdGD(:,1) + dECdGD(:,1)   ! dE/d(grad_u)
     Pd = dEXdGD(:,2) + dECdGD(:,2)   ! dE/d(grad_d)

     ! A=|Gu|^2, B=Gu·Gd, C=|Gd|^2
     A = gxu*gxu + gyu*gyu + gzu*gzu
     C = gxd*gxd + gyd*gyd + gzd*gzd
     B = gxu*gxd + gyu*gyd + gzu*gzd

     det = 2.0_dp * (A*C - B*B)
     if (abs(det) < 1.0d-20) then
        vsuu = 0.0_dp
        vsud = 0.0_dp
        vsdd = 0.0_dp
     else
        puu = Pu(1)*gxu + Pu(2)*gyu + Pu(3)*gzu
        pud = Pu(1)*gxd + Pu(2)*gyd + Pu(3)*gzd
        pdu = Pd(1)*gxu + Pd(2)*gyu + Pd(3)*gzu
        pdd = Pd(1)*gxd + Pd(2)*gyd + Pd(3)*gzd

        vsuu = ( C*puu - B*pud ) / det
        vsud = ( -2.0_dp*B*puu + 2.0_dp*A*pud ) / det
        vsdd = ( -B*pdu + A*pdd ) / det
     end if

     vals = (/ rho_u, rho_d, gxu, gyu, gzu, gxd, gyd, gzd, w, rho, grad_rho, exc, edens, vrhou, vrhod, vsuu, vsud, vsdd /)

     write(20,'(I0)', advance='no') linecount
     do j = 1, 18
        write(20,'(A,ES24.16)', advance='no') ',', vals(j)
     end do
     write(20,'(A)') ''

     if (linecount <= 10) then
        write(*,'("i=",I4," rho_u=",1PE12.5," rho_d=",1PE12.5," grad=",1PE12.5," exc=",1PE12.5, &
                  " vrhou=",1PE12.5," vrhod=",1PE12.5," vsuu=",1PE12.5," vsud=",1PE12.5," vsdd=",1PE12.5)') &
             linecount, rho_u, rho_d, grad_rho, exc, vrhou, vrhod, vsuu, vsud, vsdd
     end if

     exc_int = exc_int + w * edens
  end do

  close(10)
  close(20)

  write(*,'("Read ",I10," grid points")') linecount
  write(*,'("Fortran NN-XC integrated E_xc (spin grid) = ", F20.12)') exc_int
  write(*,'("Wrote ",A)') trim(out_csv)

end program test_xc_nn